// Render GPU audio through CoreAudio and report latency and underruns.
//
//   GpuAudio [frames] [seconds] [depth] [keepalive_ms] [voices]
//
// Frames sets device cadence; depth sets lookahead; keepalive prevents GPU idle; voices sets load.
// GPU_AUDIO_FEEDBACK=1 adds GPU timestamps and one allocation per commit.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <CoreAudio/CoreAudio.h>
#include <Metal/Metal.hpp>
#include <QuartzCore/CABase.h>
#include <mach/mach_init.h>
#include <mach/semaphore.h>
#include <mach/mach_time.h>
#include <mach/task.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <os/workgroup.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <format>
#include <print>
#include <ranges>
#include <span>
#include <thread>
#include <vector>

// Avoid allocating an NSAutoreleasePool object for every dispatch.
extern "C" void *objc_autoreleasePoolPush();
extern "C" void objc_autoreleasePoolPop(void *);

namespace {
constexpr auto SineSource = R"(
#include <metal_stdlib>
using namespace metal;

struct Params { float PhaseInc, Phase0; uint Voices; };

kernel void Sine(
    device float *out [[buffer(0)]], constant Params &p [[buffer(1)]],
    uint frame [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr float Tau = 6.283185307179586f;
    float sum = 0;
    for (uint v = lane; v < p.Voices; v += lanes) {
        const float voice = float(v + 1);
        sum += sin(fma(p.PhaseInc * voice, float(frame), fmod(p.Phase0 * voice, Tau)));
    }

    threadgroup float partials[32]; // one per simdgroup, and a threadgroup is at most 1024 threads
    sum = simd_sum(sum);
    if (simd_lane == 0) partials[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float total = 0;
        for (uint k = 0; k < (lanes + 31) / 32; ++k) total += partials[k];
        out[frame] = total / float(p.Voices);
    }
}
)";

struct Params { float PhaseInc, Phase0; uint32_t Voices; };

// Seconds. Committed is absolute; Commit covers queue calls; Gap exposes descheduling.
struct Timings { double Committed, Encode, Commit, Submit, Exec, Done, Gap; };

constexpr uint32_t SlotCount = 8;
constexpr uint32_t WarmSlots = 4; // guarded by a completion event before reuse
constexpr double MinComputation = 50e-6; // minimum accepted realtime budget
constexpr double RenderWork = 50e-6; // encode, submit, and filler CPU work
constexpr int WarmupBlocks = 100;
constexpr double Tau = 6.283185307179586;
constexpr double Amplitude = 0.2;
constexpr double FadeSeconds = 0.01;
constexpr double TailSeconds = 0.2; // drain the device pipeline after fading out
constexpr float PhaseInc = 0.01f; // radians per sample, about 76 Hz at 48 kHz

// Shared by Metal feedback and CoreAudio host timestamps.
double Now() { return CACurrentMediaTime(); }

// Drain autoreleased encoders after each dispatch.
struct AutoreleasePool {
    void *Token = objc_autoreleasePoolPush();
    ~AutoreleasePool() { objc_autoreleasePoolPop(Token); }
};

// Wake on consumption or the next keepalive deadline.
void WaitFor(semaphore_t semaphore, double seconds) {
    seconds = std::max(0., seconds);
    const auto whole = uint32_t(seconds);
    const mach_timespec_t timeout{whole, int32_t((seconds - whole) * 1e9)};
    semaphore_timedwait(semaphore, timeout);
}

void WaitUntil(MTL::SharedEvent *event, uint64_t value) { while (!event->waitUntilSignaledValue(value, 1000)) {} }

struct Semaphore {
    semaphore_t Value{SEMAPHORE_NULL};
    Semaphore() { semaphore_create(mach_task_self(), &Value, SYNC_POLICY_FIFO, 0); }
    ~Semaphore() { if (Value != SEMAPHORE_NULL) semaphore_destroy(mach_task_self(), Value); }
    Semaphore(const Semaphore &) = delete;
    Semaphore &operator=(const Semaphore &) = delete;
};

struct WorkgroupMembership {
    os_workgroup_t Group;
    os_workgroup_join_token_s Token{};
    bool Joined;

    explicit WorkgroupMembership(os_workgroup_t group)
        : Group(group), Joined(Group && os_workgroup_join(Group, &Token) == 0) {}
    ~WorkgroupMembership() { if (Joined) os_workgroup_leave(Group, &Token); }
};

struct CommandSlot {
    NS::SharedPtr<MTL4::CommandAllocator> Allocator;
    NS::SharedPtr<MTL4::CommandBuffer> Buffer;
};

template<typename T> NS::SharedPtr<T> Make() { return NS::TransferPtr(T::alloc()->init()); }

std::vector<CommandSlot> MakeCommandSlots(MTL::Device *device, uint32_t count) {
    std::vector<CommandSlot> slots;
    slots.reserve(count);
    while (slots.size() < count)
        slots.push_back({NS::TransferPtr(device->newCommandAllocator()), NS::TransferPtr(device->newCommandBuffer())});
    return slots;
}

void Encode(MTL4::CommandAllocator *allocator, MTL4::CommandBuffer *buffer, MTL4::ArgumentTable *table,
            MTL::ComputePipelineState *pipeline, MTL::Size groups, MTL::Size threads) {
    allocator->reset();
    buffer->beginCommandBuffer(allocator);
    auto *encoder = buffer->computeCommandEncoder();
    encoder->setArgumentTable(table);
    encoder->setComputePipelineState(pipeline);
    encoder->dispatchThreadgroups(groups, threads);
    encoder->endEncoding();
    buffer->endCommandBuffer();
}

void Commit(MTL4::CommandQueue *queue, MTL4::CommandBuffer *buffer, MTL::SharedEvent *event,
            uint64_t value, MTL4::CommitOptions *options = nullptr) {
    const MTL4::CommandBuffer *list[]{buffer};
    if (options) queue->commit(list, 1, options);
    else queue->commit(list, 1);
    queue->signalEvent(event, value);
}

// State shared with the IO thread.
struct Ring {
    const float *Samples{};
    uint32_t Frames{}, SampleStride{};
    uint64_t FadeFrames{}, TotalFrames{}; // transport envelope, so the kernel stays pure synthesis
    semaphore_t RenderWake{SEMAPHORE_NULL};
    bool Measure{};

    // Separate producer and consumer counters to avoid cache-line contention.
    alignas(128) std::atomic<uint64_t> Written{0};
    alignas(128) std::atomic<uint64_t> Read{0};
    uint64_t Underruns{0}, MinDepth{SlotCount}; // only the IO thread writes; read after AudioDeviceStop

    // Ramp transport edges without changing the synthesis kernel.
    double Gain(uint64_t frame) const {
        const auto edge = std::min(frame, TotalFrames > frame ? TotalFrames - frame : 0);
        return Amplitude * std::min(1., double(edge) / double(FadeFrames));
    }
};

// Specialize steady/ramped mono and stereo copies so their inner loops stay branch-free.
template<bool Ramp>
inline void CopySamples(const float *__restrict block, float *__restrict dst, UInt32 frames,
                        UInt32 channels, float gain, float step) {
    const auto sample = [&](UInt32 f) {
        return (Ramp ? gain + step * float(f) : gain) * block[f];
    };
    if (channels == 1) {
        for (UInt32 f = 0; f < frames; ++f) dst[f] = sample(f);
    } else if (channels == 2) {
        for (UInt32 f = 0; f < frames; ++f) {
            const float value = sample(f);
            dst[2 * f] = value;
            dst[2 * f + 1] = value;
        }
    } else {
        for (UInt32 f = 0; f < frames; ++f) {
            const float value = sample(f);
            for (UInt32 c = 0; c < channels; ++c) dst[f * channels + c] = value;
        }
    }
}

inline void CopyBlock(const float *__restrict block, float *__restrict dst, UInt32 frames,
                      UInt32 channels, float gain, float step) {
    if (step == 0) CopySamples<false>(block, dst, frames, channels, gain, step);
    else CopySamples<true>(block, dst, frames, channels, gain, step);
}

// Realtime context. No allocation, no locks, no logging, no Metal.
OSStatus Render(AudioObjectID, const AudioTimeStamp *, const AudioBufferList *, const AudioTimeStamp *,
                AudioBufferList *out, const AudioTimeStamp *, void *context) {
    auto &ring = *static_cast<Ring *>(context);
    const auto written = ring.Written.load(std::memory_order_acquire), read = ring.Read.load(std::memory_order_relaxed);
    const auto depth = written - read;
    if (ring.Measure && depth < ring.MinDepth) ring.MinDepth = depth;

    if (depth == 0) { // HAL zeroes every output buffer before entry, so doing nothing emits silence
        ++ring.Underruns;
        semaphore_signal(ring.RenderWake);
        return noErr;
    }

    const float *block = ring.Samples + (read % SlotCount) * ring.SampleStride;
    // Keep the transport envelope outside the measured synthesis workload.
    const auto position = read * ring.Frames;
    const float gain = float(ring.Gain(position)), step = float(ring.Gain(position + ring.Frames) - gain) / float(ring.Frames);
    for (UInt32 b = 0; b < out->mNumberBuffers; ++b) {
        auto *dst = static_cast<float *>(out->mBuffers[b].mData);
        const auto channels = out->mBuffers[b].mNumberChannels;
        const auto wanted = uint32_t(out->mBuffers[b].mDataByteSize / (sizeof(float) * channels));
        const auto frames = std::min(ring.Frames, wanted);
        CopyBlock(block, dst, frames, channels, gain, step);
        // If a device asks for more, the untouched tail retains the silence supplied by HAL.
    }
    ring.Read.store(read + 1, std::memory_order_release);
    semaphore_signal(ring.RenderWake);
    return noErr;
}

// Keep producer scheduling inside the audio deadline.
bool RequestRealtime(double period_seconds) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    const auto ticks = [&](double s) { return uint32_t(s * 1e9 * double(tb.denom) / double(tb.numer)); };
    // Budget only active CPU work; values below 50 us are rejected.
    const double computation = std::clamp(RenderWork, MinComputation, period_seconds);
    thread_time_constraint_policy_data_t policy{
        .period = ticks(period_seconds), .computation = ticks(computation),
        .constraint = ticks(std::max(computation, period_seconds / 2)), .preemptible = 1};
    return thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                             reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT) == KERN_SUCCESS;
}

double Pct(const std::vector<double> &sorted, double p) { return sorted[size_t(p * double(sorted.size() - 1))] * 1e6; } // microseconds

void Report(std::string_view name, std::span<const Timings> timings, double Timings::*field) {
    auto v = timings | std::views::transform(field) | std::ranges::to<std::vector>();
    std::ranges::sort(v);
    std::println("{:<22} p50={:8.1f}  p90={:8.1f}  p99={:8.1f}  p99.9={:8.1f}  max={:9.1f}  (us)", name,
                 Pct(v, .5), Pct(v, .9), Pct(v, .99), Pct(v, .999), Pct(v, 1.));
}

// Reads a CoreAudio scalar property, leaving the value untouched if the device does not answer.
template<typename T> T DeviceProperty(AudioObjectID device, AudioObjectPropertySelector selector, T fallback = {},
                                      AudioObjectPropertyScope scope = kAudioObjectPropertyScopeOutput) {
    AudioObjectPropertyAddress address{selector, scope, kAudioObjectPropertyElementMain};
    UInt32 size = sizeof(T);
    T value{};
    return AudioObjectGetPropertyData(device, &address, 0, nullptr, &size, &value) == noErr ? value : fallback;
}

// Stream processing latency is additive to the device latency.
UInt32 StreamLatency(AudioObjectID device) {
    AudioObjectPropertyAddress address{kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(device, &address, 0, nullptr, &size) != noErr || size == 0) return 0;
    std::vector<AudioStreamID> streams(size / sizeof(AudioStreamID));
    if (AudioObjectGetPropertyData(device, &address, 0, nullptr, &size, streams.data()) != noErr || streams.empty()) return 0;
    return DeviceProperty<UInt32>(streams[0], kAudioStreamPropertyLatency, 0, kAudioObjectPropertyScopeGlobal);
}
} // namespace

int main(int argc, char **argv) {
    const std::span args{argv, size_t(argc)};
    // Lowest setting that remained clean across long runs.
    const uint32_t frames = args.size() > 1 ? uint32_t(std::atoi(args[1])) : 32;
    const double seconds = args.size() > 2 ? std::atof(args[2]) : 10;
    const int depth = std::clamp(args.size() > 3 ? std::atoi(args[3]) : 2, 1, int(SlotCount) - 1);
    // A depth-two ring absorbs the cold-GPU wake and does not need filler work. At depth one there
    // is only a single period to finish, so keep the GPU awake unless the CLI explicitly overrides it.
    const double keepalive = args.size() > 4 ? std::atof(args[4]) * 1e-3 : (depth == 1 ? 0.25e-3 : 0.); // seconds
    const uint32_t voices = args.size() > 5 ? uint32_t(std::atoi(args[5])) : 1;
    // Production mode removes timing storage, clocks, feedback, and callback diagnostics.
    const bool measure = std::getenv("GPU_AUDIO_PRODUCTION") == nullptr;
    // GPU feedback allocates per commit, so enable it only for measurement.
    const bool want_feedback = measure && std::getenv("GPU_AUDIO_FEEDBACK") != nullptr;

    // Audio device first, since it decides the sample rate and the block size we actually get.
    const auto audio_device = DeviceProperty<AudioObjectID>(
        kAudioObjectSystemObject, kAudioHardwarePropertyDefaultOutputDevice, 0, kAudioObjectPropertyScopeGlobal);
    if (audio_device == 0)
        return std::println(stderr, "No default output device."), 1;

    AudioObjectPropertyAddress address{kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeOutput,
                                       kAudioObjectPropertyElementMain};
    UInt32 requested = frames;
    AudioObjectSetPropertyData(audio_device, &address, 0, nullptr, sizeof(requested), &requested);
    const auto block_frames = DeviceProperty<UInt32>(audio_device, kAudioDevicePropertyBufferFrameSize, frames);
    const auto sample_rate = DeviceProperty<Float64>(audio_device, kAudioDevicePropertyNominalSampleRate, 48000);
    const auto safety = DeviceProperty<UInt32>(audio_device, kAudioDevicePropertySafetyOffset);
    const auto device_latency = DeviceProperty<UInt32>(audio_device, kAudioDevicePropertyLatency);
    const auto stream_latency = StreamLatency(audio_device);
    const double block_period = double(block_frames) / sample_rate;
    // Keep GPU writes and CoreAudio reads in disjoint 128-byte cache lines even below 32 frames.
    const uint32_t sample_stride = (block_frames + 31) & ~31u;

    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    NS::Error *error{};
    auto library = NS::TransferPtr(device->newLibrary(NS::String::string(SineSource, NS::UTF8StringEncoding), nullptr, &error));
    if (!library) return std::println(stderr, "Shader: {}", error->localizedDescription()->utf8String()), 1;

    // Compiling at runtime avoids needing Xcode's Metal toolchain, which is a separate download.
    auto compiler = NS::TransferPtr(device->newCompiler(Make<MTL4::CompilerDescriptor>().get(), &error));
    if (!compiler) return std::println(stderr, "Compiler: {}", error->localizedDescription()->utf8String()), 1;
    auto function = Make<MTL4::LibraryFunctionDescriptor>();
    function->setName(MTLSTR("Sine"));
    function->setLibrary(library.get());
    auto pipeline_desc = Make<MTL4::ComputePipelineDescriptor>();
    pipeline_desc->setComputeFunctionDescriptor(function.get());
    auto pipeline = NS::TransferPtr(compiler->newComputePipelineState(pipeline_desc.get(), nullptr, &error));
    if (!pipeline) return std::println(stderr, "Pipeline: {}", error->localizedDescription()->utf8String()), 1;

    const bool warm_enabled = keepalive > 0;
    const uint32_t resource_slots = SlotCount + uint32_t(warm_enabled);
    auto samples = NS::TransferPtr(device->newBuffer(sample_stride * resource_slots * sizeof(float), MTL::ResourceStorageModeShared));
    auto params = NS::TransferPtr(device->newBuffer(resource_slots * sizeof(Params), MTL::ResourceStorageModeShared));

    // Metal 4 has no implicit residency tracking, so resources go in a set attached to the queue.
    auto residency = NS::TransferPtr(device->newResidencySet(Make<MTL::ResidencySetDescriptor>().get(), &error));
    residency->addAllocation(samples.get());
    residency->addAllocation(params.get());
    residency->commit();
    residency->requestResidency();

    auto table_desc = Make<MTL4::ArgumentTableDescriptor>();
    table_desc->setMaxBufferBindCount(2);
    auto table = NS::TransferPtr(device->newArgumentTable(table_desc.get(), &error));
    auto queue = NS::TransferPtr(device->newMTL4CommandQueue(Make<MTL4::CommandQueueDescriptor>().get(), &error));
    queue->addResidencySet(residency.get());

    // Four-way rotation avoids the allocator-reset penalty at ordinary depths. Larger requested
    // depths get one command slot per in-flight block; the sample ring remains independently eight.
    const auto command_slot_count = uint32_t(std::max(4, depth));
    auto command_slots = MakeCommandSlots(device.get(), command_slot_count);

    // Cache invariant Objective-C properties outside the render loop.
    auto *slot_params = static_cast<Params *>(params->contents());
    const auto samples_address = samples->gpuAddress(), params_address = params->gpuAddress();
    const auto max_threads = uint32_t(pipeline->maxTotalThreadsPerThreadgroup());
    const auto threads_per_group = std::clamp((voices + 31) / 32 * 32, 32u, max_threads);
    const auto bind_slot = [&](uint32_t slot) {
        table->setAddress(samples_address + slot * sample_stride * sizeof(float), 0);
        table->setAddress(params_address + slot * sizeof(Params), 1);
    };

    // Queue signalling safely publishes GPU writes to the CPU; kernel-written flags do not.
    auto ready_event = NS::TransferPtr(device->newSharedEvent());

    Semaphore render_wake;
    if (render_wake.Value == SEMAPHORE_NULL)
        return std::println(stderr, "Could not create synchronization semaphore."), 1;
    Ring ring{.Samples = static_cast<const float *>(samples->contents()), .Frames = block_frames, .SampleStride = sample_stride,
              .FadeFrames = uint64_t(FadeSeconds * sample_rate), .TotalFrames = uint64_t(seconds * sample_rate),
              .RenderWake = render_wake.Value, .Measure = measure};
    double phase = 0; // advanced in double and wrapped, so float never sees a large angle

    // Isolate filler resources and create them only when enabled.
    if (warm_enabled) slot_params[SlotCount] = {.PhaseInc = 0, .Phase0 = 0, .Voices = 1};
    auto warm_slots = MakeCommandSlots(device.get(), warm_enabled ? WarmSlots : 0);
    NS::SharedPtr<MTL::SharedEvent> warm_event;
    if (warm_enabled) warm_event = NS::TransferPtr(device->newSharedEvent());
    uint64_t warm_index = 0;
    const auto keep_warm = [&] {
        const AutoreleasePool pool;
        // A slow filler is no reason to block audio or reset an in-flight allocator. Skipping one
        // cadence lets the queue catch up and bounds this pool without relying on an observed tail.
        if (warm_index >= WarmSlots && warm_event->signaledValue() < warm_index - WarmSlots + 1) return;
        const auto warm_slot = warm_index++ % WarmSlots;
        auto &commands = warm_slots[warm_slot];
        bind_slot(SlotCount);
        Encode(commands.Allocator.get(), commands.Buffer.get(), table.get(), pipeline.get(), {1, 1, 1}, {1, 1, 1});
        Commit(queue.get(), commands.Buffer.get(), warm_event.get(), warm_index);
    };

    // Cover the silent tail so vectors never move under feedback handlers.
    const auto max_blocks = size_t((seconds + TailSeconds) / block_period) + size_t(depth) + 64;
    // Fixed storage lets both threads write separate entries without mutating vector bookkeeping.
    std::vector<Timings> timings(measure ? max_blocks : 0);
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> feedback_completed{0};
    std::atomic<bool> stop_completion{false};

    AudioDeviceIOProcID proc{};
    if (AudioDeviceCreateIOProcID(audio_device, Render, &ring, &proc) != noErr)
        return std::println(stderr, "Could not create IOProc."), 1;

    // Join the device's workgroup so the scheduler knows this thread shares the audio deadline.
    const auto workgroup = DeviceProperty<os_workgroup_t>(audio_device, kAudioDevicePropertyIOThreadOSWorkgroup);
    const WorkgroupMembership producer_membership(workgroup);
    const bool realtime = RequestRealtime(block_period);

    // Publish completions independently so the producer can keep submitting across a long GPU tail.
    bool completion_realtime{}, completion_workgroup{};
    std::thread completion_thread([&] {
        const AutoreleasePool pool;
        const WorkgroupMembership membership(workgroup);
        completion_workgroup = membership.Joined;
        completion_realtime = RequestRealtime(block_period);
        uint64_t done = 0;
        for (;;) {
            // A future timeline value can be awaited before submission, so the event also announces
            // work. Shutdown advances it once to release the final wait.
            WaitUntil(ready_event.get(), done + 1);
            if (stop_completion.load(std::memory_order_acquire) &&
                done == submitted.load(std::memory_order_acquire)) break;
            if (measure) timings[done].Done = Now() - timings[done].Committed;
            ring.Written.store(++done, std::memory_order_release);
        }
    });

    const auto stop_completion_thread = [&] {
        stop_completion.store(true, std::memory_order_release);
        ready_event->setSignaledValue(submitted.load(std::memory_order_relaxed) + 1);
        completion_thread.join();
    };

    // Fill the ring before starting the device, so the first blocks are not an underrun by construction.
    const auto submit_block = [&](uint64_t index) {
        const AutoreleasePool pool;
        const uint32_t slot = uint32_t(index % command_slot_count);
        auto &commands = command_slots[slot];
        auto *command_buffer = commands.Buffer.get();
        Timings *timing = measure ? &timings[index] : nullptr;
        const double started = measure ? Now() : 0;

        // Priming can submit more blocks than the steady-state depth. Do not recycle command memory
        // until the queue signal after its previous use has landed.
        if (index >= command_slot_count)
            WaitUntil(ready_event.get(), index - command_slot_count + 1);

        // Commit consumes its options, so feedback needs a fresh object.
        NS::SharedPtr<MTL4::CommitOptions> options;
        if (want_feedback) {
            options = Make<MTL4::CommitOptions>();
            options->addFeedbackHandler([timing, &feedback_completed](MTL4::CommitFeedback *feedback) {
                timing->Submit = feedback->GPUStartTime(); // raw stamps, converted once the run is over
                timing->Exec = feedback->GPUEndTime();
                feedback_completed.fetch_add(1, std::memory_order_release);
            });
        }

        const uint32_t sample_slot = uint32_t(index % SlotCount);
        slot_params[sample_slot] = {.PhaseInc = PhaseInc, .Phase0 = float(phase), .Voices = voices};
        phase = std::fmod(phase + PhaseInc * block_frames, Tau);
        bind_slot(sample_slot);
        Encode(commands.Allocator.get(), command_buffer, table.get(), pipeline.get(),
               {block_frames, 1, 1}, {threads_per_group, 1, 1});

        if (measure) {
            timing->Committed = Now();
            timing->Encode = timing->Committed - started;
            timing->Gap = index > 0 ? timing->Committed - timings[index - 1].Committed : 0;
        }
        Commit(queue.get(), command_buffer, ready_event.get(), index + 1, options.get());
        if (measure) timing->Commit = Now() - timing->Committed;
        submitted.store(index + 1, std::memory_order_release);
    };

    // Before start and during teardown, await both GPU completion and CPU publication.
    const auto wait_until_published = [&](uint64_t target) {
        WaitUntil(ready_event.get(), target);
        while (ring.Written.load(std::memory_order_acquire) < target) std::this_thread::yield();
    };

    // Prefill CoreAudio's startup pull; surplus blocks drain back to the requested depth.
    const uint64_t priming = (safety + device_latency + block_frames - 1) / block_frames;
    const uint64_t prefill = std::min<uint64_t>(uint64_t(depth) + priming, SlotCount - 1);
    for (uint64_t i = 0; i < prefill; ++i) submit_block(i);
    wait_until_published(prefill);
    if (AudioDeviceStart(audio_device, proc) != noErr) {
        stop_completion_thread();
        AudioDeviceDestroyIOProcID(audio_device, proc);
        return std::println(stderr, "Could not start device."), 1;
    }

    // Stay `depth` submitted blocks ahead. CoreAudio wakes this thread when it consumes a block; a
    // semaphore timeout is used only to meet the keepalive cadence or stop at the requested duration.
    const double finish = Now() + seconds + TailSeconds; // Gain is zero during the drain tail.
    double last_warm = Now();
    for (uint64_t next = prefill; next < max_blocks;) {
        const double now = Now();
        if (now >= finish) break;
        if (next - ring.Read.load(std::memory_order_acquire) < uint64_t(depth)) {
            submit_block(next++);
            continue;
        }
        if (keepalive > 0 && now - last_warm >= keepalive) {
            keep_warm();
            last_warm = Now();
            continue;
        }
        if (keepalive > 0) {
            const double until_warm = keepalive - (now - last_warm);
            WaitFor(render_wake.Value, std::min(finish - now, until_warm));
        } else {
            // The running IOProc signals once per period, including on underrun, so no deadline timer
            // is needed when there is no filler cadence to service.
            semaphore_wait(render_wake.Value);
        }
    }

    wait_until_published(submitted.load(std::memory_order_acquire));
    while (ring.Read.load(std::memory_order_acquire) < ring.Written.load(std::memory_order_acquire))
        semaphore_wait(render_wake.Value); // let the fade reach the device
    AudioDeviceStop(audio_device, proc);
    AudioDeviceDestroyIOProcID(audio_device, proc);
    stop_completion_thread();
    const auto block_count = size_t(submitted.load(std::memory_order_acquire));
    if (measure) timings.resize(block_count);
    if (want_feedback) {
        while (feedback_completed.load(std::memory_order_acquire) < timings.size()) {} // GPU timestamps arrive on the feedback queue
        for (auto &timing : timings) {
            timing.Exec -= timing.Submit; // GPU end minus GPU start, while Submit is still a raw time
            timing.Submit -= timing.Committed;
        }
    }

    // Separate code-controlled render latency from the output device path.
    const double render_latency = double(block_frames) / sample_rate * (1 + depth);
    const double device_latency_seconds = double(safety + device_latency + stream_latency) / sample_rate;
    const auto keepalive_label = keepalive > 0 ? std::format("{:.2f} ms", keepalive * 1e3) : "off";
    std::println("--- {} frames ({:.3f} ms @ {:g} Hz), {} voices, depth {}, keepalive {}, {:.0f}s, {}, realtime {}/{}, workgroup {}/{} ---",
                 block_frames, block_period * 1e3, sample_rate, voices, depth, keepalive_label, seconds,
                 measure ? "metrics" : "production",
                 realtime ? "yes" : "NO", completion_realtime ? "yes" : "NO",
                 producer_membership.Joined ? "yes" : "NO", completion_workgroup ? "yes" : "NO");
    std::println("render latency {:.2f} ms = {} block + {} lookahead frames, which is what this code sets\n"
                 "  the output device adds {:.2f} ms ({} safety + {} device + {} stream frames), which it does not,\n"
                 "  for {:.2f} ms out of the speakers on this machine",
                 render_latency * 1e3, block_frames, depth * block_frames, device_latency_seconds * 1e3,
                 safety, device_latency, stream_latency, (render_latency + device_latency_seconds) * 1e3);
    if (measure) std::println("blocks {}, underruns {}, min ring depth {} of {}",
                              block_count, ring.Underruns, ring.MinDepth, depth);
    else std::println("blocks {}, underruns {}", block_count, ring.Underruns);

    if (!measure) return 0;

    const auto steady = std::span<const Timings>(timings).subspan(std::min<size_t>(WarmupBlocks, timings.size()));
    if (steady.empty()) return 0;
    Report("cpu encode", steady, &Timings::Encode);
    Report("cpu commit+signal", steady, &Timings::Commit);
    if (want_feedback) {
        Report("commit -> GPU start", steady, &Timings::Submit);
        Report("GPU exec", steady, &Timings::Exec);
    } else {
        std::println("commit -> GPU start and GPU exec need GPU_AUDIO_FEEDBACK=1, which allocates per commit");
    }
    Report("commit -> readable", steady, &Timings::Done);
    Report("gap between commits", steady, &Timings::Gap);
    const double completion_deadline = block_period * depth;
    std::println("headroom: a tail block must be ready in {:.2f} ms at depth {}, worst was {:.2f} ms",
                 completion_deadline * 1e3, depth,
                 std::ranges::max(steady | std::views::transform(&Timings::Done)) * 1e3);
    return 0;
}
