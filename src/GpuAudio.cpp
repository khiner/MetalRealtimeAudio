// Render audio blocks on the GPU, play them through CoreAudio, and report latency and underruns.
//
//   GpuAudio [frames] [seconds] [depth] [keepalive_ms] [voices]
//
// Frames sets the device cadence; depth sets lookahead. Keepalive prevents GPU idle, and voices
// controls the synthesis load summed by each frame's threadgroup.
//
// Completion uses a queue-signalled MTLSharedEvent. GPU_AUDIO_FEEDBACK=1 adds GPU timestamps at the
// cost of one allocation per commit.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <CoreAudio/CoreAudio.h>
#include <Metal/Metal.hpp>
#include <QuartzCore/CABase.h>
#include <mach/mach_init.h>
#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <os/workgroup.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <format>
#include <print>
#include <ranges>
#include <span>
#include <vector>

// Avoid allocating an NSAutoreleasePool object for every dispatch.
extern "C" void *objc_autoreleasePoolPush();
extern "C" void objc_autoreleasePoolPop(void *);

namespace {
constexpr auto SineSource = R"(
#include <metal_stdlib>
using namespace metal;

struct Params { float PhaseInc, Phase0; uint Seq, Voices; };

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

struct Params { float PhaseInc, Phase0; uint32_t Seq, Voices; };

// Per-block seconds. Commit covers queue calls; Gap exposes producer descheduling.
struct Timings { double Encode, Commit, Submit, Exec, Done, Gap; };

constexpr uint32_t SlotCount = 8;
constexpr uint32_t WarmSlots = 16; // keep allocator reuse beyond the observed completion tail
constexpr double IdleQuantum = 60e-6;
constexpr double MinComputation = 50e-6; // minimum accepted realtime budget
constexpr double RenderWork = 200e-6; // encode plus p99 GPU round trip
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

// Avoid the timer slop of relative sleeps.
void SleepUntil(double from_now) {
    static const auto tb = [] { mach_timebase_info_data_t t; mach_timebase_info(&t); return t; }();
    mach_wait_until(mach_absolute_time() + uint64_t(from_now * 1e9 * double(tb.denom) / double(tb.numer)));
}

// State shared with the IO thread.
struct Ring {
    const float *Samples{};
    uint32_t Frames{};
    uint64_t FadeFrames{}, TotalFrames{}; // transport envelope, so the kernel stays pure synthesis

    // Separate producer and consumer counters to avoid cache-line contention.
    alignas(128) std::atomic<uint64_t> Written{0};
    alignas(128) std::atomic<uint64_t> Read{0};
    std::atomic<uint64_t> Underruns{0}; // IO-thread writers share Read's line
    std::atomic<uint64_t> MinDepth{SlotCount};

    // Ramp transport edges without changing the synthesis kernel.
    double Gain(uint64_t frame) const {
        const auto edge = std::min(frame, TotalFrames > frame ? TotalFrames - frame : 0);
        return Amplitude * std::min(1., double(edge) / double(FadeFrames));
    }
};

// Realtime context. No allocation, no locks, no logging, no Metal.
OSStatus Render(AudioObjectID, const AudioTimeStamp *, const AudioBufferList *, const AudioTimeStamp *,
                AudioBufferList *out, const AudioTimeStamp *, void *context) {
    auto &ring = *static_cast<Ring *>(context);
    const auto written = ring.Written.load(std::memory_order_acquire), read = ring.Read.load(std::memory_order_relaxed);
    const auto depth = written - read;
    if (depth < ring.MinDepth.load(std::memory_order_relaxed)) ring.MinDepth.store(depth, std::memory_order_relaxed);

    if (depth == 0) { // starved, so emit silence rather than repeating or blocking
        for (UInt32 b = 0; b < out->mNumberBuffers; ++b) std::memset(out->mBuffers[b].mData, 0, out->mBuffers[b].mDataByteSize);
        ring.Underruns.fetch_add(1, std::memory_order_relaxed);
        return noErr;
    }

    const float *block = ring.Samples + (read % SlotCount) * ring.Frames;
    // Keep the transport envelope outside the measured synthesis workload.
    const auto position = read * ring.Frames;
    const float gain = float(ring.Gain(position)), step = float(ring.Gain(position + ring.Frames) - gain) / float(ring.Frames);
    for (UInt32 b = 0; b < out->mNumberBuffers; ++b) {
        auto *dst = static_cast<float *>(out->mBuffers[b].mData);
        const auto channels = out->mBuffers[b].mNumberChannels;
        const auto wanted = out->mBuffers[b].mDataByteSize / (4 * channels);
        const auto frames = std::min(ring.Frames, wanted);
        for (UInt32 f = 0; f < frames; ++f) {
            const float sample = (gain + step * float(f)) * block[f];
            for (UInt32 c = 0; c < channels; ++c) dst[f * channels + c] = sample;
        }
        // A device asking for more than we rendered would otherwise replay whatever was in the buffer.
        if (wanted > frames) std::memset(dst + frames * channels, 0, (wanted - frames) * channels * sizeof(float));
    }
    ring.Read.store(read + 1, std::memory_order_release);
    return noErr;
}

// Keep producer scheduling inside the audio deadline.
bool RequestRealtime(double period_seconds) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    const auto ticks = [&](double s) { return uint32_t(s * 1e9 * double(tb.denom) / double(tb.numer)); };
    // Work is one fixed-cost GPU round trip; budgets below 50 us are rejected.
    const double computation = std::clamp(RenderWork, MinComputation, period_seconds);
    thread_time_constraint_policy_data_t policy{
        .period = ticks(period_seconds), .computation = ticks(computation),
        .constraint = ticks(std::max(computation, period_seconds / 2)), .preemptible = 1};
    return thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                             reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT) == KERN_SUCCESS;
}

double Pct(const std::vector<double> &sorted, double p) { return sorted[size_t(p * double(sorted.size() - 1))] * 1e6; } // microseconds

void Report(std::string_view name, const std::vector<Timings> &timings, double Timings::*field) {
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
    const double keepalive = args.size() > 4 ? std::atof(args[4]) * 1e-3 : 0.25e-3; // seconds, 0 to disable
    const uint32_t voices = args.size() > 5 ? uint32_t(std::atoi(args[5])) : 1;
    // GPU feedback allocates per commit, so enable it only for measurement.
    const bool want_feedback = std::getenv("GPU_AUDIO_FEEDBACK") != nullptr;

    // Audio device first, since it decides the sample rate and the block size we actually get.
    AudioObjectPropertyAddress address{kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    AudioObjectID audio_device = 0;
    UInt32 size = sizeof(audio_device);
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &address, 0, nullptr, &size, &audio_device) != noErr || audio_device == 0)
        return std::println(stderr, "No default output device."), 1;

    address = {kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
    UInt32 requested = frames;
    AudioObjectSetPropertyData(audio_device, &address, 0, nullptr, sizeof(requested), &requested);
    const auto block_frames = DeviceProperty<UInt32>(audio_device, kAudioDevicePropertyBufferFrameSize, frames);
    const auto sample_rate = DeviceProperty<Float64>(audio_device, kAudioDevicePropertyNominalSampleRate, 48000);
    const auto safety = DeviceProperty<UInt32>(audio_device, kAudioDevicePropertySafetyOffset);
    const auto device_latency = DeviceProperty<UInt32>(audio_device, kAudioDevicePropertyLatency);
    const auto stream_latency = StreamLatency(audio_device);
    const double block_period = double(block_frames) / sample_rate;

    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    NS::Error *error{};
    auto library = NS::TransferPtr(device->newLibrary(NS::String::string(SineSource, NS::UTF8StringEncoding), nullptr, &error));
    if (!library) return std::println(stderr, "Shader: {}", error->localizedDescription()->utf8String()), 1;

    // Compiling at runtime avoids needing Xcode's Metal toolchain, which is a separate download.
    auto compiler = NS::TransferPtr(device->newCompiler(NS::TransferPtr(MTL4::CompilerDescriptor::alloc()->init()).get(), &error));
    if (!compiler) return std::println(stderr, "Compiler: {}", error->localizedDescription()->utf8String()), 1;
    auto function = NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init());
    function->setName(MTLSTR("Sine"));
    function->setLibrary(library.get());
    auto pipeline_desc = NS::TransferPtr(MTL4::ComputePipelineDescriptor::alloc()->init());
    pipeline_desc->setComputeFunctionDescriptor(function.get());
    auto pipeline = NS::TransferPtr(compiler->newComputePipelineState(pipeline_desc.get(), nullptr, &error));
    if (!pipeline) return std::println(stderr, "Pipeline: {}", error->localizedDescription()->utf8String()), 1;

    auto samples = NS::TransferPtr(device->newBuffer(block_frames * (SlotCount + 1) * sizeof(float), MTL::ResourceStorageModeShared)); // one extra block as the filler's scratch
    auto params = NS::TransferPtr(device->newBuffer((SlotCount + 1) * sizeof(Params), MTL::ResourceStorageModeShared)); // one extra for the filler

    // Metal 4 has no implicit residency tracking, so resources go in a set attached to the queue.
    auto residency = NS::TransferPtr(device->newResidencySet(NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init()).get(), &error));
    residency->addAllocation(samples.get());
    residency->addAllocation(params.get());
    residency->commit();
    residency->requestResidency();

    auto table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    table_desc->setMaxBufferBindCount(2);
    auto table = NS::TransferPtr(device->newArgumentTable(table_desc.get(), &error));
    auto queue = NS::TransferPtr(device->newMTL4CommandQueue(NS::TransferPtr(MTL4::CommandQueueDescriptor::alloc()->init()).get(), &error));
    queue->addResidencySet(residency.get());

    std::vector<NS::SharedPtr<MTL4::CommandAllocator>> allocators(SlotCount);
    std::vector<NS::SharedPtr<MTL4::CommandBuffer>> command_buffers(SlotCount);
    for (uint32_t i = 0; i < SlotCount; ++i) {
        allocators[i] = NS::TransferPtr(device->newCommandAllocator());
        command_buffers[i] = NS::TransferPtr(device->newCommandBuffer());
    }

    // Cache invariant Objective-C properties outside the render loop.
    auto *slot_params = static_cast<Params *>(params->contents());
    const auto samples_address = samples->gpuAddress(), params_address = params->gpuAddress();
    const auto max_threads = uint32_t(pipeline->maxTotalThreadsPerThreadgroup());
    const auto threads_per_group = std::clamp((voices + 31) / 32 * 32, 32u, max_threads);

    // Queue signalling safely publishes GPU writes to the CPU; kernel-written flags do not.
    auto ready_event = NS::TransferPtr(device->newSharedEvent());

    Ring ring{.Samples = static_cast<const float *>(samples->contents()), .Frames = block_frames,
              .FadeFrames = uint64_t(FadeSeconds * sample_rate), .TotalFrames = uint64_t(seconds * sample_rate)};
    double phase = 0; // advanced in double and wrapped, so float never sees a large angle

    // Keep one-thread filler work isolated from the audio ring and synthesis load.
    slot_params[SlotCount] = {.PhaseInc = 0, .Phase0 = 0, .Seq = 0, .Voices = 1};
    // Rotate allocators so reset never reaches in-flight filler work.
    std::vector<NS::SharedPtr<MTL4::CommandAllocator>> warm_allocators(WarmSlots);
    std::vector<NS::SharedPtr<MTL4::CommandBuffer>> warm_buffers(WarmSlots);
    for (uint32_t i = 0; i < WarmSlots; ++i) {
        warm_allocators[i] = NS::TransferPtr(device->newCommandAllocator());
        warm_buffers[i] = NS::TransferPtr(device->newCommandBuffer());
    }
    uint64_t warm_index = 0;
    const auto keep_warm = [&] {
        const AutoreleasePool pool;
        const auto warm_slot = warm_index++ % WarmSlots;
        auto *warm_buffer = warm_buffers[warm_slot].get();
        warm_allocators[warm_slot]->reset();
        warm_buffer->beginCommandBuffer(warm_allocators[warm_slot].get());
        warm_buffer->useResidencySet(residency.get());
        auto *encoder = warm_buffer->computeCommandEncoder();
        table->setAddress(samples_address + SlotCount * block_frames * sizeof(float), 0);
        table->setAddress(params_address + SlotCount * sizeof(Params), 1);
        encoder->setArgumentTable(table.get());
        encoder->setComputePipelineState(pipeline.get());
        encoder->dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder->endEncoding();
        warm_buffer->endCommandBuffer();
        const MTL4::CommandBuffer *list[]{warm_buffer};
        queue->commit(list, 1);
    };

    // Cover the silent tail so vectors never move under feedback handlers.
    const auto max_blocks = size_t((seconds + TailSeconds) / block_period) + size_t(depth) + 64;
    std::vector<Timings> timings;
    timings.reserve(max_blocks);
    std::atomic<int> completed{0};
    std::vector<double> commit_times;
    commit_times.reserve(max_blocks);

    AudioDeviceIOProcID proc{};
    if (AudioDeviceCreateIOProcID(audio_device, Render, &ring, &proc) != noErr)
        return std::println(stderr, "Could not create IOProc."), 1;

    // Join the device's workgroup so the scheduler knows this thread shares the audio deadline.
    address = {kAudioDevicePropertyIOThreadOSWorkgroup, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
    os_workgroup_t workgroup{};
    size = sizeof(workgroup);
    const bool joined_workgroup = AudioObjectGetPropertyData(audio_device, &address, 0, nullptr, &size, &workgroup) == noErr && workgroup &&
                                  [&] { os_workgroup_join_token_s token; return os_workgroup_join(workgroup, &token) == 0; }();
    const bool realtime = RequestRealtime(block_period);

    // Render one block and publish it after GPU completion.
    const auto render_block = [&](uint64_t index) {
        const AutoreleasePool pool;
        const uint32_t slot = uint32_t(index % SlotCount);
        auto *command_buffer = command_buffers[slot].get();
        timings.emplace_back();
        auto &t = timings.back();
        const double started = Now();

        // Commit consumes its options, so feedback needs a fresh object.
        NS::SharedPtr<MTL4::CommitOptions> options;
        if (want_feedback) {
            options = NS::TransferPtr(MTL4::CommitOptions::alloc()->init());
            // Capture the stable index because the handler outlives this call.
            options->addFeedbackHandler([entry = timings.size() - 1, &timings, &completed](MTL4::CommitFeedback *feedback) {
                timings[entry].Submit = feedback->GPUStartTime(); // raw stamps, converted once the run is over
                timings[entry].Exec = feedback->GPUEndTime();
                completed.fetch_add(1, std::memory_order_release);
            });
        }

        allocators[slot]->reset();
        command_buffer->beginCommandBuffer(allocators[slot].get());
        command_buffer->useResidencySet(residency.get());
        slot_params[slot] = {.PhaseInc = PhaseInc, .Phase0 = float(phase), .Seq = uint32_t(index) + 1, .Voices = voices};
        phase = std::fmod(phase + PhaseInc * block_frames, Tau);
        table->setAddress(samples_address + slot * block_frames * sizeof(float), 0);
        table->setAddress(params_address + slot * sizeof(Params), 1);
        auto *encoder = command_buffer->computeCommandEncoder();
        encoder->setArgumentTable(table.get());
        encoder->setComputePipelineState(pipeline.get());
        encoder->dispatchThreadgroups({block_frames, 1, 1}, {threads_per_group, 1, 1});
        encoder->endEncoding();
        command_buffer->endCommandBuffer();

        commit_times.push_back(Now());
        t.Encode = commit_times.back() - started;
        t.Gap = commit_times.size() > 1 ? commit_times.back() - commit_times[commit_times.size() - 2] : 0;
        const MTL4::CommandBuffer *list[]{command_buffer};
        if (want_feedback) queue->commit(list, 1, options.get());
        else queue->commit(list, 1);
        queue->signalEvent(ready_event.get(), index + 1);
        t.Commit = Now() - commit_times.back();
        while (ready_event->signaledValue() < index + 1) {} // completion is what makes the block visible
        t.Done = Now() - commit_times.back();
        ring.Written.store(index + 1, std::memory_order_release);
    };

    // Prefill CoreAudio's startup pull; surplus blocks drain back to the requested depth.
    const uint64_t priming = (safety + device_latency + block_frames - 1) / block_frames;
    const uint64_t prefill = std::min<uint64_t>(uint64_t(depth) + priming, SlotCount - 1);
    for (uint64_t i = 0; i < prefill; ++i) render_block(i);
    if (AudioDeviceStart(audio_device, proc) != noErr) return std::println(stderr, "Could not start device."), 1;

    // Maintain lookahead, run optional filler, and sleep within the realtime budget.
    const double finish = Now() + seconds + TailSeconds; // gain_at is already zero past `seconds`, so the tail is silent
    double last_warm = Now();
    const double idle_quantum = keepalive > 0 ? std::min(IdleQuantum, keepalive / 4) : IdleQuantum;
    for (uint64_t next = prefill; Now() < finish && timings.size() < max_blocks;) {
        if (next - ring.Read.load(std::memory_order_acquire) >= uint64_t(depth)) {
            if (keepalive > 0 && Now() - last_warm >= keepalive) { keep_warm(); last_warm = Now(); }
            SleepUntil(idle_quantum);
            continue;
        }
        render_block(next++);
    }

    while (ring.Read.load(std::memory_order_acquire) < ring.Written.load(std::memory_order_acquire)) SleepUntil(idle_quantum); // let the fade reach the device
    AudioDeviceStop(audio_device, proc);
    AudioDeviceDestroyIOProcID(audio_device, proc);
    if (want_feedback) {
        while (completed.load(std::memory_order_acquire) < int(timings.size())) {} // GPU timestamps arrive on the feedback queue
        for (size_t i = 0; i < timings.size(); ++i) {
            timings[i].Exec -= timings[i].Submit; // GPU end minus GPU start, while Submit is still a raw time
            timings[i].Submit -= commit_times[i];
        }
    }

    // Separate code-controlled render latency from the output device path.
    const double render_latency = double(block_frames) / sample_rate * (1 + depth);
    const double device_latency_seconds = double(safety + device_latency + stream_latency) / sample_rate;
    std::println("--- {} frames ({:.3f} ms @ {:g} Hz), {} voices, depth {}, {:.0f}s, realtime {}, workgroup {} ---",
                 block_frames, block_period * 1e3, sample_rate, voices, depth, seconds,
                 realtime ? "yes" : "NO", joined_workgroup ? "yes" : "NO");
    std::println("render latency {:.2f} ms = {} block + {} lookahead frames, which is what this code sets",
                 render_latency * 1e3, block_frames, depth * block_frames);
    std::println("  the output device adds {:.2f} ms ({} safety + {} device + {} stream frames), which it does not,",
                 device_latency_seconds * 1e3, safety, device_latency, stream_latency);
    std::println("  for {:.2f} ms out of the speakers on this machine",
                 (render_latency + device_latency_seconds) * 1e3);
    std::println("blocks {}, underruns {}, min ring depth {} of {}",
                 timings.size(), ring.Underruns.load(), ring.MinDepth.load(), depth);

    const std::vector<Timings> steady(timings.begin() + std::min<size_t>(WarmupBlocks, timings.size()), timings.end());
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
    std::println("headroom: a block must be ready in {:.2f} ms, worst was {:.2f} ms", block_period * 1e3,
                 std::ranges::max(steady | std::views::transform(&Timings::Done)) * 1e3);
    return 0;
}
