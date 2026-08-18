// Measures Metal 4 compute dispatch round-trip latency, which sets the latency floor for
// GPU-generated realtime audio. Per dispatch: CPU encode, commit to GPU start, GPU execution, and
// commit to GPU end, the per-block latency that stays valid at any depth.
//
//   DispatchLatency [frames] [iters] [submit_period_ms] [depth] [keepalive_ms] [voices]
//
// Period defaults to the cadence the frame count implies, and 0 submits back to back. Depth is how
// many dispatches stay in flight. Keepalive submits a one-thread filler dispatch on its own period,
// which stops the GPU idling between blocks and is worth about 5x at audio cadences.
//
// Voices is how many sines each output sample sums, which is the load knob. One threadgroup per
// frame splits the voices across its threads and reduces, so occupancy scales with voices rather
// than being capped at one thread per frame.
//
// Completion is an MTLSharedEvent signalled on the queue and polled, so a realtime consumer never
// touches a dispatch queue. MTL4CommitFeedback is collected too, but only for its GPU timestamps.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <QuartzCore/CABase.h>
#include <mach/mach_init.h>
#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <print>
#include <ranges>
#include <span>
#include <thread>
#include <vector>

namespace {
constexpr auto SineSource = R"(
#include <metal_stdlib>
using namespace metal;

struct Params { float PhaseInc, Phase0; uint Seq, Voices; };

kernel void Sine(
    device float *out [[buffer(0)]], constant Params &p [[buffer(1)]], device uint *witness [[buffer(2)]],
    uint frame [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    float sum = 0;
    for (uint v = lane; v < p.Voices; v += lanes) sum += sin(fma(p.PhaseInc * float(v + 1), float(frame), p.Phase0));

    threadgroup float partials[32]; // one per simdgroup, and a threadgroup is at most 1024 threads
    sum = simd_sum(sum);
    if (simd_lane == 0) partials[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float total = 0;
        for (uint k = 0; k < (lanes + 31) / 32; ++k) total += partials[k];
        out[frame] = total / float(p.Voices);
        witness[frame] = p.Seq; // exact, so the visibility check cannot be confused with float precision
    }
}
)";

struct Params { float PhaseInc, Phase0; uint32_t Seq, Voices; };

// One dispatch's timings, all in seconds.
struct Timings { double Encode, Submit, Exec, Done, Ready; };

constexpr double SampleRateKHz = 48; // sets the default cadence and labels the output, nothing else
constexpr int WarmupIters = 100;
constexpr int WorstCount = 12;

// Same timebase MTL4CommitFeedback uses for GPUStartTime and GPUEndTime, so GPU and CPU times
// subtract directly. std::chrono::steady_clock has a different epoch here and would give garbage.
double Now() { return CACurrentMediaTime(); }

// Without this, a deschedule of this thread lands in the submission number and looks like driver
// latency. Preemptible, because the loop spins and must not be able to starve the machine.
bool RequestRealtime() {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    const auto ms = uint32_t(1e6 * double(tb.denom) / double(tb.numer));
    thread_time_constraint_policy_data_t policy{.period = ms, .computation = ms / 2, .constraint = ms, .preemptible = 1};
    return thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                             reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT) == KERN_SUCCESS;
}

double Pct(const std::vector<double> &sorted, double p) { return sorted[size_t(p * double(sorted.size() - 1))] * 1e6; } // microseconds

int Fail(std::string_view stage, NS::Error *error) {
    std::println(stderr, "{}: {}", stage, error->localizedDescription()->utf8String());
    return 1;
}

void Report(std::string_view name, const std::vector<Timings> &timings, double Timings::*field) {
    auto v = timings | std::views::transform(field) | std::ranges::to<std::vector>();
    std::ranges::sort(v);
    std::println("{:<26} p50={:8.1f}  p90={:8.1f}  p99={:8.1f}  max={:9.1f}  (us)", name, Pct(v, .5), Pct(v, .9), Pct(v, .99), Pct(v, 1.));
}

void SleepUntil(double deadline) {
    if (const double remaining = deadline - Now(); remaining > 0) std::this_thread::sleep_for(std::chrono::duration<double>{remaining});
}
} // namespace

int main(int argc, char **argv) {
    const std::span args{argv, size_t(argc)};
    const uint32_t frames = args.size() > 1 ? uint32_t(std::atoi(args[1])) : 128;
    const int iters = args.size() > 2 ? std::atoi(args[2]) : 4000; // about 10 s at the default cadence, enough for a p99.9
    // Submitting a block more often than it plays is a regime no audio pipeline can reach.
    const double period = args.size() > 3 ? std::atof(args[3]) * 1e-3 : frames / (SampleRateKHz * 1e3);
    // The fastest configuration measured. Pass 1 and 0 for the unpipelined, cold-GPU baseline.
    const int depth = args.size() > 4 ? std::atoi(args[4]) : 2;
    const double keepalive = args.size() > 5 ? std::atof(args[5]) * 1e-3 : 0.25e-3; // seconds, 0 to disable
    const uint32_t voices = args.size() > 6 ? uint32_t(std::atoi(args[6])) : 1; // sines summed per output sample

    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    NS::Error *error{};
    auto library = NS::TransferPtr(device->newLibrary(NS::String::string(SineSource, NS::UTF8StringEncoding), nullptr, &error));
    if (!library) return Fail("Shader", error);

    // Compiling at runtime avoids needing Xcode's Metal toolchain, which is a separate download.
    auto compiler = NS::TransferPtr(device->newCompiler(NS::TransferPtr(MTL4::CompilerDescriptor::alloc()->init()).get(), &error));
    if (!compiler) return Fail("Compiler", error);
    auto function = NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init());
    function->setName(MTLSTR("Sine"));
    function->setLibrary(library.get());
    auto pipeline_desc = NS::TransferPtr(MTL4::ComputePipelineDescriptor::alloc()->init());
    pipeline_desc->setComputeFunctionDescriptor(function.get());
    auto pipeline = NS::TransferPtr(compiler->newComputePipelineState(pipeline_desc.get(), nullptr, &error));
    if (!pipeline) return Fail("Pipeline", error);

    // One slot is legal at depth 1, but resetting the allocator that just finished costs about
    // 1.3 us more per encode than rotating, so the floor stays 4 regardless of depth.
    const uint32_t SlotCount = std::max(4, depth + 2);
    auto samples = NS::TransferPtr(device->newBuffer(frames * SlotCount * sizeof(float), MTL::ResourceStorageModeShared));
    auto params = NS::TransferPtr(device->newBuffer((SlotCount + 1) * sizeof(Params), MTL::ResourceStorageModeShared)); // one extra for the filler
    auto witness = NS::TransferPtr(device->newBuffer(frames * SlotCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));

    // Metal 4 has no implicit residency tracking, so resources go in a set attached to the queue.
    auto residency = NS::TransferPtr(device->newResidencySet(NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init()).get(), &error));
    residency->addAllocation(samples.get());
    residency->addAllocation(params.get());
    residency->addAllocation(witness.get());
    residency->commit();
    residency->requestResidency();

    auto table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    table_desc->setMaxBufferBindCount(3);
    auto table = NS::TransferPtr(device->newArgumentTable(table_desc.get(), &error));

    auto queue_desc = NS::TransferPtr(MTL4::CommandQueueDescriptor::alloc()->init());
    auto queue = NS::TransferPtr(device->newMTL4CommandQueue(queue_desc.get(), &error));
    queue->addResidencySet(residency.get());

    std::vector<NS::SharedPtr<MTL4::CommandAllocator>> allocators(SlotCount);
    std::vector<NS::SharedPtr<MTL4::CommandBuffer>> command_buffers(SlotCount);
    for (uint32_t i = 0; i < SlotCount; ++i) {
        allocators[i] = NS::TransferPtr(device->newCommandAllocator());
        command_buffers[i] = NS::TransferPtr(device->newCommandBuffer());
    }

    // Each is an objc_msgSend returning the same value every iteration, so keep them out of the loop.
    auto *slot_params = static_cast<Params *>(params->contents());
    const auto samples_address = samples->gpuAddress(), params_address = params->gpuAddress();
    const auto witness_address = witness->gpuAddress();
    // A threadgroup per frame, sized to the voice count so the whole GPU is used rather than one
    // thread per frame. Rounded to a simdgroup, since the reduction below works in simdgroups.
    const auto max_threads = uint32_t(pipeline->maxTotalThreadsPerThreadgroup());
    const auto threads_per_group = std::clamp((voices + 31) / 32 * 32, 32u, max_threads);

    // Signalled on the queue timeline, so completion is what makes the writes visible. Publishing a
    // flag from inside the kernel is faster and is not safe, which the stale counter below proves.
    auto ready_event = NS::TransferPtr(device->newSharedEvent());
    const auto *cpu_witness = static_cast<const volatile uint32_t *>(witness->contents()); // volatile, so a stale read is the GPU's and not the optimizer's
    int stale = 0;

    // Its own allocator, buffer and params, so filler never disturbs the rotation the measured blocks
    // use and stays one thread of one voice however heavy the measured load gets. Sharing the bound
    // params instead makes the filler scale with the voice count and swamp the queue.
    slot_params[SlotCount] = {.PhaseInc = 0, .Phase0 = 0, .Seq = 0, .Voices = 1};
    auto warm_allocator = NS::TransferPtr(device->newCommandAllocator());
    auto warm_buffer = NS::TransferPtr(device->newCommandBuffer());
    const auto keep_warm = [&] {
        warm_allocator->reset();
        warm_buffer->beginCommandBuffer(warm_allocator.get());
        warm_buffer->useResidencySet(residency.get());
        auto *encoder = warm_buffer->computeCommandEncoder();
        table->setAddress(samples_address, 0);
        table->setAddress(params_address + SlotCount * sizeof(Params), 1);
        table->setAddress(witness_address, 2);
        encoder->setArgumentTable(table.get());
        encoder->setComputePipelineState(pipeline.get());
        encoder->dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder->endEncoding();
        warm_buffer->endCommandBuffer();
        const MTL4::CommandBuffer *list[]{warm_buffer.get()};
        queue->commit(list, 1);
    };

    std::vector<Timings> timings(iters);
    std::vector<double> commit_times(iters);
    std::atomic<int> completed{0};
    const bool realtime = RequestRealtime();

    const auto encode_and_commit = [&](int i) {
        const uint32_t slot = uint32_t(i) % SlotCount;
        auto *command_buffer = command_buffers[slot].get();
        auto &t = timings[i];
        const double started = Now();

        // Metal consumes the options at commit, so reusing one stops the handler from firing.
        auto options = NS::TransferPtr(MTL4::CommitOptions::alloc()->init());
        options->addFeedbackHandler([&t, &completed](MTL4::CommitFeedback *feedback) {
            t.Submit = feedback->GPUStartTime(); // raw stamps, converted once the run is over
            t.Exec = feedback->GPUEndTime();
            completed.fetch_add(1, std::memory_order_release);
        });

        allocators[slot]->reset();
        command_buffer->beginCommandBuffer(allocators[slot].get());
        command_buffer->useResidencySet(residency.get());
        slot_params[slot] = {.PhaseInc = 0.01f, .Phase0 = float(i), .Seq = uint32_t(i) + 1, .Voices = voices};
        table->setAddress(samples_address + slot * frames * sizeof(float), 0);
        table->setAddress(params_address + slot * sizeof(Params), 1);
        table->setAddress(witness_address + slot * frames * sizeof(uint32_t), 2);
        auto *encoder = command_buffer->computeCommandEncoder();
        encoder->setArgumentTable(table.get());
        encoder->setComputePipelineState(pipeline.get());
        encoder->dispatchThreadgroups({frames, 1, 1}, {threads_per_group, 1, 1});
        encoder->endEncoding();
        command_buffer->endCommandBuffer();

        commit_times[i] = Now();
        t.Encode = commit_times[i] - started;
        const MTL4::CommandBuffer *list[]{command_buffer};
        queue->commit(list, 1, options.get());
        queue->signalEvent(ready_event.get(), uint64_t(i) + 1);
    };

    // At depth 1 this is submit-then-wait.
    const double run_start = Now();
    for (int submitted = 0, done = 0; done < iters;) {
        while (submitted < iters && submitted - done < depth) {
            // computeCommandEncoder autoreleases, so without a pool per dispatch the encoders pile up
            // for the whole run and the pool growing mid-run shows up in the timings.
            NS::SharedPtr<NS::AutoreleasePool> pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
            const double due = run_start + submitted * period;
            if (keepalive > 0) for (double t = Now() + keepalive; t < due; t += keepalive) { SleepUntil(t); keep_warm(); }
            if (period > 0) SleepUntil(due);
            encode_and_commit(submitted++);
        }
        const auto signalled = ready_event->signaledValue(); // spin, so nothing is added to the observed latency
        const double now = Now();
        for (; done < int(signalled); ++done) {
            timings[done].Ready = now - commit_times[done];
            // Standing check that the completion signal really means the writes landed. Must stay 0.
            const uint32_t slot = uint32_t(done) % SlotCount;
            if (cpu_witness[slot * frames + frames - 1] != uint32_t(done) + 1) ++stale;
        }
    }
    while (completed.load(std::memory_order_acquire) < iters) {} // GPU timestamps arrive on the feedback queue
    for (int i = 0; i < iters; ++i) {
        timings[i].Done = timings[i].Exec - commit_times[i]; // the only per-block latency still valid at depth > 1
        timings[i].Exec -= timings[i].Submit; // GPU end minus GPU start, while Submit is still a raw time
        timings[i].Submit -= commit_times[i];
    }

    std::println("--- {} frames ({:.3f} ms audio @{:g}k), {} voices, {} iters, period {}, depth {}, realtime {} ---", frames, frames / SampleRateKHz, SampleRateKHz, voices, iters, period > 0 ? std::format("{:.3f} ms", period * 1e3) : "hot loop", depth, realtime ? "yes" : "NO");
    Report("cpu encode", timings, &Timings::Encode);
    Report("commit -> GPU start", timings, &Timings::Submit);
    Report("GPU exec", timings, &Timings::Exec);
    Report("commit -> GPU end", timings, &Timings::Done);
    // Above depth 1 the poll runs only between submissions, so it times when the CPU looked.
    if (depth == 1) Report("commit -> readable", timings, &Timings::Ready);
    else std::println("{:<26} not meaningful at depth > 1, the poll is what limits it", "commit -> readable");
    std::println("stale reads (writes not visible when signalled): {} of {}", stale, iters);

    const int worst_count = std::min(WorstCount, iters);
    auto worst = std::views::iota(0, iters) | std::ranges::to<std::vector>();
    std::ranges::partial_sort(worst, worst.begin() + worst_count, [&](int a, int b) { return timings[a].Done > timings[b].Done; });
    std::println("\nworst {} iterations (us):  iter    encode  commit->start     exec     done    ready", worst_count);
    for (int k = 0; k < worst_count; ++k) {
        const auto &t = timings[worst[k]];
        std::println("                          {:5}  {:8.1f}  {:13.1f}  {:7.1f}  {:7.1f}  {:7.1f}", worst[k], t.Encode * 1e6, t.Submit * 1e6, t.Exec * 1e6, t.Done * 1e6, t.Ready * 1e6);
    }
    // OUTLIERS=<ms> lists slow dispatches with their offset into the run, to expose their spacing.
    if (const char *env = std::getenv("OUTLIERS")) {
        const double threshold = std::atof(env) * 1e-3;
        std::println("\noutliers over {:g} ms:      iter    at (s)    encode  commit->start     exec     done", threshold * 1e3);
        double previous = 0;
        for (int i = 0; i < iters; ++i) {
            const auto &t = timings[i];
            if (t.Done <= threshold) continue;
            const double at = commit_times[i] - run_start;
            std::println("                          {:6}  {:8.3f}  {:8.1f}  {:13.1f}  {:7.1f}  {:7.1f}   gap {:.3f}s", i, at, t.Encode * 1e6, t.Submit * 1e6, t.Exec * 1e6, t.Done * 1e6, at - previous);
            previous = at;
        }
    }

    for (double threshold : {.5e-3, 1e-3, 2e-3}) {
        int over = 0, last = -1;
        for (int k = 0; k < iters; ++k) if (timings[k].Done > threshold) { ++over; last = k; }
        std::println("done > {:.1f} ms: {:5} of {} ({:.2f}%)  last@{}", threshold * 1e3, over, iters, 100. * double(over) / iters, last);
    }

    if (iters > WarmupIters) {
        auto steady = timings | std::views::drop(WarmupIters) | std::views::transform(&Timings::Done) | std::ranges::to<std::vector>();
        std::ranges::sort(steady);
        std::println("done excluding first {} iters: p50={:.1f} p99={:.1f} p99.9={:.1f} max={:.1f} (us)", WarmupIters, Pct(steady, .5), Pct(steady, .99), Pct(steady, .999), Pct(steady, 1.));
    }
    return 0;
}
