# MetalRealtimeAudio

Latency limits for realtime GPU-generated audio.
Apple M5 Max, macOS 26.5, Metal 4 via metal-cpp, built with Homebrew clang and libc++.
The GPU renders every sample and a CoreAudio device plays it.

## Result

**On an otherwise idle M5 Max, Metal 4 can synthesize audio in real time for CoreAudio playback at 32-frame periods, with 2.00 ms of configured pre-device latency and no dropouts.**

At depth two, the producer keeps two submitted 32-frame blocks (1.33 ms) ahead of CoreAudio.
Latency begins at GPU submission, before synthesis, includes generation and synchronization within that lookahead window plus the 0.67 ms CoreAudio output block, and excludes the subsequent device path.

- A 30 s production run at default settings (see below) rendered 45,305 blocks with zero underruns.
- At 128 frames, a block becomes CPU-readable in 106 µs p50, 158 µs p99, and 197 µs p99.9.
  CPU encoding and commit cost 3.1 µs and 2.0 µs p50.
- A 1,048,576-voice (sines) dispatch takes 376.2 µs p50 and 378.8 µs p99—14.1% of a 2.667 ms block.
- Shared UMA buffers and queue-signalled events require no GPU readback, staging copy, or polling.
  Event publication produces no stale reads in 111k dispatches.

```
./run                        # 32-frame blocks, 10 s, depth 2, no keepalive filler
./run 32 20 1                # one less block of lookahead, but glitches about once in 150k blocks
./run 128 10 2 0.25 4096     # 4096 voices summed per sample
GPU_AUDIO_FEEDBACK=1 ./run   # add GPU timestamps, which cost an allocation per commit
GPU_AUDIO_PRODUCTION=1 ./run # omit timing storage, clocks and callback diagnostics
```

## Measured

Measured on an idle machine.
Tail results are sensitive to competing work.
Render latency is block plus lookahead, the part controlled here.
The built-in output path adds 16.63 ms, including 690 frames of speaker DSP, and is reported separately.

These are 20 s, one-voice runs.
CPU is process user plus system time.

| block | depth | keepalive | render latency | underruns | CPU per run | out of the speakers here |
|---|---|---|---|---|---|---|
| 128 frames | 1 | 0.25 ms | 5.33 ms | 0 in 7576 | 4.55 s | 21.96 ms |
| 64 frames | 2 | 0.25 ms | 4.00 ms | 0 in 15153 | 5.49 s | 20.63 ms |
| 64 frames | 1 | 0.25 ms | 2.67 ms | 1 in 15135 | 5.50 s | 19.29 ms |
| **32 frames** | **2** | **off** | **2.00 ms** | **0 in 30305** | **2.97 s** | **18.63 ms** |
| 32 frames | 1 | 0.25 ms | 1.33 ms | 1 in 30260 | 7.72 s | 17.96 ms |

The bold row is the default.
Depth 1 usually runs clean at 32 frames but averages about one underrun per 150k blocks.

Per block, at 128 frames and depth 2, with `GPU_AUDIO_FEEDBACK=1`.

| stage | p50 | p99 | p99.9 |
|---|---|---|---|
| CPU encode | 3.1 µs | 7.3 µs | 11 µs |
| CPU inside commit and signal | 2.0 µs | 3.6 µs | 5.5 µs |
| commit → GPU start | 84 µs | 134 µs | 167 µs |
| GPU execution | 2.2 µs | 2.3 µs | 82 µs |
| **commit → readable** | **106 µs** | **158 µs** | **197 µs** |

At 128 frames and depth 2, a filler dispatch prevents the GPU from idling between blocks.

| keepalive | commit → readable p50 | p99 |
|---|---|---|
| off | 423 µs | 488 µs |
| 0.25 ms | 135 µs | 233 µs |

Depth-two lookahead hides the cold-GPU wake, making filler pure overhead:

| block | keepalive | worst readable | system CPU | retired instructions |
|---|---|---|---|---|
| 32 frames | off | 471 µs | 0.71 s | 2.06B |
| 32 frames | 0.25 ms | 316 µs | 1.32 s | 5.50B |
| 128 frames | off | 896 µs | 0.26 s | 0.69B |
| 128 frames | 0.25 ms | 234 µs | 1.05 s | 4.30B |

The default policy uses 0.25 ms at depth one and disables filler at greater depths.
The fourth argument overrides it.
A 30 s default production run completed 45,305 blocks with zero underruns and 4.01 s process CPU.

Creating filler resources only when enabled saves 160 KiB of default RSS.

Callback-driven submission and a dedicated completion waiter keep blocks in flight without polling:

| block | rendering | underruns | process CPU / wall |
|---|---|---|---|
| 32 frames, 5 s | serial polling | 0 | 1.97 / 5.64 s |
| 32 frames, 5 s | pipelined, semaphore-driven | 0 | 1.19 / 5.35 s |
| 16 frames, 10 s | serial polling | 16 | 5.93 / 10.33 s |
| 16 frames, 10 s | pipelined, semaphore-driven | 2 | 3.20 / 10.35 s |

Using the Metal timeline as the completion wakeup removed two Mach semaphore signals per block.
An 8 s run retired 2.38B instead of 2.56B instructions and used 0.91 instead of 0.99 s system CPU.

Untimed waits without filler reduced system CPU from 1.04 to 1.01 s in matched 10 s runs.

Sizing command pools to in-flight work cut their allocation from 13.2 to 4.4 MB.

Production mode keeps the underrun count but omits 56 bytes of timing storage per rendered block.
At 32 frames that is 84 KB/s of run duration and saved 738 KB in a matched 5 s run.

Clang vectorizes the mono and stereo steady/ramp copy loops.
The process-level gain is below noise.

Below 32 frames, the device accepts the size but does not run cleanly.
These 20 s frontier runs use a 0.1 ms keepalive.
Pipelined rendering submits before the preceding block becomes readable.

| block | depth | rendering | render latency | underruns per 60k blocks |
|---|---|---|---|---|
| 16 frames | 1 | serial | 0.67 ms | 10–21 |
| 16 frames | 2 | serial | 1.00 ms | 9–22 |
| 16 frames | 2 | pipelined | 1.00 ms | 1–3 |

This halves default render latency but retains rare underruns.

Each output sample sums the requested voices across one threadgroup.

| voices | GPU exec p50 | p99 | share of a 2.667 ms block |
|---|---|---|---|
| 1 | 2.2 µs | 2.4 µs | 0.1% |
| 4096 | 5.0 µs | 5.4 µs | 0.2% |
| 65536 | 26.8 µs | 29.5 µs | 1.0% |
| 262144 | 96.6 µs | 99.2 µs | 3.6% |
| 1048576 | 376.2 µs | 378.8 µs | 14.1% |

## Constraints

Observed underruns coincided with producer descheduling—commit gaps reached 45 ms while GPU readiness stayed below 0.25 ms.
macOS rejects realtime computation budgets below 50 µs and demotes threads that overrun their declared budget.
Queue-signalled events publish GPU writes reliably.
Kernel-written flags produced 0.2–5% stale reads in 111k dispatches.
CoreAudio startup needs safety plus device-latency prefill.
HAL supplies zeroed output, and 128-byte ring strides isolate GPU writes from CoreAudio reads.
