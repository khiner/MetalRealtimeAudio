# MetalRealtimeAudio

Latency limits for realtime GPU-generated audio.
Apple M5 Max, macOS 26.5, Metal 4 via metal-cpp, built with Homebrew clang and libc++, no Objective-C.
The GPU renders every sample and a CoreAudio device plays it.

```
./run                        # 32-frame blocks, 10 s, depth 2, the lowest latency that measures clean
./run 32 20 1                # one less block of lookahead, but glitches about once in 150k blocks
./run 128 10 2 0.25 4096     # 4096 voices summed per sample
GPU_AUDIO_FEEDBACK=1 ./run   # add GPU timestamps, which cost an allocation per commit
```

## Measured

Measured on an idle machine; tail results are sensitive to competing work.
Render latency is block plus lookahead, the part controlled here. The built-in output path adds
16.63 ms, including 690 frames of speaker DSP, and is reported separately.

These are 20 s, one-voice runs. CPU is process user plus system time.

| block | depth | render latency | underruns | CPU per run | out of the speakers here |
|---|---|---|---|---|---|
| 128 frames | 1 | 5.33 ms | 0 in 7576 | 4.55 s | 21.96 ms |
| 64 frames | 2 | 4.00 ms | 0 in 15153 | 5.49 s | 20.63 ms |
| 64 frames | 1 | 2.67 ms | 1 in 15135 | 5.50 s | 19.29 ms |
| **32 frames** | **2** | **2.00 ms** | **0 in 30247** | **7.75 s** | **18.63 ms** |
| 32 frames | 1 | 1.33 ms | 1 in 30260 | 7.72 s | 17.96 ms |

The bold row is the default. Depth 1 usually runs clean at 32 frames but averages about one underrun
per 150k blocks.

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

Below 32 frames, the device accepts the size but does not run cleanly. These 20 s frontier runs use
a 0.1 ms keepalive; pipelined rendering submits before the preceding block becomes readable.

| block | depth | rendering | render latency | underruns per 60k blocks |
|---|---|---|---|---|
| 16 frames | 1 | serial | 0.67 ms | 10–21 |
| 16 frames | 2 | serial | 1.00 ms | 9–22 |
| 16 frames | 2 | pipelined | 1.00 ms | 1–2 |

This halves default render latency but retains rare underruns.

Each output sample sums the requested voices across one threadgroup.

| voices | GPU exec p50 | p99 | share of a 2.667 ms block |
|---|---|---|---|
| 1 | 2.2 µs | 2.4 µs | 0.1% |
| 4096 | 5.0 µs | 5.4 µs | 0.2% |
| 65536 | 26.8 µs | 29.5 µs | 1.0% |
| 262144 | 96.6 µs | 99.2 µs | 3.6% |
| 1048576 | 376.2 µs | 378.8 µs | 14.1% |

## Learned

| finding | evidence |
|---|---|
| One-voice compute is not the bottleneck | a block becomes readable in 0.11 ms against a 2.67 ms period |
| Scheduling caused the measured underruns | readiness stayed below 0.25 ms while commit gaps reached 45 ms |
| Realtime budgets are enforced promises | spinning past the declared budget caused scheduler demotion |
| The computation budget has a 50 µs floor | smaller values fail at periods from 0.33 to 10 ms |
| Render work is fixed per block | one GPU round trip does not shrink with the frame count |
| Absolute sleeps preserve cadence | `mach_wait_until` stayed within 70 µs and cut 20 s user CPU from 20.0 to 1.3 s |
| CoreAudio primes on startup | prefilling safety plus device latency removed three startup underruns at 32 frames |
| Device latency is a separate metric | this output adds 16.63 ms beyond code-controlled render latency |
| Stream latency is additive | the built-in stream adds 690 frames beyond the device's 60 |
| Driver scheduling dominates submission | queue calls cost 2.0 µs; GPU start arrives 84 µs after commit |
| Keepalive prevents a cold GPU | 0.25 ms filler reduced 128-frame readiness from 423 to 135 µs |
| Pipelining matters only near the period | it was neutral at 32–128 frames and reduced 16-frame underruns to 1–2 |
| Depth absorbs completion tails | a 350–500 µs tail cannot meet a 333 µs depth-one deadline |
| Queue-signalled events are the safe completion primitive | 0 stale reads in 111k dispatches versus 0.2–5% for kernel flags |
| Command storage cannot be reused in flight | command buffers are single-use and allocators require completed work before reset |
| Threadgroup occupancy dominates reduction cost | one threadgroup per frame was 27x faster at 4096 voices |
| The load knob has substantial range | 1M voices used 14% of a 2.667 ms block |

## Not measured yet

These results cover additive synthesis, not the memory behavior of convolution, modal synthesis, or
wave simulation. Rare 32- and 16-frame underruns remain unattributed. Deeper rings do not remove the
16-frame failures, so completion latency is not their cause. Callback-driven producer wakeup remains
untested and accounts for most remaining CPU.
