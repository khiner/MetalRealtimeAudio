# MetalRealtimeAudio

Latency limits for realtime GPU-generated audio.
Apple M5 Max, macOS 26.5, Metal 4 via metal-cpp, built with Homebrew clang and libc++, no Objective-C.

```
./run                        # 128-frame blocks at their real 48k cadence
./run 128 2000 2.667 2 0.25 4096   # 4096 voices summed per sample
./run 128 4000 2.667 1 0     # unpipelined, cold-GPU baseline
OUTLIERS=1 ./run 128 22000   # list dispatches slower than 1 ms
```

## Measured

`src/DispatchLatency.cpp` times one compute dispatch repeatedly, at a given block size and cadence.
Below: 128 frames at 2.667 ms, default settings.

| stage | p50 | p99.9 |
|---|---|---|
| CPU encode | 2.2 µs | 7 µs |
| commit → GPU start | 90 µs | 190 µs |
| GPU execution | 1.6 µs | 6 µs |
| **commit → GPU end** | **92 µs** | **195 µs** |

| configuration | p50 | p99.9 | worst | over 2 ms |
|---|---|---|---|---|
| depth 1, no keep-alive | 618 µs | 820 µs | 7.7 ms | 2 in 22k |
| depth 2 + 0.25 ms keep-alive | 92 µs | 195 µs | 6.8 ms | 4 in 112k |

Load, as sines summed per output sample. One threadgroup per frame splits the voices and reduces.

| voices | GPU exec p50 | p99 | share of a 2.667 ms block |
|---|---|---|---|
| 1 | 2.1 µs | 56 µs | 0.1% |
| 4096 | 4.7 µs | 22 µs | 0.2% |
| 65536 | 22.5 µs | 107 µs | 0.8% |
| 262144 | 80.0 µs | 383 µs | 3.0% |
| 1048576 | 309.5 µs | 1483 µs | 11.6% |

## Learned

| finding | evidence |
|---|---|
| Submission dominates, not compute | kernel takes ~6 µs, reaching the GPU takes 90–600 µs |
| Block size is free | 32, 128 and 512 frames cost the same per dispatch |
| An idle GPU goes cold | submitting slower than ~1.3 ms adds a fixed ~450 µs wake |
| Keep-alive recovers it | filler every 0.25 ms: 618 → 92 µs |
| Depth 2 helps, 4+ hurts | hides submission behind the previous block, deeper just queues |
| Realtime priority matters | ~20 µs of our own descheduling was counted as submission |
| Kernel-published flags are unsafe | 0.2–5% stale reads, even behind a device-visibility barrier |
| Queue-signalled `MTLSharedEvent` is safe | 0 stale in 111k, and 32 µs faster than the feedback callback |
| Command buffers are single-use | second commit aborts in the driver, re-encoding is mandatory |
| Compute contention does not hurt | a second instance saturating the GPU improved p50, by keeping it warm |
| Occupancy, not arithmetic, is the limit | one thread per frame is ~25x off, a threadgroup per frame is 27x faster at 4096 voices |
| Load is not the constraint | 1M voices per sample fits in 12% of a 2.667 ms block, still behind the ~90 µs submission |
| A residual multi-ms stall remains | bursts every 20–80 s, hitting both submission and execution |

## Not measured yet

No CoreAudio, so the HAL's fixed 2.25 ms and real underrun behavior are untouched.
The kernel is additive synthesis only, so nothing here covers convolution, modal or wave simulation, which have very different memory behavior.
