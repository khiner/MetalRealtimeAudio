# MetalRealtimeAudio

Latency limits for realtime GPU-generated audio.
Apple M5 Max, macOS 26.5, Metal 4 via metal-cpp, built with Homebrew clang and libc++, no Objective-C.

```
./run                        # 128-frame blocks at their real 48k cadence
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
| A residual multi-ms stall remains | bursts every 20–80 s, hitting both submission and execution |

## Not measured yet

No CoreAudio, so the HAL's fixed 2.25 ms and real underrun behavior are untouched.
No load scaling: the kernel is one sine per sample, and every finding above holds only while GPU execution is 2% of the round trip.
