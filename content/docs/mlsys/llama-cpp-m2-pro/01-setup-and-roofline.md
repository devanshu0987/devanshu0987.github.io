---
title: "Part 1: Setup and roofline"
weight: 1
---

# Part 1: Setup and roofline

Before profiling llama.cpp, I needed something to compare the profile against.
On Apple Silicon the number printed on the box is the wrong one.

Apple says 200 GB/s for the M2 Pro. That is the whole chip talking to memory:
CPU, GPU, neural engine, media engines. One client never gets all of it. I
measured 112 GB/s from the CPU cores and 154 GB/s from the GPU. Every roofline
in this series uses those two numbers.

## What I ran

| | |
|---|---|
| Chip | Apple M2 Pro, 8 performance + 4 efficiency cores |
| OS | macOS, MDM-managed work laptop, no Xcode at first |
| Binary | llama.cpp `llama-cli`, build `3653e6d6d` |
| Model | LFM2.5-1.2B-Instruct-Q8_0.gguf, 1.25 GB on disk |
| Workload | one prompt, greedy, long generation |

Two phases matter and they have different bottlenecks. Prefill reads your
prompt and does batched matrix-matrix multiplies, which keep the ALUs busy.
Decode generates one token at a time, which collapses those into
matrix-vector multiplies. Every weight gets read once and used once, roughly
one multiply-add per byte. This series is about decode.

## Why I needed a ceiling at all

For generating one token at a time, two limits apply:

```
memory limit:   tokens/s <= bandwidth / bytes_per_token
compute limit:  tokens/s <= FLOP/s / FLOPs_per_token
```

The compute limit is nowhere close. Two FLOPs per parameter is about 2.4
GFLOPs per token, against several TFLOP/s on this GPU. That leaves tens of
thousands of tokens per second on paper. The memory limit is the one that
bites.

So the only interesting question is which bandwidth number goes in the
denominator. With the brochure figure:

```
bytes per token = 1.24 GB
200 / 1.24      = about 161 tokens/s
```

I used that number for a while and it made llama.cpp look much worse than it
was.

## Measuring the CPU

Rules for a read benchmark that behaves like weight streaming:

- Buffer of 1 GB or more, so it does not fit in L2 (16 MB per cluster) or the
  system level cache (about 24 MB). Everything has to come from DRAM.
- `memset` the buffer first, so page faults are not part of the timing.
- Keep many loads in flight at once. One dependent chain measures latency,
  not bandwidth.
- Add every value into a sink, or the compiler deletes the loop.
- Take the best of several runs, and sweep the thread count.

The read loop, from `bw_bench.c`:

```c
static size_t BUF_BYTES = 1024ull * 1024 * 1024;  // 1 GB, past L2 and SLC

static void *read_worker(void *p) {
    targ_t *a = (targ_t *)p;
    size_t chunk = BUF_BYTES / NTHREADS;
    const uint8_t *base = g_buf + (size_t)a->tid * chunk;

    uint64x2_t acc0 = vdupq_n_u64(0), acc1 = vdupq_n_u64(0);
    uint64x2_t acc2 = vdupq_n_u64(0), acc3 = vdupq_n_u64(0);

    for (size_t off = 0; off + 256 <= chunk; off += 256) {
        const uint8_t *q = base + off;
        // 8 loads of 32B per iteration, four independent chains
        acc0 = vaddq_u64(acc0, vld1q_u64((const uint64_t *)(q +   0)));
        acc1 = vaddq_u64(acc1, vld1q_u64((const uint64_t *)(q +  16)));
        acc2 = vaddq_u64(acc2, vld1q_u64((const uint64_t *)(q +  32)));
        acc3 = vaddq_u64(acc3, vld1q_u64((const uint64_t *)(q +  48)));
        // ... same pattern through q + 240
    }
    uint64x2_t s = vaddq_u64(vaddq_u64(acc0, acc1), vaddq_u64(acc2, acc3));
    a->sink = vgetq_lane_u64(s, 0) + vgetq_lane_u64(s, 1);
    a->bytes = (double)(chunk / 256) * 256.0;
    return NULL;
}
```

Build and sweep:

```bash
clang -O3 -mcpu=apple-m2 -o bw_bench bw_bench.c
for t in 1 2 4 6 8 10 12; do echo "--- t=$t ---"; ./bw_bench $t; done
```

| threads | read GB/s | copy GB/s |
|---:|---:|---:|
| 1 | 51.8 | 84.4 |
| 2 | 71.0 | 127.4 |
| 4 | 103.6 | 139.7 |
| 6 | 111.6 | 150.2 |
| 8 | 112.3 | 152.0 |
| 12 | 112.3 | 149.8 |

Reads flatten at 6 to 8 threads and stay there. Adding cores does not open up
more of the memory controller. So 112 GB/s is the CPU ceiling.

Copy looks faster, 150 GB/s, which surprised me and I never fully explained
it. Likely non-temporal stores skipping part of the cache hierarchy. Decode
reads weights, so `read` is the right column.

The sweep confirmed a second thing for free. llama.cpp also peaked at 6
threads, for the same reason the benchmark did, so two independent
measurements agreed on the thread count.

## Measuring the GPU

Same rules, this time in Metal compute. llama.cpp keeps weights in shared
storage, so that is what I measured.

```metal
kernel void read_bw(device const float4 *src [[buffer(0)]],
                    device atomic_uint *sink [[buffer(1)]],
                    constant uint &n4 [[buffer(2)]],
                    uint tid [[thread_position_in_grid]],
                    uint nthreads [[threads_per_grid]]) {
    float4 acc = float4(0.0f);
    for (uint i = tid; i < n4; i += nthreads) {
        acc += src[i];
    }
    uint s = as_type<uint>(acc.x + acc.y + acc.z + acc.w);
    atomic_fetch_add_explicit(sink, s, memory_order_relaxed);
}
```

On the host side: one 1 GB buffer, a warmup dispatch to fault the pages, eight
passes inside each timed command buffer, and a sweep over threadgroup sizes
from 64 to 1024 and grid sizes from 256 to 4096 threadgroups.

```bash
clang -O2 -fobjc-arc -framework Metal -framework Foundation \
    -o bw_bench_gpu bw_bench_gpu.m
./bw_bench_gpu
```

| Client | Read GB/s |
|---|---:|
| CPU cores, 6 threads | 112 |
| GPU, shared storage | 154 |
| GPU, private storage | 140 |
| Apple's number | 200 |

Here read and copy agree at about 154, unlike on the CPU. Same memory chips
in both benchmarks. The difference is which client is asking.

## The corrected ceilings

A GGUF audit of the Q8 file gives 1.244 GB streamed per token, and that
includes the embedding matrix acting as the output head. Part 2 covers why.

| Ceiling used | Math | tokens/s |
|---|---|---:|
| Apple's 200 GB/s | 200 / 1.244 | 161 |
| CPU measured | 112 / 1.244 | 90 |
| GPU measured | 154 / 1.244 | 124 |

CPU decode came in at 56.4 tokens/s, which is 63% of 90 rather than 35% of
161. GPU decode came in at 115.5, which is 93% of 124. Scoring either against
200 invents headroom that does not exist, and sends you off optimizing a
kernel with nothing left to give.

## Why the GPU number is higher

Memory takes about 100 ns to answer, and bandwidth is just how many requests
you can have outstanding while you wait. The GPU runs thousands of small
threads, so when one is waiting, hundreds of others have already asked for
their data. A CPU core can only track a few dozen outstanding cache misses,
and eight of them together still only reach a few hundred. That is the whole
gap between 112 and 154. Part 5 goes through the hardware terms.

## What this part does not show

It does not show whether llama.cpp gets close to these ceilings, which is what
Part 2 measures for CPU and Part 4 for GPU. Neither 112 nor 154 is the most the
memory can do either, since the 200 GB/s figure assumes everything on the chip
reads at once. And prefill does not live under this roof, because batched
matrix multiplies reuse weights instead of streaming them once.

Next: [Part 2, profiling CPU decode](02-cpu-decode.md).
