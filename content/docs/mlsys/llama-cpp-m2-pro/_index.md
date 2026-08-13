---
weight: 1
BookCollapseSection: true
title: llama.cpp on Apple M2 Pro
---

# llama.cpp on Apple M2 Pro

- **Question** — Where does the time go when llama.cpp generates tokens on a MacBook?
- **Setup** — A few days measuring LFM2.5-1.2B on an M2 Pro, writing down what survived and what died.
- **TL;DR** — Token generation is stuck on memory bandwidth, both CPU and GPU.
  - Relevant bandwidth is what I measured: **112 GB/s** from CPU cores, **154 GB/s** from GPU — not the 200 GB/s Apple prints for the whole chip.
  - GPU generates tokens ~2× faster because it can pull harder on that same memory. The math is identical.

| | CPU (`-ngl 0`) | GPU (`-ngl 999`) |
|---|---:|---:|
| Tokens/s | 56.4 | 115.5 |
| Bandwidth reached | 110 GB/s inside the gemv | 144 GB/s over the token |
| Measured ceiling | 112 GB/s | 154 GB/s |
| Share of ceiling | 99% | 93% |
| Hot kernel | `ggml_gemv_q8_0_4x8_q8_0` | `kernel_mul_mv_q8_0_f32` |

- **Chip** — Apple M2 Pro, 8 performance + 4 efficiency cores
- **llama.cpp** — `3653e6d6d`
- **Model** — LFM2.5-1.2B-Instruct-Q8_0, ~1.24 GB per token

## Parts

- **Part 1: [Setup and roofline](01-setup-and-roofline.md)** — The 200 GB/s number is wrong for this question. Measuring what the CPU and GPU can really read, with the benchmark code.
- **Part 2: CPU decode** — Profiling with `/usr/bin/sample`, why 6 threads beat the default, and how the gemv turned out to be at 99% of the ceiling.
- **Part 3: Four dead ends** — Prefetching, dequant cost, the barrier, and work stealing. All four closed, three of them by reading code instead of running experiments.
- **Part 4: GPU decode** — Turning on Metal, capturing a trace with `xctrace`, and landing at 93% of 154 GB/s.
- **Part 5: Why the GPU pulls harder on the same memory.**
- **Part 6: Smaller weights** — Q4_K and Q5_K generate more tokens per second, but far less than the byte count predicts.
- **Part 7: Reading the Q4 Metal trace**, and what it failed to prove.
- **Part 8: Speculative decoding** — Lossless on this model, but a 350M draft model does not beat the wall on a normal prompt.
- **Part 9: What is worth doing next**, and why I did not rewrite the Metal kernel.

## The whole thing in one page

{{% details "One-page summary" %}}

### Tooling
- **`/usr/bin/sample`** — ships with macOS, answered the question in minutes.
- **`xctrace`** (Xcode) — same job for the GPU later.

### CPU decode
- Default thread count is wrong for this chip. **6 threads → 26% more t/s** than default.
- Nearly all useful time sits in one function: `ggml_gemv_q8_0_4x8_q8_0`.
- First efficiency number was wrong: dividing streamed bytes by whole token time gave **63%** of ceiling.
  - The gemv only runs **11.28 ms** of the **17.73 ms** token.
  - During the rest, no weights move at all.
  - Dividing by kernel time → **110 GB/s** against a measured **112 GB/s** ceiling → **98.8%**.
  - That killed the prefetch idea, the dequant idea, and every tiling idea in one line of arithmetic.
- Embedding matrix is **142 MB**, streams in full every token (tied to output head). Nothing to trim without changing quantization.
- Barrier + threadpool cost **4.78 ms/token (27%)**. No physical floor, but the work-stealing fix I was going to propose had already landed upstream (Oct 2025), and the adaptive barrier PR has been open 16 months because naive version makes generation **43% slower on M1**.

### GPU decode
- Widened the pipe without changing the problem.
- Same **1.24 GB/token**, **115.5 t/s**, GPU busy **99%** of the time, clocks at max.
- **93%** of busy time inside a single Metal kernel → **144 GB/s** against measured **154 GB/s**. Memory still the limit.

### Smaller weights
- The one lever that reduces bytes. Tried **Q4_K** and **Q5_K**.
- Wrote the verdict rule before running:
  - Within **10%** of `115.5 × (1.246 / size)` → memory still rules.
  - **15%+ below** → dequant became the limit.
- **Q5_K_M**: 120.9 t/s → **29% below** → tripped.
- **Q4_K_M**: 154.5 t/s → **22% below** → tripped.
- Q4 still worth shipping: **34% more t/s** (byte count had promised 71%).

### Q4 trace
- Two shaders busy: `mul_mv_q4_K_f32` at **62%** and `mul_mv_q6_K_f32` at **26%** (Q4_K_M is a mix, 30% is Q6_K, including the embedding matrix).
- Performance Limiters table exported empty — cannot say "ALU-bound" from a counter. Inferred from bandwidth falling off the roof.

### Speculative decoding
- Side experiment. Rollback on this hybrid model is lossless, greedy output matches on three prompts.
- **n-gram draft**: 5.5× faster on a copy-back prompt.
- **350M draft model**: **4% at best** on a normal prompt, got worse as draft length rose.

### Metal kernel — not rewritten
- If dequant were free, ceiling is ~**197 t/s** → **27% more**, not double.
- Cannot yet separate "sloppy code" from "K-quants cost more integer work by design."
- Part 9 lists the three things to measure before writing any Metal.

{{% /details %}}
