---
weight: 1
BookCollapseSection: true
title: Profiling LFM2.5-1.2B via llama.cpp
---

# Profiling LFM2.5-1.2B via llama.cpp

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
- **Part 2: CPU decode — the arc**
  - Baseline at t=8: gemv 42.6%, barrier 18% (the surprise)
  - Thread sweep: t=6 optimal, +25.6%. Barrier grew 1.31→3.61 ms from t=6→8
  - Per-iteration cost: 0.94 MAC/byte, ~75% stall. Not dependency-bound
  - Hypothesis: 200 GB/s is the wrong ceiling
  - Measuring it with `bw_bench.c`: **112 GB/s**, saturates at 6-8 threads
- **Part 3: Four dead ends — what died**
  - The embedding assumption was backwards (`token_embd` is the tied `lm_head`, streams full matrix)
  - Corrected efficiency: gemv runs 11.28 ms of the 17.73 ms token → 110/112 = **98.8%**
  - Prefetch, dequant, and tiling all dead in one line of arithmetic
  - Barrier = disassembled `ggml_graph_compute_thread` → 94% is an inlined `yield` spin loop
  - Upstream check: work-stealing already landed (#16833), futex barrier PR open 16 months
- **Part 4: GPU decode — the arc**
  - `-ngl 999`: CPU 99% idle, GPU invisible without Xcode
  - Metal System Trace: **115.5 t/s**, GPU busy 99%, `kernel_mul_mv_q8_0_f32` = 93% shader time
  - Measuring GPU bandwidth with `bw_bench_gpu.m`: **154 GB/s** shared ceiling → 144/154 = **93%**
  - Same wall, wider pipe
- **Part 5: Why the GPU pulls harder on the same memory** — Thousands of small threads keep the memory controller busy; a CPU core can only track a few dozen outstanding misses. The math is identical.
- **Part 6: Smaller weights — left the DRAM roof**
  - The one lever that cuts bytes
  - Verdict rule (locked before running): within 10% of `115.5 × (1.246 / size)` → memory still rules. 15%+ below → dequant became the limit
  - Q5_K_M: 120.9 t/s (−29%). Q4_K_M: 154.5 t/s (−22%). Both tripped — left the DRAM roof, now dequant-limited
- **Part 7: Reading the Q4 Metal trace, and what it failed to prove**
  - Two shaders busy: `mul_mv_q4_K_f32` (62%) and `mul_mv_q6_K_f32` (26%)
  - Performance Limiters table exported empty — bound inferred from bandwidth, not a counter
- **Part 8: Speculative decoding — lossless, not faster**
  - 350M draft is lossless on this hybrid model
  - ngram-simple: 5.5× on copy-back, 0% on open-ended
  - Draft model on v0: +4% at best, worse as `n_max` rises
  - The arithmetic doesn't close at 1.2B
- **Part 9: Metal kernel — not rewritten**
  - If dequant were free, ceiling is ~**197 t/s** → **27% more**, not double
  - Cannot yet separate "sloppy code" from "K-quants cost more integer work by design"
  - Three things to measure before writing any Metal
