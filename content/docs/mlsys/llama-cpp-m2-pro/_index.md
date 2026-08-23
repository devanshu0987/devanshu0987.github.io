---
weight: 1
BookCollapseSection: true
title: Profiling LFM2.5-1.2B via llama.cpp
---

# Profiling LFM2.5-1.2B via llama.cpp

- **Question**: where does the time go when llama.cpp generates tokens on a
  MacBook?
- **TL;DR (so far)**:
  - Thread count changes everything. t=1: 35.7 tok/s, t=6: 72.9 tok/s,
    t=12: 11.2 tok/s. Twelve threads is 3.2× slower than one.
  - Why: every graph node ends in a `ggml_barrier`, ~270 nodes per token.
    At t=12, 80.6% of CPU time is thread coordination, not arithmetic.
  - At t=1, one kernel (`ggml_gemv_q8_0_4x8_q8_0`) is 81.6% of decode,
    streaming 1.24 GB of weights per token at 55.2 GB/s effective.
  - Hypothesis: decode speed is the rate of moving weights from DRAM into
    the CPU cores. Whether 55.2 GB/s is good is an open question.
- **Status**: Part 1 written. Part 2 (the DRAM read benchmark) is next.
- **Setup**:
  - Chip: Apple M2 Pro, 8 performance + 4 efficiency cores.
  - llama.cpp `llama-cli`, build `3653e6d6d`, CPU-only (`-ngl 0`).
  - Model: LFM2.5-1.2B-Instruct-Q8_0.gguf, 1.17B params, 1.16 GiB.

## Parts

- **Part 1: [Setup + Profiling on CPU](01-setup-profiling-cpu/)**
  - The thread sweep, the barrier, the t=1 profile, the kernel disassembly.
  - Ends on the hypothesis: weight transfer from DRAM to CPU is the limit.
- **Part 2 (not yet written)**: a DRAM read benchmark for this machine, to
  settle whether 55.2 GB/s is a problem or the answer.
