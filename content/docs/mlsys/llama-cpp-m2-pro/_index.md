---
weight: 1
BookCollapseSection: true
title: Profiling LFM2.5-1.2B via llama.cpp
---

# Profiling LFM2.5-1.2B via llama.cpp

- **Question**: where does the time go when llama.cpp generates tokens on a
  MacBook?
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
  find the real CPU-reachable ceiling and compare it to 55.2 GB/s.
