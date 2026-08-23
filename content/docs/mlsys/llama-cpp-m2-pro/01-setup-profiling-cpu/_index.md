---
title: "Part 1: Setup + Profiling on CPU"
weight: 1
---

# Part 1: Setup + Profiling on CPU

- The question for the whole series: where does the time go when llama.cpp
  generates tokens?
- This part stays entirely on the CPU.

## The setup

| | |
|---|---|
| Chip | Apple M2 Pro, 8 performance + 4 efficiency cores |
| OS | macOS 26.6.1 |
| Binary | llama.cpp `llama-cli`, build `3653e6d6d` |
| Model | LFM2.5-1.2B-Instruct-Q8_0.gguf, 1.17B params, 1.16 GiB on disk |
| Workload | one prompt, greedy, `-n 1000 --ignore-eos` |

To run CPU-only:

```bash
llama-cli --model model.gguf -ngl 0 --file prompt.md --perf --single-turn \
  -n 1000 --ignore-eos --output response.md
```

Key flags:

- `-ngl 0`: no GPU layers, all tensors stay on the CPU
- `-n 1000 --ignore-eos`: generate exactly 1000 tokens, long enough for a stable steady state
- `--perf`: prints the one-line `[ Prompt: X t/s | Generation: Y t/s ]` summary
- `--single-turn` / `--file`: process the prompt file and stop

Thread count:

- The command does not pass `-t`, so llama.cpp defaults to `-t -1`.
- `-t -1` resolves to `std::thread::hardware_concurrency()`.
- On macOS that calls `sysctl(hw.logicalcpu)`.
- This machine: 12 threads (8P + 4E).
- Those 12 threads become the `ggml_graph_compute_thread` workers.

## Thread count changes everything

Same command, three thread counts, 1000 generated tokens each:

| threads | Prompt (t/s) | Generation (t/s) |
|---:|---:|---:|
| 1 | 113.1 | 35.7 |
| 6 | 624.7 | **72.9** |
| 12 | 263.5 | 11.2 |

Two observations:

1. Six threads roughly doubles single-threaded generation.
2. **Twelve threads is 3.2× slower than one.**

The prompt column follows the same shape: prefill wants threads (113.1 t/s
at t=1) but also regresses past six (263.5 vs 624.7). Barriers are per graph
node, and prefill runs the same graph.

A parallel job getting slower when you add workers doesn't make sense.
Let's scope down to two runs, `t=1` and `t=12`, and profile both.

## Two traces

How the traces were captured:

- Instruments Time Profiler, `xctrace record --launch`.
- Whole run at 1 ms resolution.
- Exported to XML, parsed from top-of-stack frames (self time, not
  cumulative).
- Two traces, same command, separate runs from the sweep above.

| | t=1 | t=12 |
|---|---:|---:|
| Wall time | 41.8 s | 81.1 s |
| Total CPU time | 40.9 s | 792.7 s |
| Threads seen in trace | 4 | 11,060 |

t=12 burns **19.4× the CPU time to be 1.9× slower**.

- t=1: one core ~98% busy for 42 s.
- t=12: nearly the whole machine for 80 s, half the throughput.
- The thread counts need context:
  - t=1: 4 threads, one persistent compute thread plus 3 short-lived
    background helpers.
  - t=12: 11,060 threads, but only ~1,905 persistent. The rest is ephemeral
    GCD churn, not 11,000 compute workers.

Which run is which:

- The sweep above and these traces are different runs of the same command.
- At t=1 they agree: 36.5 t/s from the trace, 35.7 from the sweep. About 2%
  apart.
- At t=12 they do not:
  - The trace: 81.1 s wall minus ~14 s of load and prompt leaves ~67 s of
    decode. About 15 t/s.
  - The sweep: 11.2 t/s.
- Why the gap: not measured. Two guesses:
  - The sweep runs the three thread counts back to back. The trace ran on
    its own.
  - At 12 threads the OS spreads workers over 8 P-cores and 4 E-cores.
    Each run lands differently. t=1 has no such choice, so it is stable.
- This does not change the story:
  - Throughput numbers come from the sweep. At either rate, t=12 is 2.4× to
    3.2× slower than t=1.
  - The 19.4× and 1.9× above stay within one run: the trace.
  - Where the time goes (next table) is a share of CPU time. It does not
    depend on the exact rate.

Where those cycles go (top-of-stack attribution, mutually exclusive frames):

| Top frame | t=1 | t=12 |
|---|---:|---:|
| `ggml_gemv_q8_0_4x8_q8_0` | 55.4% | 15.4% |
| `ggml_gemm_q8_0_4x8_q8_0` | 25.3% | 1.8% |
| `ggml_graph_compute_thread` | 0.9% | **62.2%** |
| `ggml_barrier` | **0%, never sampled** | **18.4%** |
| flash attention + `ggml_vec_dot_f16` | 9.3% | 1.2% |

Reading the table:

- t=1: the matrix kernels own 80% of the run.
  - The `gemm` share is the prompt phase.
  - The `gemv` is decode.
- t=12: **80.6% of all CPU time is `ggml_graph_compute_thread` +
  `ggml_barrier`**. Thread coordination, not arithmetic.
- Adding those two rows is legal because every sample has exactly one
  top-of-stack frame. A thread spinning in `ggml_barrier` lands in that row,
  never in `ggml_graph_compute_thread`'s. Zero overlap in the raw
  backtraces, so the 80.6% double counts nothing.
- The barrier does not appear at all in the t=1 trace. Zero samples.

## The reason: a barrier after every graph node

- llama.cpp executes the model as a graph of ~270 compute nodes per token.
  - Counted op by op from the model source: [counting the barriers](counting-the-barriers/).
- The per-thread worker loop in `ggml-cpu.c`:

```c
for (int node_n = 0; node_n < cgraph->n_nodes && ...; node_n++) {
    ...
    if (n_fused > 0) { node_n += n_fused; }
    else             { ggml_compute_forward(&params, node); }

    if (node_n + 1 < cgraph->n_nodes) {
        ggml_barrier(state->threadpool);   // after EVERY node
    }
}
```

So one generated token costs ~270 barriers.

- Every barrier waits for the **slowest** of 12 threads.
- Nobody moves to the next node until that thread arrives.

Where the waiting happens (trace backtraces):

- ~70% of barrier samples sit under `forward_mul_mat`.
- ~23% sit under `flash_attn_ext`.
- The picture: threads that finished their chunk of a node, spinning until the
  last straggler arrives.

Disassembled (objdump of `libggml-cpu.dylib`, build `3653e6d6d`), the barrier
is a spin loop. This snippet is the copy inlined into
`ggml_graph_compute_thread` at offset +692. In a t=6 profile, 94% of that
function's self-time samples sat exactly there:

```asm
ldr    w10, [x8, #0x100]     ; read the shared arrival counter
cmp    w10, w9               ; has everyone arrived?
b.ne   …                     ; not yet → spin again
yield                       ; hint to the core: "I am spinning"
```

- The trace's 18.4% row is the standalone `ggml_barrier` symbol: the same
  loop under its own address (`ldr` the two counters, `cmp`, `b.ne`,
  `yield`).
- Some barrier call sites inline into the worker loop, some call the
  standalone symbol. Both spin.

What it means:

- No sleeping, no futex.
- A core executes this loop flat out until the last thread shows up.
- That is the 18.4% of t=12 CPU time.
- The 62.2% in `ggml_graph_compute_thread` is the node loop run twelve times
  in parallel:
  - Node iteration, atomic chunk claiming, condition checks.
  - Once per thread, for every one of the ~270 nodes.

At t=1 the situation inverts:

- The barrier's fast path is a single atomic check (`n_threads == 1`). It is
  never even sampled.
- The graph loop itself is almost free: dispatch outside any kernel measures
  **0.017 ms/token** (0.06% of decode, ~60 ns per node).
- The graph loop is not expensive. Coordinating 12 threads through it, 270
  times per token, is expensive.

Why threads stop helping:

- Per-node work is small: a few milliseconds of streaming weights.
- The synchronization tax does not shrink. Every added thread joins every
  barrier.
- The tax explains the wasted CPU (80.6%). It does not by itself explain the
  wall-time regression. Even at 19% efficiency, 12 threads deliver ~2.3
  core-equivalents of real compute, more than t=1's one core.
- So why doesn't that extra compute win? Two reasons:
  - The gemv is capped by DRAM bandwidth, not by thread count:
    - t=1: gemv costs 22.5 CPU-ms per token (per-op table below).
    - t=12: the same bytes cost 122 CPU-ms per token (15.4% of 792.7 s,
      across 1000 tokens).
    - Perfectly overlapped, that is ~10 ms of wall time (122 ÷ 12). Only
      ~2.2× better than the single thread.
    - The pipe saturates at a thread count below 12. Past that point, extra
      threads add no bandwidth. They only add tax.
  - Four of the 12 workers land on efficiency cores (8P + 4E):
    - E-core threads finish their chunks last.
    - Every one of the ~270 barriers waits for the slowest arrival.
    - Default scheduler placement; affinity not measured.
- Threads beyond a handful cannot make decode faster. Here they make it much
  slower.
- Why exactly 6 threads wins is a bandwidth question. It comes back in Part 2.

From here on, the interesting run is t=1: if threads only add tax, what limits
a single thread?

## Profiling t=1: the 27.44 ms token

The trace splits into three phases. The kernel symbols mark them:

- batched `gemm` = prompt processing
- single-token `gemv` = generation

| Phase | Span | What happens |
|---|---:|---|
| load + warmup | 2.75 s | mmap the GGUF, repack weights |
| prompt eval | 11.51 s | ~1312 tokens at ~114 t/s (batched GEMM) |
| **decode** | **27.52 s** | 1003 tokens → **27.44 ms/token = 36.5 t/s** |

- 36.5 t/s comes from trace timestamps and a token count of the output.
- The `--perf` one-liner in the sweep above said 35.7. Run-to-run variance.

Decode-phase attribution, per op:

| op | % of decode | ms/token |
|---|---:|---:|
| `ggml_gemv_q8_0_4x8_q8_0` (all weight matmuls + output head) | **81.6%** | **22.52** |
| flash_attn_ext | 15.4% | 4.26 |
| everything else (norms, glu, rope, conv, sampling, dispatch) | ~3% | ~0.7 |

One kernel owns decode. The name:

- **gemv**: matrix-*vector* multiply. Batch size 1, one token at a time.
- **q8_0**: the weight format.
- **4x8**: the NEON tile.
- Everything else is rounding error. Perfect fusion of all small ops would
  buy ~2%.

## Inside the kernel

- Hand-written NEON: `ggml/src/ggml-cpu/arch/arm/repack.cpp:1757`.
- The profile confirms the SIMD path actually runs:
  - The `_generic` fallback has zero samples.
  - The chip advertises `FEAT_DotProd`.
- Weights are repacked at load time into an interleaved 4-row layout:
  - The inner loop's vector loads hit memory sequentially.

One iteration of the inner loop (`0xae18c`–`0xae204` in the dylib) processes
one block group of 4 columns × 32 weights:

```asm
ld1.8b    { v1, v2, v3, v4 },    [x14]      ; 32 B of quantized activations
ld1.16b   { v16, v17, v18, v19 }, [x15]     ; 64 B of weights
ld1.16b   { v20, v21, v22, v23 }, [x16]     ; 64 B of weights
ldur      d5, [x15, #-0x48]                 ; 4 × fp16 weight scales
ldur      h6, [x14, #-0x2]                  ; 1 × fp16 activation scale
sdot.4s   v7,  v20, v1                      ; ┐
sdot.4s   v24, v21, v1                      ; │
sdot.4s   v7,  v22, v2                      ; │
sdot.4s   v24, v23, v2                      ; │ 8 SDOTs
sdot.4s   v7,  v16, v3                      ; │ = 128 int8 MACs
sdot.4s   v24, v17, v3                      ; │
sdot.4s   v7,  v18, v4                      ; │
sdot.4s   v24, v19, v4                      ; ┘
addp.4s   v1, v7, v24                       ; reduce
scvtf.4s  v1, v1                            ; int → float
fcvtl/fmul                                 ; load & apply the two scales
fmla.4s   v0, v2, v1                        ; accumulate
subs      w13, w13, #0x1                    ; loop counter
b.ne      …                                 ; next block group
```

Instruments can attribute samples to individual instructions. At t=1, inside
this loop:

| Samples | Instruction | What it is |
|---:|---|---|
| 28% | `ldur h6, [x14, #-0x2]` | scalar fp16 scale load |
| 25% | `subs w13, w13, #0x1` | loop-counter decrement |
| 18% | `sdot.4s v7, v16, v3` | **the actual math** |
| 14% | `ldur d5, [x15, #-0x48]` | scale load |
| 13% | `scvtf.4s` | int→float convert |
| 2% | `ld1.16b { v16–v19 }, [x15]` | the 64 B weight load |

How to read it:

- The naive reading is wrong: "loads are 42%, optimize the loads".
- The tell is the 25% on a loop-counter *decrement*:
  - A `subs` has no cost to speak of.
  - The sampler lands on the oldest instruction that has not retired.
  - Those samples are the core **stalling**, waiting for data that has not
    arrived.
- Loads and the counter pile up. The math (`sdot`) is only 18%.
- This is the profile of a kernel limited by data arrival, not computation.

The per-iteration budget makes the same point from the other direction:

| Quantity | Value |
|---|---|
| Weight bytes per iteration | 128 B + 8 B of scales = **136 B** |
| Useful work per iteration | 4 cols × 32 weights = **128 MACs** |
| Arithmetic intensity | **0.94 MAC/byte** |

- Every weight byte is loaded from DRAM and used exactly once. That is the
  arithmetic-intensity signature of decode.
- Compute is nowhere near the limit:
  - ~2.4 GFLOP per token against multi-TFLOP/s cores.
  - Tens of thousands of tokens/s on paper.
- The only thing that can bind this kernel is the rate the bytes arrive.

## The bottleneck: weight transfer from DRAM to CPU

The whole chain:

- t=12 loses because barriers, not compute, own the machine. Threads are not
  the lever.
- At t=1, one kernel is 81.6% of decode: the gemv streaming the weights.
- Inside that kernel, the core waits for data. The arithmetic intensity
  (~1 MAC/byte) says it always will.

The byte audit:

- Decode streams **1.2439 GB of weights per token**.
  - Includes the embedding matrix, which is tied to the output head. It
    streams in full every token.
- The gemv moves those bytes in 22.52 ms.
- Effective rate: **55.2 GB/s**.

The hypothesis:

- **Decode speed is the rate of moving weights from DRAM into the CPU cores.**
- One caveat: that covers the 81.6% that streams weights. The 15.4% in
  flash_attn is a different limiter. It runs at 5.2 GB/s effective over the
  KV cache, far below any DRAM ceiling, because its cost is scalar compute
  per position, not bytes. Faster memory does not touch it.

Which raises the only question that matters: is 55.2 GB/s good?

- Apple prints 200 GB/s for the M2 Pro.
  - That is the whole chip (CPU, GPU, everything) talking to memory.
- If CPU cores could pull 200 GB/s:
  - Naive roofline: 200 ÷ 1.244 ≈ **161 tokens/s**.
  - But the naive number assumes the entire token is weight streaming. It is
    not: flash_attn plus the small ops is ~4.9 ms/token that bandwidth does
    not touch.
  - Honest roofline: 1.244 GB ÷ 200 GB/s = 6.2 ms of gemv, plus the 4.9 ms
    floor ≈ **90 tokens/s**.
  - Measured: 36.5 t/s. A ~2.5× gap, not 4.4×.
- If the real CPU-reachable ceiling is much lower:
  - The kernel may already be nearly perfect.
  - Nothing to find.

Nobody has measured it. That is Part 2: a DRAM read benchmark for this
machine, and the roofline that settles whether 55.2 GB/s is a problem or
the answer.
