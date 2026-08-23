---
title: "Counting the barriers"
weight: 10
---

# Counting the barriers

The main page says one generated token costs ~270 barriers. This page is the
arithmetic, so the number is checkable instead of folklore.

## The rule

From `ggml-cpu.c`:

- Every *executed* graph node pays one `ggml_barrier`.
- Three kinds of nodes pay nothing:
  - Empty ops: view, permute, reshape, transpose. Skipped.
  - Nodes without the COMPUTE flag. Skipped.
  - Fused ops. Several nodes merge into one execution.
- So: barriers per token = number of real compute ops.

## The anchor that does not work

- `llama-cli --verbose` prints `sched_reserve: max_nodes = 1192`.
- I first read that as "the graph has ~1,100+ nodes". Wrong.
- The source (`src/llama-context.cpp`, `graph_max_nodes()`):

```cpp
uint32_t res = std::max<uint32_t>(1024u, 8u * model.n_tensors());
```

- It is a buffer-reservation rule of thumb, proportional to tensor count.
  It says nothing about graph size.
- Why 1192 and not 8 × 148 = 1184: `n_tensors()` counts every tensor the
  loader allocates. This model has no separate `output.weight`, so the loader
  duplicates `token_embd` to serve as the tied output head. That is 148 + 1
  = 149 tensors, and 8 × 149 = 1192.

## The real count: itemize the graph

Sources: `src/models/lfm2.cpp` for the block structure, the shared builders
in `src/llama-graph.cpp` for what each builder emits.

The model is hybrid: 6 attention blocks + 10 conv blocks.

### Shared by every block

| op | count |
|---|---:|
| operator norm | 1 |
| ffn norm | 1 |
| gate proj, up proj, down proj (`mul_mat`) | 3 |
| swiglu (one fused op) | 1 |
| two residual adds | 2 |
| **shared total** | **8** |

### One attention block

| op | count |
|---|---:|
| q, k, v projections (`mul_mat`) | 3 |
| q norm, k norm | 2 |
| RoPE on q, on k | 2 |
| KV cache writes for k and v | 2 |
| flash_attn_ext | 1 |
| output projection (`mul_mat`) | 1 |
| shared | 8 |
| **total** | **19** |

### One conv block

| op | count |
|---|---:|
| in_proj (`mul_mat`) | 1 |
| mul (b·x gating) | 1 |
| concat (conv state prepended to input) | 1 |
| cpy (write new conv state back) | 1 |
| ssm_conv | 1 |
| mul (c·conv_out) | 1 |
| out_proj (`mul_mat`) | 1 |
| shared | 8 |
| **total** | **15** |

### Global, outside all blocks

| op | count |
|---|---:|
| embedding get_rows | 1 |
| final norm | 1 |
| output head (`mul_mat`) | 1 |

### The sum

```
 6 attention blocks × 19 = 114
10 conv blocks     × 15 = 150
global                   =   3
                         ≈ 267
```

- ~270 barriers per token.
- Slightly fewer in practice: the CPU backend fuses each RMS_NORM + MUL pair
  into one op.
  - This graph has 45 weighted norms: 16 input norms, 16 ffn norms, 12 q/k
    norms (attention blocks only), 1 output norm (stored as `token_embd_norm`
    in this GGUF).
  - Each pair is two nodes; fusion removes one. Up to 45 nodes can merge,
    so the floor is closer to 267 − 45 ≈ 220 barriers.

## Sanity: cost per barrier

- t=6, measured barrier self-time: 1.31 ms/token.
  - From a separately sampled t=6 run (that run measured 56.4 t/s).
- 1.31 ms ÷ 270 ≈ **5 µs per barrier**.
  - With the fused floor from above (~220 barriers), it is ~6 µs. Same
    ballpark either way.
- Plausible for a spin barrier waiting on 6 memory-starved threads.
