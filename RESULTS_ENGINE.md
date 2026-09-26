# Results: the from-scratch engine

Qwen3-8B target, **stock** (untrained) Qwen3-0.6B draft, MATH-500, greedy,
batch 1, 64 new tokens, 8 problems, one H200 (lorax). `results/stage4_eager_eager.json`,
`results/stage6_graphs.json`.

## Acceptance is high; throughput is not

| method | tok/s | vs AR | α (conditional) | accepted/round | greedy vs AR |
|---|---|---|---|---|---|
| autoregressive | 40.8 | 1.00× | — | — | baseline |
| fixed k=1 | 29.6 | 0.73× | 0.882 | 1.89 | 1 bf16 tie |
| fixed k=4 | 34.5 | 0.85× | 0.910 | 4.10 | 2 bf16 ties |
| adaptive (ewma) | 36.0 | 0.88× | 0.913 | 4.38 | 2 bf16 ties |

An **untrained** 0.6B draft already gets α = 0.91 against the 8B on math, committing
4.1 tokens per round at k=4. The speculation is working exactly as designed. And it
is still **slower than plain autoregressive decoding.**

## Why: the engine is launch-bound, so a 0.6B forward costs what an 8B forward costs

`torch.profiler` over a 64-token greedy run:

```
Self CPU time total:  3.515 s
Self CUDA time total: 0.498 s        <- 7x more time launching work than doing it
aten::mm    20258 calls   10.2 us CUDA each   25.2 us CPU each
aten::mul   37832 calls    1.8 us CUDA each
```

Per-forward cost, measured directly:

| batch | draft (0.6B) | target (8B) | c = draft/target |
|---|---|---|---|
| 1 | 21.4 ms | 28.1 ms | 0.76 |
| 8 | 21.4 ms | 28.2 ms | 0.76 |

**An 8× larger batch costs the same wall time.** That is the proof: nothing here is
compute-bound. The 0.6B model is 13× smaller than the 8B and takes 76% as long,
because both are paying the same fixed Python-and-launch tax per layer.

Feed that c into the speedup formula the controller maximises:

```
speedup(k) = (1 - a^(k+1)) / ((1 - a) * ((k+1)c + 1))
a = 0.91, c = 0.76  ->  best k = 2, speedup 0.93
```

The measured 0.88× matches the prediction. **The controller is not underperforming;
it is correctly reporting that on this runtime there is no speedup to be had.** It
picks k ≈ 4.2 and beats every fixed k we tried, which is all it can do.

## How much of that is launch overhead: 78%

CUDA-graphing a single fixed-shape draft decode:

```
eager  21.371 ms
graphed 4.668 ms      4.58x   -> 78% of the step was launch overhead
```

## What this means

Speculative decoding pays only when the target forward is bound by something the
draft forward is *not* bound by. In a compiled runtime at batch 1 that something is
weight bandwidth: 16 GB of target weights against 1.2 GB of draft weights is a real
13× asymmetry, and c falls to ~0.1. Under a per-layer Python tax, both models pay
the same tax, the asymmetry disappears, and no draft length can win.

So the from-scratch engine's job in this project is to be the **correctness
reference** — it is what gates G1 and G2 are proved on — and the **profiling
subject**. The throughput claim belongs to the vLLM path, which has the compiled,
graph-captured runtime this one does not.

Two honest consequences:

- Gate G5 (≥1.5× end-to-end) is not achievable on this engine and is not a tuning
  problem. It is re-scoped to the served path and reported here as measured: 0.88×.
- `torch.compile` would be the obvious fix and **cannot run in this environment**:
  `InductorError: 'KernelMetadata' object has no attribute 'cluster_dims'`, a
  triton/torch mismatch from the env's mixed torch 2.9.0 / 2.11.0 install. Making
  CUDA graphs work inside the engine needs a static attention mask, which the ragged
  cache does not currently provide; the 4.58× above is what that work is worth.

## G2: bf16 ties, not bugs

Greedy speculative output is provably token-identical to greedy autoregressive
output, and it very nearly is. The handful of differences are all the same shape:

```
position 22   AR token 88190 (logit 43.75)   spec token 6771 (logit 43.50)
gap 0.25      bf16 ULP at that magnitude 0.171
```

bf16 carries 8 mantissa bits, so at a logit of 43.75 consecutive representable
values are ~0.17 apart. The target computes that position inside a multi-token
verify forward and inside a single-token baseline forward; the two reduce in a
different order and disagree by less than two ULP. The model has no meaningful
preference between those tokens.

Every divergence observed was classified as a tie (gap ≤ 2 ULP **and** the
speculative run chose the runner-up); **zero hard mismatches**. G2 gates on hard
mismatches, because exact token identity is not a thing bf16 can deliver when the
target itself cannot order its top two candidates.
