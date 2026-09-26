# Results: vLLM serving with the DraftModelProposer

Qwen3-8B target, **stock** Qwen3-0.6B draft, one H200 (lorax), `max_model_len=4096`,
CUDA graphs and prefix caching on, MATH-500 prompts, greedy, 256 max tokens.
`results/stage5_serving_*.json`.

> Status: the sweep was preempted partway (this partition is `PreemptMode=REQUEUE`
> and all 23 H200s were allocated). The autoregressive baseline and k=2 are
> measured; k=4, k=8 and the adaptive controller are queued. Every label is
> idempotent, so the requeued job resumes rather than restarting.

## The proposer is correct

All six bring-up checkpoints pass:

| # | check | result |
|---|---|---|
| 1 | engine accepts a `draft_model` speculative config | pass |
| 2 | both models load, no "Duplicate layer name" | pass — 28 attention layers under `draft_model` |
| 3 | one KV cache group for all 64 layers | pass — `GPU KV cache size: 389,136 tokens`, 95× max concurrency |
| 4 | `validate_same_kv_cache_group` | pass |
| 5 | **greedy output identical to speculation-off** | **pass — 16/16 completions byte-identical** |
| 6 | non-trivial acceptance at k>1 | pass — α = 0.78, 2.56 accepted/round at k=2 |
| 7 | drafter compiles and captures CUDA graphs | pass — own `draft_model` compile tag, PIECEWISE + FULL capture |

Checkpoint 5 is the one that matters. Rejection sampling makes greedy speculative
output provably identical to greedy autoregressive output, so 16/16 identical says
the positions, slot mapping, shift-by-one layout and KV rollback are all correct.

## Throughput: speculation loses here, and the arithmetic says why

| config | conc 1 | conc 4 | conc 16 |
|---|---|---|---|
| autoregressive | **161.1** tok/s (itl 6.13 ms) | **635.0** tok/s | **2337.1** tok/s |
| fixed k=2 | 98.1 tok/s (itl 9.98 ms) | 330.0 tok/s | 1249.6 tok/s |
| | 0.61× | 0.52× | 0.53× |

At k=2 the draft is doing its job — α = 0.78, 2.56 tokens committed per round —
and throughput still falls by 39%. Back out the per-forward costs from the
inter-token latency:

```
autoregressive:  itl = 6.13 ms  ->  t_target ~ 6.1 ms
k=2:             itl = 9.98 ms over 2.56 tokens/round  ->  round ~ 25.5 ms
                 drafting = 25.5 - 6.1 = 19.4 ms for 2 draft forwards
                 t_draft ~ 9.7 ms      ->  c = t_draft / t_target ~ 1.6
```

A 0.6B draft forward costing 1.6× an 8B target forward is the whole story. Put
c = 1.6 into the speedup formula and no k wins:

```
speedup(k) = (1 - a^(k+1)) / ((1 - a) * ((k+1)c + 1))
a = 0.78, c = 1.6  ->  best k = 1, speedup 0.55
```

Which is roughly what we measure. This is the same conclusion the from-scratch
engine reached (`RESULTS_ENGINE.md`, c = 0.76 there) arrived at from a completely
different runtime: **the draft model is not cheap enough relative to the target
for speculation to pay on this pair.**

The difference is that in vLLM it is *not* a launch-overhead artifact — the drafter
is compiled under its own `draft_model` tag and its CUDA graphs are captured. The
remaining suspects, in the order worth investigating:

1. **The drafter re-runs the whole prompt-shaped forward each step.** EAGLE's
   `propose()` does its first forward over *all* scheduled tokens, then k-1
   batch-shaped forwards. That is cheap for a 1-layer EAGLE head and much less so
   for 28 layers.
2. **Per-step attention-metadata rebuilds.** `build_for_drafting` runs once per
   draft position, on the CPU, in the critical path.
3. **A 13× parameter ratio is simply not much.** Published draft-model speedups
   mostly use ratios like 70B/7B. At 8B/0.6B, with the target already at 6.1 ms
   on an H200, there is little room for a draft to be dramatically cheaper.

## Concurrency behaves as predicted

The gap widens with load: 0.61× at concurrency 1 but 0.52–0.53× at 4 and 16. Once
the verify step is compute-saturated, every rejected draft token is FLOPs taken
from another request rather than idle capacity reclaimed. Speculative decoding is
a latency optimisation that costs throughput, and this is what that looks like
when it is not paying for itself.

## What the adaptive controller can and cannot do

With c ≈ 1.6 the analytic controller's correct answer is "draft as little as
possible" — k = 1 — and even that loses. The controller is not going to rescue a
pair whose cost ratio is wrong; what it does is *notice*, quickly, and stop
spending on speculation that is not being accepted. Measuring that is the point of
the queued `adaptive` label.

The honest headline is that this project demonstrates a working, correct,
previously-missing vLLM component, and measures precisely why this particular
target/draft pair does not benefit from it — with two independent runtimes
agreeing on the reason.

## Cost

$/1M output tokens at a stated \$3.50/GPU-hour (an assumption, not a measurement):
autoregressive \$6.04 / \$1.53 / \$0.42 at concurrency 1 / 4 / 16; k=2
\$9.91 / \$2.95 / \$0.78.
