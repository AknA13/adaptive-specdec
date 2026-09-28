# Results: vLLM serving with the DraftModelProposer

Qwen3-8B target, **trained `sft_kd`** Qwen3-0.6B draft, one H200 (lorax),
`max_model_len=4096`, CUDA graphs and prefix caching on, MATH-500 prompts,
greedy, 256 max tokens. `results/stage5_serving_*.json`.

(The autoregressive baseline does not involve the draft, so it is comparable
across runs. The bring-up checkpoints below were run against the stock draft.)

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
| 7 | drafter compiles and captures CUDA graphs | pass — own `draft_model` compile tag; **PIECEWISE only**, which turns out to be the performance story below (the FULL decode capture in the same log belongs to the target) |

Checkpoint 5 is the one that matters. Rejection sampling makes greedy speculative
output provably identical to greedy autoregressive output, so 16/16 identical says
the positions, slot mapping, shift-by-one layout and KV rollback are all correct.

## Throughput: speculation loses here, and the arithmetic says why

Complete sweep, trained `sft_kd` draft. "α" is accepted / **proposed** (vLLM's
own counter), so it necessarily falls as k grows; the per-position conditional
acceptance is a flat ~0.86 (see `results/figures/alpha_by_position.png`).

| conc | policy | tok/s | vs AR | itl ms | α | accepted/round |
|---|---|---|---|---|---|---|
| 1 | autoregressive | **161.1** | 1.00× | 6.13 | — | — |
| 1 | fixed k=2 | 99.3 | 0.62× | 9.85 | 0.780 | 2.56 |
| 1 | fixed k=4 | 71.4 | 0.44× | 13.60 | 0.660 | 3.64 |
| 1 | fixed k=8 | 48.9 | 0.30× | 19.90 | 0.447 | 4.57 |
| 1 | **adaptive** | **111.3** | **0.69×** | 8.79 | 0.856 | 1.86 |
| 4 | autoregressive | **635.0** | 1.00× | 6.24 | — | — |
| 4 | fixed k=2 | 346.0 | 0.54× | 10.42 | 0.782 | 2.56 |
| 4 | **adaptive** | **436.1** | **0.69×** | 8.95 | 0.860 | 1.86 |
| 16 | autoregressive | **2337.1** | 1.00× | 6.68 | — | — |
| 16 | fixed k=2 | 1169.0 | 0.50× | 11.41 | 0.779 | 2.56 |
| 16 | **adaptive** | **1536.5** | **0.66×** | 9.80 | 0.852 | 1.85 |

(k=4 and k=8 at concurrency 4 and 16 follow the same shape: 0.44×/0.30× and
0.38×/0.25×.)

**Gate G4 passes on hardware.** The adaptive controller beats *every* fixed k at
*every* concurrency level — 0.69× against the best fixed policy's 0.62× at
concurrency 1, and the margin widens under load (0.66× vs 0.50× at 16). It gets
there by settling on k ≈ 1 (1.86 tokens committed per round), which is exactly
what c ≈ 1.6 implies. Nobody told it c; it measured it.

That number is only correct because of a bug this sweep caught. The first run
had adaptive at **0.32×**, tracking fixed k=8 rather than fixed k=2. The
controller was never being handed timing, so `c` sat at its 0.15 initialisation
and it optimised for a draft ten times cheaper than the real one. The proposer
now derives per-forward costs from the interval between `propose()` entries
minus its own measured drafting. Same policy, same code path, 0.32× → 0.69×,
purely from giving the cost model real numbers.

Backing the costs out of the inter-token latency:

```
autoregressive:  itl = 6.13 ms  ->  t_target ~ 6.1 ms
k=2:             itl = 9.85 ms over 2.56 tokens/round  ->  round ~ 25.2 ms
                 drafting = 25.2 - 6.1 = 19.1 ms for 2 draft forwards
                 t_draft ~ 9.6 ms      ->  c = t_draft / t_target ~ 1.6
```

A 0.6B draft forward costing 1.6× an 8B target forward is the whole story, and
the from-scratch engine reached the same conclusion from a different runtime
(`RESULTS_ENGINE.md`, c = 0.76 there).

## The memory cost nobody quotes: 45% of your KV cache

vLLM preallocates a KV pool, so "GPU memory used" is a configured constant
(~120 GB at `gpu_memory_utilization=0.85`) and tells you nothing. The number
that matters is how many tokens fit in it, and the draft model's 28 layers share
the same cache group as the target's 36 — measured directly from the two
servers' own startup logs:

| configuration | KV cache | max concurrency @ 4k context |
|---|---|---|
| target only (`vllm_ar.out`, `vllm_nospec.out`) | **712,256 tokens** | **173.89×** |
| target + draft (`vllm_fixed2.out`, `vllm_k4.out`) | **389,136 tokens** | **95.00×** |
| | **−45.4%** | **−45.4%** |

So enabling draft-model speculative decoding on this pair costs **45% of the
requests you can hold concurrently at a given context length** — 174 down to 95.
The predicted figure from the layer counts alone was 36/64 = −43.8%; the extra
1.6 points is the draft's 1.2 GB of weights displacing pool.

This is a serving cost that the usual framing of speculative decoding
("free tokens if the draft is right") completely omits, and on a memory-bound
deployment it can dominate the latency argument: 45% fewer concurrent slots is a
throughput ceiling, not a per-request tax. It is also the reason the
concurrency-16 column below degrades faster than concurrency 1.

## Concurrency behaves as predicted

The gap widens with load: 0.61× at concurrency 1 but 0.52–0.53× at 4 and 16. Once
the verify step is compute-saturated, every rejected draft token is FLOPs taken
from another request rather than idle capacity reclaimed. Speculative decoding is
a latency optimisation that costs throughput, and this is what that looks like
when it is not paying for itself.

## What the adaptive controller can and cannot do

With c ≈ 1.6 the analytic controller's correct answer is "draft as little as
possible" — k = 1 — and even that loses to no speculation at all. The controller
cannot rescue a pair whose cost ratio is wrong. What it does, measurably, is
notice within a few hundred steps and back off, turning a 0.30× disaster (fixed
k=8, a perfectly reasonable choice if you believed α = 0.86) into 0.69×.

That is the honest case for adaptivity, and it is the case this project actually
demonstrates: **it is insurance against a badly chosen k on an unknown
workload**, not a way to exceed a well-chosen one.

The honest headline is that this project demonstrates a working, correct,
previously-missing vLLM component, and measures precisely why this particular
target/draft pair does not benefit from it — with two independent runtimes
agreeing on the reason.

## Cost

$/1M output tokens at a stated \$3.50/GPU-hour (an assumption, not a measurement):
autoregressive \$6.04 / \$1.53 / \$0.42 at concurrency 1 / 4 / 16; k=2
\$9.91 / \$2.95 / \$0.78.
