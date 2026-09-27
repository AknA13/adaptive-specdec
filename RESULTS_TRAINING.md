# Results: training the draft (gate G3)

3,000 gsm8k train problems → Qwen3-8B thinking-mode traces with top-8 teacher
logprobs → filter → FSDP2 fine-tune of Qwen3-0.6B on 2 H200s.
`results/stage2_filter.json`, `results/stage3_train_*.json`, `results/stage4_g3*.json`.

## The corpus

| | |
|---|---|
| generated | 3,000 traces, 18 min on one H200 |
| kept | **2,015 (67%)** |
| rejected: over the length window | 712 |
| rejected: truncated (hit max_tokens) | 262 |
| rejected: **wrong answer** | **11** |
| completion length | mean 1,352, p50 1,343, p90 1,835 |

Only 11 of 3,000 traces got gsm8k wrong — Qwen3-8B is close to saturated on this
benchmark, which matters for the result below: there is very little signal in
"learn to be correct" here, so the fine-tune is almost purely a style/distribution
match.

Both arms trained in ~5 minutes on 2 GPUs, 126 optimizer steps, loss 0.41 → 0.24
(`sft`) with the KD term active at 0.14–0.18 (`sft_kd`).

## G3: training helps in-domain and hurts out-of-domain

Conditional acceptance α at k=4, greedy, against Qwen3-8B:

| draft | MATH-500 (out-of-domain) | gsm8k test (in-domain) |
|---|---|---|
| stock Qwen3-0.6B | **0.867** | 0.843 |
| `sft` (λ=0) | 0.850 (**−0.017**) | **0.864** (**+0.021**) |
| `sft_kd` (λ=0.5) | 0.853 (−0.013) | 0.863 (+0.021) |

**G3 as written (≥ +0.05 absolute over stock) fails.** What actually happened is
more interesting than a threshold:

- **In-domain the fine-tune works,** +0.021 for both arms. The draft agrees with
  the target more often on the distribution it was trained on.
- **Out-of-domain it costs more than it gains,** −0.017 / −0.013. Training on
  grade-school arithmetic moved the draft away from competition math, and
  acceptance is a similarity metric, so any specialisation away from the target's
  behaviour on the eval distribution shows up directly.

## Why the headroom is so small

Qwen3-0.6B is not an arbitrary small model — it is a member of the same family,
trained by its authors with distillation from larger Qwen3 models. It is
*already* close to the best available draft for Qwen3-8B, and α = 0.867 out of
the box says so. Two thousand gsm8k traces and 126 steps cannot add much to that,
and can easily subtract by narrowing the distribution.

The lesson generalises: **when the draft is already a distilled sibling of the
target, the default assumption should be that fine-tuning is a domain-adaptation
tool, not an acceptance-improvement tool.** It pays when you know the serving
distribution and can train on it; it costs when you guess wrong.

## KD vs SFT

`sft_kd` is marginally better than `sft` out-of-domain (0.853 vs 0.850) and
indistinguishable in-domain (0.863 vs 0.864). The direction matches the theory —
acceptance is exactly `1 - TV(p, q)`, and KD matches the distribution while
cross-entropy only matches the argmax, so KD should generalise slightly better —
but a 0.003 gap on 24 problems is not evidence. Calling this a win for KD would
be overreading it.

A fair test would need the eval distribution held out from training, several
seeds, and enough problems to resolve 0.01. That is the experiment to run next,
and it is cheap: the whole pipeline is ~25 GPU-minutes end to end.

## What this does not change

Acceptance was never the binding constraint. At α = 0.867 the speedup formula
still returns < 1 because the cost ratio c ≈ 1.6 dominates (see
`RESULTS_SERVING.md`). Raising α from 0.84 to 0.87 moves nothing that matters
while a draft forward costs more than a target forward.
