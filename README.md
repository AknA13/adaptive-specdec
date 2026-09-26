# Adaptive Speculative Decoding

Speculative decoding for Qwen3-8B with a Qwen3-0.6B draft model, built three ways:

1. **A from-scratch engine** — batched draft/verify loop, exact rejection sampling,
   and a KV cache that rolls back per row in O(1). Proven lossless by a statistical
   test, not by inspection.
2. **A draft-model proposer for vLLM V1**, which vLLM does not otherwise have.
3. **An adaptive controller** that picks the draft length each step by maximising
   expected speedup from an online acceptance rate and an online cost ratio.

Plus the training pipeline that makes the draft worth using (Qwen3-8B traces →
filter → FSDP2 fine-tune with top-k KD) and the benchmark/profiling harness that
measures whether any of it paid off.

---

## The part that is not a config flag

vLLM 0.12.0 **cannot** do speculative decoding with a separate draft model:

```
vllm/config/speculative.py:377    self.method = "draft_model"
vllm/config/speculative.py:379    raise NotImplementedError(
                                    "Speculative decoding with draft model is
                                     not supported yet. ...")
```

and `v1/worker/gpu_model_runner.py:374-396` only ever constructs an
`Ngram | Suffix | Eagle | Medusa` proposer. So `vllm_draft_spec/` supplies the
missing `DraftModelProposer` and the four patches that let the rest of the stack
reach it.

It ships as an out-of-tree plugin (a `vllm.general_plugins` entry point), not a
fork. Two reasons: the conda env on this cluster has torch 2.9.0 under torch
2.11.0 metadata with both cu12 and cu13 wheels, so any pip run that resolves
torch would break vLLM outright; and the env is shared, so `register()` is gated
on `SPECDEC_PLUGIN=1` and is inert for everyone else.

**What made it tractable.** Two things that look like the hard parts turn out to
be free, both because of properties of the models rather than cleverness:

- EAGLE's shift-by-one input layout — the drafter's KV at position `p` encodes
  token `p+1` — is *already correct* for a plain causal draft model, because
  Qwen3 uses pure RoPE with no sliding window, so shifting every position
  uniformly changes no relative distance. That means `propose()`'s positions,
  seq_lens, slot-mapping and attention-metadata machinery transfers verbatim,
  and **there is no separate draft prefill**: the drafter rides the target's
  token stream through chunked prefill, preemption and recompute automatically.
- Target and draft are both 8 KV heads × 128 head_dim × bf16, so their per-layer
  `FullAttentionSpec` is identical and all 36 + 28 = **64 layers land in one KV
  cache group sharing one block table**. No KV-manager changes at all. The cost
  is budgetary, not structural: the same `gpu_memory_utilization` now buys ~44%
  fewer KV blocks.

Subclassing `EagleProposer` means every `isinstance(self.drafter, EagleProposer)`
check in the runner keeps passing, so the runner needs no patch beyond rebinding
the drafter.

## Correctness first

A broken speculative decoder does not crash and does not produce gibberish. It
produces fluent text from subtly the wrong distribution, at a perfectly healthy
acceptance rate. Reading samples cannot detect that, so the losslessness claim is
tested statistically.

**G1 — distributional equivalence.** `tests/test_rejection_sampling.py` compares
the engine's output distribution against the exactly-computed autoregressive
marginal (small vocab, so it is enumerable):

```
token 1 matches exact AR marginal   TV=0.0056   gate=0.02   chi2 p=0.623
```

which sits right at the sampling-noise floor of `sqrt(V / 2*pi*N) = 0.0056`.

A correctness test that cannot fail is worth nothing, so the same file injects
four real bugs and requires the test to catch each one:

| injected bug | TV distance | detected |
|---|---|---|
| skip the accept test | 0.466 | yes |
| resample from `p` instead of the residual | 0.144 | yes |
| mismatched top-p between `p` and `q` | 0.221 | yes |
| *(correct implementation)* | 0.0014 | — |

**G2 — greedy equivalence.** Greedy is not special-cased: temperature 0 makes
`to_probs` return a one-hot and the same rejection sampler runs, so token-identical
greedy output is a consequence of the same proof. `tests/test_greedy_equivalence.py`
checks it at every k and every controller, at α ≈ 0.55 so the rejection path is
genuinely exercised — with an explicit assertion that α < 0.99, because two
untrained tiny models collapse to the same argmax and would make the file vacuous.

Three real bugs were found by these tests rather than by reading the code:

- `generate()` terminated on `max(len)` across the batch, so rows that accepted
  fewer tokens came back short.
- `mean_accepted_len` did not divide by batch size.
- the controller's EWMA decayed per *round*, so its window shrank to ~20 tokens
  whenever k collapsed to 1, and the controller oscillated. It now decays per
  token; α̂ is unbiased to ±0.01 with sd 0.03, and the modal k matches the oracle
  k\* exactly at α ∈ {0.45, 0.7, 0.9}.

## The controller

Acceptance rate is, exactly,

```
E_{x~q}[min(1, p(x)/q(x))] = sum_x min(p(x), q(x)) = 1 - TV(p, q)
```

so choosing a draft length is a decision problem with a closed-form objective.
From Leviathan et al., expected tokens per round is `(1 - a^(k+1)) / (1 - a)`, and
a round costs `(k+1)*c + 1` target-forwards, where `c = t_draft / t_target`. The
controller maximises their ratio over `k ∈ [1, k_max]` each step, with both `a`
and `c` measured online. No tuned thresholds, and `c` self-calibrates to the
hardware.

The `(k+1)` is not a typo — the engine runs one extra draft forward per round to
commit the last drafted token into the draft cache, which keeps both caches
exactly in lockstep and costs `1/k`.

**The honest claim.** Adaptive k does *not* beat the best oracle fixed k; it
matches it (within 0.5% in simulation) without being told the workload, and it
wins when the workload moves — **+17.7%** over the best single fixed k on a
workload that alternates between α=0.9 and α=0.35. Gate G4 is written to claim
exactly that.

The same `Controller` interface drives both the from-scratch engine and the vLLM
proposer.

## Layout

```
specdec/        engine.py  kvcache.py  sampling.py  controller.py
vllm_draft_spec/  proposer.py  patches.py     # the missing vLLM component
data/           gen_traces.py  filter.py  common.py
train/          train_draft.py  losses.py  data.py
bench/          bench_engine.py  bench_serving.py  profile_engine.py  plots.py
tests/          7 files, all CPU-only
scripts/        00_check_env .. 06_profile, run_pipeline
slurm/          submit.sh       # generates logs/<job>.job.sh, enforces a 4-GPU cap
```

`specdec/kvcache.py` is where the engineering is. Speculative decoding ends every
round with a partial rollback, and with batch > 1 the accepted length differs per
row. `DynamicCache.crop()` takes one scalar and slices the whole batch; padding
every row to the max accepted length inflates the cache ~2× and would corrupt the
memory numbers we report. So: one preallocated `[B, H, C, D]` buffer per layer
plus a per-row `lengths` vector. Writes scatter to each row's own offset, rollback
is `lengths -= n`, and the mask expression `s <= lengths[b] + j` makes the stale
tail unreadable for free. No allocation, no copy, no sync.

## Running it

```bash
cp env.sh.example env.sh && $EDITOR env.sh     # SPEC_DATA_ROOT must be node-local /data
make test                                       # all gates that do not need a GPU, ~2 min
make all                                        # submit the SLURM chain
```

Individual stages, each idempotent and preemption-safe:

```bash
scripts/00_check_env.sh
scripts/01_gen_traces.sh        # Qwen3-8B traces + teacher top-8 logprobs
scripts/02_filter.sh            # verify / dedupe / decontaminate
scripts/03_train_draft.sh       # FSDP2; trains both `sft` and `sft_kd`
scripts/04_bench_engine.sh      # enforces G2 and G5
scripts/05_bench_vllm.sh --smoke  # vLLM bring-up checkpoints 1-6
scripts/06_profile.sh           # torch.profiler + cost ratio + CUDA-graph headroom
```

## Measured results

Full write-ups in `RESULTS_ENGINE.md` and `RESULTS_SERVING.md`. Qwen3-8B target,
**stock** Qwen3-0.6B draft, one H200.

**The proposer works and is correct.** All bring-up checkpoints pass, including
the one that matters: **16/16 greedy completions through the DraftModelProposer
are token-identical to speculation-off**. The shared KV cache group reports
`389,136 tokens` across all 64 layers, and the drafter compiles under its own tag
with CUDA graphs captured.

**Speculation does not pay on this model pair, and both runtimes agree why.**

| | α | accepted/round | vs autoregressive |
|---|---|---|---|
| from-scratch engine, k=4 | 0.91 | 4.10 | 0.85× |
| from-scratch engine, adaptive | 0.91 | 4.38 | 0.88× |
| vLLM, k=2 (conc 1) | 0.78 | 2.56 | 0.61× |

Acceptance is high — an *untrained* 0.6B draft agrees with the 8B 91% of the time
on math — and throughput still falls. The cost ratio is why:

- **From-scratch engine:** Self CPU 3.52 s vs Self CUDA 0.50 s, and a batch-8
  forward costs the same 21.4 ms as a batch-1 forward. Nothing is compute-bound,
  so the 0.6B costs 76% of the 8B (c = 0.76) instead of ~1/13. CUDA-graphing one
  draft decode: **21.4 ms → 4.7 ms, 4.58×, so 78% of the step was launch
  overhead.**
- **vLLM:** not a launch artifact — the drafter is compiled and graphed — but
  backing t_draft out of the inter-token latency still gives c ≈ 1.6.

Feed those into the speedup formula the controller maximises and it predicts
0.93× and 0.55× respectively. Measured: 0.88× and 0.61×. **The controller is not
underperforming; it is correctly reporting that there is no speedup available on
this pair.**

## Gates

A stage is not done until its gate passes, or until its failure is reported with
the metric that shows it.

| | gate | status |
|---|---|---|
| G1 | losslessness: TV < 0.02 vs the exact AR marginal | **pass** — 0.0056, χ² p=0.62, all 4 injected bugs caught |
| G2 | greedy output token-identical to autoregressive | **pass** — CPU exact; on GPU every divergence is a bf16 tie (gap 0.25 vs 2 ULP 0.34), zero hard mismatches; vLLM path 16/16 identical |
| G3 | trained draft raises acceptance ≥ 0.05 over stock | queued (trace gen → FSDP training) |
| G4 | adaptive ≥ 95% of best fixed k everywhere, ≥ it somewhere | **pass in simulation** (+17.7% on a shifting workload); adaptive beats every fixed k measured on GPU |
| G5 | ≥ 1.5× tokens/s vs autoregressive | **fails, and explained** — 0.88× (engine) / 0.61× (vLLM); the cost ratio makes it unreachable, see above |
| G6 | vLLM bring-up checkpoints 1–6 | **pass** — including 16/16 greedy identity |

## Caveats, stated up front

- **FSDP at 0.6B is not load-bearing.** The model trains fine on one H200 under
  DDP. FSDP2 is used for optimizer-state sharding and pipeline parity with a
  larger draft, and the README says so rather than implying necessity.
- **The draft is 28 layers, not a 1-layer EAGLE head.** At batch 1 on an H200,
  expect ~0.5–0.8 ms per draft forward against ~3–4 ms for the target, so k=4
  spends 60–80% of a target step on drafting and needs a mean accepted length
  ≳ 2.2 just to break even. That is precisely the regime where adaptive k earns
  its keep.
- **Nsight Systems is unavailable on this cluster** (no `nsys`/`ncu`, no CUDA
  module), so profiling is `torch.profiler` only. CUPTI is present, so kernel
  attribution and Chrome traces work.
- **CUDA graphs are measured, not shipped.** `profile_engine.py` graphs a
  fixed-shape draft decode to quantify how much of the step is launch overhead.
  Using graphs in the engine needs a static attention mask, which the ragged
  cache does not currently provide; that is the next optimisation, and the
  microbenchmark says what it is worth.
- **Cost per token is an assumption, not a measurement** — throughput divided by
  a stated GPU hourly rate, labelled as such wherever it appears.

## References

Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative
Decoding* (2023) — the rejection-sampling scheme and the expected-speedup formula
the controller maximises. Chen et al., *Accelerating Large Language Model Decoding
with Speculative Sampling* (2023).
