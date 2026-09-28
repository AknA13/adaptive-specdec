# Status

Last updated 2026-09-26, after the first GPU runs.

## Done

Everything that can be built and verified without a GPU allocation.

- `specdec/` — engine, ragged KV cache, sampling, controllers. Gates G1 and the
  CPU half of G2 pass.
- `vllm_draft_spec/` — `DraftModelProposer` + 4 patches, installed as a plugin
  (`pip install -e . --no-deps --no-build-isolation`, already done in the `rl`
  env). Bring-up checkpoint 1 passes on the login node.
- `data/`, `train/` — trace generation, filtering, FSDP2 trainer with SFT + top-k
  KD. Losses and the collator are unit-tested; the trainer itself has not run.
- `bench/`, `scripts/`, `slurm/` — full harness, all stages idempotent.
- 7 CPU test files, `make test` green.

## Measured so far

The pipeline has run end to end. See `RESULTS_ENGINE.md`, `RESULTS_TRAINING.md`
and `RESULTS_SERVING.md`.

- **G1, G2, G4, G6 pass.** Losslessness (TV 0.0056 with all four injected bugs
  caught), greedy identity (16/16 through the vLLM proposer), the adaptive
  controller beating every fixed k at every concurrency, and all six vLLM
  bring-up checkpoints.
- **G3 fails informatively**: +0.021 acceptance in-domain, −0.017
  out-of-domain. Qwen3-0.6B is already distilled from the Qwen3 family, so
  fine-tuning is domain adaptation rather than an acceptance win.
- **G5 fails for a measured reason**: the draft forward costs ~1.6× the target
  forward, because vLLM's drafter is limited to PIECEWISE CUDA graphs while the
  target gets FULL ones. No draft length wins at that cost ratio.

## Next, in order

1. **Let the drafter take FULL cudagraphs.** This is the single highest-value
   change and it is upstream, not local: `eagle.py` hard-codes PIECEWISE at
   :295, :398, :801, :1173. A 28-layer draft launches ~56 eager attention
   kernels per round; an EAGLE head launches 1, which is why nobody hit this.
2. **Instrument `propose()` with CUDA events** to attribute the 9.6 ms draft
   step between the two forwards, metadata rebuilds, and attention launches,
   rather than inferring it from inter-token latency.
3. **A shallower draft.** Depth, not parameter count, drives per-step latency
   under PIECEWISE. Qwen3-0.6B is 28 layers; a 1.7B model with fewer layers
   could well be the cheaper drafter.
4. **Re-run G3 properly**: held-out eval distribution, several seeds, and enough
   problems to resolve 0.01. The whole pipeline is ~25 GPU-minutes, so this is
   affordable. The current KD-vs-SFT gap (0.853 vs 0.850) is not resolvable at
   n=24.
5. Static-shape mask in the ragged cache so the engine can use CUDA graphs; the
   microbenchmark says that is worth 4.58× on the draft step.

## Known gaps

- The G3 comparison is n=24 problems per cell. Directionally clear (+0.021 vs
  -0.017 is well outside noise at that size) but the KD-vs-SFT difference is
  not.
- The serving sweep used the trained `sft_kd` draft; the bring-up checkpoints
  used the stock one. The autoregressive baseline is draft-independent, so the
  comparison holds, but the two are not from one run.
- The vLLM controller is batch-uniform. Per-request `k_i` is supported by
  everything downstream (`SpecDecodeMetadata.num_draft_tokens` is a `list[int]`,
  `rejection_sample` is driven by `cu_num_draft_tokens`, `DraftTokenIds` is
  `list[list[int]]`) — it needs `propose()` to return a trimmed `list[list[int]]`
  and nothing else.
- `ConfidenceEarlyExit` makes one decision for the whole batch (the mean running
  confidence). Exact at batch 1, a compromise above it.
- No tree drafting. `EagleProposer.propose_tree` exists and is inherited but is
  not wired up; it is a separate project.
- `bench_engine.py`'s long-prompt variant prepends sibling problems as context.
  Realistic for prefill cost, but it is not a natural long-context workload.

## Cluster notes

- **The conda env must be node-local.** 2026-09-28, measured on horton via
  srun: `import torch` from the /scratch (NFS) env took 690 s; stat of 200 env
  files 0.19 s; reading 100 MB from /data 0.06 s; load 10.8. The NFS mount's
  bulk-read throughput to the compute nodes is what failed. The full 8.1 GB
  `rl` env is now replicated to `/data/$USER/envs/rl_node` on horton and lorax
  with the plugin installed; `scripts/lib.sh` picks it automatically. If a new
  node is added, copy the env there before running anything on it. (The older
  `rl_local` on horton is a hollow skeleton with empty `torch/` and `vllm/`
  and must not be used.)
- **thidwick is unusable for this env.** 2026-09-28: the env check's
  `packages` step (import torch) took 52 minutes there, `models` (read a
  config.json) 43 minutes, and a vLLM server produced no log output in 30
  minutes. The same steps take seconds on horton and lorax. Both benchmark jobs
  that landed there were cancelled after wasting ~2.6 GPU-hours. The conda env
  lives on /scratch (NFS from oz); whatever is wrong is between thidwick and
  that mount. `SPEC_SLURM_EXCLUDE` now defaults to `thidwick`.
- **QOS: use `preemptive`, not `normal`.** `sacctmgr` shows `normal` is the
  lowest tier on this cluster (displaced by `preemptive` and `preemptive_high`);
  `preemptive` is only displaced by `preemptive_high`. The first `gaps` attempt
  on `normal` was bumped 2.5 minutes in.
- **Priority queue reality.** Even at #2/#3 in the partition, a single array job
  ahead consumed every freed GPU for hours. Floating across horton and lorax
  (checkpoints staged to both) is the only lever left.

- All 23 `berkeleynlp` H200s were allocated when this was written; expect to
  queue. Every partition is `PreemptMode=REQUEUE`, so every stage skips work
  that already exists on disk and every job uses `--requeue`.
- `$HOME` has ~2 GB free and `/scratch` ~10 GB. Nothing large may go there;
  `SPEC_DATA_ROOT` must be `/data/$USER/...`. `scripts/00_check_env.sh` refuses
  to run otherwise.
- The `rl` env is fragile: torch 2.9.0 with torch 2.11.0 dist-info, cu12 and
  cu13 wheels side by side. **Do not run a pip install that resolves torch.**
  The plugin was installed with `--no-deps --no-build-isolation` for this reason.
- A bare `df -h` hangs on the login node — it walks all 15 `/net` automounts.
  Scope it (`df -h /net/horton/data`). Same hazard for `find`/`du` from `/`.
