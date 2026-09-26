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

See `RESULTS_ENGINE.md` and `RESULTS_SERVING.md`. Short version: the engine and
the vLLM proposer are both correct (G1, G2, G6 pass, including 16/16 greedy
identity through the proposer), and speculation is slower than autoregressive on
this model pair because the 0.6B draft is not cheap enough relative to the 8B
target. Two independent runtimes give the same diagnosis from different
directions.

## Queued right now

All 23 berkeleynlp H200s and all four jsteinhardt H200 nodes were allocated at
the time of writing, so these are pending on priority:

| job | what | note |
|---|---|---|
| 3612394 | serving sweep (k=4, k=8, adaptive) | was preempted once; now resumes per label |
| 3612402 | trace generation, 3000 gsm8k train problems | |
| 3612403 | filter | after traces |
| 3612416 | FSDP training, `sft` and `sft_kd` | after filter, 2 GPUs |

Together those close G3 and fill in the rest of the serving table.

## Next, in order

1. Let the queued jobs land, then `python -m bench.plots` and fold the numbers
   into `RESULTS_SERVING.md`.
2. Chase the drafter cost in vLLM. It is compiled and graphed, so the suspects
   are structural: `propose()` runs its first forward over *all* scheduled
   tokens (cheap for a 1-layer EAGLE head, much less so for 28 layers), and
   `build_for_drafting` rebuilds attention metadata on the CPU once per draft
   position. Instrumenting `propose()` with CUDA events would settle it.
3. A smaller or shallower draft is the obvious lever on c. Qwen3-0.6B is 28
   layers; depth, not parameter count, is what costs per-step latency.
4. Re-run the engine benchmark once a static-shape mask makes CUDA graphs usable
   there; the microbenchmark says that is worth 4.58x on the draft step.

## Known gaps

- Both `sft` and `sft_kd` are untrained so far: every number above uses the
  **stock** Qwen3-0.6B. Training should raise alpha, but alpha is not the
  binding constraint here -- the cost ratio is -- so expect G3 to pass without
  moving G5.
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
