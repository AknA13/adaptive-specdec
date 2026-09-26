# Status

Last updated 2026-09-26.

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

## Next, in order

1. `scripts/00_check_env.sh` on horton. Qwen3-0.6B is cached **only on horton**
   and `/data` is node-local, so every stage must pin `-w horton` or pre-stage
   the weights.
2. `make all` — the SLURM chain. Budget ~7 GPU-hours total.
3. `scripts/05_bench_vllm.sh --smoke` is the one to watch: checkpoint 5 (greedy
   output at k=1 token-identical to no-spec) is the correctness oracle for the
   whole vLLM path. **If it fails, no speedup number from that path means
   anything** — it indicates a positions / slot-mapping / shift bug, not
   something to tune.
4. Fill in G3 and G5 from `results/stage4_*.json`, then `python -m bench.plots`
   and write `RESULTS_*.md`.

## Known gaps

- The vLLM proposer has never executed a forward pass. Checkpoints 2–7 in the
  README are untested, and `propose()` is the most likely place for a bug.
  Debug at `--max-num-seqs 1 --enforce-eager --no-enable-prefix-caching
  --max-model-len 4096` first, then re-enable one feature at a time.
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
