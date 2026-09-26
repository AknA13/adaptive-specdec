#!/bin/bash
# Stage 4: from-scratch engine benchmark. --quick for a smoke-sized grid.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
QUICK=0; [ "${1:-}" = "--quick" ] && { QUICK=1; shift; }
NG=$(n_gpus); [ "$NG" -lt 1 ] && die "no GPU visible"

if [ "$QUICK" = 1 ]; then
  step "quick engine check (greedy identity + speedup on a few problems)"
  run "$PY" -m bench.bench_engine --draft "$DRAFT" --draft-label stock \
      --n 8 --max-new 64 --k-grid 1,4 --controllers ewma \
      --modes greedy --prompt-lens short --tag quick "$@"
  exit 0
fi

# Stock draft first: it is the baseline G3 measures the trained draft against.
step "engine benchmark: stock draft (baseline for gate G3)"
run_soft "$PY" -m bench.bench_engine --draft "$DRAFT" --draft-label stock \
    --n "${SPEC_BENCH_N:-40}" --max-new "${SPEC_BENCH_NEW:-256}" "$@"

for name in sft sft_kd; do
  ck="$DATA/ckpt/$name"
  [ -s "$ck/config.json" ] || { warn "no checkpoint $ck -- skipping"; continue; }
  step "engine benchmark: $name"
  run_soft "$PY" -m bench.bench_engine --draft "$ck" --draft-label "$name" \
      --n "${SPEC_BENCH_N:-40}" --max-new "${SPEC_BENCH_NEW:-256}" "$@"
done
report_soft_fails
