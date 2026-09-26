#!/bin/bash
# Stage 3: FSDP2 draft training. Trains both ablation arms unless --name is given.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need_file "$DATA/traces/filtered.jsonl" "run scripts/02_filter.sh first"
NG=$(n_gpus); [ "$NG" -lt 1 ] && die "no GPU visible"
PORT="${SPEC_MASTER_PORT:-$((29500 + RANDOM % 1000))}"
# Leave a margin under the SLURM limit so the run checkpoints instead of dying.
STOP="${SPEC_STOP_AFTER_SEC:-0}"

train_one() {
  local name="$1"; local lam="$2"; shift 2
  if [ -s "$DATA/ckpt/$name/config.json" ]; then
    info "ckpt/$name exists -- skipping"; return 0
  fi
  step "training $name (lambda_kd=$lam) on $NG GPU(s)"
  torchrun --nproc_per_node="$NG" --master_port="$PORT" -m train.train_draft \
      --name "$name" --lam "$lam" --stop-after-sec "$STOP" "$@"
}

if [ $# -gt 0 ]; then
  # explicit passthrough, e.g. scripts/03_train_draft.sh --name sft_kd --lam 0.5
  step "training (explicit args)"
  run torchrun --nproc_per_node="$NG" --master_port="$PORT" -m train.train_draft "$@"
else
  # The ablation the acceptance-rate claim rests on: same data, same steps,
  # only the objective differs.
  train_one sft 0.0    || die "sft failed"
  train_one sft_kd 0.5 || die "sft_kd failed"
fi
echo "[03] done"
