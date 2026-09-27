#!/bin/bash
# Stage 1: teacher reasoning traces + top-k logprobs. Idempotent; --resume safe.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need_data
DATASET="${SPEC_TRACE_DATASET:-gsm8k}"; SPLIT="${SPEC_TRACE_SPLIT:-train}"
N="${SPEC_TRACE_N:-0}"; SHARDS="${SPEC_TRACE_SHARDS:-$(n_gpus)}"
[ "${SHARDS:-0}" -lt 1 ] && SHARDS=1

if have_output "$DATA/traces/filtered.jsonl"; then
  info "filtered.jsonl already exists -- stage 1 and 2 are done"; exit 0
fi
step "generating $DATASET/$SPLIT traces with $TARGET across $SHARDS shard(s)"
wait_free 40000
pids=()
for s in $(seq 0 $((SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$s "$PY" -m data.gen_traces \
      --model "$TARGET" --dataset "$DATASET" --split "$SPLIT" --n "$N" \
      --shard-id "$s" --num-shards "$SHARDS" --resume \
      --gpu-mem-util "$(vgmu)" &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
clean_vllm
[ $rc -eq 0 ] || die "trace generation failed"
echo "[01] done"
