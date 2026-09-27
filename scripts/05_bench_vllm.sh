#!/bin/bash
# Stage 5: vLLM serving benchmark through the DraftModelProposer plugin.
#
#   scripts/05_bench_vllm.sh --smoke    # bring-up checkpoints only
#   scripts/05_bench_vllm.sh            # full concurrency sweep
#
# Each configuration gets its own server, started and torn down here. vLLM
# cannot change speculative settings on a live engine, and leaving a 16 GB
# model resident between runs would distort the memory numbers.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need_data
SMOKE=0; [ "${1:-}" = "--smoke" ] && { SMOKE=1; shift; }
NG=$(n_gpus); [ "$NG" -lt 1 ] && die "no GPU visible"
wait_free "${SPEC_MIN_FREE_MIB:-40000}"

CK="$DATA/ckpt/sft_kd"; [ -s "$CK/config.json" ] || CK="$DRAFT"
PORT="${SPEC_VLLM_PORT:-8${RANDOM:0:3}}"
URL="http://127.0.0.1:$PORT"
MAXLEN="${SPEC_VLLM_MAXLEN:-4096}"
KMAX="${SPEC_K_MAX:-8}"
TMP="${TMPDIR:-/tmp}/specdec-serve-$$"
mkdir -p "$TMP"

# The plugin is inert unless this is set, so other jobs sharing the `rl` env
# are never patched.
export SPECDEC_PLUGIN=1
# INFO so the bring-up checkpoints are actually observable: the "GPU KV cache
# size" line is what proves all 64 layers landed in one cache group.
# Forced, not defaulted: slurm/submit.sh already exports WARNING into the job
# script, so a ${VAR:-INFO} default would never take effect and the bring-up
# checkpoints would be invisible.
export VLLM_LOGGING_LEVEL=INFO
# Stable, node-local torch.compile cache. slurm/submit.sh points VLLM_CACHE_ROOT
# at a per-job TMPDIR so concurrent jobs cannot corrupt each other's inductor
# cache, but that also means every server in this stage recompiles both models
# from scratch (~50 s each, x5 servers, x every requeue). /data is node-local,
# so a per-node cache is safe here and survives preemption.
export VLLM_CACHE_ROOT="${SPEC_VLLM_CACHE:-$DATA/vllm-cache}"
mkdir -p "$VLLM_CACHE_ROOT"

SERVER_PID=""
stop_server() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  clean_vllm
}
trap 'stop_server' EXIT

start_server() {   # start_server <logname> [extra vllm args...]
  local name="$1"; shift
  local logf="$REPO/logs/vllm_${name}.out"
  info "starting server: $name (log $logf)"
  "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$TARGET" --served-model-name target --port "$PORT" \
      --max-model-len "$MAXLEN" --gpu-memory-utilization "${SPEC_VLLM_GMU:-0.85}" \
      --tensor-parallel-size "${SPEC_VLLM_TP:-1}" "$@" > "$logf" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 180); do
    if curl -sf "$URL/v1/models" >/dev/null 2>&1; then
      info "server up after ${i}0s"; return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "--- server died, tail of $logf ---"; tail -40 "$logf"; return 1
    fi
    sleep 10
  done
  echo "--- server never became ready, tail of $logf ---"; tail -40 "$logf"; return 1
}

spec_cfg() {  # spec_cfg <k>
  printf '{"model": "%s", "method": "draft_model", "num_speculative_tokens": %s}' "$CK" "$1"
}

# ---- bring-up checkpoints -------------------------------------------------
step "checkpoint 1-4: engine starts with a draft model and one KV cache group"
start_server "k1" --speculative-config "$(spec_cfg 1)" --enforce-eager || die "server failed to start with draft_model"
grep -iE "GPU KV cache size|Duplicate layer name|DraftModelProposer|adaptive-specdec|maximum concurrency" \
     "$REPO/logs/vllm_k1.out" | tail -12

step "checkpoint 5: greedy identity at k=1"
run "$PY" -m bench.smoke_identity --base-url "$URL" --dump "$TMP/spec_k1.json" --n 16
stop_server

start_server "nospec" --enforce-eager || die "baseline server failed"
run "$PY" -m bench.smoke_identity --base-url "$URL" --dump "$TMP/nospec.json" --n 16
stop_server
"$PY" -m bench.smoke_identity --compare "$TMP/nospec.json" "$TMP/spec_k1.json" \
  || die "checkpoint 5 FAILED: greedy output differs -- do not trust any speedup number until this passes"

if [ "$SMOKE" = 1 ]; then
  step "checkpoint 6: acceptance at k=4"
  start_server "k4" --speculative-config "$(spec_cfg 4)" --enforce-eager || die "k=4 server failed"
  run "$PY" -m bench.bench_serving --base-url "$URL" --label smoke_k4 --n 16 --concurrency 1
  grep -i "SpecDecoding metrics\|acceptance" "$REPO/logs/vllm_k4.out" | tail -5
  stop_server
  echo "[05] smoke checks passed"
  exit 0
fi

# ---- full sweep -----------------------------------------------------------
CONC="${SPEC_BENCH_CONC:-1,4,16,64}"
N="${SPEC_BENCH_REQS:-64}"

# Every label is skipped if its result already exists: this partition preempts,
# and a requeued sweep should resume rather than redo 40 minutes of servers.
done_label() { [ -s "$REPO/results/stage5_serving_$1.json" ]; }

sweep_one() {   # sweep_one <label> [server args...]
  local label="$1"; shift
  if done_label "$label"; then info "results/stage5_serving_$label.json exists -- skipping"; return 0; fi
  step "serving sweep: $label"
  start_server "$label" "$@" || { warn "$label server failed"; return 1; }
  run_soft "$PY" -m bench.bench_serving --base-url "$URL" --label "$label" --n "$N" --concurrency "$CONC"
  stop_server
}

sweep_one ar
for k in 2 4 8; do
  sweep_one "fixed$k" --speculative-config "$(spec_cfg $k)"
done

if ! done_label adaptive; then
  step "serving sweep: adaptive (k chosen per step from live acceptance)"
  SPECDEC_CONTROLLER=ewma start_server "adaptive" --speculative-config "$(spec_cfg $KMAX)" \
    && run_soft "$PY" -m bench.bench_serving --base-url "$URL" --label adaptive \
         --n "$N" --concurrency "$CONC"
  stop_server
fi

report_soft_fails
