#!/bin/bash
# Common helpers for every scripts/*.sh entry point. Source it, don't run it.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
#
# Adapted from encrypted-reasoning/scripts/lib.sh.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [ -f "$REPO/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$REPO/env.sh"
elif [ -z "${SPEC_QUIET_ENV:-}" ]; then
  echo "[lib] no env.sh found -- using defaults. cp env.sh.example env.sh and edit it." >&2
fi

PY="${SPEC_PY:-python}"
DATA="${SPEC_DATA_ROOT:-$REPO/runs/default}"
TARGET="${SPEC_TARGET_ID:-Qwen/Qwen3-8B}"
DRAFT="${SPEC_DRAFT_ID:-Qwen/Qwen3-0.6B}"
export SPEC_DATA_ROOT="$DATA" SPEC_TARGET_ID="$TARGET" SPEC_DRAFT_ID="$DRAFT"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
# torch >=2.9 renamed this; export both so either version picks it up
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- logging ---------------------------------------------------------------
step()  { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }
info()  { echo "[info] $*"; }
warn()  { echo "[warn] $*" >&2; }
die()   { echo "[FATAL] $*" >&2; exit 1; }

run() {
  echo "+ $*"
  "$@" || die "stage failed: $*"
}

SOFT_FAILS=()
run_soft() {
  echo "+ $*"
  if ! "$@"; then
    warn "step FAILED (continuing): $*"
    SOFT_FAILS+=("$*")
  fi
}
report_soft_fails() {
  if [ ${#SOFT_FAILS[@]} -eq 0 ]; then
    echo "[ok] all steps completed"
  else
    echo "[summary] ${#SOFT_FAILS[@]} step(s) failed:"
    printf '  - %s\n' "${SOFT_FAILS[@]}"
    return 1
  fi
}

# ---- guards ----------------------------------------------------------------
need_file() { [ -e "$1" ] || die "missing required path: $1${2:+  ($2)}"; }
need_cmd()  { command -v "$1" >/dev/null 2>&1 || die "command not found: $1"; }

# Stages are idempotent: skip if the output already exists (preemption-safe).
# usage:  have_output "$DATA/traces/filtered.jsonl" && { info "skip"; exit 0; }
have_output() { [ -s "$1" ]; }

n_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0
  else
    echo 0
  fi
}

# SLURM scopes nvidia-smi to your cgroup, so never pass -i with a physical index.
gpu_free()  { nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1 || echo 0; }
gpu_total() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1 || echo 140000; }

# vLLM gpu_memory_utilization sized from what is actually free right now.
# NOTE: with the draft model resident there are 64 KV layers instead of 36, so
# the same utilization buys ~44% fewer KV blocks. Budget accordingly.
vgmu() {
  local f t
  f=$(gpu_free); t=$(gpu_total)
  "$PY" -c "print(round(max(0.55,min(0.88,0.90*$f/$t)),3))" 2>/dev/null || echo 0.80
}

wait_free() {
  local need="$1" tries=0 f
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  while [ $tries -lt 90 ]; do
    f=$(gpu_free)
    if [ "${f:-0}" -ge "$need" ]; then echo "[mem] gpu free=${f}MiB >= ${need}MiB"; return 0; fi
    echo "[mem] gpu free=${f}MiB < ${need}MiB -- waiting 60s (try $tries)"; sleep 60
    tries=$((tries+1))
  done
  die "GPU never freed ${need}MiB"
}

# kill only THIS job's leftover vLLM engine processes (never a sibling job's)
clean_vllm() {
  local sid; sid=$(ps -o sess= -p $$ | tr -d ' ')
  [ -n "$sid" ] && { pkill -s "$sid" -f EngineCore 2>/dev/null; pkill -s "$sid" -f vllm 2>/dev/null; }
  sleep 3; return 0
}

mkdir -p "$REPO/logs" "$REPO/results"
cd "$REPO"
