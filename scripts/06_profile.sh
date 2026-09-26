#!/bin/bash
# Stage 6: torch.profiler traces + cost ratio + CUDA-graph headroom.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
NG=$(n_gpus); [ "$NG" -lt 1 ] && die "no GPU visible"
wait_free "${SPEC_MIN_FREE_MIB:-40000}"
CK="$DATA/ckpt/sft_kd"; [ -s "$CK/config.json" ] || CK="$DRAFT"
step "profiling with draft=$CK"
run "$PY" -m bench.profile_engine --draft "$CK" "$@"
