#!/bin/bash
# Stage 2: verify / dedupe / decontaminate. CPU only.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need_data
if have_output "$DATA/traces/filtered.jsonl"; then
  info "filtered.jsonl exists -- skipping"; exit 0
fi
step "filtering traces"
run "$PY" -m data.filter "$@"
