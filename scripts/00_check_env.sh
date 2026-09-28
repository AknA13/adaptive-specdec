#!/bin/bash
# Fail fast on anything that would waste a GPU hour.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
step "environment"
info "python   : $PY"
case "$PY" in
  /data/*) info "           (node-local env)" ;;
  /scratch/*) warn "python is on /scratch (NFS). On 2026-09-28 import torch took 690 s from there on horton; stage the env to /data/\$USER/envs/rl_node" ;;
esac
"$PY" -c "import torch, transformers, vllm" 2>/dev/null || die "the resolved interpreter cannot import torch/transformers/vllm: $PY"
info "repo     : $REPO"
info "data root: $DATA"
"$PY" -c "import sys; sys.path.insert(0,'$REPO'); import config as C; print('[cfg] '+C.describe())"

case "$DATA" in
  /accounts/*|/scratch/*)
    die "SPEC_DATA_ROOT=$DATA is on a quota'd filesystem (home ~2G free, /scratch ~10G).
     Use node-local /data/\$USER/... -- edit env.sh." ;;
esac
mkdir -p "$DATA" 2>/dev/null || die "cannot create $DATA (is /data present on this node?)"
info "writable : $DATA"

step "gpu"
NG=$(n_gpus); info "visible GPUs: $NG"
[ "$NG" -ge 1 ] || warn "no GPU visible -- CPU-only checks will still run"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

step "packages"
"$PY" - <<'PY'
import importlib, sys
bad = []
for m, want in (("torch", None), ("transformers", None), ("vllm", None), ("datasets", None)):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:14s} MISSING ({e})"); bad.append(m)
import torch
print(f"  cuda available {torch.cuda.is_available()}")
try:
    import flash_attn  # noqa
    print("  flash_attn     present")
except Exception:
    print("  flash_attn     absent (expected here; code uses sdpa)")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || die "missing packages"

step "models"
"$PY" - <<PY
import os, sys
from transformers import AutoConfig
ok = True
for name, mid in (("target", os.environ["SPEC_TARGET_ID"]), ("draft", os.environ["SPEC_DRAFT_ID"])):
    try:
        c = AutoConfig.from_pretrained(mid)
        print(f"  {name:6s} {mid}  layers={c.num_hidden_layers} hidden={c.hidden_size} "
              f"vocab={c.vocab_size} kv_heads={c.num_key_value_heads} head_dim={getattr(c,'head_dim',None)}")
    except Exception as e:
        print(f"  {name:6s} {mid}  UNAVAILABLE: {type(e).__name__}: {e}"); ok = False
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || die "model weights not reachable (Qwen3-0.6B is cached only on horton; check HF_HOME and --nodelist)"

step "weights are actually present (not just a config stub)"
# A cache dir can hold config.json + tokenizer and no weights at all -- that is
# what lorax's Qwen3-8B looked like, and AutoConfig.from_pretrained is perfectly
# happy with it. The failure then lands 90 seconds into the benchmark instead of
# here. Resolve the snapshot and check for real tensor files.
"$PY" - <<PY
import os, sys, glob
from pathlib import Path
from transformers.utils import cached_file
ok = True
for name, mid in (("target", os.environ["SPEC_TARGET_ID"]), ("draft", os.environ["SPEC_DRAFT_ID"])):
    try:
        cfg = cached_file(mid, "config.json")
        snap = Path(cfg).parent
        wts = (sorted(snap.glob("*.safetensors")) + sorted(snap.glob("*.bin")))
        size = sum(w.stat().st_size for w in wts if w.exists()) / 1e9
        if not wts or size < 0.1:
            print(f"  {name:6s} NO WEIGHTS in {snap} (found {len(wts)} files, {size:.2f} GB)")
            ok = False
        else:
            print(f"  {name:6s} {len(wts)} weight file(s), {size:.1f} GB")
    except Exception as e:
        print(f"  {name:6s} FAILED to resolve: {type(e).__name__}: {e}"); ok = False
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || die "model weights missing from the cache on this node (a config-only stub resolves fine but cannot load)"

step "tokenizer compatibility (speculative decoding is undefined otherwise)"
"$PY" - <<PY
import os, sys
from transformers import AutoConfig
t = AutoConfig.from_pretrained(os.environ["SPEC_TARGET_ID"])
d = AutoConfig.from_pretrained(os.environ["SPEC_DRAFT_ID"])
if t.vocab_size != d.vocab_size:
    print(f"  FATAL vocab mismatch {t.vocab_size} vs {d.vocab_size}"); sys.exit(1)
print(f"  shared vocab {t.vocab_size}")
same_kv = (t.num_key_value_heads == d.num_key_value_heads
           and getattr(t,'head_dim',None) == getattr(d,'head_dim',None))
print(f"  identical KV geometry: {same_kv} -> "
      f"{'one vLLM KV cache group' if same_kv else 'MULTIPLE groups; the vLLM proposer needs work'}")
PY

step "ok"
echo "environment looks good"
