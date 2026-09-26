"""Draft-model speculative decoding for vLLM V1, as an out-of-tree plugin.

vLLM 0.12.0 does not support speculative decoding with a separate draft model.
`vllm/config/speculative.py:377` sets method="draft_model" and immediately
raises NotImplementedError, and `gpu_model_runner.py:374` only ever constructs
an Ngram / Suffix / Eagle / Medusa proposer. This package supplies the missing
proposer and the patches that let the rest of the stack reach it.

It is a plugin rather than a fork for a practical reason: the conda env on this
cluster has torch 2.9.0 installed with torch 2.11.0 metadata and both cu12 and
cu13 wheel sets, so any pip operation that resolves torch is likely to break
vLLM entirely. `pip install -e . --no-deps` only drops a .pth and a dist-info.

Usage:
    vllm serve Qwen/Qwen3-8B --speculative-config \
      '{"model": "<draft>", "method": "draft_model", "num_speculative_tokens": 8}'

Enable the adaptive controller with SPECDEC_CONTROLLER=ewma (see proposer.py).
"""


def register():
    """Entry point for vllm.general_plugins.

    Opt-in by design. The `rl` conda env is shared with other projects on this
    cluster, and a vllm.general_plugins entry point loads in EVERY vLLM process
    in the env -- so an unconditional register() would monkeypatch
    SpeculativeConfig for unrelated jobs. Nothing happens unless
    SPECDEC_PLUGIN=1 is set, which scripts/05_bench_vllm.sh does.
    """
    import os
    if os.environ.get("SPECDEC_PLUGIN", "").strip() not in ("1", "true", "yes", "on"):
        return
    from vllm_draft_spec.patches import apply_all
    apply_all()
