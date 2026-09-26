"""Bring-up checkpoint 1: vLLM accepts a draft_model speculative config.

Runs on CPU -- it only builds configs, never a model -- so it catches a vLLM
upgrade breaking the patches without queueing for an H200. Skips cleanly if the
weights are not reachable from this node.

Also asserts the property the whole KV-cache design rests on: target and draft
have identical per-layer KV geometry, so all 36 + 28 layers land in one cache
group sharing one block table. If a future draft model breaks that, the
proposer needs real work and this test says so early.
"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

FAIL = []
HUB = os.environ.get(
    "SPEC_HF_HUB", "/net/horton/data/akshay_anand/huggingface-cache/hub")


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def snapshot(repo_dir):
    p = pathlib.Path(HUB) / repo_dir / "snapshots"
    if not p.is_dir():
        return None
    snaps = [d for d in p.iterdir() if (d / "config.json").exists()]
    return str(snaps[0]) if snaps else None


def main():
    target = snapshot("models--Qwen--Qwen3-8B")
    draft = snapshot("models--Qwen--Qwen3-0.6B")
    if not target or not draft:
        print(f"[vllm] SKIP: Qwen3-8B/0.6B not found under {HUB}")
        print("       (Qwen3-0.6B is cached only on horton; set SPEC_HF_HUB)")
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["SPECDEC_PLUGIN"] = "1"
    os.environ.pop("TRANSFORMERS_CACHE", None)

    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.config import ModelConfig, ParallelConfig
    from vllm.config.speculative import SpeculativeConfig

    print("[vllm] stock vLLM 0.12 refuses draft_model; the plugin must make it work")
    tgt = ModelConfig(model=target, tokenizer=target, max_model_len=4096,
                      dtype="bfloat16", trust_remote_code=False)
    sc = SpeculativeConfig(target_model_config=tgt, target_parallel_config=ParallelConfig(),
                           model=draft, method="draft_model", num_speculative_tokens=4)
    check("config builds without NotImplementedError", sc.method == "draft_model")
    check("use_eagle() is true (buys lookahead slots + prefix-cache boundary)",
          sc.use_eagle())
    check("num_speculative_tokens preserved", sc.num_speculative_tokens == 4)
    check("speculative_token_tree was generated (the raise had skipped it)",
          sc.speculative_token_tree is not None, str(sc.speculative_token_tree))
    check("draft_parallel_config was generated", sc.draft_parallel_config is not None)
    check("draft max_model_len clamped to the target's",
          sc.draft_model_config.max_model_len == 4096,
          f"{sc.draft_model_config.max_model_len}")

    print("[vllm] the draft config is the draft's, not a copy of the target's")
    dh, th = sc.draft_model_config.hf_config, tgt.hf_config
    check("draft has its own layer count", dh.num_hidden_layers != th.num_hidden_layers,
          f"draft={dh.num_hidden_layers} target={th.num_hidden_layers}")
    check("draft has its own hidden size", dh.hidden_size != th.hidden_size,
          f"draft={dh.hidden_size} target={th.hidden_size}")

    print("[vllm] KV geometry matches, so all layers share one cache group")
    check("same vocab (required for speculative decoding at all)",
          dh.vocab_size == th.vocab_size, f"{dh.vocab_size}")
    check("same num_key_value_heads", dh.num_key_value_heads == th.num_key_value_heads,
          f"{dh.num_key_value_heads}")
    check("same head_dim", dh.head_dim == th.head_dim, f"{dh.head_dim}")
    check("neither uses sliding window (the shift-by-one layout needs pure RoPE)",
          not getattr(dh, "use_sliding_window", False)
          and not getattr(th, "use_sliding_window", False))
    total = dh.num_hidden_layers + th.num_hidden_layers
    print(f"       -> one KV cache group of {total} layers; the same "
          f"gpu_memory_utilization buys ~{100*(1-th.num_hidden_layers/total):.0f}% "
          f"fewer KV blocks than target-only")

    print("[vllm] the proposer subclasses EagleProposer compatibly")
    import inspect
    from vllm.v1.spec_decode.eagle import EagleProposer
    from vllm_draft_spec.proposer import DraftModelProposer
    check("DraftModelProposer is an EagleProposer",
          issubclass(DraftModelProposer, EagleProposer))
    for m in ("propose", "load_model", "dummy_run", "prepare_inputs_padded"):
        a = list(inspect.signature(getattr(DraftModelProposer, m)).parameters)
        b = list(inspect.signature(getattr(EagleProposer, m)).parameters)
        check(f"{m}() signature matches the parent", a == b)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("vLLM patch checks passed (bring-up checkpoint 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
