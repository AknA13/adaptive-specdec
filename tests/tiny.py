"""A pair of tiny random Qwen3 models, so the correctness tests run on CPU.

Small vocab is deliberate: the statistical test in test_rejection_sampling.py
needs enough samples per token that sampling noise sits well under the gate, and
noise scales like sqrt(V/N).
"""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

VOCAB = 16


def tiny_config(hidden=32, layers=2):
    return Qwen3Config(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256, tie_word_embeddings=True,
    )


def tiny_pair(seed_target=0, seed_draft=1):
    """Target and draft with different shapes -- proves the engine never assumes
    matching layer counts or hidden sizes (the real pair is 36 vs 28 layers)."""
    torch.manual_seed(seed_target)
    target = Qwen3ForCausalLM(tiny_config(32, 2)).eval()
    torch.manual_seed(seed_draft)
    draft = Qwen3ForCausalLM(tiny_config(16, 1)).eval()
    return target, draft


def tiny_pair_noisy(noise=0.35, seed=0):
    """Draft = target + gaussian noise, which is what a distilled draft actually
    looks like: usually right, sometimes not.

    Two *independently* initialised tiny transformers are near-constant
    functions and happen to collapse to the same argmax, giving alpha = 1.0 and
    a greedy test that never once exercises the rejection path. Perturbing a
    copy gives a tunable, genuinely non-trivial acceptance rate.
    """
    import copy
    torch.manual_seed(seed)
    target = Qwen3ForCausalLM(tiny_config(32, 2)).eval()
    draft = copy.deepcopy(target).eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for prm in draft.parameters():
            prm.add_(torch.randn(prm.shape, generator=g) * noise * prm.std().clamp_min(1e-3))
    return target, draft


class TinyTok:
    """Just enough of a tokenizer for the engine (it only reads eos_token_id)."""
    eos_token_id = None
