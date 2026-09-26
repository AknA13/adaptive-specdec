"""RaggedCache invariant: after a rollback the cache is indistinguishable from
one that only ever processed the accepted prefix.

If this breaks, speculative decoding silently conditions on tokens that were
rejected -- the output stays fluent and the acceptance rate even looks fine, so
nothing downstream would catch it.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from transformers import Qwen3ForCausalLM
from specdec.kvcache import RaggedCache
from tests.tiny import tiny_config

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def logits_via_ragged(m, cache, tokens, valid=None):
    q = tokens.shape[1]
    out = m(input_ids=tokens, position_ids=cache.positions(q),
            attention_mask=cache.attn_mask(q, torch.float32, valid=valid),
            past_key_values=cache, cache_position=cache.cache_position(q), use_cache=True)
    return out.logits


def main():
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(tiny_config()).eval()
    B, P = 3, 7
    ids = torch.randint(0, 16, (B, P))
    plen = torch.tensor([7, 5, 6])

    print("[rollback] prefill matches per-row unpadded reference")
    c = RaggedCache(m.config.num_hidden_layers, B, 128, torch.device("cpu"))
    lg = logits_via_ragged(m, c, ids, valid=plen)
    c.set_lengths(plen)
    for b in range(B):
        ref = m(input_ids=ids[b:b + 1, :plen[b]], use_cache=True).logits[0, -1]
        d = (lg[b, plen[b] - 1] - ref).abs().max().item()
        check(f"prefill row{b}", d < 1e-4, f"maxdiff={d:.2e}")

    # Speculate 4 tokens, then keep a different number per row.
    spec = torch.randint(0, 16, (B, 4))
    keep = torch.tensor([4, 1, 2])          # row1 rejects almost everything
    print("[rollback] decode 4, keep [4,1,2], then one more token")
    logits_via_ragged(m, c, spec)
    c.advance(4)
    c.rollback(4 - keep)
    check("lengths after rollback", c.lengths.tolist() == (plen + keep).tolist(),
          f"{c.lengths.tolist()} vs {(plen + keep).tolist()}")

    nxt = torch.randint(0, 16, (B, 1))
    got = logits_via_ragged(m, c, nxt)

    # Reference: a fresh cache that only ever saw prompt + kept speculation.
    print("[rollback] post-rollback logits match a never-rolled-back cache")
    for b in range(B):
        seq = torch.cat([ids[b, :plen[b]], spec[b, :keep[b]], nxt[b]]).unsqueeze(0)
        ref = m(input_ids=seq, use_cache=True).logits[0, -1]
        d = (got[b, -1] - ref).abs().max().item()
        check(f"row{b} after rollback", d < 1e-4, f"maxdiff={d:.2e}")

    # Stale entries past a row's length must never be readable again.
    # Build two caches in the same state, corrupt everything beyond the live
    # length in one of them, and check the next forward cannot tell.
    print("[rollback] stale tail is unreachable")
    clean = RaggedCache(m.config.num_hidden_layers, B, 128, torch.device("cpu"))
    dirty = RaggedCache(m.config.num_hidden_layers, B, 128, torch.device("cpu"))
    outs = []
    for cache, corrupt in ((clean, False), (dirty, True)):
        logits_via_ragged(m, cache, ids, valid=plen)
        cache.set_lengths(plen)
        logits_via_ragged(m, cache, spec)
        cache.advance(4)
        cache.rollback(4 - keep)
        if corrupt:
            for layer in cache.layers:
                for b in range(B):
                    layer.keys[b, :, int(cache.lengths[b]):, :] = 99.0
                    layer.values[b, :, int(cache.lengths[b]):, :] = 99.0
        outs.append(logits_via_ragged(m, cache, nxt).clone())
    d = (outs[0] - outs[1]).abs().max().item()
    check("garbage beyond live length does not leak", d < 1e-4, f"maxdiff={d:.2e}")

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("all kv-rollback checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
