"""Sampling transforms and exact rejection sampling for speculative decoding.

THE correctness rule of this file
---------------------------------
The acceptance test compares the *transformed* distributions. Whatever you do to
the target logits (temperature, top-k, top-p) you must do identically to the
draft logits, and you must do it BEFORE the accept/reject arithmetic.

Speculative sampling (Leviathan et al. 2023, Chen et al. 2023) is lossless for
*any* pair of distributions (p, q): draw x ~ q, accept with probability
min(1, p(x)/q(x)), and on rejection draw from norm((p - q)_+). The result is
distributed exactly as p. Nothing requires q to be close to p -- closeness only
buys throughput. But it does require that the p and q you feed the test are the
same objects you would have sampled from.

The classic bug is applying top-p to the draft only (or with a different cutoff),
which silently changes the output distribution. Text still looks fine, which is
why this needs the statistical test in tests/test_rejection_sampling.py rather
than eyeballing.

Greedy is not a special case
----------------------------
temperature == 0 returns a one-hot distribution, and the generic rejection
sampler then reduces exactly to "accept iff the draft token is the target's
argmax, else emit the target's argmax". So greedy and sampling share one code
path, and greedy equivalence to autoregressive decoding (gate G2) is a
consequence of the same proof rather than a separate implementation.
"""
import torch

__all__ = ["to_probs", "sample_from_probs", "residual_probs", "gather_prob"]


def to_probs(logits, temperature=1.0, top_p=1.0, top_k=0):
    """Logits -> a proper probability distribution over the full vocab.

    Returns float32 [..., V] summing to 1 along the last dim, with zeros on
    tokens excluded by top-k / top-p. Keeping the full-width vector (rather than
    a compacted one) is what lets p and q be compared elementwise even when
    their nuclei differ.
    """
    logits = logits.float()
    if temperature is None or temperature <= 0.0:
        # Greedy: degenerate one-hot. Ties broken by argmax's first-index rule,
        # which is the same rule the autoregressive baseline uses, so G2 holds.
        out = torch.zeros_like(logits)
        out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return out

    probs = torch.softmax(logits / temperature, dim=-1)

    if top_k and top_k > 0 and top_k < probs.shape[-1]:
        kth = probs.topk(top_k, dim=-1).values[..., -1:]
        probs = torch.where(probs < kth, torch.zeros_like(probs), probs)

    if top_p is not None and 0.0 < top_p < 1.0:
        srt, idx = torch.sort(probs, dim=-1, descending=True)
        cum = srt.cumsum(dim=-1)
        # Keep every token up to and including the one that crosses top_p, so the
        # kept mass is always >= top_p and the nucleus is never empty.
        keep = cum - srt < top_p
        keep[..., 0] = True
        mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, idx, keep)
        probs = torch.where(mask, probs, torch.zeros_like(probs))

    total = probs.sum(dim=-1, keepdim=True)
    return probs / total.clamp_min(1e-12)


def sample_from_probs(probs, generator=None):
    """Multinomial sample. probs: [..., V] -> ids: [...]"""
    flat = probs.reshape(-1, probs.shape[-1])
    ids = torch.multinomial(flat, num_samples=1, generator=generator)
    return ids.reshape(probs.shape[:-1])


def residual_probs(p, q):
    """norm((p - q)_+), the distribution to draw from after a rejection.

    Degenerate case: if p and q are identical one-hots (greedy, accepted-token
    path) the positive part is all zeros. That can only be reached when the
    token would have been accepted, but clamp to p anyway so a numerical edge
    can never produce a NaN or a uniform draw over the whole vocab.
    """
    r = (p - q).clamp_min(0.0)
    total = r.sum(dim=-1, keepdim=True)
    degenerate = total <= 1e-9
    r = torch.where(degenerate, p, r)
    total = torch.where(degenerate, p.sum(dim=-1, keepdim=True), total)
    return r / total.clamp_min(1e-12)


def gather_prob(probs, ids):
    """probs: [..., V], ids: [...] -> probs at ids, shape [...]"""
    return probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
