"""Gate G1: speculative decoding must not change the output distribution.

This is the only test that can catch the failure mode that matters. A broken
speculative decoder does not crash and does not produce gibberish -- it produces
fluent text from subtly the wrong distribution, at a perfectly healthy-looking
acceptance rate. Eyeballing samples cannot detect it. So we measure.

Part A -- the rejection rule in isolation, against synthetic p and q.
  Also runs three *deliberately broken* variants and requires that they FAIL.
  A correctness test that never fires is worth nothing; this shows it has power.

Part B -- the full engine with tiny Qwen3 models, k>1, top-p on.
  Compares the empirical distribution of the second generated token against the
  exactly-computed autoregressive marginal (small vocab makes it enumerable).

Sampling noise: E[TV] under the null is about sqrt(V / (2*pi*N)). With V=16 and
N~8e4 that is ~0.006, comfortably under the 0.02 gate, so a failure means a bug
rather than bad luck.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import config as C
from specdec.sampling import to_probs, sample_from_probs, residual_probs, gather_prob
from specdec.engine import SpecDecodeEngine, SamplingConfig
from tests.tiny import tiny_pair, TinyTok, VOCAB

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def tv(a, b):
    return 0.5 * float((a - b).abs().sum())


def empirical(ids, v):
    return torch.bincount(ids.reshape(-1), minlength=v).float() / ids.numel()


def chi2(obs_counts, exp_probs):
    n = float(obs_counts.sum())
    keep = exp_probs > 1e-6
    e = exp_probs[keep] * n
    o = obs_counts[keep].float()
    stat = float((((o - e) ** 2) / e).sum())
    df = int(keep.sum()) - 1
    try:
        from scipy.stats import chi2 as _c
        return stat, df, float(_c.sf(stat, df))
    except Exception:
        return stat, df, float("nan")


# --------------------------------------------------------------------------
# Part A: the accept/reject rule, and proof that the test can detect breakage.
# --------------------------------------------------------------------------
def simulate(p, q, n, mode="correct", gen=None):
    """Draw n tokens through one speculative step. Returns sampled ids."""
    P = p.expand(n, -1).contiguous()
    Q = q.expand(n, -1).contiguous()
    x = sample_from_probs(Q, gen)
    ratio = gather_prob(P, x) / gather_prob(Q, x).clamp_min(1e-10)
    u = torch.rand(n, generator=gen)
    if mode == "always_accept":
        return x
    acc = u < ratio.clamp(max=1.0)
    if mode == "resample_from_p":
        # forgot to subtract q: draws from p instead of the residual
        y = sample_from_probs(P, gen)
    else:
        y = sample_from_probs(residual_probs(P, Q), gen)
    return torch.where(acc, x, y)


def part_a():
    print("[A] rejection rule on synthetic distributions")
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(7)
    N = 200_000
    p = to_probs(torch.randn(1, VOCAB) * 1.5, 1.0)
    q = to_probs(torch.randn(1, VOCAB) * 1.5, 1.0)
    print(f"      TV(p,q) = {tv(p[0], q[0]):.3f}  (draft is genuinely different)")

    got = empirical(simulate(p, q, N, "correct", gen), VOCAB)
    d = tv(got, p[0])
    stat, df, pv = chi2(torch.bincount(simulate(p, q, N, "correct", gen), minlength=VOCAB), p[0])
    check("correct rule reproduces p", d < 0.01, f"TV={d:.4f}  chi2 p={pv:.3f}")

    # Negative controls: each of these is a real bug people ship.
    for mode, label in (("always_accept", "skip the accept test"),
                        ("resample_from_p", "resample from p, not the residual")):
        dbad = tv(empirical(simulate(p, q, N, mode, gen), VOCAB), p[0])
        check(f"detects bug: {label}", dbad > 0.02, f"TV={dbad:.4f}")

    # The subtlest one: top-p applied with a different cutoff on the draft.
    # The draft must actually resemble the target here. With independent random
    # logits the two nuclei come out disjoint, acceptance probability is 0, the
    # residual collapses to p, and the output is correct no matter which q the
    # token was drawn from -- a vacuous control that would always "pass". So
    # build q as a noisy copy of p, and assert the instance is non-degenerate
    # before trusting what it says.
    lp = torch.randn(1, VOCAB) * 1.5
    lq = lp + torch.randn(1, VOCAB) * 0.8          # a plausible draft model
    p_full = to_probs(lp, 1.0, top_p=0.9)
    q_right = to_probs(lq, 1.0, top_p=0.9)         # same transform as p: correct
    q_mismatch = to_probs(lq, 1.0, top_p=0.5)      # different nucleus: the bug
    overlap = float(torch.minimum(p_full, q_right).sum())
    check("control instance is non-degenerate", overlap > 0.2,
          f"accept mass={overlap:.3f} (0 would make the control vacuous)")

    d_ok = tv(empirical(simulate(p_full, q_right, N, "correct", gen), VOCAB), p_full[0])
    # sample x from the differently-truncated q, but test it against q_right
    x = sample_from_probs(q_mismatch.expand(N, -1).contiguous(), gen)
    ratio = gather_prob(p_full.expand(N, -1), x) / gather_prob(q_right.expand(N, -1), x).clamp_min(1e-10)
    acc = torch.rand(N, generator=gen) < ratio.clamp(max=1.0)
    y = sample_from_probs(residual_probs(p_full, q_right).expand(N, -1).contiguous(), gen)
    d_bad = tv(empirical(torch.where(acc, x, y), VOCAB), p_full[0])
    check("consistent top-p is lossless", d_ok < 0.01, f"TV={d_ok:.4f}")
    check("detects bug: mismatched top-p between p and q", d_bad > 0.02, f"TV={d_bad:.4f}")


# --------------------------------------------------------------------------
# Part B: the real engine, multi-token rounds, against an exact AR marginal.
# --------------------------------------------------------------------------
def exact_second_token_marginal(target, prompt, sp):
    """sum over t0 of P(t0) * P(t1 | t0), computed exactly (vocab is tiny)."""
    with torch.inference_mode():
        lg = target(input_ids=prompt).logits[0, -1]
        p0 = sp.probs(lg.unsqueeze(0))[0]
        marg = torch.zeros(VOCAB)
        for t0 in range(VOCAB):
            if p0[t0] <= 0:
                continue
            seq = torch.cat([prompt[0], torch.tensor([t0])]).unsqueeze(0)
            l1 = target(input_ids=seq).logits[0, -1]
            marg += p0[t0] * sp.probs(l1.unsqueeze(0))[0]
    return p0, marg


def part_b():
    print("[B] full engine vs exact autoregressive marginal")
    target, draft = tiny_pair()
    eng = SpecDecodeEngine(target, draft, TinyTok(), device=torch.device("cpu"), capacity=64)
    sp = SamplingConfig(temperature=1.0, top_p=0.9)
    torch.manual_seed(0)
    prompt = torch.randint(0, VOCAB, (1, 5))
    p0, marg = exact_second_token_marginal(target, prompt, sp)

    B, REPS = 2048, 40
    ids = prompt.expand(B, -1).contiguous()
    plen = torch.full((B,), prompt.shape[1], dtype=torch.long)
    first, second = [], []
    accs = []
    for r in range(REPS):
        out, st = eng.generate(ids, plen, max_new_tokens=2, sampling=sp, k_fixed=4, seed=1000 + r)
        first += [o[0] for o in out]
        second += [o[1] for o in out]
        accs.append(st.alpha_conditional)
    n = len(second)
    print(f"      N={n}  mean alpha={sum(accs)/len(accs):.3f}  "
          f"(expected TV noise ~{(VOCAB / (2 * 3.14159 * n)) ** 0.5:.4f})")

    e0 = empirical(torch.tensor(first), VOCAB)
    e1 = empirical(torch.tensor(second), VOCAB)
    d0, d1 = tv(e0, p0), tv(e1, marg)
    _, _, pv = chi2(torch.bincount(torch.tensor(second), minlength=VOCAB), marg)
    # token 0 comes straight from the target: a control on the harness itself
    check("token 0 matches target (control)", d0 < C.G1_TV_MAX, f"TV={d0:.4f}")
    check("token 1 matches exact AR marginal", d1 < C.G1_TV_MAX,
          f"TV={d1:.4f}  gate={C.G1_TV_MAX}  chi2 p={pv:.3f}")


def main():
    part_a()
    part_b()
    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print(f"G1 PASSED: speculative decoding is distributionally lossless (TV gate {C.G1_TV_MAX})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
