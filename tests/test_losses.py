"""SFT/KD loss checks, chiefly the next-token shift.

The teacher distribution stored at completion position t produced token t, so
it must be matched against the logits at t-1. Off-by-one here still trains --
the loss goes down, perplexity looks plausible -- but the draft learns to
predict the token before the one it is asked for, and acceptance collapses.
That failure is invisible until you measure acceptance, so it gets a test.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from train.losses import sft_loss, kd_loss, combined

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def teacher_setup(B=2, L=8, V=32, K=4, seed=0):
    """Teacher top-k with DISTINCT ids per position (duplicates would make the
    renormalised top-k and the full-vocab reconstruction disagree)."""
    g = torch.Generator().manual_seed(seed)
    tk_ids = torch.stack([torch.stack([torch.randperm(V, generator=g)[:K]
                                       for _ in range(L)]) for _ in range(B)])
    tk_lp = torch.randn(B, L, K, generator=g)
    tk_ok = torch.ones(B, L, K, dtype=torch.bool)
    loss_mask = torch.zeros(B, L, dtype=torch.bool)
    loss_mask[:, 3:] = True
    return tk_ids, tk_lp, tk_ok, loss_mask


def student_matching(tk_ids, tk_lp, V, shift=1):
    """Full-vocab logits whose distribution equals the teacher's, placed so that
    logits[t-shift] carries the teacher distribution stored at position t."""
    B, L, K = tk_ids.shape
    lg = torch.full((B, L, V), -1e4)
    for b in range(B):
        for t in range(L):
            src = t - shift
            if 0 <= src < L:
                for k in range(K):
                    lg[b, src, tk_ids[b, t, k]] = tk_lp[b, t, k]
    return lg


def main():
    V = 32
    tk_ids, tk_lp, tk_ok, lm = teacher_setup(V=V)
    B, L, K = tk_ids.shape

    print("[kd] a student that already matches the teacher has ~zero KD loss")
    lg = student_matching(tk_ids, tk_lp, V, shift=1)
    l0 = float(kd_loss(lg, tk_ids, tk_lp, tk_ok, lm))
    check("KD == 0 at the correct alignment", l0 < 1e-3, f"kd={l0:.2e}")

    print("[kd] and a large one if the alignment is off by one either way")
    for s in (0, 2):
        lgs = student_matching(tk_ids, tk_lp, V, shift=s)
        ls = float(kd_loss(lgs, tk_ids, tk_lp, tk_ok, lm))
        check(f"KD detects shift={s} misalignment", ls > 1.0, f"kd={ls:.2f}")

    print("[kd] masked-out teacher slots are ignored")
    ok2 = tk_ok.clone()
    ok2[:, :, 2:] = False
    lg2 = torch.zeros(B, L, V)
    a = float(kd_loss(lg2, tk_ids, tk_lp, ok2, lm))
    b = float(kd_loss(lg2, tk_ids, tk_lp.clone().masked_fill(~ok2, 123.0), ok2, lm))
    check("values under a false mask never reach the loss", abs(a - b) < 1e-5,
          f"{a:.5f} vs {b:.5f}")

    print("[sft] loss is taken on the completion only")
    ids = torch.randint(0, V, (B, L))
    lg3 = torch.randn(B, L, V)
    base = float(sft_loss(lg3, ids, lm))
    lg4 = lg3.clone()
    lg4[:, :2] = torch.randn(B, 2, V) * 10       # scramble prompt-region logits
    check("prompt-region logits do not affect the loss",
          abs(base - float(sft_loss(lg4, ids, lm))) < 1e-5)
    lg5 = lg3.clone()
    lg5[:, 4:] = torch.randn(B, L - 4, V) * 10   # scramble completion-region logits
    check("completion-region logits do affect the loss",
          abs(base - float(sft_loss(lg5, ids, lm))) > 1e-3)

    print("[combined] lambda interpolates")
    batch = {"input_ids": ids, "loss_mask": lm, "tk_ids": tk_ids,
             "tk_lp": tk_lp, "tk_ok": tk_ok}
    _, p0 = combined(lg3, batch, lam=0.0)
    tot, p5 = combined(lg3, batch, lam=0.5)
    check("lam=0 disables KD", p0["kd"] == 0.0)
    check("lam=0.5 mixes both", abs(tot - (0.5 * p5["ce"] + 0.5 * p5["kd"])) < 1e-4,
          f"ce={p5['ce']:.3f} kd={p5['kd']:.3f}")

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("all loss checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
