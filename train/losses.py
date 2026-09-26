"""SFT and top-k KD losses for the draft model.

Why KD and not just cross-entropy
---------------------------------
The acceptance rate of speculative decoding is, exactly,

    E_x~q[min(1, p(x)/q(x))] = sum_x min(p(x), q(x)) = 1 - TV(p, q)

so training the draft is literally a distribution-matching problem, and total
variation is the objective. Cross-entropy on sampled tokens only pushes the
argmax around; it is indifferent to how the remaining mass is arranged. KD
against the teacher's distribution attacks TV directly, and Pinsker bounds it:

    TV(p, q) <= sqrt(KL(p || q) / 2)

so every nat of KL we remove buys a bound on the acceptance gap. That is the
reason to expect the KD ablation to win on acceptance even where it does not
win on validation perplexity, and it is the thing gate G3 checks.

We only have the teacher's top-8, so this is the usual truncated
approximation: renormalise the teacher over its top-k support and match the
student there. With k=8 on a reasoning trace the top-8 typically holds most of
the mass, and the tail the draft gets wrong is tail the verifier rejects anyway.
"""
import torch
import torch.nn.functional as F


def sft_loss(logits, input_ids, loss_mask):
    """Next-token cross-entropy on the completion region."""
    pred = logits[:, :-1]
    tgt = input_ids[:, 1:]
    m = loss_mask[:, 1:]
    if not m.any():
        return logits.sum() * 0.0
    ce = F.cross_entropy(pred[m].float(), tgt[m], reduction="mean")
    return ce


def kd_loss(logits, tk_ids, tk_lp, tk_ok, loss_mask, temperature=1.0):
    """KL(teacher_topk || student) on the completion region.

    Shifted the same way as the SFT loss: the teacher distribution stored at
    completion position t is the distribution that produced token t, so it is
    predicted by the logits at t-1.
    """
    pred = logits[:, :-1]
    ids = tk_ids[:, 1:]
    lp = tk_lp[:, 1:]
    ok = tk_ok[:, 1:]
    m = loss_mask[:, 1:] & ok.any(-1)
    if not m.any():
        return logits.sum() * 0.0

    pred = pred[m].float() / temperature            # [N, V]
    ids, lp, ok = ids[m], lp[m], ok[m]              # [N, K]

    t_logits = torch.where(ok, lp, torch.full_like(lp, -1e4))
    t_prob = torch.softmax(t_logits, dim=-1)        # renormalised over the top-k
    s_logprob = torch.log_softmax(pred, dim=-1).gather(-1, ids)
    t_logprob = torch.log(t_prob.clamp_min(1e-9))
    kl = (t_prob * (t_logprob - s_logprob)).sum(-1)
    return (kl * (temperature ** 2)).mean()


def combined(logits, batch, lam=0.5, temperature=1.0):
    ce = sft_loss(logits, batch["input_ids"], batch["loss_mask"])
    if lam <= 0:
        return ce, {"ce": float(ce.detach()), "kd": 0.0}
    kd = kd_loss(logits, batch["tk_ids"], batch["tk_lp"], batch["tk_ok"],
                 batch["loss_mask"], temperature)
    return (1 - lam) * ce + lam * kd, {"ce": float(ce.detach()), "kd": float(kd.detach())}
