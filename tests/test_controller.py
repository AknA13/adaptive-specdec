"""Controller maths and adaptation behaviour. CPU-only, no models.

The point of EWMAAnalytic is that it has no tuned thresholds: given a measured
acceptance rate and a measured draft/target cost ratio it computes the k that
maximises expected speedup. So the things worth testing are that the optimiser
is actually an optimiser, that the estimator it feeds is the right estimator,
and that it converges to the oracle k when the workload is stationary.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import random
import statistics

from specdec.controller import (expected_speedup, best_k, FixedK, EWMAAnalytic,
                                ConfidenceEarlyExit, UCBBandit, make_controller)

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def simulate(controller, alpha, c, rounds=4000, k_max=8, seed=0):
    """Run a controller against a synthetic geometric acceptance process."""
    rng = random.Random(seed)
    total_tokens = 0.0
    total_cost = 0.0
    for _ in range(rounds):
        k = min(controller.propose_k(), k_max)
        a = 0
        while a < k and rng.random() < alpha:
            a += 1
        cost = (k + 1) * c + 1.0
        total_tokens += a + 1
        total_cost += cost
        controller.update(k, [a], cost, t_draft_fwd=c, t_target_fwd=1.0)
    return total_tokens / total_cost


def main():
    print("[math] expected_speedup / best_k")
    check("speedup rises with alpha",
          all(expected_speedup(4, a, 0.15) < expected_speedup(4, a + 0.1, 0.15)
              for a in (0.4, 0.5, 0.6, 0.7, 0.8)))
    check("speedup falls as the draft gets more expensive",
          all(expected_speedup(4, 0.8, c) > expected_speedup(4, 0.8, c + 0.05)
              for c in (0.05, 0.1, 0.15, 0.2)))
    check("k* is non-decreasing in alpha",
          all(best_k(a, 0.15) <= best_k(a + 0.05, 0.15)
              for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))
    check("k* is non-increasing in cost",
          all(best_k(0.85, c) >= best_k(0.85, c + 0.05) for c in (0.05, 0.1, 0.15, 0.2)))
    check("k*=1 when the draft is worthless", best_k(0.05, 0.3) == 1,
          f"got {best_k(0.05, 0.3)}")
    check("k* saturates when the draft is nearly perfect", best_k(0.98, 0.05, 8) == 8,
          f"got {best_k(0.98, 0.05, 8)}")
    # brute force agreement
    ok = True
    for a in (0.3, 0.55, 0.7, 0.9):
        for c in (0.05, 0.12, 0.25):
            bf = max(range(1, 9), key=lambda k: expected_speedup(k, a, c))
            ok &= best_k(a, c, 8) == bf
    check("best_k agrees with brute force over the grid", ok)

    print("[cost model] extra_fwd distinguishes the two runtimes")
    # The engine runs k+1 draft forwards (the last only commits the final
    # drafted token into the draft cache); vLLM's drafter rides the target's
    # token stream and runs k. That fixed extra forward is a sunk cost per
    # round, so the engine amortises it by drafting LONGER: engine k* >= vLLM
    # k* at the same alpha and c. (Asserting it the other way round is the
    # intuitive-but-wrong reading, and this test caught exactly that.)
    for a in (0.6, 0.78, 0.9):
        for c in (0.15, 0.5, 1.0, 1.6):
            ke, kv = best_k(a, c, 8, 1), best_k(a, c, 8, 0)
            check(f"alpha={a} c={c}: engine k*={ke} >= vllm k*={kv}", ke >= kv)
    check("a cheap draft wants a long run", best_k(0.78, 0.15, 8, 0) >= 4,
          f"k*={best_k(0.78, 0.15, 8, 0)}")
    check("a draft costing more than the target wants k=1",
          best_k(0.78, 1.6, 8, 0) == 1, f"k*={best_k(0.78, 1.6, 8, 0)}")

    print("[estimator] alpha uses positions reached, not positions proposed")
    ctl = EWMAAnalytic(beta=0.5, warmup=0)
    # 4 proposed, 1 accepted => positions evaluated = 2, alpha = 1/2, not 1/4.
    for _ in range(40):
        ctl.update(4, [1], 1.0, t_draft_fwd=0.1, t_target_fwd=1.0)
    check("alpha = accepted / reached", abs(ctl.alpha - 0.5) < 0.02,
          f"alpha={ctl.alpha:.3f} (0.25 would mean it counted untested positions)")

    print("[adapt] converges to the oracle k on a stationary workload")
    # Assert on the modal k and the time-averaged alpha, not the instantaneous
    # ones: alpha is an estimate from a finite window, so k jitters by +-1 by
    # design. A test on ctl._last_k alone would be a coin flip.
    for alpha, c in ((0.45, 0.12), (0.7, 0.12), (0.9, 0.08)):
        oracle = best_k(alpha, c)
        ctl = EWMAAnalytic(warmup=20)
        rng = random.Random(0)
        ks, ahat = [], []
        for r in range(3000):
            k = ctl.propose_k()
            a = 0
            while a < k and rng.random() < alpha:
                a += 1
            ctl.update(k, [a], 1.0, t_draft_fwd=c, t_target_fwd=1.0)
            if r > 1000:
                ks.append(k)
                ahat.append(ctl.alpha)
        modal = statistics.mode(ks)
        mean_a = statistics.mean(ahat)
        check(f"alpha={alpha} c={c}: modal k == oracle k*={oracle}", modal == oracle,
              f"modal k={modal}")
        check(f"alpha={alpha}: alpha_hat is unbiased", abs(mean_a - alpha) < 0.03,
              f"mean alpha_hat={mean_a:.3f} sd={statistics.pstdev(ahat):.3f}")

    print("[adapt] tracks the oracle without being told the workload")
    for alpha, c in ((0.4, 0.15), (0.75, 0.1), (0.92, 0.1)):
        oracle = best_k(alpha, c)
        best_fixed = max(simulate(FixedK(k), alpha, c) for k in range(1, 9))
        adaptive = simulate(EWMAAnalytic(beta=0.95), alpha, c)
        ratio = adaptive / best_fixed
        # The honest claim: adaptive should be within a few percent of the best
        # fixed k chosen WITH oracle knowledge, not beat it.
        check(f"alpha={alpha}: within 5% of best fixed k (oracle k*={oracle})",
              ratio > 0.95, f"adaptive/best_fixed={ratio:.3f}")

    print("[adapt] beats a fixed k when the workload shifts under it")
    # A k tuned for the easy phase is badly wrong in the hard phase.
    def two_phase(ctl, seed=0, rounds=4000, c=0.12):
        rng = random.Random(seed)
        tok = cost = 0.0
        for r in range(rounds):
            alpha = 0.9 if (r // 250) % 2 == 0 else 0.35
            k = min(ctl.propose_k(), 8)
            a = 0
            while a < k and rng.random() < alpha:
                a += 1
            cst = (k + 1) * c + 1.0
            tok += a + 1
            cost += cst
            ctl.update(k, [a], cst, t_draft_fwd=c, t_target_fwd=1.0)
        return tok / cost

    fixed_best = max(two_phase(FixedK(k)) for k in range(1, 9))
    adaptive = two_phase(EWMAAnalytic(beta=0.9))
    check("adaptive >= best single fixed k on a shifting workload",
          adaptive >= fixed_best, f"adaptive={adaptive:.3f} best_fixed={fixed_best:.3f} "
                                  f"(+{100 * (adaptive / fixed_best - 1):.1f}%)")

    print("[misc] wiring")
    ee = ConfidenceEarlyExit(FixedK(8), tau=0.5)
    import torch
    check("early exit fires below tau", ee.should_stop_drafting(2, torch.tensor([0.2])))
    check("early exit holds above tau", not ee.should_stop_drafting(2, torch.tensor([0.9])))
    check("early exit respects min_k", not ee.should_stop_drafting(0, torch.tensor([0.01])))
    # UCB only marks an arm as seen when update() is called, so a propose-only
    # loop would return arm 1 forever -- the earlier version of this check did
    # exactly that and proved nothing.
    b = UCBBandit(k_max=4)
    seen = []
    for _ in range(4):
        k = b.propose_k()
        seen.append(k)
        b.update(k, [1], 1.0)
    check("ucb tries all arms before exploiting", sorted(seen) == [1, 2, 3, 4], f"{seen}")
    best_arm = 3
    b = UCBBandit(k_max=4)
    for _ in range(600):
        k = b.propose_k()
        b.update(k, [1], 1.0 + abs(k - best_arm))    # dt is a DURATION: shortest at k=3
    check("ucb converges on the best arm", max(b.mu, key=b.mu.get) == best_arm,
          f"mu={ {k: round(v, 2) for k, v in b.mu.items()} }")
    for spec in ("fixed3", "ewma", "ewma+exit", "ucb"):
        check(f"make_controller({spec!r})", make_controller(spec).propose_k() >= 1)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("all controller checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
