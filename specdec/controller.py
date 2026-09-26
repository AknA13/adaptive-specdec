"""Draft-length controllers.

The question every round is: how many tokens should the draft propose?

Too few and you pay a full target forward to commit barely more than one token.
Too many and you burn draft forwards on tokens that get thrown away -- and, at
serving scale, you also inflate the verify batch.

The classic answer is to pick one k offline and live with it. That is fine until
the workload moves: acceptance on a memorised arithmetic prefix looks nothing
like acceptance halfway through a novel derivation, and a k tuned on one is
wrong on the other. The honest claim for adaptivity is therefore NOT that it
beats the best oracle fixed k -- it usually cannot -- but that it tracks the
oracle without being told the workload, and avoids the cliff when acceptance
collapses. That is what gate G4 measures.

Cost model
----------
Leviathan et al. give the expected number of tokens committed per round as

    E[tokens] = (1 - a^(k+1)) / (1 - a)

with a the per-token acceptance probability *conditional on reaching that
position*. The cost of a round, in units of one target forward, is

    cost(k) = (k + 1) * c + 1

where c = t_draft_forward / t_target_forward. The (k+1) rather than k is not a
typo: engine.py runs one extra draft forward per round to commit the last
drafted token into the draft cache (see its module docstring).

So the optimal k maximises

    speedup(k) = (1 - a^(k+1)) / ((1 - a) * ((k+1) * c + 1))

There is no closed form for the maximiser over the integers, and k_max is 8, so
EWMAAnalytic just evaluates all of them. Both a and c are measured online, which
is the point: no thresholds to tune, and c self-calibrates to the hardware
instead of being assumed.
"""
import math

__all__ = ["Controller", "FixedK", "EWMAAnalytic", "ConfidenceEarlyExit",
           "UCBBandit", "expected_speedup", "best_k", "make_controller"]


def expected_speedup(k, alpha, c):
    """Expected tokens per round divided by the cost of that round."""
    k = max(1, int(k))
    if alpha >= 1.0 - 1e-9:
        tokens = float(k + 1)
    else:
        tokens = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)
    return tokens / ((k + 1) * c + 1.0)


def best_k(alpha, c, k_max=8):
    """argmax over k in [1, k_max]. k_max is small, so just enumerate."""
    return max(range(1, int(k_max) + 1), key=lambda k: expected_speedup(k, alpha, c))


class Controller:
    """Interface shared by the from-scratch engine and the vLLM proposer."""

    name = "base"

    def propose_k(self) -> int:
        raise NotImplementedError

    def should_stop_drafting(self, i, conf) -> bool:
        """Optional mid-round abort. i = drafts produced so far, conf = running
        product of the sampled tokens' draft probabilities, [B]."""
        return False

    def update(self, k, n_accepted, dt, t_draft_fwd=None, t_target_fwd=None):
        """n_accepted: list of per-row accepted counts for the round just finished."""

    def state(self):
        return {"name": self.name}


class FixedK(Controller):
    """Baseline. Also the thing adaptivity has to be measured against."""

    def __init__(self, k):
        self.k = int(k)
        self.name = f"fixed{self.k}"

    def propose_k(self):
        return self.k


class EWMAAnalytic(Controller):
    """Track alpha and c online, then pick the k that maximises expected speedup.

    alpha is estimated as accepted / positions-reached, not accepted / proposed.
    Positions after a rejection were never evaluated, so counting them as
    failures would conflate "the draft was wrong here" with "we never asked" and
    drive k down for the wrong reason.

    Decay is per *token observed*, not per round. That matters: a round at k=1
    yields one Bernoulli sample and a round at k=8 yields up to eight, so a
    per-round decay silently shrinks the effective window exactly when k
    collapses to 1 -- alpha then rattles around by +-0.15, drags k back up, and
    the controller oscillates instead of settling. Weighting by the number of
    positions actually evaluated keeps the window a fixed number of tokens
    (about 1/(1-beta)) whatever k is doing.
    """

    name = "ewma-analytic"

    def __init__(self, k_max=8, beta=0.99, k_init=4, c_init=0.15, warmup=3):
        self.k_max = int(k_max)
        self.beta = float(beta)
        self.k_init = int(k_init)
        self.c = float(c_init)
        self.warmup = int(warmup)
        self._acc = 0.0        # token-decayed sum of accepted tokens
        self._reached = 0.0    # token-decayed sum of positions evaluated
        self._rounds = 0
        self._last_k = self.k_init

    @property
    def alpha(self):
        if self._reached <= 0:
            return 0.7         # only used before the first update
        return min(0.999, max(0.0, self._acc / self._reached))

    def propose_k(self):
        if self._rounds < self.warmup:
            self._last_k = self.k_init
        else:
            self._last_k = best_k(self.alpha, self.c, self.k_max)
        return self._last_k

    def update(self, k, n_accepted, dt, t_draft_fwd=None, t_target_fwd=None):
        acc = float(sum(n_accepted))
        reached = float(sum(min(a + 1, k) for a in n_accepted))
        w = self.beta ** reached          # per-token decay, not per-round
        self._acc = w * self._acc + acc
        self._reached = w * self._reached + reached
        if t_draft_fwd and t_target_fwd and t_target_fwd > 0:
            c_obs = t_draft_fwd / t_target_fwd
            self.c = 0.9 * self.c + 0.1 * c_obs
        self._rounds += 1

    def state(self):
        return {"name": self.name, "alpha": self.alpha, "c": self.c,
                "k": self._last_k, "rounds": self._rounds}


class ConfidenceEarlyExit(Controller):
    """Wrap any controller with a mid-round abort.

    propose_k sets the ceiling; this stops short of it as soon as the draft's
    own confidence says the remaining tokens are unlikely to survive. It is the
    cheap half of adaptivity: it cuts wasted *draft* work within a round without
    waiting for the next round's acceptance signal to move the ceiling.

    conf is the running product of q(x_i) over the tokens drafted so far, i.e. a
    crude upper bound on the probability the whole run gets accepted. The batch
    shares one decision, so we use the mean; at batch size 1 that is exact.
    """

    def __init__(self, base, tau=0.3, min_k=1):
        self.base = base
        self.tau = float(tau)
        self.min_k = int(min_k)
        self.name = f"{base.name}+exit{self.tau}"

    def propose_k(self):
        return self.base.propose_k()

    def should_stop_drafting(self, i, conf):
        if i < self.min_k:
            return False
        return float(conf.mean()) < self.tau

    def update(self, k, n_accepted, dt, t_draft_fwd=None, t_target_fwd=None):
        self.base.update(k, n_accepted, dt, t_draft_fwd, t_target_fwd)

    def state(self):
        s = dict(self.base.state())
        s.update({"name": self.name, "tau": self.tau})
        return s


class UCBBandit(Controller):
    """Model-free alternative: UCB1 over k, reward = tokens committed per second.

    Included as a check on EWMAAnalytic rather than as a serious contender --
    if the analytic policy is doing its job, a bandit that assumes nothing
    should converge to roughly the same k, just more slowly.
    """

    name = "ucb"

    def __init__(self, k_max=8, cexp=0.5):
        self.ks = list(range(1, int(k_max) + 1))
        self.n = {k: 0 for k in self.ks}
        self.mu = {k: 0.0 for k in self.ks}
        self.t = 0
        self.cexp = float(cexp)
        self._last_k = self.ks[0]

    def propose_k(self):
        self.t += 1
        unseen = [k for k in self.ks if self.n[k] == 0]
        self._last_k = unseen[0] if unseen else max(
            self.ks, key=lambda k: self.mu[k] + self.cexp * math.sqrt(math.log(self.t) / self.n[k]))
        return self._last_k

    def update(self, k, n_accepted, dt, t_draft_fwd=None, t_target_fwd=None):
        emitted = sum(a + 1 for a in n_accepted)
        r = emitted / dt if dt > 0 else 0.0
        self.n[k] += 1
        self.mu[k] += (r - self.mu[k]) / self.n[k]

    def state(self):
        return {"name": self.name, "k": self._last_k,
                "mu": {str(k): round(v, 2) for k, v in self.mu.items()}}


def make_controller(spec, k_max=8, tau=0.3):
    """Build a controller from a benchmark-grid label.

    'ar' | 'fixed3' | 'ewma' | 'ewma+exit' | 'ucb'
    """
    s = spec.strip().lower()
    exit_on = s.endswith("+exit")
    if exit_on:
        s = s[: -len("+exit")]
    if s.startswith("fixed"):
        base = FixedK(int(s[len("fixed"):]))
    elif s == "ewma":
        base = EWMAAnalytic(k_max=k_max)
    elif s == "ucb":
        base = UCBBandit(k_max=k_max)
    else:
        raise ValueError(f"unknown controller spec: {spec!r}")
    return ConfidenceEarlyExit(base, tau=tau) if exit_on else base
