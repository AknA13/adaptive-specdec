"""From-scratch batched speculative decoding over two HuggingFace causal LMs.

Round structure
---------------
Invariant at the top of every round: both KV caches hold exactly the committed
sequence (length n), and `pending` is the already-emitted token n+1 that neither
model has consumed yet.

  1. feed `pending` to the draft            -> q_1, sample x_1
  2. feed x_i to the draft, i = 1..k-1      -> q_{i+1}, sample x_{i+1}
  3. feed x_k to the draft                  (commit-only; its logits are dropped)
  4. feed [pending, x_1..x_k] to the target -> p_1..p_{k+1}, ONE forward
  5. accept a tokens by rejection sampling; roll both caches back by k-a
  6. y = residual draw (a<k) or bonus draw from p_{k+1} (a==k); pending <- y

Step 3 is the one non-obvious cost: k+1 draft forwards, not k. The draft never
needs to *run* on x_k to propose it, but it does need x_k in its cache to keep
going if x_k is accepted. Doing that forward unconditionally keeps the two
caches exactly in lockstep -- both are length n+1+k before the rollback and both
roll back by k-a -- which removes a whole class of off-by-one bugs at a cost of
1/k extra draft work. The alternative (catching up lazily, only on the
all-accepted path) makes the catch-up width differ per row, i.e. ragged forwards
across the batch, for a saving the controller's cost model absorbs anyway.

The controller sees this: the cost model in controller.py charges (k+1)*c_draft
+ c_target per round, not the textbook k*c + 1.

Losslessness comes from sampling.py; this file is bookkeeping. The statistical
proof that the bookkeeping did not break it is tests/test_rejection_sampling.py.
"""
import time
from dataclasses import dataclass, field

import torch

from specdec.kvcache import RaggedCache
from specdec.sampling import to_probs, sample_from_probs, residual_probs, gather_prob


@dataclass
class SamplingConfig:
    temperature: float = 0.0     # 0 => greedy, and the generic path still applies
    top_p: float = 1.0
    top_k: int = 0

    def probs(self, logits):
        return to_probs(logits, self.temperature, self.top_p, self.top_k)


@dataclass
class RunStats:
    """Everything the benchmarks and the acceptance-rate plots need."""
    rounds: int = 0
    batch: int = 1
    proposed: int = 0                 # draft tokens offered to the verifier
    accepted: int = 0                 # of those, how many survived
    emitted: int = 0                  # tokens actually returned to the caller
    draft_forwards: int = 0
    target_forwards: int = 0
    t_draft: float = 0.0
    t_target: float = 0.0
    t_total: float = 0.0
    # per-position: how often position i was offered, and how often it was kept.
    offered_at: list = field(default_factory=list)
    accepted_at: list = field(default_factory=list)
    k_history: list = field(default_factory=list)
    accept_history: list = field(default_factory=list)

    def _grow(self, n):
        while len(self.offered_at) < n:
            self.offered_at.append(0)
            self.accepted_at.append(0)

    def observe(self, k, n_accepted_per_row):
        """Record one round.

        A position only counts as *offered* if the round actually reached it.
        With a accepted out of k, positions 0..a were evaluated (the one at
        index a is where the rejection happened) and positions after that were
        never tested. Counting all k as offered would fold the geometric decay
        of "did we get this far" into alpha and make later positions look far
        worse than they are.
        """
        self._grow(k)
        for a in n_accepted_per_row:
            for i in range(k):
                if a >= i:
                    self.offered_at[i] += 1
                if a > i:
                    self.accepted_at[i] += 1

    @property
    def acceptance_rate(self):
        """Fraction of drafted tokens that survived -- the "was the draft work
        wasted" number. Denominator is every token proposed."""
        return self.accepted / max(1, self.proposed)

    @property
    def alpha_conditional(self):
        """P(accept at position i | position i was reached) -- the alpha that
        appears in the Leviathan speedup formula. Denominator is positions
        actually evaluated."""
        reached = sum(self.offered_at)
        return self.accepted / max(1, reached)

    @property
    def mean_accepted_len(self):
        """Tokens committed per sequence per round -- the quantity that must
        beat 1.0 for speculation to be worth anything. Divided by batch as well
        as rounds, or it would just scale with batch size."""
        return self.emitted / max(1, self.rounds * self.batch)

    def alpha_by_position(self):
        return [a / o if o else 0.0 for a, o in zip(self.accepted_at, self.offered_at)]

    def to_dict(self):
        return {
            "rounds": self.rounds, "batch": self.batch, "proposed": self.proposed, "accepted": self.accepted,
            "emitted": self.emitted, "acceptance_rate": self.acceptance_rate,
            "alpha_conditional": self.alpha_conditional,
            "mean_accepted_len": self.mean_accepted_len,
            "draft_forwards": self.draft_forwards, "target_forwards": self.target_forwards,
            "t_draft": self.t_draft, "t_target": self.t_target, "t_total": self.t_total,
            "tokens_per_s": self.emitted / self.t_total if self.t_total else 0.0,
            "alpha_by_position": self.alpha_by_position(),
            "offered_at": list(self.offered_at), "accepted_at": list(self.accepted_at),
            "reached": sum(self.offered_at),
            "k_history": self.k_history, "accept_history": self.accept_history,
        }


class SpecDecodeEngine:
    """Speculative decoding for a (target, draft) pair sharing a tokenizer."""

    def __init__(self, target, draft, tokenizer, device=None, capacity=4096, nvtx=False,
                 time_phases=False):
        self.target = target.eval()
        self.draft = draft.eval()
        self.tok = tokenizer
        self.device = device or next(target.parameters()).device
        self.capacity = capacity
        self.nvtx = nvtx
        # Per-phase timing needs a device sync around the draft loop and the
        # verify forward. Two syncs per round is a real cost at these token
        # rates, so it is opt-in: the profiler turns it on, benchmarks do not.
        self.time_phases = time_phases
        tcfg, dcfg = target.config, draft.config
        if tcfg.vocab_size != dcfg.vocab_size:
            raise ValueError(
                f"target vocab {tcfg.vocab_size} != draft vocab {dcfg.vocab_size}; "
                "speculative decoding requires a shared tokenizer")
        self.vocab_size = tcfg.vocab_size
        self.eos_ids = self._eos_ids(tokenizer, tcfg)

    @staticmethod
    def _eos_ids(tok, cfg):
        ids = set()
        for src in (getattr(tok, "eos_token_id", None), getattr(cfg, "eos_token_id", None)):
            if isinstance(src, (list, tuple)):
                ids.update(int(i) for i in src)
            elif src is not None:
                ids.add(int(src))
        return ids

    # ---- plumbing ----------------------------------------------------------
    def _mk_cache(self, model, batch):
        n_layers = model.config.num_hidden_layers
        return RaggedCache(n_layers, batch, self.capacity, self.device)

    def _forward(self, model, cache, tokens, valid=None):
        """One forward of `tokens` [B, q] against `cache`; returns logits [B, q, V]."""
        q = tokens.shape[1]
        pos = cache.positions(q)
        mask = cache.attn_mask(q, model.dtype, valid=valid)
        out = model(
            input_ids=tokens,
            position_ids=pos,
            attention_mask=mask,
            past_key_values=cache,
            cache_position=cache.cache_position(q),
            use_cache=True,
        )
        return out.logits

    def _range(self, name):
        return torch.profiler.record_function(name) if self.nvtx else _NullCtx()

    # ---- prefill -----------------------------------------------------------
    def _prefill(self, model, cache, input_ids, plen):
        """Right-padded prefill. Returns logits at each row's last real token."""
        logits = self._forward(model, cache, input_ids, valid=plen)
        cache.set_lengths(plen)                     # drop the pad from the length
        idx = (plen - 1).view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        return logits.gather(1, idx).squeeze(1)     # [B, V]

    # ---- main loop ---------------------------------------------------------
    @torch.inference_mode()
    def generate(self, input_ids, plen, max_new_tokens, sampling=None, controller=None,
                 seed=None, k_fixed=None):
        """input_ids [B, P] right-padded, plen [B] real prompt lengths.

        Returns (list of generated id lists, RunStats).
        """
        sp = sampling or SamplingConfig()
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        B = input_ids.shape[0]
        input_ids = input_ids.to(self.device)
        plen = plen.to(self.device)

        tcache = self._mk_cache(self.target, B)
        dcache = self._mk_cache(self.draft, B)
        st = RunStats(batch=B)
        t0 = time.perf_counter()

        with self._range("prefill"):
            tlog = self._prefill(self.target, tcache, input_ids, plen)
            self._prefill(self.draft, dcache, input_ids, plen)
            st.target_forwards += 1
            st.draft_forwards += 1

        # First emitted token comes straight from the target: no speculation is
        # possible before the draft has a conditioning token.
        pending = sample_from_probs(sp.probs(tlog), gen)          # [B]
        out = [[int(pending[b])] for b in range(B)]
        done = torch.tensor([int(pending[b]) in self.eos_ids for b in range(B)],
                            device=self.device)
        st.emitted += B

        # Terminate on `done` alone. Rows are marked done individually when they
        # hit EOS or max_new_tokens; a `max(len(...)) < max_new_tokens` guard here
        # would stop the whole batch as soon as the FASTEST row finished, and
        # rows that accepted fewer tokens would come back short.
        while not bool(done.all()):
            k = int(k_fixed) if k_fixed is not None else controller.propose_k()
            k = max(1, k)
            r0 = time.perf_counter()

            # ---- 1..3: draft k tokens, then commit the last one -------------
            xs, qs = [], []
            feed = pending.view(B, 1)
            conf = torch.ones(B, device=self.device)
            d0 = time.perf_counter()
            with self._range("draft_loop"):
                for i in range(k):
                    logits = self._forward(self.draft, dcache, feed)
                    dcache.advance(1)
                    st.draft_forwards += 1
                    q = sp.probs(logits[:, -1, :])
                    x = sample_from_probs(q, gen)
                    qs.append(q)
                    xs.append(x)
                    feed = x.view(B, 1)
                    if controller is not None:
                        conf = conf * gather_prob(q, x)
                        if controller.should_stop_drafting(i + 1, conf):
                            break
                # commit-only forward: puts the final drafted token in the cache
                self._forward(self.draft, dcache, feed)
                dcache.advance(1)
                st.draft_forwards += 1
            if self.time_phases and self.device.type == "cuda":
                torch.cuda.synchronize()
            st.t_draft += time.perf_counter() - d0

            k_eff = len(xs)
            draft_ids = torch.stack(xs, dim=1)                     # [B, k_eff]

            # ---- 4: one target forward verifies all of them -----------------
            v0 = time.perf_counter()
            with self._range("verify"):
                verify_in = torch.cat([pending.view(B, 1), draft_ids], dim=1)  # [B, k+1]
                vlog = self._forward(self.target, tcache, verify_in)
                tcache.advance(k_eff + 1)
                st.target_forwards += 1
            if self.time_phases and self.device.type == "cuda":
                torch.cuda.synchronize()
            st.t_target += time.perf_counter() - v0

            # p_1..p_{k+1}: p_1 is the distribution after `pending`.
            ps = [sp.probs(vlog[:, i, :]) for i in range(k_eff + 1)]

            # ---- 5: rejection sampling --------------------------------------
            with self._range("accept"):
                n_acc = torch.zeros(B, dtype=torch.long, device=self.device)
                alive = ~done
                for i in range(k_eff):
                    p_i, q_i, x_i = ps[i], qs[i], xs[i]
                    ratio = gather_prob(p_i, x_i) / gather_prob(q_i, x_i).clamp_min(1e-10)
                    u = torch.rand(B, device=self.device, generator=gen)
                    ok = (u < ratio.clamp(max=1.0)) & alive & (n_acc == i)
                    n_acc = n_acc + ok.long()
                    alive = alive & ok

                # y: residual draw where we stopped, bonus draw where we ran out.
                all_acc = n_acc == k_eff
                pick = n_acc.clamp(max=k_eff - 1)
                p_stop = torch.stack(ps[:k_eff], 1)[torch.arange(B), pick]
                q_stop = torch.stack(qs, 1)[torch.arange(B), pick]
                y_res = sample_from_probs(residual_probs(p_stop, q_stop), gen)
                y_bonus = sample_from_probs(ps[k_eff], gen)
                y = torch.where(all_acc, y_bonus, y_res)

                # One host sync per round, here. The accepted counts are needed
                # on the CPU anyway to decide which tokens to emit, so reuse
                # that transfer for the rollback instead of handing the cache a
                # CUDA tensor and paying for a second one.
                n_acc_l = n_acc.tolist()
                drop = [max(0, k_eff - a) for a in n_acc_l]
                tcache.rollback(drop)
                dcache.rollback(drop)

            # ---- record ------------------------------------------------------
            st.rounds += 1
            st.proposed += k_eff * B
            st.accepted += sum(n_acc_l)
            st.observe(k_eff, n_acc_l)
            st.k_history.append(k_eff)
            st.accept_history.append(n_acc_l)

            y_l = y.tolist()
            xs_l = draft_ids.tolist()
            for b in range(B):
                if bool(done[b]):
                    continue
                for j in range(n_acc_l[b]):
                    out[b].append(xs_l[b][j])
                    st.emitted += 1
                    if xs_l[b][j] in self.eos_ids or len(out[b]) >= max_new_tokens:
                        done[b] = True
                        break
                if bool(done[b]):
                    continue
                out[b].append(y_l[b])
                st.emitted += 1
                if y_l[b] in self.eos_ids:
                    done[b] = True
                if len(out[b]) >= max_new_tokens:
                    done[b] = True
            pending = y

            if controller is not None:
                controller.update(k_eff, n_acc_l, time.perf_counter() - r0,
                                  t_draft_fwd=st.t_draft / max(1, st.draft_forwards),
                                  t_target_fwd=st.t_target / max(1, st.target_forwards))

        st.t_total = time.perf_counter() - t0
        return out, st

    # ---- baseline ----------------------------------------------------------
    @torch.inference_mode()
    def generate_autoregressive(self, input_ids, plen, max_new_tokens, sampling=None, seed=None):
        """Target-only decoding through the identical cache/mask/sampling path.

        Using the same machinery as the speculative path is deliberate: it makes
        the greedy-equivalence gate (G2) a test of the speculation logic rather
        than a test of two unrelated implementations.
        """
        sp = sampling or SamplingConfig()
        gen = torch.Generator(device=self.device).manual_seed(int(seed)) if seed is not None else None
        B = input_ids.shape[0]
        input_ids, plen = input_ids.to(self.device), plen.to(self.device)
        cache = self._mk_cache(self.target, B)
        st = RunStats(batch=B)
        t0 = time.perf_counter()
        logits = self._prefill(self.target, cache, input_ids, plen)
        st.target_forwards += 1
        nxt = sample_from_probs(sp.probs(logits), gen)
        out = [[int(nxt[b])] for b in range(B)]
        done = torch.tensor([int(nxt[b]) in self.eos_ids for b in range(B)], device=self.device)
        st.emitted += B
        while not bool(done.all()):
            logits = self._forward(self.target, cache, nxt.view(B, 1))
            cache.advance(1)
            st.target_forwards += 1
            nxt = sample_from_probs(sp.probs(logits[:, -1, :]), gen)
            nl = nxt.tolist()
            for b in range(B):
                if bool(done[b]):
                    continue
                out[b].append(nl[b])
                st.emitted += 1
                if nl[b] in self.eos_ids or len(out[b]) >= max_new_tokens:
                    done[b] = True
        st.t_total = time.perf_counter() - t0
        st.t_target = st.t_total
        return out, st


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
