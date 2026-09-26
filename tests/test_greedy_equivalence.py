"""Gate G2 (logic half): greedy speculative decoding == greedy autoregressive,
token for token.

Greedy is not a special case in this codebase -- temperature 0 just makes
to_probs return a one-hot, and the same rejection sampler runs. So this is a
direct consequence of the same proof as G1, which is exactly why it is a good
test: if the bookkeeping (cache rollback, position ids, the p_1..p_{k+1}
alignment) is off by one, greedy output diverges immediately and visibly.

Runs on CPU with tiny models, at every k, including the early-exit controller.
The GPU half of G2 -- the same assertion on Qwen3-8B over 200 MATH-500 prompts
-- lives in bench/bench_engine.py.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from specdec.engine import SpecDecodeEngine, SamplingConfig
from specdec.controller import make_controller, FixedK
from tests.tiny import tiny_pair, tiny_pair_noisy, TinyTok, VOCAB

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def main():
    # A draft that sometimes disagrees, so the rejection path is actually taken.
    # Two independently initialised tiny models give alpha = 1.00, and
    # this whole file would pass without ever rejecting a token.
    target, draft = tiny_pair_noisy(noise=4.0)
    eng = SpecDecodeEngine(target, draft, TinyTok(), device=torch.device("cpu"), capacity=128)
    sp = SamplingConfig(temperature=0.0)
    torch.manual_seed(0)

    B, P, NEW = 4, 6, 24
    ids = torch.randint(0, VOCAB, (B, P))
    plen = torch.tensor([P, P - 1, P - 2, P])       # ragged prompts on purpose

    ar, ar_st = eng.generate_autoregressive(ids, plen, NEW, sampling=sp)
    ar = [o[:NEW] for o in ar]
    print(f"[greedy] autoregressive reference: {ar_st.target_forwards} target forwards")

    for k in (1, 2, 3, 4, 6, 8):
        out, st = eng.generate(ids, plen, NEW, sampling=sp, k_fixed=k)
        out = [o[:NEW] for o in out]
        same = out == ar
        check(f"k={k} identical to AR", same,
              f"alpha={st.alpha_conditional:.2f} rounds={st.rounds} "
              f"accepted/round={st.mean_accepted_len:.2f}"
              + ("" if same else f"\n       AR  {ar[0][:12]}\n       SD  {out[0][:12]}"))

    for spec in ("ewma", "ewma+exit", "ucb"):
        ctl = make_controller(spec, k_max=8, tau=0.5)
        out, st = eng.generate(ids, plen, NEW, sampling=sp, controller=ctl)
        out = [o[:NEW] for o in out]
        check(f"controller {spec} identical to AR", out == ar,
              f"k used={sorted(set(st.k_history))} alpha={st.alpha_conditional:.2f}")

    # A single-row run must agree too: batching must not change results.
    out1, _ = eng.generate(ids[:1], plen[:1], NEW, sampling=sp, k_fixed=5)
    check("batch of 1 agrees with batch of 4", out1[0][:NEW] == ar[0])

    # Different architectures (1 layer vs 2, hidden 16 vs 32), as the real pair
    # is 28 vs 36 layers. Catches any assumption that the caches are congruent.
    t2, d2 = tiny_pair()
    eng2 = SpecDecodeEngine(t2, d2, TinyTok(), device=torch.device("cpu"), capacity=128)
    ar2, _ = eng2.generate_autoregressive(ids, plen, NEW, sampling=sp)
    sd2, _ = eng2.generate(ids, plen, NEW, sampling=sp, k_fixed=4)
    check("mismatched draft architecture still identical",
          [o[:NEW] for o in sd2] == [o[:NEW] for o in ar2])

    # Non-vacuity: if nothing was ever rejected, none of the above tested the
    # thing this file exists to test.
    _, st8 = eng.generate(ids, plen, NEW, sampling=sp, k_fixed=8)
    check("rejection path was exercised", st8.alpha_conditional < 0.99,
          f"alpha={st8.alpha_conditional:.3f} (1.0 would make this file vacuous)")

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("G2 (logic) PASSED: greedy speculative output is token-identical to autoregressive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
