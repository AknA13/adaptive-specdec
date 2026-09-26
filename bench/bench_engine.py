"""Stage 4: benchmark the from-scratch engine. Single-stream latency.

Grid: {autoregressive, fixed k, adaptive} x prompt length x sampling mode.

Two gates are enforced here rather than merely reported:

  G2  in greedy mode every speculative configuration must emit output that is
      token-identical to the autoregressive baseline. This is the real-model
      half of the CPU test in tests/test_greedy_equivalence.py, and it is the
      only end-to-end check that the engine is lossless on the actual models.

  G5  end-to-end tokens/s at batch 1 greedy must beat autoregressive by the
      configured factor.

Reported but not gated: acceptance by position, which is the empirical
justification for adaptive k -- if alpha were flat in position, a fixed k would
be fine.

  python -m bench.bench_engine --draft <ckpt> --n 40 --max-new 256
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from data.common import load_problems, build_prompt, verify_answer


def log(m):
    print(f"[bench] {m}", flush=True)


def make_batch(tok, problems, pad_to=0, filler=""):
    """Right-padded prompt batch. `filler` lengthens the prefill realistically."""
    import torch
    texts = [build_prompt(tok, (filler + p["problem"]) if filler else p["problem"])
             for p in problems]
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    L = max(max(len(e) for e in enc), pad_to)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(enc), L), pad, dtype=torch.long)
    plen = torch.tensor([len(e) for e in enc], dtype=torch.long)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e)
    return ids, plen


def accuracy(tok, outs, problems):
    ok = 0
    for o, p in zip(outs, problems):
        if verify_answer(tok.decode(o, skip_special_tokens=True), p["answer"]):
            ok += 1
    return ok / max(1, len(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=C.TARGET_ID)
    ap.add_argument("--draft", default=C.DRAFT_ID)
    ap.add_argument("--draft-label", default="trained")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--k-grid", default="1,2,3,4,6,8")
    ap.add_argument("--controllers", default="ewma,ewma+exit")
    ap.add_argument("--modes", default="greedy,sample")
    ap.add_argument("--prompt-lens", default="short,long")
    ap.add_argument("--long-filler-tokens", type=int, default=1500)
    ap.add_argument("--capacity", type=int, default=4096)
    ap.add_argument("--tag", default="engine")
    ap.add_argument("--skip-ar", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from specdec.engine import SpecDecodeEngine, SamplingConfig
    from specdec.controller import make_controller

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.target)
    log(f"loading target {args.target}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    log(f"loading draft {args.draft}")
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    eng = SpecDecodeEngine(target, draft, tok, device=dev, capacity=args.capacity)

    problems = load_problems(args.dataset, args.split, args.n)
    # A realistic long prefill: other problems from the same set as context,
    # not random tokens, so attention has something plausible to look at.
    filler = ""
    if "long" in args.prompt_lens:
        pool = load_problems(args.dataset, args.split, 0)
        buf = []
        while len(tok(" ".join(buf), add_special_tokens=False)["input_ids"]) < args.long_filler_tokens:
            buf.append(pool[len(buf) % len(pool)]["problem"])
        filler = ("Here are some related problems for context.\n"
                  + "\n".join(buf) + "\n\nNow solve this problem.\n")

    rows = []
    gate_fail = []
    for mode in args.modes.split(","):
        sp = SamplingConfig(temperature=0.0) if mode == "greedy" else \
             SamplingConfig(temperature=C.GEN_TEMPERATURE, top_p=C.GEN_TOP_P, top_k=C.GEN_TOP_K)
        for plen_label in args.prompt_lens.split(","):
            f = filler if plen_label == "long" else ""
            baseline_out = None

            def run(label, **kw):
                outs_all, agg = [], None
                t0 = time.perf_counter()
                for s in range(0, len(problems), args.batch_size):
                    chunk = problems[s:s + args.batch_size]
                    ids, pl = make_batch(tok, chunk, filler=f)
                    if kw.get("ar"):
                        o, st = eng.generate_autoregressive(
                            ids, pl, args.max_new, sampling=sp, seed=kw.get("seed"))
                    else:
                        o, st = eng.generate(
                            ids, pl, args.max_new, sampling=sp, seed=kw.get("seed"),
                            k_fixed=kw.get("k"), controller=kw.get("controller"))
                    outs_all += [x[:args.max_new] for x in o]
                    d = st.to_dict()
                    if agg is None:
                        agg = {"rounds": 0, "proposed": 0, "accepted": 0, "emitted": 0,
                               "reached": 0, "draft_forwards": 0, "target_forwards": 0,
                               "t_draft": 0.0, "t_target": 0.0, "t_total": 0.0,
                               "offered_at": [], "accepted_at": [], "k_history": []}
                    for k_ in ("rounds", "proposed", "accepted", "emitted", "reached",
                               "draft_forwards", "target_forwards",
                               "t_draft", "t_target", "t_total"):
                        agg[k_] += d[k_]
                    # Pool the per-position COUNTS; averaging per-batch rates
                    # would weight a 2-round batch the same as a 200-round one.
                    for name in ("offered_at", "accepted_at"):
                        cur, new_ = agg[name], d[name]
                        if len(new_) > len(cur):
                            cur.extend([0] * (len(new_) - len(cur)))
                        for i, v in enumerate(new_):
                            cur[i] += v
                    agg["k_history"] += d["k_history"]
                wall = time.perf_counter() - t0
                agg["wall_s"] = wall
                agg["tokens_per_s"] = agg["emitted"] / wall
                agg["acceptance_rate"] = agg["accepted"] / max(1, agg["proposed"])
                # alpha as it appears in the speedup formula: conditional on the
                # position having been reached, not on it having been proposed.
                agg["alpha_conditional"] = agg["accepted"] / max(1, agg["reached"])
                agg["alpha_by_position"] = [
                    a / o if o else 0.0
                    for a, o in zip(agg["accepted_at"], agg["offered_at"])]
                agg["mean_accepted_len"] = agg["emitted"] / max(1, agg["rounds"] * args.batch_size)
                agg["mean_k"] = (sum(agg["k_history"]) / len(agg["k_history"])
                                 if agg["k_history"] else 0)
                agg["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                agg["accuracy"] = accuracy(tok, outs_all, problems)
                return outs_all, agg

            torch.cuda.reset_peak_memory_stats()
            if not args.skip_ar:
                baseline_out, ar = run("ar", ar=True, seed=0)
                ar.update(method="ar", mode=mode, prompt_len=plen_label,
                          batch_size=args.batch_size, draft=args.draft_label)
                rows.append(ar)
                log(f"{mode:6s} {plen_label:5s} ar            "
                    f"{ar['tokens_per_s']:7.2f} tok/s  acc={ar['accuracy']:.3f}")
                ar_tps = ar["tokens_per_s"]
            else:
                ar_tps = None

            specs = [("fixed" + k, dict(k=int(k))) for k in args.k_grid.split(",") if k]
            specs += [(c, dict(controller=None, _spec=c))
                      for c in args.controllers.split(",") if c]
            for label, kw in specs:
                if "_spec" in kw:
                    kw = dict(controller=make_controller(kw["_spec"], k_max=C.K_MAX,
                                                         tau=C.EARLY_EXIT_TAU))
                torch.cuda.reset_peak_memory_stats()
                outs, r = run(label, seed=0, **kw)
                r.update(method=label, mode=mode, prompt_len=plen_label,
                         batch_size=args.batch_size, draft=args.draft_label)
                if ar_tps:
                    r["speedup_vs_ar"] = r["tokens_per_s"] / ar_tps
                if mode == "greedy" and baseline_out is not None:
                    same = outs == baseline_out
                    r["greedy_identical"] = same
                    if not same:
                        n_diff = sum(1 for a, b in zip(outs, baseline_out) if a != b)
                        gate_fail.append(f"G2 {mode}/{plen_label}/{label}: "
                                         f"{n_diff}/{len(outs)} sequences differ")
                rows.append(r)
                log(f"{mode:6s} {plen_label:5s} {label:13s} "
                    f"{r['tokens_per_s']:7.2f} tok/s  "
                    f"x{r.get('speedup_vs_ar', float('nan')):.2f}  "
                    f"alpha={r['alpha_conditional']:.3f}  "
                    f"acc/round={r['mean_accepted_len']:.2f}  "
                    f"k={r['mean_k']:.1f}  acc={r['accuracy']:.3f}  "
                    f"{'IDENTICAL' if r.get('greedy_identical') else ''}")

    out = {"rows": rows, "args": vars(args), "gate_failures": gate_fail}
    # G5: best speculative config at greedy must beat AR by the gate factor.
    g5 = [r for r in rows if r["mode"] == "greedy" and r.get("speedup_vs_ar")]
    if g5:
        best = max(r["speedup_vs_ar"] for r in g5)
        out["g5_best_speedup"] = best
        if best < C.G5_SPEEDUP:
            gate_fail.append(f"G5: best greedy speedup {best:.2f}x < {C.G5_SPEEDUP}x")
    p = C.publish_result(f"stage4_{args.tag}_{args.draft_label}", out)
    log(f"wrote {p}")
    if gate_fail:
        log("GATE FAILURES:")
        for g in gate_fail:
            log(f"  {g}")
        return 1
    log("all engine gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
