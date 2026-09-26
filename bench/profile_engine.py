"""Stage 6: profile the engine with torch.profiler, and measure the cost ratio
the controller depends on.

Nsight Systems is not available on this cluster (no nsys/ncu binary, no CUDA
module), so this is torch.profiler only -- CUPTI is present, so kernel-level
attribution and Chrome traces work.

Three things get measured:

1. Where the time goes inside a round: draft loop vs the single verify forward
   vs the accept/rollback bookkeeping, via record_function ranges the engine
   already emits when nvtx=True.

2. The cost ratio c = t_draft_forward / t_target_forward, at several batch
   sizes. This is the number that decides the optimal k, and it is not a
   constant: a 0.6B model at batch 1 is kernel-launch-bound rather than
   compute-bound, so it costs far less than its 1/13th parameter ratio would
   suggest, and k* is correspondingly higher. At large batch both models become
   compute-bound and c rises toward the parameter ratio, pushing k* down. That
   is the mechanism behind the concurrency crossover in bench_serving.py.

3. How much of the draft cost is launch overhead, by CUDA-graphing a
   fixed-shape single-token draft decode and comparing. This is a
   microbenchmark of the headroom, not a change to the engine: graph capture
   needs static shapes, and the engine's attention mask grows by one column per
   token. Making that static is the obvious next optimisation and this
   quantifies what it would buy.

  python -m bench.profile_engine --draft <ckpt> --batch-sizes 1,4,16
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from data.common import load_problems, build_prompt


def log(m):
    print(f"[prof] {m}", flush=True)


def timed_forward(fn, iters=50, warmup=10):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=C.TARGET_ID)
    ap.add_argument("--draft", default=C.DRAFT_ID)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batch-sizes", default="1,4,16")
    ap.add_argument("--ctx", type=int, default=512, help="context length for the microbenchmarks")
    ap.add_argument("--capacity", type=int, default=2048)
    ap.add_argument("--tag", default="profile")
    args = ap.parse_args()

    import torch
    from torch.profiler import profile, ProfilerActivity
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from specdec.engine import SpecDecodeEngine, SamplingConfig
    from specdec.controller import best_k

    C.ensure_data_dirs()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    out = {"args": vars(args)}

    # ---- 1. where the time goes inside a round --------------------------
    log("profiling a speculative run")
    eng = SpecDecodeEngine(target, draft, tok, device=dev, capacity=args.capacity, nvtx=True)
    problems = load_problems(args.dataset, "test", args.n)
    texts = [build_prompt(tok, p["problem"]) for p in problems]
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    L = max(len(e) for e in enc)
    pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.full((len(enc), L), pad, dtype=torch.long)
    plen = torch.tensor([len(e) for e in enc])
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e)
    sp = SamplingConfig(temperature=0.0)

    eng.generate(ids[:1], plen[:1], 16, sampling=sp, k_fixed=args.k)     # warm up
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        _, st = eng.generate(ids[:1], plen[:1], args.max_new, sampling=sp, k_fixed=args.k)
    trace = C.PROFILE_DIR / f"{args.tag}_k{args.k}.json"
    prof.export_chrome_trace(str(trace))
    log(f"chrome trace -> {trace}")
    tbl = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
    (C.PROFILE_DIR / f"{args.tag}_k{args.k}_kernels.txt").write_text(tbl)
    print(tbl)

    d = st.to_dict()
    out["run"] = {k: d[k] for k in ("rounds", "emitted", "t_draft", "t_target", "t_total",
                                    "draft_forwards", "target_forwards",
                                    "acceptance_rate", "alpha_conditional",
                                    "mean_accepted_len", "tokens_per_s")}
    out["run"]["alpha_by_position"] = d["alpha_by_position"]
    unaccounted = d["t_total"] - d["t_draft"] - d["t_target"]
    out["run"]["t_other"] = unaccounted
    log(f"round split: draft {d['t_draft']:.2f}s  verify {d['t_target']:.2f}s  "
        f"other {unaccounted:.2f}s  ({100*unaccounted/max(d['t_total'],1e-9):.1f}% "
        f"bookkeeping/sampling)")
    log(f"alpha by position: {[round(a,3) for a in d['alpha_by_position']]}")

    # ---- 2. cost ratio c, and the k* it implies --------------------------
    log("measuring per-forward cost at several batch sizes")
    from specdec.kvcache import RaggedCache
    ratios = []
    for B in [int(b) for b in args.batch_sizes.split(",") if b]:
        entry = {"batch": B}
        for name, model in (("draft", draft), ("target", target)):
            cache = RaggedCache(model.config.num_hidden_layers, B, args.capacity, dev)
            prompt = torch.randint(0, 1000, (B, args.ctx), device=dev)
            with torch.inference_mode():
                model(input_ids=prompt, position_ids=cache.positions(args.ctx),
                      attention_mask=cache.attn_mask(args.ctx, model.dtype),
                      past_key_values=cache, cache_position=cache.cache_position(args.ctx),
                      use_cache=True)
                cache.advance(args.ctx)
                step = torch.randint(0, 1000, (B, 1), device=dev)

                def one(model=model, cache=cache, step=step):
                    model(input_ids=step, position_ids=cache.positions(1),
                          attention_mask=cache.attn_mask(1, model.dtype),
                          past_key_values=cache,
                          cache_position=cache.cache_position(1), use_cache=True)
                entry[f"t_{name}_ms"] = timed_forward(one) * 1e3
            del cache
            torch.cuda.empty_cache()
        entry["c"] = entry["t_draft_ms"] / entry["t_target_ms"]
        entry["k_star"] = {str(a): best_k(a, entry["c"], C.K_MAX)
                           for a in (0.5, 0.6, 0.7, 0.8, 0.9)}
        ratios.append(entry)
        log(f"  bs={B:3d}  draft {entry['t_draft_ms']:6.3f} ms  "
            f"target {entry['t_target_ms']:6.3f} ms  c={entry['c']:.3f}  "
            f"k* at alpha=0.8 -> {entry['k_star']['0.8']}")
    out["cost_ratio"] = ratios

    # ---- 3. how much of the draft step is launch overhead ----------------
    log("CUDA-graph microbenchmark of a single-token draft decode")
    try:
        B = 1
        cache = RaggedCache(draft.config.num_hidden_layers, B, args.capacity, dev)
        prompt = torch.randint(0, 1000, (B, args.ctx), device=dev)
        with torch.inference_mode():
            draft(input_ids=prompt, position_ids=cache.positions(args.ctx),
                  attention_mask=cache.attn_mask(args.ctx, draft.dtype),
                  past_key_values=cache,
                  cache_position=cache.cache_position(args.ctx), use_cache=True)
            cache.advance(args.ctx)
            # Static inputs: a graph cannot capture a mask whose width grows, so
            # freeze it at the current context. This is exactly the constraint
            # the engine would have to satisfy to use graphs for real.
            step = torch.randint(0, 1000, (B, 1), device=dev)
            pos = cache.positions(1).clone()
            mask = cache.attn_mask(1, draft.dtype).clone()
            cpos = cache.cache_position(1).clone()

            def one():
                draft(input_ids=step, position_ids=pos, attention_mask=mask,
                      past_key_values=cache, cache_position=cpos, use_cache=True)
            eager_ms = timed_forward(one) * 1e3

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    one()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                one()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                g.replay()
            torch.cuda.synchronize()
            graph_ms = (time.perf_counter() - t0) / 50 * 1e3
        out["cuda_graph"] = {"eager_ms": eager_ms, "graph_ms": graph_ms,
                             "speedup": eager_ms / graph_ms,
                             "launch_overhead_frac": max(0.0, 1 - graph_ms / eager_ms)}
        log(f"  draft decode eager {eager_ms:.3f} ms -> graphed {graph_ms:.3f} ms "
            f"({eager_ms/graph_ms:.2f}x; "
            f"{100*max(0,1-graph_ms/eager_ms):.0f}% of the step was launch overhead)")
    except Exception as e:
        log(f"  CUDA graph capture failed ({type(e).__name__}: {e}); reporting eager only")
        out["cuda_graph"] = {"error": f"{type(e).__name__}: {e}"}

    p = C.publish_result(f"stage6_{args.tag}", out)
    log(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
