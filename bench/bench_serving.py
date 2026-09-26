"""Stage 5: serving benchmark against a running `vllm serve` endpoint.

Poisson arrivals at a target concurrency, streaming responses, measuring what
actually matters for serving: time to first token, inter-token latency, and
end-to-end throughput -- not just aggregate tokens/s, which hides tail latency.

Concurrency is the axis where speculative decoding stops being free. At batch 1
the GPU is memory-bandwidth-bound and the draft's extra forwards are nearly
invisible; at high concurrency the verify step is already compute-saturated and
every rejected draft token is wasted FLOPs that would otherwise have served
another request. Expect the speedup to shrink, and possibly invert, as
concurrency rises -- that crossover is the interesting result, and it is what
motivates a controller that can back k off under load.

  python -m bench.bench_serving --base-url http://127.0.0.1:8000 \
      --label adaptive --concurrency 1,4,16,64
"""
import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from data.common import load_problems, MATH_INSTRUCTION


def log(m):
    print(f"[serve] {m}", flush=True)


async def one_request(session, url, model, prompt, max_tokens, temperature):
    """Stream one completion, timestamping every token."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None
    stamps = []
    ntok = 0
    text = []
    async with session.post(f"{url}/v1/chat/completions", json=body) as resp:
        if resp.status != 200:
            return {"ok": False, "error": (await resp.text())[:200]}
        async for raw in resp.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    stamps.append(now)
                    text.append(piece)
                    ntok += 1
    end = time.perf_counter()
    itls = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"ok": ntok > 0, "ttft": ttft, "e2e": end - t0, "tokens": ntok,
            "itls": itls, "text": "".join(text)}


async def run_level(url, model, prompts, concurrency, rate, max_tokens, temperature):
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def worker(p):
        async with sem:
            results.append(await one_request(session, url, model, p, max_tokens, temperature))

    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        t0 = time.perf_counter()
        tasks = []
        for p in prompts:
            if rate:      # Poisson arrivals; without this you measure a burst,
                          # which is a different (easier) scheduling problem
                await asyncio.sleep(random.expovariate(rate))
            tasks.append(asyncio.create_task(worker(p)))
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return results, wall


def summarize(results, wall, concurrency):
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return {"concurrency": concurrency, "requests": len(results), "ok": 0,
                "error": (results[0].get("error") if results else "no results")}
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    e2es = sorted(r["e2e"] for r in ok)
    itls = [x for r in ok for x in r["itls"]]
    toks = sum(r["tokens"] for r in ok)

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")

    return {
        "concurrency": concurrency, "requests": len(results), "ok": len(ok),
        "wall_s": wall, "output_tokens": toks,
        "output_tok_per_s": toks / wall, "req_per_s": len(ok) / wall,
        "ttft_p50": pct(ttfts, 0.5), "ttft_p95": pct(ttfts, 0.95),
        "itl_mean": statistics.mean(itls) if itls else float("nan"),
        "itl_p95": pct(sorted(itls), 0.95),
        "e2e_p50": pct(e2es, 0.5), "e2e_p95": pct(e2es, 0.95),
    }


def gpu_memory_gb():
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.split()
        return max(int(x) for x in out) / 1024
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None, help="model name as served; default = query /v1/models")
    ap.add_argument("--label", required=True, help="ar | fixed4 | adaptive ...")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=64, help="requests per concurrency level")
    ap.add_argument("--concurrency", default="1,4,16,64")
    ap.add_argument("--rate", type=float, default=0.0,
                    help="Poisson arrivals/s; 0 = release as fast as the semaphore allows")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cost-per-gpu-hour", type=float, default=3.50,
                    help="only used to turn throughput into $/1M tokens; stated as an assumption")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import aiohttp  # noqa: F401  (fail fast if missing)
    import urllib.request
    model = args.model
    if model is None:
        with urllib.request.urlopen(f"{args.base_url}/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]["id"]
    log(f"model={model} label={args.label}")

    random.seed(args.seed)
    problems = load_problems(args.dataset, args.split, args.n)
    prompts = [p["problem"].strip() + MATH_INSTRUCTION for p in problems]

    rows = []
    for conc in [int(c) for c in args.concurrency.split(",") if c]:
        log(f"concurrency {conc}: {len(prompts)} requests")
        results, wall = asyncio.run(run_level(
            args.base_url, model, prompts, conc, args.rate, args.max_tokens, args.temperature))
        s = summarize(results, wall, conc)
        s["label"] = args.label
        s["gpu_mem_gb"] = gpu_memory_gb()
        if s.get("output_tok_per_s"):
            # $/1M output tokens at the stated GPU price. An assumption, not a
            # measurement -- labelled as such wherever it is reported.
            s["usd_per_1m_tokens"] = (args.cost_per_gpu_hour / 3600.0) / \
                s["output_tok_per_s"] * 1e6
            s["cost_assumption_usd_gpu_hour"] = args.cost_per_gpu_hour
        rows.append(s)
        if s.get("ok"):
            log(f"  {s['output_tok_per_s']:7.1f} tok/s  {s['req_per_s']:6.2f} req/s  "
                f"ttft p50={s['ttft_p50']*1000:6.1f}ms p95={s['ttft_p95']*1000:6.1f}ms  "
                f"itl={s['itl_mean']*1000:5.2f}ms  ${s['usd_per_1m_tokens']:.2f}/1M")
        else:
            log(f"  FAILED: {s.get('error')}")

    p = C.publish_result(f"stage5_serving_{args.label}", {"rows": rows, "args": vars(args)})
    log(f"wrote {p}")
    return 0 if any(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
