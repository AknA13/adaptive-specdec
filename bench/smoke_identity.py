"""Bring-up checkpoint 5: greedy output through the draft-model proposer must be
token-identical to greedy output with speculative decoding off.

Rejection sampling guarantees this, so a mismatch is not a tuning issue -- it
means the drafter's positions, slot mapping or shift are wrong. Running it at
k=1 first isolates the single-step path from the drafting loop.

Used by scripts/05_bench_vllm.sh, which starts each server in turn and passes
the two dumps here.

  python -m bench.smoke_identity --dump out.json --base-url http://127.0.0.1:8000
  python -m bench.smoke_identity --compare nospec.json spec.json
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from data.common import load_problems, MATH_INSTRUCTION


def post(url, body, timeout=1200):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--dump")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        diffs = [i for i, (x, y) in enumerate(zip(a["texts"], b["texts"])) if x != y]
        print(f"[smoke] {len(a['texts'])} prompts, {len(diffs)} differ")
        for i in diffs[:3]:
            print(f"  prompt {i}:\n    no-spec: {a['texts'][i][:160]!r}\n"
                  f"    spec   : {b['texts'][i][:160]!r}")
        C.publish_result("stage5_greedy_identity",
                         {"n": len(a["texts"]), "n_diff": len(diffs),
                          "identical": not diffs,
                          "a": args.compare[0], "b": args.compare[1]})
        if diffs:
            print("[smoke] FAIL: greedy speculative output diverges from no-spec "
                  "-- positions/slot-mapping/shift bug, not a tuning problem")
            return 1
        print("[smoke] PASS: greedy output is token-identical (checkpoint 5)")
        return 0

    model = json.load(urllib.request.urlopen(f"{args.base_url}/v1/models",
                                             timeout=60))["data"][0]["id"]
    probs = load_problems("math500", "test", args.n)
    texts = []
    for p in probs:
        r = post(f"{args.base_url}/v1/chat/completions",
                 {"model": model,
                  "messages": [{"role": "user", "content": p["problem"].strip() + MATH_INSTRUCTION}],
                  "max_tokens": args.max_tokens, "temperature": 0.0, "seed": 0})
        texts.append(r["choices"][0]["message"]["content"])
    Path(args.dump).write_text(json.dumps({"model": model, "texts": texts}, indent=2))
    print(f"[smoke] wrote {args.dump} ({len(texts)} completions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
