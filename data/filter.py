"""Stage 2: filter raw teacher traces down to a clean training set.

Every rejection reason is counted and published, because the kept fraction is
the first thing that tells you the generation settings were wrong (a low
"correct" rate means the sampler or the prompt is off; a high "truncated" rate
means --max-tokens is too small).

Decontamination is not optional here. The draft model is judged by its
acceptance rate against the target on gsm8k test and MATH-500; if a training
trace is really an eval problem, acceptance on that problem goes up for reasons
that have nothing to do with the draft being good.

  python -m data.filter --out filtered.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from data.common import load_problems, text_hash, verify_answer, extract_boxed


def log(m):
    print(f"[filter] {m}", flush=True)


def eval_hashes():
    """Normalised-text hashes of every problem in every eval set."""
    hs = set()
    for name, split in (("gsm8k", "test"), ("math500", "test")):
        try:
            for p in load_problems(name, split):
                hs.add(text_hash(p["problem"]))
        except Exception as e:
            log(f"WARNING could not load {name}/{split} for decontamination: {e}")
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="raw_*.jsonl")
    ap.add_argument("--out", default="filtered.jsonl")
    ap.add_argument("--max-completion-tokens", type=int, default=C.MAX_SEQ_LEN)
    ap.add_argument("--min-completion-tokens", type=int, default=16)
    ap.add_argument("--require-think", action="store_true", default=True)
    ap.add_argument("--no-require-think", dest="require_think", action="store_false")
    args = ap.parse_args()

    C.ensure_data_dirs()
    srcs = sorted(C.TRACE_DIR.glob(args.glob))
    if not srcs:
        log(f"no inputs matching {C.TRACE_DIR}/{args.glob}")
        return 1
    out_path = C.TRACE_DIR / args.out
    log(f"{len(srcs)} shard(s) -> {out_path}")

    contam = eval_hashes()
    log(f"{len(contam)} eval problem hashes loaded for decontamination")

    reasons = Counter()
    seen_problem = set()
    kept = 0
    total = 0
    lens = []
    with open(out_path, "w") as fo:
        for src in srcs:
            with open(src) as fi:
                for line in fi:
                    try:
                        r = json.loads(line)
                    except Exception:
                        reasons["malformed_line"] += 1
                        continue
                    total += 1
                    n_tok = len(r.get("token_ids") or [])
                    text = r.get("text") or ""
                    h = text_hash(r.get("problem", ""))

                    if r.get("finish_reason") != "stop":
                        reasons["truncated"] += 1
                        continue
                    if n_tok < args.min_completion_tokens:
                        reasons["too_short"] += 1
                        continue
                    if n_tok > args.max_completion_tokens:
                        # kept out of training rather than truncated: a clipped
                        # trace ends mid-reasoning and teaches the draft to stop early
                        reasons["too_long"] += 1
                        continue
                    if args.require_think and C.THINK_CLOSE not in text:
                        reasons["no_think_close"] += 1
                        continue
                    if extract_boxed(text) is None:
                        reasons["no_boxed_answer"] += 1
                        continue
                    if h in contam:
                        reasons["eval_contamination"] += 1
                        continue
                    if h in seen_problem:
                        reasons["duplicate_problem"] += 1
                        continue
                    if not verify_answer(text, r.get("answer")):
                        reasons["wrong_answer"] += 1
                        continue

                    seen_problem.add(h)
                    lens.append(n_tok)
                    fo.write(json.dumps(r) + "\n")
                    kept += 1

    lens.sort()
    stats = {
        "total": total, "kept": kept,
        "kept_frac": kept / max(1, total),
        "rejected": dict(reasons.most_common()),
        "completion_tokens": {
            "mean": sum(lens) / max(1, len(lens)),
            "p50": lens[len(lens) // 2] if lens else 0,
            "p90": lens[int(0.9 * len(lens))] if lens else 0,
            "max": lens[-1] if lens else 0,
        },
        "out": str(out_path),
    }
    log(json.dumps(stats, indent=2))
    C.publish_result("stage2_filter", stats)
    if kept == 0:
        log("FATAL nothing survived the filter")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
