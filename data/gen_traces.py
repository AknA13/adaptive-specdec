"""Stage 1: generate Qwen3-8B reasoning traces with vLLM, plus teacher top-k
logprobs for the KD stage.

Dumping the teacher's top-k distribution here is close to free -- vLLM already
has the logits -- and it is the only cheap moment to get it. Recomputing it in
the training loop would mean holding the 8B resident alongside the student for
the whole run.

Operational shape follows bijection-reasoning/data/gen_base_cot.py: strided
sharding so N GPUs can each take 1/N of the problems, chunked appends so a
preempted job loses at most one chunk, and --resume to skip what is already on
disk.

  python -m data.gen_traces --dataset gsm8k --split train --shard-id 0 --num-shards 1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from data.common import load_problems, build_prompt


def log(m):
    print(f"[gen] {m}", flush=True)


def done_indices(path):
    """Indices already written, so --resume can skip them."""
    seen = set()
    if not path.exists():
        return seen
    with open(path) as f:
        for line in f:
            try:
                seen.add(json.loads(line)["idx"])
            except Exception:
                continue          # a torn last line from a preempted write
    return seen


def pack_logprobs(step_logprobs, tok_ids, topk):
    """vLLM per-step {id: Logprob} -> parallel id/logprob lists, rounded.

    Rounded to 3 decimals: the KD loss is a softmax over 8 entries, so the
    third decimal of a logprob is far below anything that changes a gradient,
    and it roughly halves the file.
    """
    ids, lps = [], []
    for t, d in enumerate(step_logprobs or []):
        if not d:
            ids.append([int(tok_ids[t])])
            lps.append([0.0])
            continue
        items = sorted(d.items(), key=lambda kv: -kv[1].logprob)[:topk]
        ids.append([int(k) for k, _ in items])
        lps.append([round(float(v.logprob), 3) for _, v in items])
    return ids, lps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.TARGET_ID)
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=0, help="0 = all problems")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gen-chunk", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=C.GEN_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=C.GEN_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=C.GEN_TOP_P)
    ap.add_argument("--top-k", type=int, default=C.GEN_TOP_K)
    ap.add_argument("--logprobs", type=int, default=C.GEN_TOP_LOGPROBS,
                    help="teacher top-k to store for KD; 0 disables")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    C.ensure_data_dirs()
    out = Path(args.out) if args.out else (
        C.TRACE_DIR / f"raw_{args.dataset}_{args.split}_{args.shard_id}of{args.num_shards}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    probs = load_problems(args.dataset, args.split, args.n)
    mine = probs[args.shard_id::args.num_shards]          # strided: even hard/easy mix
    skip = done_indices(out) if args.resume else set()
    todo = [p for p in mine if p["idx"] not in skip]
    log(f"{len(probs)} problems, shard {args.shard_id}/{args.num_shards} -> {len(mine)}, "
        f"{len(skip)} already done, {len(todo)} to go -> {out}")
    if not todo:
        log("nothing to do")
        return 0

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, enforce_eager=False)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens,
                        logprobs=args.logprobs if args.logprobs > 0 else None)

    t0 = time.time()
    written = 0
    mode = "a" if args.resume and out.exists() else "w"
    with open(out, mode) as fh:
        for s in range(0, len(todo), args.gen_chunk):
            chunk = todo[s:s + args.gen_chunk]
            prompts = [build_prompt(tok, p["problem"]) for p in chunk]
            outs = llm.generate(prompts, sp)
            for p, prompt, o in zip(chunk, prompts, outs):
                comp = o.outputs[0]
                rec = {
                    "idx": p["idx"], "source": p["source"],
                    "problem": p["problem"], "answer": p["answer"],
                    "prompt": prompt,
                    "text": comp.text,
                    "token_ids": list(comp.token_ids),
                    "finish_reason": comp.finish_reason,
                }
                if args.logprobs > 0:
                    ids, lps = pack_logprobs(comp.logprobs, comp.token_ids, args.logprobs)
                    rec["tk_ids"], rec["tk_lps"] = ids, lps
                fh.write(json.dumps(rec) + "\n")
                written += 1
            fh.flush()
            os.fsync(fh.fileno())      # a preempted job must not lose a flushed chunk
            el = time.time() - t0
            log(f"{written}/{len(todo)} written  {el/60:.1f} min  "
                f"{written/max(el,1e-9):.2f} rec/s")
    log(f"DONE {written} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
