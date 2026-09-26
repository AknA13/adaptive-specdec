"""Dataset loading, prompt building, and answer verification.

Adapted from bijection-reasoning/eval/common.py and data/answer_check.py, cut
down to the two benchmarks this cluster actually has cached offline (gsm8k and
MATH-500) and to the math/\\boxed{} case.

Split discipline, which matters because the draft model is trained on teacher
traces and then evaluated on acceptance rate:

  gsm8k TRAIN  -> teacher traces -> draft training
  gsm8k TEST   -> evaluation only
  MATH-500     -> evaluation only (it ships as a test split; never train on it)

filter.py decontaminates the training traces against both eval sets by
normalised-text hash, so a memorised problem cannot inflate acceptance.
"""
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

MATH_INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


# ---- text hashing (decontamination) ---------------------------------------
def norm_text(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def text_hash(s):
    return hashlib.sha256(norm_text(s).encode()).hexdigest()


# ---- datasets --------------------------------------------------------------
def _gsm8k_answer(raw):
    s = str(raw)
    return s.split("####")[-1].strip().replace(",", "") if "####" in s else s.strip()


def load_problems(name, split, n=0):
    """name in {'gsm8k','math500'}. Returns [{idx, problem, answer}]."""
    from datasets import load_dataset
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        recs = [(r["question"], _gsm8k_answer(r["answer"])) for r in ds]
    elif name == "math500":
        ds = load_dataset(C.MATH500_ID, split=split)
        recs = [(r["problem"], str(r["answer"])) for r in ds]
    else:
        raise ValueError(f"unknown dataset {name!r} (cached here: gsm8k, math500)")
    out = [{"idx": i, "problem": p, "answer": a, "source": f"{name}/{split}"}
           for i, (p, a) in enumerate(recs)]
    return out[:n] if n else out


def build_prompt(tokenizer, problem, instruction=MATH_INSTRUCTION, thinking=True):
    """Qwen3 chat template with thinking mode. Falls back for templates that do
    not accept enable_thinking (R1-Distill / QwQ always think)."""
    msgs = [{"role": "user", "content": problem.strip() + instruction}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---- answer checking -------------------------------------------------------
def extract_boxed(text):
    """Content of the last \\boxed{...}, or None."""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth, start = 0, i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    return None


def answer_after_think(text):
    k = text.rfind(C.THINK_CLOSE)
    return text[k + len(C.THINK_CLOSE):] if k != -1 else text


def _norm_ans(s):
    s = (s or "").strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\,", ""), ("\\!", ""),
                 ("\\ ", ""), ("$", ""), ("%", ""), (",", "")):
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s).rstrip(".")


def verify_answer(pred_text, gold):
    """True if the prediction matches gold. math-verify first, string fallback."""
    pred = extract_boxed(pred_text)
    if pred is None:
        pred = answer_after_think(pred_text).strip()
    gold_str = str(gold)
    gb = extract_boxed(gold_str)
    if gb is not None:
        gold_str = gb
    try:
        from math_verify import parse, verify
        if verify(parse("\\boxed{" + gold_str + "}"), parse("\\boxed{" + (pred or "") + "}")):
            return True
    except Exception:
        pass
    return pred is not None and _norm_ans(pred) == _norm_ans(gold_str)
