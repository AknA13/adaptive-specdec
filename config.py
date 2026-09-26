"""Central configuration for the adaptive speculative decoding project.

Everything machine-specific is read from the environment so a fresh clone runs
anywhere. Set these in your shell (see `env.sh.example`):

  SPEC_DATA_ROOT   where large artifacts live: traces, checkpoints, profiles.
                   MUST NOT be $HOME or /scratch on this cluster -- both are
                   quota'd to a couple of GB free. Use node-local /data.
  SPEC_TARGET_ID   HF id (or local dir) of the target/teacher model.
  SPEC_DRAFT_ID    HF id (or local dir) of the draft model to fine-tune.
  HF_HOME          HuggingFace cache root.

Code and small result JSONs stay in the repo (`results/`); anything large goes
under SPEC_DATA_ROOT.

Never hard-code vocab size / n_layers / head_dim here -- read them from the
model config at runtime. The one thing we *do* assert is that target and draft
share a tokenizer, because speculative decoding is undefined otherwise.
"""
import os
from pathlib import Path

# ---- roots -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("SPEC_DATA_ROOT", str(REPO_ROOT / "runs" / "default")))

TRACE_DIR = DATA_ROOT / "traces"      # raw.jsonl + filtered.jsonl + teacher top-k
CKPT_DIR = DATA_ROOT / "ckpt"         # trained draft checkpoints (HF format)
PROFILE_DIR = DATA_ROOT / "profiles"  # chrome traces (tens of MB each)
BENCH_DIR = DATA_ROOT / "bench"       # raw per-request benchmark records

LOG_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"   # small JSONs, committed

HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))

# ---- models ----------------------------------------------------------------
TARGET_ID = os.environ.get("SPEC_TARGET_ID", "Qwen/Qwen3-8B")
DRAFT_ID = os.environ.get("SPEC_DRAFT_ID", "Qwen/Qwen3-0.6B")

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# ---- data ------------------------------------------------------------------
# Only gsm8k and MATH-500 are cached locally on this cluster. gsm8k *train* is
# the trace source; MATH-500 and gsm8k *test* are eval only and are decontaminated
# against the training traces.
TRAIN_SOURCES = ["openai/gsm8k"]
EVAL_SETS = ["gsm8k", "math500"]
MATH500_ID = "HuggingFaceH4/MATH-500"

# Qwen3 thinking-mode recommended sampling for trace generation.
GEN_TEMPERATURE = 0.6
GEN_TOP_P = 0.95
GEN_TOP_K = 20
GEN_MAX_TOKENS = 4096        # weekend budget; reasoning traces rarely exceed this on gsm8k
GEN_TOP_LOGPROBS = 8         # teacher top-k dumped for the KD stage

# ---- training --------------------------------------------------------------
KD_LAMBDA = 0.5              # loss = (1-l)*CE + l*T^2*KL(teacher_topk || student)
KD_TEMPERATURE = 1.0
LR = 1e-4
EPOCHS = 2
MAX_SEQ_LEN = 2048

# ---- speculative decoding --------------------------------------------------
K_MAX = 8                    # controller upper bound; also vLLM num_speculative_tokens
EWMA_BETA = 0.99             # acceptance smoothing, decayed PER TOKEN (~100-token window)
EARLY_EXIT_TAU = 0.3         # ConfidenceEarlyExit: abort a draft round below this

# ---- gates (see plan; a stage is not done until its gate passes) -----------
G1_TV_MAX = 0.02             # losslessness: total-variation vs target distribution
G2_GREEDY_PROMPTS = 200      # greedy outputs must be token-identical to AR on this many
G3_ACCEPT_GAIN = 0.05        # trained draft vs stock draft, absolute mean-acceptance gain
G4_ADAPTIVE_FLOOR = 0.95     # adaptive must reach this fraction of best fixed-k everywhere
G5_SPEEDUP = 1.5             # end-to-end tok/s vs autoregressive at bs=1 greedy


def ensure_data_dirs():
    """Create the artifact subdirs (call on the node where SPEC_DATA_ROOT is mounted)."""
    for d in (TRACE_DIR, CKPT_DIR, PROFILE_DIR, BENCH_DIR):
        d.mkdir(parents=True, exist_ok=True)


def publish_result(name, obj):
    """Write a small JSON into RESULTS_DIR (in-repo, always readable)."""
    import json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / (name if name.endswith(".json") else name + ".json")
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    return p


def describe():
    return (f"target={TARGET_ID} draft={DRAFT_ID} data_root={DATA_ROOT} "
            f"hf_home={HF_HOME} k_max={K_MAX}")


if __name__ == "__main__":
    print(describe())
    for k in ("TRACE_DIR", "CKPT_DIR", "PROFILE_DIR", "BENCH_DIR", "RESULTS_DIR"):
        print(f"  {k:12s} {globals()[k]}")
