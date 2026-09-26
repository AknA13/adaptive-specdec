"""Static repo checks. CPU-only, no models, runs in a second.

This is the test that catches a broken rename or a stale path before it burns a
GPU hour queueing behind 23 busy H200s. Adapted from
encrypted-reasoning/tests/test_repo_integrity.py.
"""
import ast
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FAIL = []
SKIP_DIRS = {".git", "__pycache__", "runs", "logs", ".egg-info"}


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def py_files():
    for p in REPO.rglob("*.py"):
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in p.parts):
            continue
        yield p


def main():
    files = sorted(py_files())
    print(f"[integrity] {len(files)} python files under {REPO}")

    print("[integrity] every module parses")
    trees = {}
    for p in files:
        try:
            trees[p] = ast.parse(p.read_text(), filename=str(p))
        except SyntaxError as e:
            check(f"parse {p.relative_to(REPO)}", False, f"line {e.lineno}: {e.msg}")
    if FAIL:
        return 1
    check(f"all {len(files)} modules parse", True)

    print("[integrity] intra-repo imports resolve")
    pkgs = {"config", "specdec", "data", "train", "bench", "tests", "vllm_draft_spec"}
    bad = []
    for p, tree in trees.items():
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.Import):
                mod = node.names[0].name
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mod = node.module
            if not mod:
                continue
            root = mod.split(".")[0]
            if root not in pkgs:
                continue
            target = REPO / pathlib.Path(mod.replace(".", "/"))
            if not (target.with_suffix(".py").exists() or (target / "__init__.py").exists()
                    or target.is_dir()):
                bad.append(f"{p.relative_to(REPO)} -> {mod}")
    check("no dangling internal imports", not bad, "; ".join(bad[:5]))

    print("[integrity] shell scripts exist, are executable, and parse")
    for sh in sorted((REPO / "scripts").glob("*.sh")) + sorted((REPO / "slurm").glob("*.sh")):
        ok = os.access(sh, os.X_OK) or sh.name == "lib.sh"
        rc = os.system(f"bash -n {sh!s} 2>/dev/null")
        check(f"{sh.relative_to(REPO)}", ok and rc == 0,
              "" if ok else "(not executable)")

    print("[integrity] scripts reference real entry points")
    missing = []
    for sh in sorted((REPO / "scripts").glob("*.sh")):
        txt = sh.read_text()
        for tok in txt.split():
            if tok.endswith(".py") and "/" in tok and not tok.startswith("-"):
                cand = REPO / tok.lstrip("$").lstrip("{").replace("PY}", "")
                if tok.count("$") == 0 and not (REPO / tok).exists():
                    missing.append(f"{sh.name} -> {tok}")
    check("no script points at a missing .py", not missing, "; ".join(missing[:5]))

    print("[integrity] config is importable and its gates are sane")
    import config as C
    check("config imports", True, C.describe())
    check("gates present", all(hasattr(C, g) for g in
                               ("G1_TV_MAX", "G2_GREEDY_PROMPTS", "G3_ACCEPT_GAIN",
                                "G4_ADAPTIVE_FLOOR", "G5_SPEEDUP")))
    # $HOME has ~2G free and /scratch ~10G: neither can hold a checkpoint or a
    # trace dump. Only an *explicitly set* bad root is an error; unset just
    # means this shell did not source env.sh, which is fine for CPU tests.
    explicit = os.environ.get("SPEC_DATA_ROOT")
    if explicit:
        check("SPEC_DATA_ROOT is not on a quota'd filesystem",
              not explicit.startswith(("/accounts", "/scratch")),
              f"{explicit} (home ~2G free, /scratch ~10G -- use /data/$USER/...)")
    else:
        print("  note SPEC_DATA_ROOT unset; GPU stages must be launched via "
              "scripts/*.sh so env.sh is sourced")

    print("[integrity] env.sh.example documents every SPEC_ var the code reads")
    example = (REPO / "env.sh.example").read_text()
    used = set()
    for p, tree in trees.items():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("get", "getenv") and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value.startswith("SPEC_")):
                used.add(node.args[0].value)
    # Shell stages read SPEC_ vars too, and those are the ones an operator is
    # most likely to want to set, so scan them as well.
    import re
    for sh in list((REPO / "scripts").glob("*.sh")) + list((REPO / "slurm").glob("*.sh")):
        used |= set(re.findall(r"SPEC_[A-Z0-9_]+", sh.read_text()))
    # SPEC_SLURM_ is a prefix fragment from a comment, not a variable.
    used -= {"SPEC_SLURM_", "SPEC_QUIET_ENV"}
    undocumented = sorted(v for v in used if v not in example)
    check("no undocumented SPEC_ env var", not undocumented, ", ".join(undocumented))

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("repo integrity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
