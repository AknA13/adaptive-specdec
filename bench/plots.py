"""Regenerate every figure in RESULTS_*.md from results/*.json.

Colors come from a validated categorical palette, assigned in fixed slot order
so a series keeps its hue when another is added or filtered out. Three of the
slots sit under 3:1 contrast on the light surface, so every chart carries a
legend and direct value labels rather than relying on color alone.

  python -m bench.plots
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

# Validated (node scripts/validate_palette.js, light surface #fcfcfb):
# lightness band PASS, chroma floor PASS, CVD separation PASS (worst adjacent
# dE 9.1 protan), normal-vision floor PASS (19.6).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e4e4e0"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 11, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 5,
    })
    return plt


def load(pattern):
    out = []
    for p in sorted(C.RESULTS_DIR.glob(pattern)):
        try:
            out.append((p.stem, json.loads(p.read_text())))
        except Exception:
            continue
    return out


def fig_alpha_by_position(plt):
    """Per-position conditional acceptance, one line per trained draft.

    Titled for what the data shows, not for what was expected. The geometric
    model behind the controller's speedup formula assumes a single per-position
    acceptance probability; if that held only loosely, the analytic k* would be
    built on sand. It holds here, which is the finding.

    Only the three draft variants on one dataset -- earlier exploratory runs
    (eager/compiled/quick tags) share the results directory and would put seven
    near-identical lines on one axis.
    """
    keep = {"stock", "sft", "sft_kd"}
    series = {}
    for stem, d in load("stage4_*.json"):
        if d.get("args", {}).get("dataset") != "math500":
            continue
        best = {}
        for r in d.get("rows", []):
            m = r.get("method", "")
            if (m.startswith("fixed") and m[5:].isdigit()
                    and r.get("mode") == "greedy" and r.get("draft") in keep):
                key = r["draft"]
                if int(m[5:]) >= best.get(key, (0, None))[0]:
                    best[key] = (int(m[5:]), r.get("alpha_by_position") or [])
        for key, (_, ab) in best.items():
            if ab:
                series[key] = ab
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, label in enumerate(["stock", "sft", "sft_kd"]):
        ys = series.get(label)
        if not ys:
            continue
        ax.plot(range(1, len(ys) + 1), ys, color=SERIES[i % len(SERIES)],
                marker="o", label=label)
    ax.set_xlabel("draft position within a round")
    ax.set_ylabel("acceptance | position reached")
    ax.set_title("Per-position acceptance is flat, as the geometric model assumes")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, max(len(v) for v in series.values()) + 1))
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    fig.tight_layout()
    return fig, "alpha_by_position.png"


def fig_engine_throughput(plt):
    rows = [r for _, d in load("stage4_*.json") for r in d.get("rows", [])
            if r.get("mode") == "greedy" and r.get("prompt_len") == "short"]
    if not rows:
        return None
    by_draft = {}
    for r in rows:
        by_draft.setdefault(r.get("draft", "?"), {})[r["method"]] = r["tokens_per_s"]
    order = ["ar", "fixed1", "fixed2", "fixed3", "fixed4", "fixed6", "fixed8",
             "ewma", "ewma+exit"]
    methods = [m for m in order if any(m in v for v in by_draft.values())]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    n = len(by_draft)
    w = 0.8 / max(1, n)
    for i, (draft, vals) in enumerate(sorted(by_draft.items())):
        xs = [j + i * w - 0.4 + w / 2 for j in range(len(methods))]
        ys = [vals.get(m, 0) for m in methods]
        ax.bar(xs, ys, width=w * 0.92, color=SERIES[i % len(SERIES)], label=draft,
               linewidth=0)
        for x, y in zip(xs, ys):
            if y:
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylabel("output tokens / s")
    ax.set_title("Single-stream throughput, greedy, short prompts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, "engine_throughput.png"


def fig_serving_concurrency(plt):
    """The crossover: speculation stops paying as the batch saturates."""
    runs = load("stage5_serving_*.json")
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, (stem, d) in enumerate(runs):
        rows = [r for r in d.get("rows", []) if r.get("ok")]
        if not rows:
            continue
        label = rows[0].get("label", stem)
        xs = [r["concurrency"] for r in rows]
        ys = [r["output_tok_per_s"] for r in rows]
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], marker="o", label=label)
        if ys:
            ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(6, 0), color=INK, fontsize=8, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel("output tokens / s")
    ax.set_title("Serving throughput vs concurrency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, "serving_throughput.png"


def fig_serving_ttft(plt):
    runs = load("stage5_serving_*.json")
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    any_row = False
    for i, (stem, d) in enumerate(runs):
        rows = [r for r in d.get("rows", []) if r.get("ok")]
        if not rows:
            continue
        any_row = True
        label = rows[0].get("label", stem)
        ax.plot([r["concurrency"] for r in rows],
                [r["ttft_p95"] * 1000 for r in rows],
                color=SERIES[i % len(SERIES)], marker="o", label=label)
    if not any_row:
        return None
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel("time to first token, p95 (ms)")
    ax.set_title("Tail latency vs concurrency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, "serving_ttft.png"


def fig_cost_ratio(plt):
    """c is not a constant, and k* follows it."""
    runs = load("stage6_*.json")
    rows = []
    for _, d in runs:
        rows += d.get("cost_ratio", [])
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    xs = [r["batch"] for r in rows]
    ys = [r["c"] for r in rows]
    ax.plot(xs, ys, color=SERIES[0], marker="o", label="c = t_draft / t_target")
    # Annotate below the marks: above collides with the title at the last point,
    # which is exactly the one worth reading.
    for r in rows:
        ax.annotate(f"c={r['c']:.2f}\nk*={r['k_star']['0.8']}", (r["batch"], r["c"]),
                    textcoords="offset points", xytext=(0, -20), ha="center",
                    va="top", fontsize=7, color=INK)
    ax.margins(y=0.28)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("draft cost as a fraction of a target forward")
    ax.set_title("Draft/target cost ratio, and the k* it implies at alpha=0.8")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, "cost_ratio.png"


def main():
    plt = _style()
    outdir = C.RESULTS_DIR / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig_alpha_by_position, fig_engine_throughput, fig_serving_concurrency,
               fig_serving_ttft, fig_cost_ratio):
        try:
            res = fn(plt)
        except Exception as e:
            print(f"[plots] {fn.__name__} failed: {type(e).__name__}: {e}")
            continue
        if res is None:
            print(f"[plots] {fn.__name__}: no data yet, skipping")
            continue
        fig, name = res
        fig.savefig(outdir / name, dpi=160)
        plt.close(fig)
        made.append(name)
        print(f"[plots] wrote {outdir / name}")
    if not made:
        print("[plots] nothing to plot -- run the benchmark stages first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
