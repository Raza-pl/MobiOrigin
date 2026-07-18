#!/usr/bin/env python3
"""Generate comparison figures from benchmark evaluation results.

Reads the TSVs produced by evaluate.py and outputs publication-quality figures:

  01_f1_overall.png       — grouped bar chart: F1 per tool (overall)
  02_pr_tradeoff.png      — precision vs recall scatter per tool
  03_f1_by_length.png     — heatmap: F1 by (tool, length tier)
  04_mcc_by_taxon.png     — heatmap: MCC by (tool, taxon)
  05_timing.png           — bar chart: wall-clock seconds per tool

Usage
-----
    python scripts/benchmark/plot_results.py \\
        --eval  data/benchmark/eval/tier1_all \\
        --out   data/benchmark/figures/tier1_all
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOOL_COLORS = {
    "plasflow2": "#1a7a4a",  # PlasFlow v2 — green (ours)
    "genomad": "#2471a3",  # geNomad — blue
    "plasclass": "#e67e22",  # PlasClass — orange
    "rfplasmid": "#8e44ad",  # RFPlasmid — purple
    "mobrecon": "#c0392b",  # MOB-recon — red
}
TOOL_LABELS = {
    "plasflow2": "PlasFlow v2 (ours)",
    "genomad": "geNomad",
    "plasclass": "PlasClass",
    "rfplasmid": "RFPlasmid",
    "mobrecon": "MOB-recon",
}


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        import sys

        sys.exit("matplotlib is required: pip install matplotlib")


def _load_tsv(path: Path) -> list[dict]:
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _tool_order(rows: list[dict]) -> list[str]:
    """Return tools sorted by overall F1 descending."""
    by_f1 = sorted(rows, key=lambda r: -float(r.get("f1", 0)))
    seen: list[str] = []
    for r in by_f1:
        t = r["tool"]
        if t not in seen:
            seen.append(t)
    return seen


# ── Plot 1: Overall F1, Precision, Recall ─────────────────────────────────────


def plot_overall(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    tools = _tool_order(rows)
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(tools))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        vals = [float(next(r[metric] for r in rows if r["tool"] == t)) for t in tools]
        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=metric.capitalize(),
            color=["#1a7a4a", "#2471a3", "#e67e22"][i],
            alpha=0.85,
        )
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([TOOL_LABELS.get(t, t) for t in tools], rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Plasmid Classification — Overall Precision / Recall / F1", fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "01_f1_overall.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


# ── Plot 2: Precision vs Recall scatter ────────────────────────────────────────


def plot_pr_scatter(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))

    for row in rows:
        tool = row["tool"]
        p = float(row["precision"])
        r = float(row["recall"])
        color = TOOL_COLORS.get(tool, "#888")
        label = TOOL_LABELS.get(tool, tool)
        ax.scatter(r, p, s=180, color=color, zorder=5, label=label)
        ax.annotate(
            label,
            (r, p),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=color,
        )
        # F1 iso-curves
    for f1_target in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        r_vals = [v / 100 for v in range(1, 101)]
        p_vals = [
            f1_target * rv / (2 * rv - f1_target) if (2 * rv - f1_target) > 0 else None
            for rv in r_vals
        ]
        rv_f = [rv for rv, pv in zip(r_vals, p_vals) if pv is not None and 0 <= pv <= 1]
        pv_f = [pv for rv, pv in zip(r_vals, p_vals) if pv is not None and 0 <= pv <= 1]
        if rv_f:
            ax.plot(rv_f, pv_f, "--", color="#cccccc", linewidth=0.7, zorder=1)
            ax.text(rv_f[-1], pv_f[-1], f"F1={f1_target}", fontsize=6.5, color="#aaa")

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision vs Recall — Plasmid Detection", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out_path = out_dir / "02_pr_tradeoff.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


# ── Plot 3: F1 by length tier (heatmap) ───────────────────────────────────────

LENGTH_TIER_ORDER = ["<2 kb", "2-5 kb", "5-10 kb", "10-50 kb", ">50 kb"]


def plot_by_length(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    tools = list(TOOL_COLORS.keys())
    tiers = [t for t in LENGTH_TIER_ORDER if any(r.get("length_tier") == t for r in rows)]
    if not tiers:
        logger.warning("No length-tier data — skipping plot 3")
        return

    matrix = np.zeros((len(tools), len(tiers)))
    for i, tool in enumerate(tools):
        for j, tier in enumerate(tiers):
            match = next(
                (r for r in rows if r["tool"] == tool and r.get("length_tier") == tier), None
            )
            matrix[i, j] = float(match["f1"]) if match else 0.0

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="F1 score")

    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers)
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels([TOOL_LABELS.get(t, t) for t in tools])

    for i in range(len(tools)):
        for j in range(len(tiers)):
            val = matrix[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    ax.set_title("F1 Score by Contig Length Tier", fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "03_f1_by_length.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


# ── Plot 4: MCC by taxon (heatmap) ────────────────────────────────────────────


def plot_by_taxon(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    tools = list(TOOL_COLORS.keys())
    taxa = sorted({r["taxon"] for r in rows if r.get("taxon")})
    if not taxa:
        logger.warning("No taxon data — skipping plot 4")
        return

    matrix = np.zeros((len(tools), len(taxa)))
    for i, tool in enumerate(tools):
        for j, taxon in enumerate(taxa):
            match = next((r for r in rows if r["tool"] == tool and r.get("taxon") == taxon), None)
            matrix[i, j] = float(match["mcc"]) if match else 0.0

    fig, ax = plt.subplots(figsize=(max(8, len(taxa) * 1.4), 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="MCC")

    short_taxa = [t.split()[-1] for t in taxa]  # just genus abbreviation
    ax.set_xticks(range(len(taxa)))
    ax.set_xticklabels(short_taxa, rotation=30, ha="right")
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels([TOOL_LABELS.get(t, t) for t in tools])

    for i in range(len(tools)):
        for j in range(len(taxa)):
            val = matrix[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    ax.set_title("MCC by Taxon", fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "04_mcc_by_taxon.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


# ── Plot 5: Runtime ────────────────────────────────────────────────────────────


def plot_timing(timing_path: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not timing_path.exists():
        logger.warning("timing.tsv not found — skipping runtime plot")
        return

    rows = _load_tsv(timing_path)
    tools, times = [], []
    for row in rows:
        if row.get("status") in ("ok", ""):
            t = float(row.get("wallclock_sec", -1))
            if t >= 0:
                tools.append(row["tool"])
                times.append(t / 60)  # convert to minutes

    if not tools:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [TOOL_COLORS.get(t, "#888") for t in tools]
    bars = ax.barh(tools, times, color=colors, alpha=0.85)
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} min",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Wall-clock time (minutes)")
    ax.set_title("Runtime Comparison", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticklabels([TOOL_LABELS.get(t, t) for t in tools])

    plt.tight_layout()
    out_path = out_dir / "05_timing.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    _require_matplotlib()

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--eval", required=True, type=Path, help="Evaluation directory produced by evaluate.py."
    )
    p.add_argument("--out", required=True, type=Path, help="Output directory for figures.")
    p.add_argument(
        "--timing", type=Path, default=None, help="timing.tsv from run_tools.sh (optional)."
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args.out.mkdir(parents=True, exist_ok=True)

    overall = _load_tsv(args.eval / "metrics_overall.tsv")
    by_length = _load_tsv(args.eval / "metrics_by_length.tsv")
    by_taxon = _load_tsv(args.eval / "metrics_by_taxon.tsv")
    timing_path = args.timing or (args.eval.parent / "timing.tsv")

    if overall:
        plot_overall(overall, args.out)
        plot_pr_scatter(overall, args.out)
    if by_length:
        plot_by_length(by_length, args.out)
    if by_taxon:
        plot_by_taxon(by_taxon, args.out)
    plot_timing(timing_path, args.out)

    logger.info("All figures saved to %s", args.out)


if __name__ == "__main__":
    main()
