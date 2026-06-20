#!/usr/bin/env python3
"""
tune_thresholds_k7.py — Find optimal per-length-bin plasmid thresholds for
the k=7 canonical model.

Reads:
  data/benchmark/results/plasflow2_predictions.tsv  — plasmid_score per contig
  data/benchmark/ground_truth.tsv                    — true_label + length

Outputs:
  Optimal threshold per length bin (to update LENGTH_THRESHOLD_TIERS in predict.py)
  Score distributions for FP/FN analysis

Usage:
  python scripts/tune_thresholds_k7.py
  python scripts/tune_thresholds_k7.py --predictions data/k7_experiment/...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import csv
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Length bins matching predict.py LENGTH_THRESHOLD_TIERS
# ─────────────────────────────────────────────────────────────────────────────
BINS = [
    ("1-2kb",   1_000,   2_000),
    ("2-5kb",   2_000,   5_000),
    ("5-10kb",  5_000,  10_000),
    ("10-20kb", 10_000, 20_000),
    (">20kb",   20_000, float("inf")),
]


def length_bin(L: int) -> str:
    for name, lo, hi in BINS:
        if lo <= L < hi:
            return name
    return ">20kb"


def f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def sweep_thresholds(scores: np.ndarray, labels: np.ndarray,
                     thresholds: np.ndarray) -> dict:
    """Return per-threshold {tp, fp, fn, precision, recall, f1} for the plasmid class."""
    results = []
    n_pos = labels.sum()
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1v  = f1(tp, fp, fn)
        results.append(dict(threshold=t, tp=tp, fp=fp, fn=fn,
                            precision=prec, recall=rec, f1=f1v))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune per-length-bin plasmid thresholds")
    parser.add_argument(
        "--predictions", default="data/benchmark/results/plasflow2_predictions.tsv",
        help="TSV from run_benchmark_evaluation.py (default: data/benchmark/results/plasflow2_predictions.tsv)",
    )
    parser.add_argument(
        "--ground-truth", default="data/benchmark/ground_truth.tsv",
        help="Ground truth TSV with true_label + length columns",
    )
    parser.add_argument(
        "--min-f1-plasmid", type=float, default=0.0,
        help="Minimum plasmid F1 required (used to exclude bins with no plasmids)",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    gt_path   = Path(args.ground_truth)

    if not pred_path.exists():
        sys.exit(f"ERROR: Predictions file not found: {pred_path}")
    if not gt_path.exists():
        sys.exit(f"ERROR: Ground truth file not found: {gt_path}")

    # ── Load ground truth ─────────────────────────────────────────────────────
    gt: dict[str, tuple[str, int]] = {}  # contig_id → (true_label, length)
    with open(gt_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gt[row["contig_id"]] = (row["true_label"], int(row["length"]))

    # ── Load predictions ──────────────────────────────────────────────────────
    records: list[dict] = []
    with open(pred_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cid = row["contig_id"]
            if cid not in gt:
                continue
            true_label, length = gt[cid]
            records.append({
                "contig_id":    cid,
                "plasmid_score": float(row["plasmid_score"]),
                "true_label":   true_label,
                "length":       length,
                "bin":          length_bin(length),
            })

    print(f"Loaded {len(records)} contigs ({sum(1 for r in records if r['true_label']=='plasmid')} plasmids)\n")

    thresholds = np.arange(0.50, 1.001, 0.01)

    # ── Global sweep ─────────────────────────────────────────────────────────
    all_scores = np.array([r["plasmid_score"] for r in records], dtype=np.float64)
    all_labels = np.array([1 if r["true_label"] == "plasmid" else 0 for r in records])
    global_results = sweep_thresholds(all_scores, all_labels, thresholds)
    best_global = max(global_results, key=lambda x: x["f1"])
    print("=" * 65)
    print("GLOBAL (all lengths combined)")
    print("=" * 65)
    print(f"  Best global F1:  {best_global['f1']:.4f}  "
          f"P={best_global['precision']:.4f}  R={best_global['recall']:.4f}  "
          f"threshold={best_global['threshold']:.2f}")
    print(f"  (TP={best_global['tp']} FP={best_global['fp']} FN={best_global['fn']})")
    print()

    # ── Per-bin sweep ─────────────────────────────────────────────────────────
    print("=" * 65)
    print("PER-LENGTH-BIN THRESHOLD SWEEP")
    print("=" * 65)

    optimal: dict[str, float] = {}  # bin_name → best threshold

    for bin_name, lo, hi in BINS:
        bin_records = [r for r in records if r["bin"] == bin_name]
        n_total = len(bin_records)
        n_plas  = sum(1 for r in bin_records if r["true_label"] == "plasmid")
        n_chrom = n_total - n_plas

        if n_total == 0:
            print(f"\n{bin_name}: no sequences — skipping")
            optimal[bin_name] = 0.99
            continue

        scores = np.array([r["plasmid_score"] for r in bin_records])
        labels = np.array([1 if r["true_label"] == "plasmid" else 0 for r in bin_records])

        results = sweep_thresholds(scores, labels, thresholds)
        best = max(results, key=lambda x: x["f1"])

        # Score distribution for plasmids vs chromosomes
        plas_scores  = scores[labels == 1]
        chrom_scores = scores[labels == 0]

        print(f"\n{bin_name}  ({n_plas} plasmids / {n_chrom} chromosomes)")
        print(f"  Plasmid score:     "
              f"mean={plas_scores.mean():.3f}  "
              f"p10={np.percentile(plas_scores, 10):.3f}  "
              f"p50={np.percentile(plas_scores, 50):.3f}  "
              f"p90={np.percentile(plas_scores, 90):.3f}" if len(plas_scores) > 0 else "  No plasmids")
        print(f"  Chromosome score:  "
              f"mean={chrom_scores.mean():.3f}  "
              f"p90={np.percentile(chrom_scores, 90):.3f}  "
              f"p95={np.percentile(chrom_scores, 95):.3f}  "
              f"p99={np.percentile(chrom_scores, 99):.3f}  "
              f"max={chrom_scores.max():.3f}")

        if n_plas == 0:
            print(f"  No plasmids in this bin — threshold has no effect")
            optimal[bin_name] = 0.99
        else:
            best = max(results, key=lambda x: x["f1"])
            print(f"  Best F1: {best['f1']:.4f}  "
                  f"P={best['precision']:.4f}  R={best['recall']:.4f}  "
                  f"threshold={best['threshold']:.2f}  "
                  f"(TP={best['tp']} FP={best['fp']} FN={best['fn']})")

            # Show a few threshold levels for comparison
            check_thresholds = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
            rows = [r for r in results if any(abs(r["threshold"] - t) < 0.005 for t in check_thresholds)]
            print(f"  {'thresh':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'P':>6}  {'R':>6}  {'F1':>6}")
            for r in rows:
                print(f"  {r['threshold']:>7.2f}  {r['tp']:>5}  {r['fp']:>5}  {r['fn']:>5}  "
                      f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f1']:>6.4f}")
            optimal[bin_name] = best["threshold"]

    # ── Recommendation ────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("RECOMMENDED LENGTH_THRESHOLD_TIERS FOR predict.py (k=7 model)")
    print("=" * 65)
    print("""
# (plasmid_threshold, chromosome_threshold, phage_threshold)
# chromosome/phage thresholds remain as-is — only plasmid column is swept here.
# Adjust chromosome/phage thresholds independently if needed.
""")
    print("LENGTH_THRESHOLD_TIERS = [")
    # Map bin names to predict.py boundary format
    boundaries = [
        ("1-2kb",   2_000,  "0.95, 0.80"),
        ("2-5kb",   4_999,  "0.90, 0.75"),
        ("5-10kb",  9_999,  "0.87, 0.72"),
        ("10-20kb", 19_999, "0.85, 0.70"),
        (">20kb",   None,   "0.82, 0.68"),
    ]
    for bin_name, boundary, other in boundaries:
        opt_t = optimal.get(bin_name, 0.95)
        # Round to 2 decimal places and ensure [0.50, 0.99]
        opt_t = max(0.50, min(0.99, round(opt_t, 2)))
        if boundary is not None:
            print(f"    ({boundary:>7},  {opt_t:.2f},  {other}),   # {bin_name}")
        else:
            print(f"    (float('inf'), {opt_t:.2f},  {other}),  # {bin_name}")
    print("]")

    print()
    print("NOTE: After updating predict.py, re-run run_benchmark.sh with --force")
    print("      (or delete data/benchmark/results/plasflow2_predictions.tsv)")
    print()

    # ── Chromosome score distribution at key bins ─────────────────────────────
    print("=" * 65)
    print("CHROMOSOME FP ANALYSIS — top-scoring chromosome windows")
    print("(helps identify which organisms are causing FPs)")
    print("=" * 65)
    for bin_name, lo, hi in BINS:
        bin_records = [r for r in records if r["bin"] == bin_name and r["true_label"] == "chromosome"]
        if not bin_records:
            continue
        top = sorted(bin_records, key=lambda x: x["plasmid_score"], reverse=True)[:10]
        if top[0]["plasmid_score"] < 0.70:
            continue
        print(f"\n{bin_name} — top chromosome FPs:")
        print(f"  {'score':>7}  contig_id")
        for r in top:
            if r["plasmid_score"] >= 0.70:
                print(f"  {r['plasmid_score']:>7.4f}  {r['contig_id']}")

    # ── Plasmid FN analysis ────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("PLASMID FN ANALYSIS — missed plasmids (score < 0.85)")
    print("=" * 65)
    fn_records = [r for r in records if r["true_label"] == "plasmid" and r["plasmid_score"] < 0.85]
    fn_records.sort(key=lambda x: x["plasmid_score"])
    print(f"  {len(fn_records)} plasmids score < 0.85")
    for bin_name, lo, hi in BINS:
        bin_fn = [r for r in fn_records if r["bin"] == bin_name]
        if bin_fn:
            scores_arr = np.array([r["plasmid_score"] for r in bin_fn])
            print(f"  {bin_name:>8}: {len(bin_fn):>3} missed  "
                  f"(mean={scores_arr.mean():.3f}  min={scores_arr.min():.3f}  max={scores_arr.max():.3f})")


if __name__ == "__main__":
    main()
