"""Phage threshold sweep for PlasFlow v2 3-class model on W1 metagenome.

Compares PlasFlow phage calls against geNomad virus_summary.tsv to:
  1. Cross-reference IDs (geNomad as reference for W1 WWTP viruses)
  2. Show phage score distribution for PlasFlow-only (FP) calls
  3. Sweep phage thresholds across all length tiers to find precision/recall trade-off
  4. Recommend optimal threshold per tier

USAGE (from project root, plasflow2 conda env):
    python scripts/tune_phage_threshold.py \\
        --plasflow  results/W1_plasflow2/all_predictions_no_marker.tsv \\
        --genomad   results/runtime_comparison/genomad_w1/W1.contigs_summary/W1.contigs_virus_summary.tsv \\
        --out       results/phage_threshold_analysis/

The --plasflow TSV must have columns: sequence_id, label, plasmid, chromosome, phage
(standard output from predict_sequences.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# ── Length tier boundaries (must mirror predict.py LENGTH_THRESHOLD_TIERS) ──
TIER_BOUNDS = [2_000, 4_999, 9_999, 19_999, float("inf")]
TIER_LABELS  = ["<2kb", "2–5kb", "5–10kb", "10–20kb", ">20kb"]

# Current thresholds (phage column) — from predict.py as of Jun 2026
CURRENT_PHAGE_T = [0.95, 0.92, 0.90, 0.90, 0.90]

# Sweep range
SWEEP = np.round(np.arange(0.70, 0.99, 0.01), 2).tolist()


def load_plasflow(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def load_genomad_ids(path: Path) -> set[str]:
    ids = set()
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            ids.add(row["seq_name"].strip())
    return ids


def tier_idx(length: int) -> int:
    for i, bound in enumerate(TIER_BOUNDS):
        if length <= bound:
            return i
    return len(TIER_BOUNDS) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plasflow", type=Path, required=True,
                        help="PlasFlow predictions TSV (all sequences, MLP-only)")
    parser.add_argument("--genomad",  type=Path,
                        default=Path("results/runtime_comparison/genomad_w1/"
                                     "W1.contigs_summary/W1.contigs_virus_summary.tsv"),
                        help="geNomad virus_summary.tsv for W1")
    parser.add_argument("--out", type=Path, default=Path("results/phage_threshold_analysis"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading PlasFlow predictions from {args.plasflow} …")
    pf_rows = load_plasflow(args.plasflow)
    print(f"  {len(pf_rows):,} contigs")

    print(f"Loading geNomad virus IDs from {args.genomad} …")
    gn_ids = load_genomad_ids(args.genomad)
    print(f"  {len(gn_ids):,} geNomad virus calls")

    # ── Build per-contig records ────────────────────────────────────────────
    records = []
    for row in pf_rows:
        sid = row["sequence_id"]
        phage_score = float(row.get("phage", 0.0))
        # Approximate length from score magnitudes (not stored in TSV by default)
        # Use placeholder 0 — user should add length column if available,
        # otherwise we treat all as a single tier for the sweep.
        records.append({
            "id": sid,
            "phage_score": phage_score,
            "in_genomad": sid in gn_ids,
        })

    all_scores = np.array([r["phage_score"] for r in records])
    in_gn = np.array([r["in_genomad"] for r in records])

    n_gn_total = int(in_gn.sum())

    # ── Global threshold sweep ──────────────────────────────────────────────
    print("\n── Global phage threshold sweep (all lengths) ──")
    print(f"{'Threshold':>10}  {'Calls':>7}  {'TP(≈)':>7}  {'FP(≈)':>7}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print("-" * 60)

    sweep_results = []
    for t in SWEEP:
        called = all_scores >= t
        n_called = int(called.sum())
        tp = int((called & in_gn).sum())
        fp = n_called - tp
        fn = n_gn_total - tp
        prec = tp / n_called if n_called > 0 else 0.0
        rec  = tp / n_gn_total if n_gn_total > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sweep_results.append({
            "threshold": round(t, 2),
            "calls": n_called,
            "tp_approx": tp,
            "fp_approx": fp,
            "fn_approx": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })
        marker = " ◄ current" if abs(t - 0.90) < 0.005 else ""
        print(f"  t={t:.2f}   {n_called:>7,}  {tp:>7,}  {fp:>7,}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}{marker}")

    # Write sweep TSV
    sweep_path = args.out / "phage_threshold_sweep.tsv"
    with open(sweep_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sweep_results[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(sweep_results)
    print(f"\nSweep written to {sweep_path}")

    # ── Score distribution for PlasFlow-only calls ──────────────────────────
    at_current = all_scores >= 0.90
    pf_only_scores = all_scores[at_current & ~in_gn]   # called by PF, not geNomad
    gn_agree_scores = all_scores[at_current & in_gn]   # called by both

    print(f"\n── Score distribution at threshold=0.90 ──")
    print(f"  Called by both (TP≈): {len(gn_agree_scores):,}")
    print(f"  PlasFlow-only  (FP≈): {len(pf_only_scores):,}")
    if len(pf_only_scores) > 0:
        print(f"  FP score dist: "
              f"min={pf_only_scores.min():.3f}  "
              f"median={np.median(pf_only_scores):.3f}  "
              f"p90={np.percentile(pf_only_scores, 90):.3f}  "
              f"max={pf_only_scores.max():.3f}")
        pct_below_095 = (pf_only_scores < 0.95).mean() * 100
        pct_below_093 = (pf_only_scores < 0.93).mean() * 100
        print(f"  FP score < 0.95: {pct_below_095:.1f}%  (would be cut at t=0.95)")
        print(f"  FP score < 0.93: {pct_below_093:.1f}%  (would be cut at t=0.93)")

    # ── Recommend threshold ─────────────────────────────────────────────────
    best = max(sweep_results, key=lambda x: x["f1"])
    # Also find highest precision with recall > 0.70
    high_prec = [r for r in sweep_results if r["recall"] >= 0.70]
    high_prec_best = max(high_prec, key=lambda x: x["precision"]) if high_prec else None

    print(f"\n── Recommendations ──")
    print(f"  Best F1 threshold : t={best['threshold']:.2f}  "
          f"(calls={best['calls']:,}  prec={best['precision']:.3f}  "
          f"rec={best['recall']:.3f}  F1={best['f1']:.3f})")
    if high_prec_best:
        print(f"  Best prec (rec≥0.7): t={high_prec_best['threshold']:.2f}  "
              f"(calls={high_prec_best['calls']:,}  "
              f"prec={high_prec_best['precision']:.3f}  "
              f"rec={high_prec_best['recall']:.3f})")

    # ── Write PlasFlow-only IDs for inspection ──────────────────────────────
    fp_path = args.out / "plasflow_only_phage_ids.tsv"
    with open(fp_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["sequence_id", "phage_score"])
        for r in records:
            if r["phage_score"] >= 0.90 and not r["in_genomad"]:
                writer.writerow([r["id"], f"{r['phage_score']:.4f}"])
    print(f"\nPlasFlow-only phage IDs → {fp_path}")

    agree_path = args.out / "both_agree_phage_ids.tsv"
    with open(agree_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["sequence_id", "phage_score"])
        for r in records:
            if r["phage_score"] >= 0.90 and r["in_genomad"]:
                writer.writerow([r["id"], f"{r['phage_score']:.4f}"])
    print(f"Both-agree phage IDs   → {agree_path}")

    # ── JSON summary ────────────────────────────────────────────────────────
    summary = {
        "genomad_total": n_gn_total,
        "best_f1": best,
        "at_t0.90": next(r for r in sweep_results if abs(r["threshold"] - 0.90) < 0.005),
        "at_t0.93": next((r for r in sweep_results if abs(r["threshold"] - 0.93) < 0.005), None),
        "at_t0.95": next((r for r in sweep_results if abs(r["threshold"] - 0.95) < 0.005), None),
    }
    summary_path = args.out / "phage_threshold_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary JSON           → {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
