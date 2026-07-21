#!/usr/bin/env python3
"""Characterize how far below Stage-1 candidacy the missed true plasmids sit.

Context: the entire biological-evidence pipeline (ARG/VF/MGE/ICE via DIAMOND,
mob-suite mobility, PLSDB match, geNomad, marker-XGBoost rescoring) only ever
runs on contigs the Stage-1 MLP already labels "plasmid" (pipeline.py's
`plasmid_records = [r for r in records if pred_by_id[r.id].label == "plasmid"]`).
Any true plasmid the MLP scores as chromosome/phage/unclassified is invisible
to every downstream evidence source, regardless of how strong the biology is.

The 2026-07 confirmation run found 170/394 true plasmids fall in this bucket.
Before widening the candidate gate (which costs real annotation runtime --
geNomad already times out at ~290 candidates), this script measures exactly
how far below the Stage-1 bar those 170 contigs sit, and what beat them, so
the widening policy can be sized deliberately instead of guessed.

For each true-plasmid contig that was NOT a Stage-1 candidate, reports:
  - plasmid_score vs. the tier threshold for its length (the deficit)
  - which class actually won (chromosome/phage/unclassified) and by how much
  - length bucket

Usage
-----
    python scripts/analyze_missed_candidates.py \\
        --predictions /tmp/pf2_default_v2/all_predictions.tsv \\
        --ground-truth data/benchmark/ground_truth.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

LENGTH_THRESHOLD_TIERS = [
    (2_000, 0.862, 0.95, 0.75),
    (4_999, 0.864, 0.92, 0.68),
    (9_999, 0.859, 0.90, 0.65),
    (19_999, 0.857, 0.90, 0.63),
    (float("inf"), 0.809, 0.90, 0.62),
]
LEGACY_PLASMID_FLOOR = 0.95  # historical CLI default, applied below 5kb


def _tier_plasmid_threshold(length: int) -> float:
    for max_len, plas_t, _phage_t, _chr_t in LENGTH_THRESHOLD_TIERS:
        if length <= max_len:
            if length < 5_000:
                return max(plas_t, LEGACY_PLASMID_FLOOR)
            return plas_t
    return LEGACY_PLASMID_FLOOR if length < 5_000 else 0.80


def _flt(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _len_bucket(length: int) -> str:
    if length < 2_000:
        return "<2kb"
    if length < 5_000:
        return "2-5kb"
    if length < 10_000:
        return "5-10kb"
    if length < 20_000:
        return "10-20kb"
    if length < 50_000:
        return "20-50kb"
    return ">=50kb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()

    gt: dict[str, str] = {}
    with open(args.ground_truth) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            gt[row["contig_id"]] = row["true_label"]

    rows: list[dict] = []
    with open(args.predictions) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cid = row["contig_id"]
            if gt.get(cid) != "plasmid":
                continue

            length = int(_flt(row.get("length", "0")))
            plas_s = _flt(row.get("plasmid_score"))
            chr_s = _flt(row.get("chromosome_score"))
            phg_s = _flt(row.get("phage_score"))
            thresh = _tier_plasmid_threshold(length)

            was_candidate = plas_s >= chr_s and plas_s >= phg_s and plas_s >= thresh
            if was_candidate:
                continue  # only care about the missed bucket here

            winner = max(
                [("chromosome", chr_s), ("phage", phg_s), ("plasmid", plas_s)],
                key=lambda t: t[1],
            )[0]
            deficit_vs_threshold = thresh - plas_s  # how far below the bar
            margin_vs_winner = max(chr_s, phg_s) - plas_s  # how far behind 2nd place

            rows.append(
                {
                    "contig_id": cid,
                    "length": length,
                    "len_bucket": _len_bucket(length),
                    "plasmid_score": plas_s,
                    "threshold": thresh,
                    "deficit_vs_threshold": deficit_vs_threshold,
                    "winner": winner,
                    "margin_vs_winner": margin_vs_winner,
                }
            )

    if not rows:
        print("No missed true-plasmid contigs found (nothing to analyze).")
        return

    rows.sort(key=lambda r: r["margin_vs_winner"])

    print(f"Missed true plasmids (not a Stage-1 candidate): {len(rows)}\n")

    # Margin-vs-winner distribution -- this is what determines how wide a
    # "near-miss" widening window needs to be to catch each contig.
    print("How close were they to being the argmax winner (plasmid_score vs. runner-up)?")
    buckets = [0.01, 0.05, 0.10, 0.20, 0.50, float("inf")]
    labels = ["<0.01", "0.01-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.50", ">=0.50"]
    counts = [0] * len(buckets)
    for r in rows:
        m = r["margin_vs_winner"]
        for i, b in enumerate(buckets):
            if m < b:
                counts[i] += 1
                break
    for label, c in zip(labels, counts):
        pct = 100 * c / len(rows)
        print(f"  margin {label:>10}: {c:4d}  ({pct:5.1f}%)")

    print("\nWho won instead (the class that beat plasmid)?")
    winner_counts: dict[str, int] = {}
    for r in rows:
        winner_counts[r["winner"]] = winner_counts.get(r["winner"], 0) + 1
    for k, v in sorted(winner_counts.items(), key=lambda t: -t[1]):
        print(f"  {k:>12}: {v:4d}  ({100 * v / len(rows):5.1f}%)")

    print("\nLength-bucket breakdown:")
    len_counts: dict[str, int] = {}
    for r in rows:
        len_counts[r["len_bucket"]] = len_counts.get(r["len_bucket"], 0) + 1
    for k in ["<2kb", "2-5kb", "5-10kb", "10-20kb", "20-50kb", ">=50kb"]:
        if k in len_counts:
            print(f"  {k:>8}: {len_counts[k]:4d}")

    print("\n20 closest-to-candidacy examples (smallest margin_vs_winner first):")
    for r in rows[:20]:
        print(
            f"  {r['contig_id']:<30} len={r['length']:>7}  "
            f"plas={r['plasmid_score']:.3f}  thresh={r['threshold']:.3f}  "
            f"winner={r['winner']:<10} margin={r['margin_vs_winner']:.3f}"
        )


if __name__ == "__main__":
    main()
