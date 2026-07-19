#!/usr/bin/env python3
"""Score a plasflow2 all_predictions.tsv against a ground-truth TSV.

Lightweight, single-purpose scorer for validating changes against
data/benchmark/ground_truth.tsv (or any TSV with contig_id + true_label
columns) — a much smaller footprint than scripts/benchmark/evaluate.py,
which expects a multi-tool comparison directory layout.

Usage
-----
    python scripts/score_against_ground_truth.py \\
        --predictions results/all_predictions.tsv \\
        --ground-truth data/benchmark/ground_truth.tsv

    # Compare two runs side by side (e.g. default vs --lenient):
    python scripts/score_against_ground_truth.py \\
        --predictions /tmp/pf2_default/all_predictions.tsv \\
        --ground-truth data/benchmark/ground_truth.tsv \\
        --label "default"

    python scripts/score_against_ground_truth.py \\
        --predictions /tmp/pf2_lenient/all_predictions.tsv \\
        --ground-truth data/benchmark/ground_truth.tsv \\
        --label "lenient"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def score(predictions_path: Path, ground_truth_path: Path, target_class: str) -> dict[str, float]:
    gt: dict[str, str] = {}
    with open(ground_truth_path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            gt[row["contig_id"]] = row["true_label"]

    tp = fp = fn = tn = unclassified_total = unclassified_positive = 0
    n_matched = 0
    with open(predictions_path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            cid = row.get("contig_id") or row.get("sequence_id", "")
            true = gt.get(cid)
            if true is None:
                continue
            n_matched += 1
            pred = row.get("label", "unclassified")
            if pred == "unclassified":
                unclassified_total += 1
                if true == target_class:
                    unclassified_positive += 1
                    fn += 1
                continue
            if pred == target_class and true == target_class:
                tp += 1
            elif pred == target_class and true != target_class:
                fp += 1
            elif pred != target_class and true == target_class:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n_matched": n_matched,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unclassified_total": unclassified_total,
        "unclassified_positive": unclassified_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--target-class",
        default="plasmid",
        help="Class to compute precision/recall/F1 for (default: plasmid).",
    )
    parser.add_argument("--label", default=None, help="Optional label to print with the results.")
    args = parser.parse_args()

    result = score(args.predictions, args.ground_truth, args.target_class)

    header = f"=== {args.label} ===" if args.label else f"=== {args.predictions} ==="
    print(header)
    print(f"matched contigs:        {result['n_matched']}")
    print(
        f"TP={result['tp']}  FP={result['fp']}  FN={result['fn']}  TN={result['tn']}  "
        f"unclassified={result['unclassified_total']} "
        f"(of which true {args.target_class}: {result['unclassified_positive']})"
    )
    print(
        f"{args.target_class} precision={result['precision']:.4f}  "
        f"recall={result['recall']:.4f}  f1={result['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
