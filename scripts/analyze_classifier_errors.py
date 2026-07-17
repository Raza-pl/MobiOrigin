#!/usr/bin/env python3
"""Summarize classifier errors by confusion, length tier, and taxon."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def analyze_errors(
    predictions_path: Path,
    out_dir: Path,
    *,
    prediction_column: str = "argmax_prediction",
) -> dict[str, object]:
    """Write grouped error counts and the most confident incorrect predictions."""

    with predictions_path.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise ValueError("Prediction file is empty")
    if prediction_column not in rows[0]:
        raise ValueError(f"Missing prediction column: {prediction_column}")

    errors = [row for row in rows if row["true_label"] != row[prediction_column]]
    grouped = Counter(
        (
            row["true_label"],
            row[prediction_column],
            row.get("length_tier", "unknown") or "unknown",
            row.get("taxon", "unknown") or "unknown",
        )
        for row in errors
    )

    def predicted_confidence(row: dict[str, str]) -> float:
        score_name = f"{row[prediction_column]}_score"
        try:
            return float(row.get(score_name, 0.0))
        except ValueError:
            return 0.0

    high_confidence_errors = sorted(
        errors,
        key=predicted_confidence,
        reverse=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "error_summary.tsv").open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["true_label", "predicted_label", "length_tier", "taxon", "count"])
        for key, count in sorted(grouped.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([*key, count])

    output_fields = list(rows[0]) + ["predicted_confidence"]
    with (out_dir / "high_confidence_errors.tsv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_fields, delimiter="\t")
        writer.writeheader()
        for row in high_confidence_errors[:1000]:
            writer.writerow(
                {
                    **row,
                    "predicted_confidence": predicted_confidence(row),
                }
            )

    summary: dict[str, object] = {
        "n_rows": len(rows),
        "n_errors": len(errors),
        "error_rate": len(errors) / len(rows),
        "prediction_column": prediction_column,
        "top_confusions": [
            {
                "true_label": key[0],
                "predicted_label": key[1],
                "length_tier": key[2],
                "taxon": key[3],
                "count": count,
            }
            for key, count in grouped.most_common(20)
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prediction-column", default="argmax_prediction")
    args = parser.parse_args()

    summary = analyze_errors(
        args.predictions,
        args.out,
        prediction_column=args.prediction_column,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
