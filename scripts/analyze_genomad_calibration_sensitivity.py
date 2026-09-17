#!/usr/bin/env python3
"""Compare calibrated and uncalibrated geNomad calls on the external cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

BIOLOGICAL_CLASSES = ("chromosome", "plasmid", "phage")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parent_id(identifier: str) -> str:
    return identifier.split("|provirus_", 1)[0]


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metrics(truth: list[str], prediction: list[str]) -> dict[str, object]:
    if len(truth) != len(prediction):
        raise ValueError("Truth and prediction lengths differ")
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for class_name in BIOLOGICAL_CLASSES:
        tp = sum(t == class_name and p == class_name for t, p in zip(truth, prediction))
        fp = sum(t != class_name and p == class_name for t, p in zip(truth, prediction))
        fn = sum(t == class_name and p != class_name for t, p in zip(truth, prediction))
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[class_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "sensitivity": recall,
            "f1": f1,
        }
        f1_values.append(f1)
        recalls.append(recall)

    plasmid = per_class["plasmid"]
    confusion = {
        true_class: {
            predicted_class: sum(
                t == true_class and p == predicted_class for t, p in zip(truth, prediction)
            )
            for predicted_class in (*BIOLOGICAL_CLASSES, "unclassified")
        }
        for true_class in BIOLOGICAL_CLASSES
    }
    return {
        "records": len(truth),
        "prediction_coverage": safe_divide(
            sum(p in BIOLOGICAL_CLASSES for p in prediction), len(prediction)
        ),
        "macro_f1": sum(f1_values) / len(f1_values),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "plasmid_precision": plasmid["precision"],
        "plasmid_sensitivity": plasmid["sensitivity"],
        "plasmid_f1": plasmid["f1"],
        "per_class": per_class,
        "confusion_truth_by_prediction": confusion,
        "label_counts": dict(sorted(Counter(prediction).items())),
    }


def standardize_uncalibrated(
    truth_rows: list[dict[str, str]],
    score_rows: list[dict[str, str]],
    plasmid_rows: list[dict[str, str]],
    virus_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    ordered_ids = [row["opaque_contig_id"] for row in truth_rows]
    score_by_id = {row["seq_name"]: row for row in score_rows}
    plasmid_ids = {parent_id(row["seq_name"]) for row in plasmid_rows}
    virus_ids = {parent_id(row["seq_name"]) for row in virus_rows}

    if len(score_by_id) != len(ordered_ids) or set(score_by_id) != set(ordered_ids):
        raise RuntimeError("Uncalibrated score identifiers do not match the external cohort")

    standardized: list[dict[str, object]] = []
    for identifier in ordered_ids:
        is_plasmid = identifier in plasmid_ids
        is_virus = identifier in virus_ids
        if is_plasmid and is_virus:
            label = "unclassified"
            status = "ambiguous_dual_call"
        elif is_plasmid:
            label = "plasmid"
            status = "called_plasmid"
        elif is_virus:
            label = "phage"
            status = "called_phage"
        else:
            label = "chromosome"
            status = "not_detected"
        score = score_by_id[identifier]
        standardized.append(
            {
                "sequence_id": identifier,
                "predicted_label": label,
                "prediction_status": status,
                "chromosome_score": score["chromosome_score"],
                "plasmid_score": score["plasmid_score"],
                "virus_score": score["virus_score"],
            }
        )
    return standardized


def ordered_predictions(
    truth_rows: list[dict[str, str]],
    rows: list[dict[str, str]],
    identifier_field: str,
    label_field: str,
) -> list[str]:
    prediction_by_id = {row[identifier_field]: row[label_field] for row in rows}
    ordered_ids = [row["opaque_contig_id"] for row in truth_rows]
    if len(prediction_by_id) != len(ordered_ids) or set(prediction_by_id) != set(ordered_ids):
        raise RuntimeError(f"Prediction identifiers do not match: {identifier_field}")
    return [prediction_by_id[identifier] for identifier in ordered_ids]


def flatten_metrics(system: str, values: dict[str, object]) -> dict[str, object]:
    return {
        "system": system,
        "records": values["records"],
        "prediction_coverage": values["prediction_coverage"],
        "macro_f1": values["macro_f1"],
        "balanced_accuracy": values["balanced_accuracy"],
        "plasmid_precision": values["plasmid_precision"],
        "plasmid_sensitivity": values["plasmid_sensitivity"],
        "plasmid_f1": values["plasmid_f1"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    run_dir = args.run_dir.resolve()
    ext = root / "evaluation/release_audit/mobiorigin_external_validation_20260817"
    truth_path = (
        ext
        / "024_corrected_final_external_cohort_freeze"
        / "mobiorigin_external_validation_label_map.sealed.tsv"
    )
    calibrated_path = (
        ext
        / "029_dual_external_prediction_execution/genomad_prediction_output"
        / "standardized_predictions.tsv"
    )
    mobiorigin_path = (
        ext
        / "029_dual_external_prediction_execution/mobiorigin_prediction_output"
        / "predictions.tsv"
    )
    stem = "mobiorigin_external_validation_cohort_3000"
    uncalibrated_root = run_dir / "raw_uncalibrated"
    score_path = (
        uncalibrated_root
        / f"{stem}_aggregated_classification"
        / f"{stem}_aggregated_classification.tsv"
    )
    summary_root = uncalibrated_root / f"{stem}_summary"
    plasmid_path = summary_root / f"{stem}_plasmid_summary.tsv"
    virus_path = summary_root / f"{stem}_virus_summary.tsv"

    inputs = [
        truth_path,
        calibrated_path,
        mobiorigin_path,
        score_path,
        plasmid_path,
        virus_path,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise RuntimeError("Missing required input(s):\n" + "\n".join(missing))

    truth_rows = read_tsv(truth_path)
    calibrated_rows = read_tsv(calibrated_path)
    mobiorigin_rows = read_tsv(mobiorigin_path)
    uncalibrated = standardize_uncalibrated(
        truth_rows,
        read_tsv(score_path),
        read_tsv(plasmid_path),
        read_tsv(virus_path),
    )
    if len(truth_rows) != 3000:
        raise RuntimeError(f"Expected 3,000 external records, observed {len(truth_rows)}")

    truth = [row["class"] for row in truth_rows]
    calibrated = ordered_predictions(truth_rows, calibrated_rows, "contig_id", "predicted_label")
    mobiorigin = ordered_predictions(truth_rows, mobiorigin_rows, "sequence_id", "prediction")
    uncalibrated_labels = [str(row["predicted_label"]) for row in uncalibrated]

    all_metrics = {
        "mobiorigin_v0.1.6": metrics(truth, mobiorigin),
        "genomad_calibrated": metrics(truth, calibrated),
        "genomad_uncalibrated": metrics(truth, uncalibrated_labels),
    }

    comparison_rows = [flatten_metrics(name, value) for name, value in all_metrics.items()]
    metric_fields = [
        "system",
        "records",
        "prediction_coverage",
        "macro_f1",
        "balanced_accuracy",
        "plasmid_precision",
        "plasmid_sensitivity",
        "plasmid_f1",
    ]
    write_tsv(run_dir / "system_metrics.tsv", comparison_rows, metric_fields)
    write_tsv(
        run_dir / "genomad_uncalibrated_standardized.tsv",
        uncalibrated,
        [
            "sequence_id",
            "predicted_label",
            "prediction_status",
            "chromosome_score",
            "plasmid_score",
            "virus_score",
        ],
    )

    changes: list[dict[str, object]] = []
    for truth_row, calibrated_label, uncalibrated_label in zip(
        truth_rows, calibrated, uncalibrated_labels
    ):
        if calibrated_label != uncalibrated_label:
            changes.append(
                {
                    "sequence_id": truth_row["opaque_contig_id"],
                    "true_class": truth_row["class"],
                    "length_bin": truth_row["length_bin"],
                    "source_accession": truth_row["source_accession"],
                    "calibrated_label": calibrated_label,
                    "uncalibrated_label": uncalibrated_label,
                }
            )
    write_tsv(
        run_dir / "genomad_calibration_label_changes.tsv",
        changes,
        [
            "sequence_id",
            "true_class",
            "length_bin",
            "source_accession",
            "calibrated_label",
            "uncalibrated_label",
        ],
    )

    length_rows: list[dict[str, object]] = []
    for length_bin in sorted({row["length_bin"] for row in truth_rows}):
        indices = [i for i, row in enumerate(truth_rows) if row["length_bin"] == length_bin]
        bin_truth = [truth[i] for i in indices]
        for system, predictions in (
            ("mobiorigin_v0.1.6", mobiorigin),
            ("genomad_calibrated", calibrated),
            ("genomad_uncalibrated", uncalibrated_labels),
        ):
            result = metrics(bin_truth, [predictions[i] for i in indices])
            row = flatten_metrics(system, result)
            row["length_bin"] = length_bin
            length_rows.append(row)
    write_tsv(
        run_dir / "system_metrics_by_length.tsv",
        length_rows,
        ["length_bin", *metric_fields],
    )

    report = {
        "status": "PASS",
        "external_records": len(truth_rows),
        "systems": all_metrics,
        "genomad_labels_changed_by_calibration": len(changes),
        "genomad_change_rate": len(changes) / len(truth_rows),
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
    }
    report_path = run_dir / "genomad_calibration_sensitivity.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("===== geNomad calibration sensitivity =====")
    for system in ("mobiorigin_v0.1.6", "genomad_calibrated", "genomad_uncalibrated"):
        result = all_metrics[system]
        print(f"{system}:")
        print(f"  macro-F1: {result['macro_f1']:.4f}")
        print(f"  plasmid F1: {result['plasmid_f1']:.4f}")
        print(f"  plasmid precision: {result['plasmid_precision']:.4f}")
        print(f"  plasmid sensitivity: {result['plasmid_sensitivity']:.4f}")
        print(f"  prediction coverage: {result['prediction_coverage']:.4f}")
        print(f"  label counts: {result['label_counts']}")
    print(f"geNomad labels changed by calibration: {len(changes)} / {len(truth_rows)}")
    print(f"Results: {run_dir}")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
