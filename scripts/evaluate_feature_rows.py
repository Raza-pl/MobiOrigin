#!/usr/bin/env python3
"""Evaluate a saved MLP on selected rows of a memory-mapped feature matrix."""

# Imports intentionally follow the thread-pool environment setup below.
# ruff: noqa: E402

from __future__ import annotations

import os as _os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_name, "1")

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from plasflow2.classify.model import load_model
from plasflow2.utils.device import IDX_TO_CLASS, get_device
from sklearn.metrics import (  # type: ignore[import]
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


def _expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        if in_bin.any():
            error += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidences[in_bin].mean())
            )
    return error


def _load_manifest_rows(
    manifest_path: Path,
    *,
    split: str | None = None,
    ids_path: Path | None = None,
) -> list[dict[str, str]]:
    with manifest_path.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]

    id_to_feature_row: dict[str, int] = {}
    if ids_path is not None:
        id_to_feature_row = {
            sequence_id.strip(): i
            for i, sequence_id in enumerate(ids_path.read_text().splitlines())
            if sequence_id.strip()
        }

    for row in rows:
        if id_to_feature_row:
            row["_feature_row_index"] = str(id_to_feature_row[row["sequence_id"]])
        elif row.get("feature_row_index", ""):
            row["_feature_row_index"] = row["feature_row_index"]
        else:
            row["_feature_row_index"] = row["row_index"]
    return rows


def evaluate_feature_rows(
    features_path: Path,
    model_path: Path,
    manifest_path: Path,
    out_dir: Path,
    *,
    split: str | None = None,
    ids_path: Path | None = None,
    batch_size: int = 512,
) -> dict[str, object]:
    """Score selected feature rows and write predictions plus calibration metrics."""

    rows = _load_manifest_rows(manifest_path, split=split, ids_path=ids_path)
    if not rows:
        raise ValueError("No manifest rows selected for evaluation")
    feature_rows = np.array([int(row["_feature_row_index"]) for row in rows], dtype=np.int64)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    features = np.load(features_path, mmap_mode="r")
    scaled_log_lengths = np.asarray(features[feature_rows, -1], dtype=np.float64)
    lengths = np.rint(10 ** (3.0 + 3.0 * scaled_log_lengths)).astype(np.int64)

    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(1)
    model = load_model(model_path, device=device)
    model.eval()

    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(feature_rows), batch_size):
            batch_rows = feature_rows[start : start + batch_size]
            read_order = np.argsort(batch_rows)
            inverse_order = np.argsort(read_order)
            batch = np.ascontiguousarray(features[batch_rows[read_order]], dtype=np.float32)
            probabilities = (
                torch.softmax(model(torch.from_numpy(batch).to(device)), dim=-1).cpu().numpy()
            )
            all_probabilities.append(probabilities[inverse_order])
    probability_matrix = np.concatenate(all_probabilities)
    predictions = probability_matrix.argmax(axis=1).astype(np.int64)
    class_labels = list(range(probability_matrix.shape[1]))
    class_names = [IDX_TO_CLASS.get(label, str(label)) for label in class_labels]
    observed_labels = sorted(int(label) for label in np.unique(labels))
    observed_class_recall = [
        float((predictions[labels == label] == label).mean()) for label in observed_labels
    ]

    metrics: dict[str, object] = {
        "n_rows": len(rows),
        "split": split,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy_observed_classes": float(np.mean(observed_class_recall)),
        "macro_f1_observed_classes": float(
            f1_score(labels, predictions, labels=observed_labels, average="macro")
        ),
        "log_loss": float(log_loss(labels, probability_matrix, labels=class_labels)),
        "multiclass_brier": float(
            np.mean(
                np.sum(
                    (probability_matrix - np.eye(len(class_labels), dtype=np.float32)[labels]) ** 2,
                    axis=1,
                )
            )
        ),
        "expected_calibration_error": _expected_calibration_error(probability_matrix, labels),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_labels).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=class_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        out_dir / "predictions.npz",
        feature_row_index=feature_rows,
        lengths=lengths,
        labels=labels,
        predictions=predictions,
        probabilities=probability_matrix,
    )
    with (out_dir / "predictions.tsv").open("w", newline="") as fh:
        fieldnames = [
            "feature_row_index",
            "sequence_id",
            "source_group",
            "length",
            "true_label",
            "predicted_label",
            *[f"{name}_score" for name in class_names],
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row, feature_row, length, label, prediction, probabilities in zip(
            rows,
            feature_rows,
            lengths,
            labels,
            predictions,
            probability_matrix,
        ):
            writer.writerow(
                {
                    "feature_row_index": int(feature_row),
                    "sequence_id": row["sequence_id"],
                    "source_group": row["source_group"],
                    "length": int(length),
                    "true_label": IDX_TO_CLASS.get(int(label), str(int(label))),
                    "predicted_label": IDX_TO_CLASS.get(int(prediction), str(int(prediction))),
                    **{
                        f"{name}_score": float(probabilities[i])
                        for i, name in enumerate(class_names)
                    },
                }
            )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument(
        "--ids",
        type=Path,
        help="Original sequence-ID file; use to recover feature rows from legacy manifests.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    metrics = evaluate_feature_rows(
        args.features,
        args.model,
        args.manifest,
        args.out,
        split=args.split,
        ids_path=args.ids,
        batch_size=args.batch_size,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
