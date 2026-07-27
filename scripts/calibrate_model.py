#!/usr/bin/env python3
"""Fit validation-only temperature scaling and create a calibrated checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from plasflow2.utils.device import IDX_TO_CLASS
from scipy.optimize import minimize_scalar  # type: ignore[import]
from sklearn.metrics import log_loss, precision_recall_curve  # type: ignore[import]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    log_probabilities = np.log(np.clip(probabilities, 1e-8, 1.0))
    return _softmax(log_probabilities / temperature)


def _expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return error


def _threshold_recommendations(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    recommendations: dict[str, dict[str, float | int | None]] = {}
    for class_index in range(probabilities.shape[1]):
        class_name = IDX_TO_CLASS.get(class_index, str(class_index))
        truth = labels == class_index
        n_positive = int(truth.sum())
        n_negative = int((~truth).sum())
        if n_positive == 0:
            recommendations[class_name] = {
                "n_positive": 0,
                "n_negative": n_negative,
                "max_f1_threshold": None,
            }
            continue
        precision, recall, thresholds = precision_recall_curve(truth, probabilities[:, class_index])
        if len(thresholds) == 0:
            continue
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best_index = int(np.argmax(f1))
        class_result: dict[str, float | int | None] = {
            "n_positive": n_positive,
            "n_negative": n_negative,
            "max_f1_threshold": float(thresholds[best_index]),
            "max_f1": float(f1[best_index]),
            "precision_at_max_f1": float(precision[best_index]),
            "recall_at_max_f1": float(recall[best_index]),
        }
        for target in (0.95, 0.98):
            candidates = np.flatnonzero(precision[:-1] >= target)
            key = f"threshold_for_{int(target * 100)}pct_precision"
            if len(candidates):
                chosen = int(candidates[np.argmax(recall[candidates])])
                class_result[key] = float(thresholds[chosen])
            else:
                class_result[key] = None
        recommendations[class_name] = class_result
    return recommendations


def calibrate_model(
    validation_scores_path: Path,
    model_path: Path,
    out_model_path: Path,
    report_path: Path,
) -> dict[str, object]:
    """Fit temperature on validation predictions and scale the final layer."""

    validation = np.load(validation_scores_path)
    probabilities = np.asarray(validation["probabilities"], dtype=np.float64)
    labels = np.asarray(validation["labels"], dtype=np.int64)

    if probabilities.ndim != 2:
        raise ValueError("Calibration probabilities must be a two-dimensional matrix.")
    if len(probabilities) != len(labels):
        raise ValueError("Calibration probabilities and labels have different row counts.")
    if probabilities.shape[1] != len(IDX_TO_CLASS):
        raise ValueError(
            "Primary-model calibration requires exactly three probability "
            f"columns; observed {probabilities.shape[1]}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Calibration probabilities contain non-finite values.")

    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Calibration probability rows must have a positive sum.")
    probabilities = probabilities / row_sums

    class_labels = list(range(probabilities.shape[1]))
    observed_classes = {int(value) for value in np.unique(labels)}
    expected_classes = set(class_labels)
    if observed_classes != expected_classes:
        raise ValueError(
            "Calibration requires positive validation rows for every class; "
            f"expected {sorted(expected_classes)}, "
            f"observed {sorted(observed_classes)}."
        )

    def objective(temperature: float) -> float:
        return float(
            log_loss(
                labels,
                _temperature_scale(probabilities, temperature),
                labels=class_labels,
            )
        )

    optimization = minimize_scalar(
        objective,
        bounds=(0.05, 10.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    temperature = float(optimization.x)
    calibrated = _temperature_scale(probabilities, temperature)
    length_thresholds: dict[str, object] = {}
    if "lengths" in validation:
        lengths = np.asarray(validation["lengths"], dtype=np.int64)
        length_bins = {
            "<=2 kb": (0, 2_001),
            "2-5 kb": (2_001, 5_000),
            "5-10 kb": (5_000, 10_000),
            "10-20 kb": (10_000, 20_000),
            ">20 kb": (20_000, np.iinfo(np.int64).max),
        }
        for name, (lower, upper) in length_bins.items():
            mask = (lengths >= lower) & (lengths < upper)
            if mask.any():
                length_thresholds[name] = {
                    "n_rows": int(mask.sum()),
                    "threshold_recommendations": _threshold_recommendations(
                        calibrated[mask], labels[mask]
                    ),
                }

    source_model_sha256 = _sha256(model_path)
    validation_scores_sha256 = _sha256(validation_scores_path)

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    for required_key in ("net.11.weight", "net.11.bias"):
        if required_key not in state:
            raise ValueError(f"Model checkpoint is missing required tensor {required_key!r}.")

    state["net.11.weight"] = state["net.11.weight"] / temperature
    state["net.11.bias"] = state["net.11.bias"] / temperature
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_model_path)
    calibrated_model_sha256 = _sha256(out_model_path)

    report: dict[str, object] = {
        "calibration_method": "scalar_temperature_final_logits",
        "temperature": temperature,
        "class_names": [IDX_TO_CLASS[index] for index in class_labels],
        "class_counts": {
            IDX_TO_CLASS[index]: int((labels == index).sum()) for index in class_labels
        },
        "source_model_sha256": source_model_sha256,
        "validation_scores_sha256": validation_scores_sha256,
        "calibrated_model_sha256": calibrated_model_sha256,
        "validation_rows": int(len(labels)),
        "nll_before": float(log_loss(labels, probabilities, labels=class_labels)),
        "nll_after": float(log_loss(labels, calibrated, labels=class_labels)),
        "ece_before": _expected_calibration_error(probabilities, labels),
        "ece_after": _expected_calibration_error(calibrated, labels),
        "threshold_recommendations": _threshold_recommendations(calibrated, labels),
        "length_threshold_recommendations": length_thresholds,
        "source_model": str(model_path),
        "calibrated_model": str(out_model_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-scores", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = calibrate_model(
        args.validation_scores,
        args.model,
        args.out_model,
        args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
