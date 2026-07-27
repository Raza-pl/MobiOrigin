"""Tests for validation-only temperature scaling."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from plasflow2.classify.model import PlasFlowMLP, save_model

from scripts.calibrate_model import calibrate_model


def test_calibrate_model_writes_scaled_checkpoint_and_report(tmp_path) -> None:
    scores = tmp_path / "validation.npz"
    model_path = tmp_path / "model.pt"
    calibrated_path = tmp_path / "model_calibrated.pt"
    report_path = tmp_path / "calibration.json"
    np.savez_compressed(
        scores,
        labels=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        lengths=np.array([1000, 1500, 3000, 4000, 6000, 12000], dtype=np.int64),
        probabilities=np.array(
            [
                [0.99, 0.005, 0.005],
                [0.80, 0.10, 0.10],
                [0.10, 0.80, 0.10],
                [0.10, 0.85, 0.05],
                [0.10, 0.10, 0.80],
                [0.01, 0.01, 0.98],
            ],
            dtype=np.float32,
        ),
    )
    model = PlasFlowMLP(input_dim=4, num_classes=3, hidden_dims=(8, 6, 4))
    save_model(model, model_path)
    original = torch.load(model_path, map_location="cpu", weights_only=True)

    report = calibrate_model(scores, model_path, calibrated_path, report_path)

    calibrated = torch.load(calibrated_path, map_location="cpu", weights_only=True)
    assert report["temperature"] > 0
    assert report["nll_after"] <= report["nll_before"]
    assert not torch.equal(original["net.11.weight"], calibrated["net.11.weight"])
    saved_report = json.loads(report_path.read_text())
    assert saved_report["validation_rows"] == 6
    assert saved_report["class_counts"] == {
        "plasmid": 2,
        "chromosome": 2,
        "phage": 2,
    }
    assert len(saved_report["source_model_sha256"]) == 64
    assert len(saved_report["validation_scores_sha256"]) == 64
    assert len(saved_report["calibrated_model_sha256"]) == 64
    assert "<=2 kb" in report["length_threshold_recommendations"]
    assert "2-5 kb" in report["length_threshold_recommendations"]
    assert "5-10 kb" in report["length_threshold_recommendations"]
    assert "10-20 kb" in report["length_threshold_recommendations"]


def test_calibrate_model_rejects_missing_validation_class(tmp_path) -> None:
    scores = tmp_path / "missing_class_validation.npz"
    model_path = tmp_path / "model.pt"

    np.savez_compressed(
        scores,
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        probabilities=np.array(
            [
                [0.90, 0.08, 0.02],
                [0.80, 0.15, 0.05],
                [0.10, 0.85, 0.05],
                [0.15, 0.80, 0.05],
            ],
            dtype=np.float32,
        ),
    )

    save_model(
        PlasFlowMLP(
            input_dim=4,
            num_classes=3,
            hidden_dims=(8, 6, 4),
        ),
        model_path,
    )

    with pytest.raises(
        ValueError,
        match="requires positive validation rows for every class",
    ):
        calibrate_model(
            scores,
            model_path,
            tmp_path / "calibrated.pt",
            tmp_path / "calibration.json",
        )
