"""Tests for validation-only temperature scaling."""

from __future__ import annotations

import json

import numpy as np
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
    original = torch.load(model_path, map_location="cpu", weights_only=False)

    report = calibrate_model(scores, model_path, calibrated_path, report_path)

    calibrated = torch.load(calibrated_path, map_location="cpu", weights_only=False)
    assert report["temperature"] > 0
    assert report["nll_after"] <= report["nll_before"]
    assert not torch.equal(original["net.11.weight"], calibrated["net.11.weight"])
    assert json.loads(report_path.read_text())["validation_rows"] == 6
    assert "<2 kb" in report["length_threshold_recommendations"]
