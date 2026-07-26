"""Tests for leakage-resistant model training helpers."""

from __future__ import annotations

import json

import numpy as np
import torch

from scripts.train_model import (
    _evaluate_mlp_mmap,
    _exclude_source_groups,
    _train_mlp_mmap,
)


def test_evaluate_mlp_mmap_writes_heldout_metrics(tmp_path, monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            score = x[:, 0]
            return torch.stack((1.0 - score, score), dim=1)

    monkeypatch.setattr(
        "plasflow2.classify.model.load_model",
        lambda model_path, device=None: DummyModel().to(device),
    )
    monkeypatch.setattr("plasflow2.utils.device.get_device", lambda: torch.device("cpu"))

    features = tmp_path / "features.npy"
    metrics_path = tmp_path / "heldout_test_metrics.json"
    np.save(
        features,
        np.array([[0.0, 2.0], [1.0, 2.0], [1.0, 3.0], [0.0, 3.0]], dtype=np.float32),
    )
    labels = np.array([0, 1, 1, 0], dtype=np.int64)

    metrics = _evaluate_mlp_mmap(
        model_path=tmp_path / "model.pt",
        data_path=str(features),
        idx_te=np.arange(4, dtype=np.int64),
        y_te=labels,
        class_names=["zero", "one"],
        out_path=metrics_path,
        batch_size=2,
    )

    saved = json.loads(metrics_path.read_text())
    assert metrics["macro_f1"] == 1.0
    assert saved["accuracy"] == 1.0
    assert saved["confusion_matrix"] == [[2, 0], [0, 2]]


def test_training_writes_and_resumes_recovery_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("plasflow2.utils.device.get_device", lambda: torch.device("cpu"))
    features = tmp_path / "features.npy"
    model_path = tmp_path / "mlp_v2.pt"
    matrix = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.9, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.9, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    np.save(features, matrix)

    kwargs = {
        "data_path": str(features),
        "idx_tr": np.arange(6, dtype=np.int64),
        "y_tr": labels,
        "X_va": matrix,
        "y_va": labels,
        "epochs": 1,
        "batch_size": 3,
        "out_path": model_path,
        "num_classes": 3,
        "torch_threads": 1,
        "hidden_dims": (8, 6, 4),
    }
    _train_mlp_mmap(**kwargs)

    checkpoint_path = tmp_path / "training_checkpoint.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert "best_val_macro_f1" in checkpoint
    model_path.unlink()

    _train_mlp_mmap(**kwargs)

    assert model_path.exists()


def test_exclude_source_groups_removes_benchmark_accessions() -> None:
    indices = np.arange(4, dtype=np.int64)
    labels = np.array([0, 0, 1, 2], dtype=np.int64)
    sequence_ids = [
        "KEEP_w1000_s0",
        "LOCKED_w1000_s0",
        "KEEP2_w1000_s0",
        "LOCKED_w2000_s1000",
    ]

    kept_indices, kept_labels, kept_ids, kept_groups = _exclude_source_groups(
        indices,
        labels,
        sequence_ids,
        {"LOCKED"},
    )

    assert kept_indices.tolist() == [0, 2]
    assert kept_labels.tolist() == [0, 1]
    assert kept_ids == ["KEEP_w1000_s0", "KEEP2_w1000_s0"]
    assert kept_groups == ["KEEP", "KEEP2"]
