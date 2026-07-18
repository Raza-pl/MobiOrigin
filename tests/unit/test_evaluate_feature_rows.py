"""Tests for evaluation on selected memory-mapped feature rows."""

from __future__ import annotations

import json

import numpy as np
from plasflow2.classify.model import PlasFlowMLP, save_model

from scripts.evaluate_feature_rows import evaluate_feature_rows


def test_evaluate_feature_rows_writes_metrics_and_predictions(tmp_path) -> None:
    features = tmp_path / "features.npy"
    manifest = tmp_path / "manifest.tsv"
    model_path = tmp_path / "model.pt"
    out = tmp_path / "evaluation"
    matrix = np.zeros((4, 4), dtype=np.float32)
    np.save(features, matrix)
    manifest.write_text(
        "row_index\tsequence_id\tsource_group\tlabel\n"
        "0\tseq0\tgroup0\t0\n"
        "1\tseq1\tgroup1\t1\n"
        "2\tseq2\tgroup2\t2\n"
        "3\tseq3\tgroup3\t0\n"
    )
    model = PlasFlowMLP(input_dim=4, num_classes=3, hidden_dims=(8, 6, 4))
    save_model(model, model_path)

    metrics = evaluate_feature_rows(features, model_path, manifest, out, batch_size=2)

    assert metrics["n_rows"] == 4
    assert (out / "metrics.json").exists()
    assert (out / "predictions.npz").exists()
    assert len((out / "predictions.tsv").read_text().splitlines()) == 5
    assert json.loads((out / "metrics.json").read_text())["n_rows"] == 4
    assert np.load(out / "predictions.npz")["lengths"].tolist() == [1000] * 4
