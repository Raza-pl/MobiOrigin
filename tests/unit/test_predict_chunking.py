"""Tests for bounded-memory MLP inference."""

from __future__ import annotations

import numpy as np
import plasflow2.classify.predict as predict_module
import torch
from plasflow2.classify.features import FEATURE_DIM


def test_mlp_scores_extracts_features_in_chunks(monkeypatch) -> None:
    chunk_sizes: list[int] = []

    def fake_extract(sequences, **kwargs):
        chunk_sizes.append(len(sequences))
        return np.zeros((len(sequences), FEATURE_DIM), dtype=np.float32)

    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros((len(x), 3), dtype=torch.float32)
            logits[:, 0] = 1.0
            return logits

    monkeypatch.setattr(predict_module, "extract_features", fake_extract)
    scores = predict_module._mlp_scores_chunked(
        ["A" * 1000] * 5,
        [f"seq_{i}" for i in range(5)],
        DummyModel(),
        torch.device("cpu"),
        batch_size=2,
    )

    assert chunk_sizes == [2, 2, 1]
    assert len(scores) == 5
    assert all(score["plasmid"] > score["chromosome"] for score in scores)
