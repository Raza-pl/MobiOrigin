"""Tests for portable PlasFlow MLP checkpoints."""

from __future__ import annotations

import torch
from plasflow2.classify.model import PlasFlowMLP, load_model, save_model


def test_load_model_infers_compact_hidden_dimensions(tmp_path) -> None:
    path = tmp_path / "compact.pt"
    model = PlasFlowMLP(input_dim=12, num_classes=3, hidden_dims=(16, 8, 4))
    save_model(model, path)

    loaded = load_model(path)

    assert loaded.net[0].weight.shape == torch.Size([16, 12])
    assert loaded.net[4].weight.shape == torch.Size([8, 16])
    assert loaded.net[8].weight.shape == torch.Size([4, 8])
    assert loaded.net[11].weight.shape == torch.Size([3, 4])
