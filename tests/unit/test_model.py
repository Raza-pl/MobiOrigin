"""Tests for portable PlasFlow MLP checkpoints."""

from __future__ import annotations

import pytest
import torch
from plasflow2.classify.model import (
    ModelLoadError,
    PlasFlowMLP,
    load_model,
    save_model,
)


def test_load_model_infers_compact_hidden_dimensions(tmp_path) -> None:
    path = tmp_path / "compact.pt"
    model = PlasFlowMLP(input_dim=12, num_classes=3, hidden_dims=(16, 8, 4))
    save_model(model, path)

    loaded = load_model(path)

    assert loaded.net[0].weight.shape == torch.Size([16, 12])
    assert loaded.net[4].weight.shape == torch.Size([8, 16])
    assert loaded.net[8].weight.shape == torch.Size([4, 8])
    assert loaded.net[11].weight.shape == torch.Size([3, 4])


def test_load_model_uses_restricted_deserialization(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "safe.pt"
    save_model(
        PlasFlowMLP(input_dim=12, hidden_dims=(16, 8, 4)),
        path,
    )

    real_load = torch.load
    observed = {}

    def recording_load(*args, **kwargs):
        observed.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    load_model(path)

    assert observed["weights_only"] is True


def test_load_model_rejects_non_mapping_checkpoint(tmp_path) -> None:
    path = tmp_path / "not_a_state_dict.pt"
    torch.save(["unexpected", "objects"], path)

    with pytest.raises(ModelLoadError, match="state-dictionary mapping"):
        load_model(path)


def test_load_model_rejects_incomplete_state_dict(tmp_path) -> None:
    path = tmp_path / "incomplete.pt"
    torch.save({"net.0.weight": torch.zeros((4, 3))}, path)

    with pytest.raises(ModelLoadError, match="missing required weights"):
        load_model(path)


def test_load_model_rejects_wrong_class_count(tmp_path) -> None:
    path = tmp_path / "two_class.pt"
    save_model(
        PlasFlowMLP(
            input_dim=12,
            num_classes=2,
            hidden_dims=(16, 8, 4),
        ),
        path,
    )

    with pytest.raises(
        ModelLoadError,
        match="declared contract value 3",
    ):
        load_model(path, expected_num_classes=3)


def test_load_model_enforces_declared_input_dimension(tmp_path) -> None:
    path = tmp_path / "dimension_mismatch.pt"
    save_model(
        PlasFlowMLP(input_dim=12, hidden_dims=(16, 8, 4)),
        path,
    )

    with pytest.raises(ModelLoadError, match="declared contract value 13"):
        load_model(path, expected_input_dim=13)
