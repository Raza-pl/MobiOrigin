"""Standalone MobiOrigin MLP architecture and safe checkpoint loader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

INPUT_DIM = 9_574
NUM_CLASSES = 3
HIDDEN_DIMS = (2_048, 512, 128)


class MobiOriginMLP(nn.Module):
    """Frozen MobiOrigin marker-fusion network."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        num_classes: int = NUM_CLASSES,
        hidden_dims: tuple[int, int, int] = HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        first, second, third = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.BatchNorm1d(first),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(first, second),
            nn.BatchNorm1d(second),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(second, third),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(third, num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values.float())


class ModelLoadError(ValueError):
    """Raised when a checkpoint is not a safe compatible state dictionary."""


def load_model(path: Path, *, input_dim: int = INPUT_DIM) -> MobiOriginMLP:
    """Load tensor weights only and reconstruct the declared architecture."""
    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ModelLoadError(f"Unable to safely load checkpoint {path}: {error}") from error
    if not isinstance(raw, Mapping) or not raw:
        raise ModelLoadError("Checkpoint must be a non-empty state-dictionary mapping")
    state = dict(raw)
    if any(not isinstance(key, str) for key in state):
        raise ModelLoadError("Checkpoint contains a non-string parameter name")
    if any(not torch.is_tensor(value) for value in state.values()):
        raise ModelLoadError("Checkpoint contains a non-tensor value")
    required = ("net.0.weight", "net.4.weight", "net.8.weight", "net.11.weight")
    if any(key not in state for key in required):
        raise ModelLoadError("Checkpoint is missing required linear-layer weights")
    weights = [state[key] for key in required]
    if any(value.ndim != 2 for value in weights):
        raise ModelLoadError("Checkpoint contains invalid linear-layer dimensions")
    first, second, third, output = weights
    hidden_1, observed_input = first.shape
    hidden_2, second_input = second.shape
    hidden_3, third_input = third.shape
    classes, output_input = output.shape
    if (
        int(observed_input) != input_dim
        or int(classes) != NUM_CLASSES
        or second_input != hidden_1
        or third_input != hidden_2
        or output_input != hidden_3
    ):
        raise ModelLoadError("Checkpoint architecture differs from the frozen contract")
    model = MobiOriginMLP(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        hidden_dims=(int(hidden_1), int(hidden_2), int(hidden_3)),
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ModelLoadError(f"Checkpoint state dictionary is incompatible: {error}") from error
    model.eval()
    return model
