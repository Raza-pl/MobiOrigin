"""PyTorch MLP classifier for 3-class sequence classification.

Rev 4 — k=7 canonical architecture.
Input: 9557 dims (k=1–5 + k=7 canonical + length)
Architecture: MLP (9557→2048→512→128→3) with BatchNorm, GELU, Dropout.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from plasflow2.utils.device import NUM_CLASSES

logger = logging.getLogger(__name__)

INPUT_DIM = 9557  # 1364 (k=1–5) + 8192 (k=7 canonical) + 1 (length)
DEFAULT_HIDDEN_DIMS = (2048, 512, 128)


class PlasFlowMLP(nn.Module):
    """Three-hidden-layer MLP for plasmid/chromosome/phage classification.

    Wider first layer (2048) to handle the expanded k=7 canonical input.

    Note:
        All inputs must be float32 (MPS does not support float64).
        Call model.forward(x.float()) or ensure tensors are already float32.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        num_classes: int = NUM_CLASSES,
        hidden_dims: tuple[int, int, int] = DEFAULT_HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        hidden_1, hidden_2, hidden_3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_2, hidden_3),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())  # float32 required for MPS


class ModelLoadError(ValueError):
    """Raised when an MLP checkpoint cannot be loaded safely."""


def save_model(model: PlasFlowMLP, path: Path | str) -> None:
    """Save model weights to CPU (safe across platforms).

    Args:
        model: Trained PlasFlowMLP.
        path: Destination .pt file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), str(path))
    logger.info("Saved MLP weights to %s", path)


def load_model(
    path: Path | str,
    device: torch.device | None = None,
    *,
    expected_input_dim: int | None = None,
    expected_num_classes: int | None = None,
) -> PlasFlowMLP:
    """Safely load a tensor-only MLP state dictionary.

    PyTorch pickle object reconstruction is disabled. The checkpoint must be
    a plain mapping of string parameter names to tensors and must describe the
    expected PlasFlow MLP architecture.
    """
    checkpoint = Path(path)

    try:
        raw_state = torch.load(
            str(checkpoint),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ModelLoadError(
            f"Unable to safely load tensor-only MLP checkpoint {checkpoint}: {error}"
        ) from error

    if not isinstance(raw_state, Mapping):
        raise ModelLoadError(
            f"MLP checkpoint {checkpoint} must contain a state-dictionary mapping."
        )

    state = dict(raw_state)
    if not state:
        raise ModelLoadError(f"MLP checkpoint {checkpoint} contains no parameters.")

    if any(not isinstance(key, str) for key in state):
        raise ModelLoadError(f"MLP checkpoint {checkpoint} contains a non-string parameter name.")

    non_tensor_keys = [key for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensor_keys:
        raise ModelLoadError(
            f"MLP checkpoint {checkpoint} contains non-tensor values: "
            + ", ".join(sorted(non_tensor_keys)[:10])
        )

    required_weights = (
        "net.0.weight",
        "net.4.weight",
        "net.8.weight",
        "net.11.weight",
    )
    missing = [key for key in required_weights if key not in state]
    if missing:
        raise ModelLoadError(
            f"MLP checkpoint {checkpoint} is missing required weights: " + ", ".join(missing)
        )

    weights = [state[key] for key in required_weights]
    if any(weight.ndim != 2 for weight in weights):
        raise ModelLoadError(
            f"MLP checkpoint {checkpoint} contains invalid linear-layer dimensions."
        )

    first, second, third, output = weights
    hidden_1, input_dim = first.shape
    hidden_2, second_input = second.shape
    hidden_3, third_input = third.shape
    num_classes, output_input = output.shape

    dimensions = (
        input_dim,
        hidden_1,
        hidden_2,
        hidden_3,
        num_classes,
    )
    if any(int(value) <= 0 for value in dimensions):
        raise ModelLoadError(f"MLP checkpoint {checkpoint} contains a zero-sized architecture.")

    if second_input != hidden_1 or third_input != hidden_2 or output_input != hidden_3:
        raise ModelLoadError(
            f"MLP checkpoint {checkpoint} has incompatible adjacent layer dimensions."
        )

    if expected_input_dim is not None and input_dim != expected_input_dim:
        raise ModelLoadError(
            f"MLP checkpoint {checkpoint} input dimension {input_dim} does not "
            f"match its declared contract value {expected_input_dim}."
        )

    if expected_num_classes is not None and num_classes != expected_num_classes:
        raise ModelLoadError(
            f"MLP checkpoint {checkpoint} output dimension {num_classes} does not "
            f"match its declared contract value {expected_num_classes}."
        )

    model = PlasFlowMLP(
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        hidden_dims=(
            int(hidden_1),
            int(hidden_2),
            int(hidden_3),
        ),
    )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ModelLoadError(
            f"MLP checkpoint {checkpoint} does not match the PlasFlow architecture: {error}"
        ) from error

    model.eval()
    if device is not None:
        model = model.to(device)

    logger.info(
        "Safely loaded MLP from %s " "(input_dim=%d, hidden_dims=%s, num_classes=%d)",
        checkpoint,
        input_dim,
        (hidden_1, hidden_2, hidden_3),
        num_classes,
    )
    return model
