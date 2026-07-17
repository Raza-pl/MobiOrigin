"""PyTorch MLP classifier for 3-class sequence classification.

Rev 4 — k=7 canonical architecture.
Input: 9557 dims (k=1–5 + k=7 canonical + length)
Architecture: MLP (9557→2048→512→128→3) with BatchNorm, GELU, Dropout.
"""

from __future__ import annotations

import logging
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


def load_model(path: Path | str, device: torch.device | None = None) -> PlasFlowMLP:
    """Load MLP weights from a .pt file.

    The input dimension is inferred from the checkpoint's first-layer weight
    shape, so the function remains correct even if FEATURE_DIM changes between
    releases.

    Args:
        path: Path to .pt file.
        device: Target device (defaults to CPU if not specified).

    Returns:
        PlasFlowMLP in eval mode.
    """
    # weights_only=False: model .pt files are our own trusted weights (not
    # user-supplied), so pickle-based loading is safe here. Explicit False
    # suppresses the FutureWarning in PyTorch >= 2.4.
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    # Infer input_dim from the saved first-layer weight rather than hardcoding
    # INPUT_DIM — this survives feature-dimension changes without manual updates.
    input_dim = state["net.0.weight"].shape[1]
    # Infer hidden widths and num_classes so compact/full candidates share the
    # same portable checkpoint format without a sidecar configuration file.
    hidden_dims = (
        state["net.0.weight"].shape[0],
        state["net.4.weight"].shape[0],
        state["net.8.weight"].shape[0],
    )
    num_classes = state["net.11.weight"].shape[0]
    model = PlasFlowMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(state)
    model.eval()
    if device is not None:
        model = model.to(device)
    logger.info(
        "Loaded MLP from %s  (input_dim=%d, hidden_dims=%s, num_classes=%d)",
        path,
        input_dim,
        hidden_dims,
        num_classes,
    )
    return model
