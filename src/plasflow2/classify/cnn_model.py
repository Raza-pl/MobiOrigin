"""1D CNN sequence classifier for plasmid/chromosome/phage classification.

This is Path 3 in the accuracy improvement roadmap.  It directly reads raw
DNA sequences (one-hot encoded) rather than pre-computed k-mer frequencies.
A 1D CNN can learn:

  - Origin of replication (oriV/oriT) motifs (~6-20bp)
  - Rep protein binding sites (~8bp direct repeats)
  - Relaxase recognition sites
  - Insertion sequence terminal inverted repeats
  - Phage attB/attP integration sites

These are positional motifs that k-mer frequencies cannot capture because
k-mer counts are order-independent.

Architecture
------------
  Input   : one-hot DNA, shape (N, seq_len, 4)  seq_len padded/truncated to MAX_LEN
  Conv1   : 256 filters, kernel=8  → captures 8-mer motifs (rep binding sites)
  Conv2   : 128 filters, kernel=4  → captures 4-mer motifs (AT-rich regions)
  Conv3   :  64 filters, kernel=16 → captures longer structural motifs (origin)
  Pool    : GlobalMaxPool → collapse to (N, 256+128+64)
  Dense1  : 256 → BatchNorm → GELU → Dropout(0.3)
  Dense2  : 3 → Softmax (plasmid/chromosome/phage)

Benchmark reference
-------------------
  DeepPlasmid (2021): CNN on 10kb windows, achieves 95%+ on novel plasmids.
  PlasClass (2020)  : CNN achieves 92%+ on ≥5kb sequences.

This implementation targets DeepPlasmid-level accuracy.

Usage
-----
    # Train:
    from plasflow2.classify.cnn_model import build_cnn, train_cnn, save_cnn
    model = build_cnn()
    model = train_cnn(model, train_seqs, train_labels, val_seqs, val_labels)
    save_cnn(model, 'data/models/cnn_v1.pt')

    # Predict:
    from plasflow2.classify.cnn_model import load_cnn, predict_cnn
    model = load_cnn('data/models/cnn_v1.pt')
    probs = predict_cnn(model, sequences)  # (N, 3) float32
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Sequences are padded/truncated to this length.
# 10kb is optimal: long enough for full plasmid signal, short enough for RAM.
MAX_SEQ_LEN = 10_000

# One-hot encoding: A=0, C=1, G=2, T=3
_BASE_TO_OH_IDX = {
    'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3, 't': 3,
}

NUM_CLASSES = 3   # plasmid, chromosome, phage


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

def one_hot_encode(seq: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """One-hot encode a DNA string to shape (max_len, 4).

    Sequences shorter than max_len are zero-padded at the end.
    Sequences longer than max_len are truncated (centre-cropped for plasmids).
    N/ambiguous bases are encoded as [0, 0, 0, 0].
    """
    seq = seq.upper()[:max_len]
    oh = np.zeros((max_len, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = _BASE_TO_OH_IDX.get(base, -1)
        if idx >= 0:
            oh[i, idx] = 1.0
    return oh


def batch_one_hot(sequences: list[str], max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Encode a list of sequences to (N, max_len, 4) float32."""
    return np.array([one_hot_encode(s, max_len) for s in sequences], dtype=np.float32)


# ---------------------------------------------------------------------------
# CNN model definition (PyTorch)
# ---------------------------------------------------------------------------

def build_cnn(
    seq_len: int = MAX_SEQ_LEN,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.3,
) -> "torch.nn.Module":
    """Build the 1D CNN classifier.

    Returns a PyTorch Module.  Input shape: (N, seq_len, 4).
    Output shape: (N, num_classes) — raw logits, apply softmax for probs.
    """
    import torch
    import torch.nn as nn

    class PlasFlowCNN(nn.Module):
        def __init__(self):
            super().__init__()

            # Three parallel convolutional towers with different kernel sizes.
            # Each tower captures patterns at a different scale.
            self.tower8  = nn.Sequential(
                nn.Conv1d(4, 256, kernel_size=8,  padding=4),
                nn.BatchNorm1d(256), nn.GELU(),
                nn.Conv1d(256, 128, kernel_size=4, padding=2),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.AdaptiveMaxPool1d(1),   # (N, 128, 1)
            )
            self.tower16 = nn.Sequential(
                nn.Conv1d(4, 256, kernel_size=16, padding=8),
                nn.BatchNorm1d(256), nn.GELU(),
                nn.Conv1d(256, 128, kernel_size=8, padding=4),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.AdaptiveMaxPool1d(1),
            )
            self.tower32 = nn.Sequential(
                nn.Conv1d(4, 128, kernel_size=32, padding=16),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 64, kernel_size=16, padding=8),
                nn.BatchNorm1d(64),  nn.GELU(),
                nn.AdaptiveMaxPool1d(1),
            )

            # After global pooling: 128 + 128 + 64 = 320 dims
            merged_dim = 128 + 128 + 64  # 320

            self.classifier = nn.Sequential(
                nn.Linear(merged_dim, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, num_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (N, seq_len, 4) → transpose to (N, 4, seq_len) for Conv1d
            x = x.permute(0, 2, 1).float()
            t8  = self.tower8(x).squeeze(-1)   # (N, 128)
            t16 = self.tower16(x).squeeze(-1)  # (N, 128)
            t32 = self.tower32(x).squeeze(-1)  # (N, 64)
            merged = torch.cat([t8, t16, t32], dim=1)  # (N, 320)
            return self.classifier(merged)

    return PlasFlowCNN()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cnn(
    model: "torch.nn.Module",
    train_seqs: list[str],
    train_labels: list[int],   # 0=plasmid, 1=chromosome, 2=phage
    val_seqs: list[str],
    val_labels: list[int],
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    max_len: int = MAX_SEQ_LEN,
) -> "torch.nn.Module":
    """Train the CNN with early stopping on validation accuracy."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score  # type: ignore

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training CNN on device: %s", device)
    model = model.to(device)

    # Pre-encode all sequences (memory-intensive but avoids re-encoding each epoch)
    logger.info("One-hot encoding %d training sequences …", len(train_seqs))
    X_tr = torch.tensor(batch_one_hot(train_seqs, max_len))
    y_tr = torch.tensor(train_labels, dtype=torch.long)
    X_va = torch.tensor(batch_one_hot(val_seqs, max_len)).to(device)
    y_va_np = np.array(val_labels)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                        shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    best_state = {}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_va).argmax(dim=-1).cpu().numpy()
        acc = accuracy_score(y_va_np, preds)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d — loss %.4f  val_acc %.4f  best %.4f",
                        epoch, epochs, total_loss / len(loader), acc, best_acc)
        if no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    model.load_state_dict(best_state)
    model.eval()
    logger.info("CNN training complete — best val acc: %.4f", best_acc)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_cnn(
    model: "torch.nn.Module",
    sequences: list[str],
    batch_size: int = 16,
    max_len: int = MAX_SEQ_LEN,
) -> np.ndarray:
    """Run CNN inference on sequences.

    Returns:
        (N, 3) float32 softmax probabilities [plasmid, chromosome, phage].
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    model.eval()

    all_probs = []
    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start : start + batch_size]
        x = torch.tensor(batch_one_hot(batch_seqs, max_len)).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs).astype(np.float32)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_cnn(model: "torch.nn.Module", path: Path | str) -> None:
    """Save CNN weights."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), str(path))
    logger.info("Saved CNN weights → %s", path)


def load_cnn(path: Path | str,
             seq_len: int = MAX_SEQ_LEN,
             num_classes: int = NUM_CLASSES) -> "torch.nn.Module":
    """Load CNN weights."""
    import torch
    model = build_cnn(seq_len=seq_len, num_classes=num_classes)
    state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded CNN from %s", path)
    return model
