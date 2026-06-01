"""Inference: run classifier on sequences and return predictions with confidence.

Week 2 — Days 11–12 implementation target.

Class-specific thresholds
-------------------------
The MLP is trained on a balanced dataset (~25 % per class), but real
metagenome assemblies contain only ~2–5 % plasmid contigs.  Using a single
confidence threshold therefore overestimates plasmid prevalence because the
model has never learned that plasmid is a rare class.

To correct for this prior imbalance we apply *class-specific* thresholds:

* **plasmid** — default 0.95 (high bar; false positives are costly).
* **chromosome / phage / archaea** — default 0.70 (lower bar; these are
  abundant and the cost of a missed call is lower).

Users can override these via the CLI flags ``--threshold`` (all non-plasmid
classes) and ``--plasmid-threshold`` (plasmid only).

Argmax fallback (--min-confidence)
------------------------------------
When ``argmax_fallback=True`` (activated via ``--min-confidence`` on the CLI),
sequences that fall below the applicable threshold are assigned the
**argmax class** (highest-scoring class) rather than "unclassified", with
their actual confidence retained.  This trades precision for recall — useful
when the unclassified rate is unacceptably high and you prefer a best-guess
assignment over no assignment.

Typical use:
    plasflow2 run ... --min-confidence 0.70
    # contigs where no class hits 0.95 (plasmid) / 0.70 (others) get the
    # argmax label instead of 'unclassified'.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plasflow2.classify.features import extract_features
from plasflow2.utils.device import IDX_TO_CLASS, get_device

logger = logging.getLogger(__name__)

# Default confidence thresholds (class-specific)
DEFAULT_THRESHOLD = 0.70  # chromosome / phage / archaea
DEFAULT_PLASMID_THRESHOLD = 0.95  # plasmid — higher bar to correct for class-prior imbalance


@dataclass
class Prediction:
    """Single-sequence prediction result."""

    sequence_id: str
    label: str  # plasmid | chromosome | phage | archaea | unclassified
    confidence: float  # max softmax probability (after temperature scaling)
    scores: dict[str, float]  # per-class probabilities


def predict(
    sequences: list[str],
    sequence_ids: list[str],
    model_path: Path | str,
    threshold: float = DEFAULT_THRESHOLD,
    plasmid_threshold: float = DEFAULT_PLASMID_THRESHOLD,
    batch_size: int = 512,
    argmax_fallback: bool = False,
) -> list[Prediction]:
    """Classify sequences using a trained MLP with class-specific thresholds.

    For each sequence the model's argmax class is selected, then a
    *class-specific* confidence threshold is applied:

    * ``plasmid_threshold`` governs plasmid calls (default 0.95).
    * ``threshold`` governs all other classes (default 0.70).

    Sequences whose winning class falls below the applicable threshold are
    labelled ``unclassified`` **unless** ``argmax_fallback=True``, in which
    case they receive the argmax label with their actual (below-threshold)
    confidence.  Use this to reduce the unclassified rate at the cost of
    slightly lower precision on borderline contigs.

    Args:
        sequences: DNA strings.
        sequence_ids: Identifiers corresponding to each sequence.
        model_path: Path to saved .pt weights.
        threshold: Minimum confidence for chromosome / phage / archaea calls.
        plasmid_threshold: Minimum confidence for plasmid calls (higher than
            ``threshold`` to compensate for class-prior imbalance).
        batch_size: Inference batch size.
        argmax_fallback: When True, contigs below threshold receive the argmax
            class instead of "unclassified".  Activated by ``--min-confidence``
            on the CLI.

    Returns:
        List of Prediction objects, one per input sequence.
    """
    import torch
    import torch.nn as nn

    from plasflow2.classify.model import load_model

    device = get_device()
    model = load_model(model_path, device=device)

    # Multi-GPU: wrap in DataParallel when multiple CUDA devices are available.
    # DataParallel splits the batch across all GPUs and merges results.
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        logger.info("Using %d GPUs via DataParallel for MLP inference", n_gpus)
        model = nn.DataParallel(model)
    elif n_gpus == 1:
        logger.info("Using 1 GPU for MLP inference")
    else:
        logger.info("GPU not available — running MLP inference on CPU")

    model.eval()

    X = extract_features(sequences)
    results: list[Prediction] = []

    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start : start + batch_size]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for i, prob_row in enumerate(probs):
            idx = int(np.argmax(prob_row))
            confidence = float(prob_row[idx])
            best_class = IDX_TO_CLASS[idx]
            # Apply class-specific threshold: plasmid requires higher confidence
            # to compensate for class-prior imbalance (model trained ~25% plasmid
            # but real metagenomes have ~2–5% plasmid).
            applicable_threshold = plasmid_threshold if best_class == "plasmid" else threshold
            if confidence >= applicable_threshold:
                label = best_class
            elif argmax_fallback:
                label = best_class   # best-guess instead of "unclassified"
            else:
                label = "unclassified"

            results.append(
                Prediction(
                    sequence_id=sequence_ids[start + i],
                    label=label,
                    confidence=confidence,
                    scores={IDX_TO_CLASS[j]: float(prob_row[j]) for j in range(len(prob_row))},
                )
            )

    n_unclassified = sum(1 for r in results if r.label == "unclassified")
    logger.info(
        "Classified %d sequences (plasmid_threshold=%.2f, threshold=%.2f, "
        "argmax_fallback=%s, unclassified=%d)",
        len(results),
        plasmid_threshold,
        threshold,
        argmax_fallback,
        n_unclassified,
    )
    return results
