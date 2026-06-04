"""Inference: run 3-class MLP classifier on sequences.

Classes: plasmid | chromosome | phage
Archaea is NOT predicted here — it is detected post-classification in the
pipeline by comparing archaeal vs bacterial ORF hits from DIAMOND taxonomy
(paper criteria: archaea_hits > bacteria_hits AND archaea_hits >= 5).

Class-specific thresholds
-------------------------
The MLP is trained on a balanced dataset (~33 % per class), but real
metagenome assemblies contain only ~2–5 % plasmid contigs.  We apply
*class-specific* thresholds to compensate:

* **plasmid** — default 0.95 (high bar; false positives are costly).
* **chromosome / phage** — default 0.70 (lower bar).

Argmax fallback (--min-confidence)
------------------------------------
When ``argmax_fallback=True``, sequences below threshold receive the argmax
class instead of "unclassified".  Activated via ``--min-confidence`` on CLI.
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
DEFAULT_THRESHOLD = 0.70          # chromosome / phage
DEFAULT_PLASMID_THRESHOLD = 0.95  # plasmid — higher bar to correct for class-prior imbalance

# Class prior distributions by sample context.
# The MLP is trained on balanced classes (33%/33%/33%) but real metagenomes
# are dominated by chromosomal sequences. Bayesian correction:
#   corrected[c] = mlp_score[c] * prior[c] / sum_k(mlp_score[k] * prior[k])
# This reduces false-positive plasmid and phage calls substantially.
CONTEXT_PRIORS: dict[str, dict[str, float]] = {
    "wastewater":    {"plasmid": 0.030, "chromosome": 0.930, "phage": 0.040},
    "clinical":      {"plasmid": 0.050, "chromosome": 0.900, "phage": 0.050},
    "environmental": {"plasmid": 0.020, "chromosome": 0.950, "phage": 0.030},
    "unspecified":   {"plasmid": 0.333, "chromosome": 0.334, "phage": 0.333},
}


def apply_prior_correction(
    scores: dict[str, float],
    context: str = "unspecified",
) -> dict[str, float]:
    """Apply Bayesian class-prior correction to MLP softmax scores.

    Multiplies each class score by its expected frequency in the given sample
    context, then renormalises. For 'unspecified' the prior is uniform (no
    correction). This is equivalent to geNomad's score calibration step.

    Args:
        scores:  Dict of class → probability from MLP softmax.
        context: Sample context — 'wastewater', 'clinical', 'environmental',
                 or 'unspecified' (default, no correction).

    Returns:
        Corrected and renormalised score dict.
    """
    prior = CONTEXT_PRIORS.get(context, CONTEXT_PRIORS["unspecified"])
    corrected = {c: scores.get(c, 0.0) * prior.get(c, 1.0) for c in scores}
    total = sum(corrected.values()) or 1.0
    return {c: v / total for c, v in corrected.items()}


@dataclass
class Prediction:
    """Single-sequence prediction result."""

    sequence_id: str
    label: str  # plasmid | chromosome | phage | archaea | unclassified
    # Note: 'archaea' is assigned post-classification by the pipeline,
    # not by the MLP itself.
    confidence: float  # max softmax probability
    scores: dict[str, float]  # per-class probabilities (3-class MLP output)


def predict(
    sequences: list[str],
    sequence_ids: list[str],
    model_path: Path | str,
    threshold: float = DEFAULT_THRESHOLD,
    plasmid_threshold: float = DEFAULT_PLASMID_THRESHOLD,
    batch_size: int = 512,
    argmax_fallback: bool = False,
    source_context: str = "unspecified",
    apply_prior: bool = True,
) -> list[Prediction]:
    """Classify sequences using the 3-class MLP (plasmid / chromosome / phage).

    Archaea is not a model output — it is assigned post-classification by
    the pipeline using DIAMOND taxonomy ORF voting.

    Args:
        sequences: DNA strings.
        sequence_ids: Identifiers corresponding to each sequence.
        model_path: Path to saved .pt weights.
        threshold: Minimum confidence for chromosome / phage calls (default 0.70).
        plasmid_threshold: Minimum confidence for plasmid calls (default 0.95).
        batch_size: Inference batch size.
        argmax_fallback: When True, contigs below threshold receive the argmax
            class instead of "unclassified".  Activated by ``--min-confidence``.

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
            raw_scores = {IDX_TO_CLASS[j]: float(prob_row[j]) for j in range(len(prob_row))}
            # Apply Bayesian prior correction before thresholding
            if apply_prior and source_context != "unspecified":
                scores = apply_prior_correction(raw_scores, source_context)
            else:
                scores = raw_scores
            best_class = max(scores, key=scores.get)
            confidence = float(scores[best_class])
            # Apply class-specific threshold
            if best_class == "plasmid":
                applicable_threshold = plasmid_threshold
            else:
                applicable_threshold = threshold
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
                    scores=scores,  # prior-corrected scores
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
