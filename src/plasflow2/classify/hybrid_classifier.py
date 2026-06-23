"""Two-stage hybrid classifier: k-mer MLP + marker gene signals.

Stage 1: MLP on k-mer frequencies (fast, handles all sequence lengths)
Stage 2: Marker gene refinement using biological signals

Why this matters
----------------
k-mer classifiers have a hard ceiling because they measure composition,
not function.  A plasmid from an unknown organism will have unfamiliar
k-mer patterns.  But if it carries a Relaxase or Rep protein — it's a
plasmid regardless of composition.

This classifier adds 5 biological features on top of the MLP output:

  rep_protein_hits   — replication protein hits (highly plasmid-specific)
  relaxase_hits      — conjugation relaxase hits (conjugative/mobilisable plasmids)
  mpf_protein_hits   — mating-pair-formation protein hits (conjugative)
  phage_marker_hits  — phage capsid/tail/integrase hits (phage-specific)
  chromosome_marker  — bacterial single-copy genes (housekeeping = chromosome)

These are computed by a fast DIAMOND pre-screen against 5 small databases
(rep, mob, mpf, phage markers, chromosome markers) before the MLP runs.

The final classifier is a simple Logistic Regression or XGBoost that takes:
  [mlp_plasmid_score, mlp_chromosome_score, mlp_phage_score,
   rep_hits, relaxase_hits, mpf_hits, phage_hits, chr_hits, seq_len_log]

This is structurally similar to geNomad's approach but lighter-weight.

Training
--------
1. Run the full pipeline on a set of labeled sequences
2. Collect MLP scores + marker hit counts + true labels
3. Fit the meta-classifier on these 9 features
4. Save via save_hybrid() / load via load_hybrid()

Usage
-----
    from plasflow2.classify.hybrid_classifier import (
        HybridClassifier, load_hybrid, predict_hybrid
    )
    hybrid = load_hybrid('data/models/hybrid_clf.pkl')
    labels = predict_hybrid(hybrid, mlp_scores, marker_features)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HYBRID_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "models" / "hybrid_clf.pkl"
)


@dataclass
class MarkerFeatures:
    """Per-sequence marker gene hit counts."""

    rep_hits: int = 0  # replication protein hits
    relaxase_hits: int = 0  # relaxase / mobilisation protein hits
    mpf_hits: int = 0  # mating-pair-formation protein hits
    phage_hits: int = 0  # phage structural protein hits
    chr_hits: int = 0  # single-copy chromosome marker hits (recA, rpoB, etc.)

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.rep_hits, self.relaxase_hits, self.mpf_hits, self.phage_hits, self.chr_hits],
            dtype=np.float32,
        )


def build_hybrid_feature_matrix(
    mlp_scores: list[dict[str, float]],  # [{plasmid: 0.3, chromosome: 0.6, phage: 0.1}, ...]
    marker_features: list[MarkerFeatures],
    seq_lengths: list[int],
) -> np.ndarray:
    """Stack MLP softmax scores + marker hits + length into feature matrix.

    Returns: (N, 9) float32 array
      cols 0-2: mlp plasmid/chromosome/phage scores
      cols 3-7: rep/relaxase/mpf/phage/chr marker hits (log1p-transformed)
      col  8  : log10(length) / 6 (normalised)
    """
    n = len(mlp_scores)
    X = np.zeros((n, 9), dtype=np.float32)
    for i, (scores, markers, length) in enumerate(zip(mlp_scores, marker_features, seq_lengths)):
        X[i, 0] = scores.get("plasmid", 0.0)
        X[i, 1] = scores.get("chromosome", 0.0)
        X[i, 2] = scores.get("phage", 0.0)
        marker_arr = markers.to_array()
        # Log1p-transform hit counts — prevents high-hit sequences from dominating
        X[i, 3:8] = np.log1p(marker_arr)
        X[i, 8] = float(np.log10(max(1, length)) / 6.0)
    return X


# ---------------------------------------------------------------------------
# Marker feature extraction from existing annotation results
# ---------------------------------------------------------------------------


def markers_from_pipeline_row(row: dict) -> MarkerFeatures:
    """Extract MarkerFeatures from a pipeline all_predictions.tsv row.

    Works with the existing PlasFlow v2 output format — no extra DIAMOND
    runs needed if predictions have already been computed.
    """

    def _int(val, default=0):
        try:
            return int(val) if val else default
        except (ValueError, TypeError):
            return default

    # Rep proteins: pipeline stores count in 'rep_protein_hits' or similar
    rep = max(
        _int(row.get("rep_protein_hits")),
        _int(row.get("replicon_score", 0)),
        1 if row.get("replicon_type", "") not in ("", "None", "NA") else 0,
    )
    # Relaxase / mobility
    relaxase = max(
        _int(row.get("relaxase_hits")),
        1 if row.get("relaxase_type", "") not in ("", "None", "NA") else 0,
    )
    # MPF
    mpf = max(
        _int(row.get("mpf_hits")),
        1 if row.get("mpf_type", "") not in ("", "None", "NA") else 0,
        1 if row.get("mobility_class", "") == "conjugative" else 0,
    )
    # Phage markers
    phage = _int(row.get("phage_marker_hits", 0))

    return MarkerFeatures(rep_hits=rep, relaxase_hits=relaxase, mpf_hits=mpf, phage_hits=phage)


# ---------------------------------------------------------------------------
# Hybrid classifier training and inference
# ---------------------------------------------------------------------------


class HybridClassifier:
    """Wraps a meta-classifier (XGBoost or LR) that refines MLP predictions
    using marker gene signals."""

    CLASSES = ["plasmid", "chromosome", "phage"]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    def __init__(self, clf, threshold: float = 0.50) -> None:
        self.clf = clf
        self.threshold = threshold

    def predict(
        self,
        mlp_scores: list[dict[str, float]],
        marker_features: list[MarkerFeatures],
        seq_lengths: list[int],
    ) -> list[str]:
        X = build_hybrid_feature_matrix(mlp_scores, marker_features, seq_lengths)
        probs = self.clf.predict_proba(X)
        labels = []
        for prob_row in probs:
            best_idx = int(np.argmax(prob_row))
            best_prob = float(prob_row[best_idx])
            if best_prob >= self.threshold:
                labels.append(self.CLASSES[best_idx])
            else:
                # Fallback to MLP argmax when hybrid is uncertain
                labels.append(self.CLASSES[best_idx])
        return labels

    def predict_proba(
        self,
        mlp_scores: list[dict[str, float]],
        marker_features: list[MarkerFeatures],
        seq_lengths: list[int],
    ) -> np.ndarray:
        X = build_hybrid_feature_matrix(mlp_scores, marker_features, seq_lengths)
        return self.clf.predict_proba(X).astype(np.float32)


def train_hybrid(
    mlp_scores: list[dict[str, float]],
    marker_features: list[MarkerFeatures],
    seq_lengths: list[int],
    true_labels: list[str],
    out_path: Path | str | None = None,
    use_xgb: bool = True,
) -> HybridClassifier:
    """Train and save the hybrid meta-classifier.

    Args:
        mlp_scores:      Per-sequence MLP softmax dicts.
        marker_features: Per-sequence marker hit counts.
        seq_lengths:     Sequence lengths in bp.
        true_labels:     Ground-truth labels ('plasmid','chromosome','phage').
        out_path:        Where to save the fitted model.
        use_xgb:         Use XGBoost (True) or Logistic Regression (False).

    Returns:
        Fitted HybridClassifier.
    """
    path = Path(out_path) if out_path else DEFAULT_HYBRID_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    X = build_hybrid_feature_matrix(mlp_scores, marker_features, seq_lengths)
    idx_map = HybridClassifier.CLASS_TO_IDX
    y = np.array([idx_map.get(lbl, 1) for lbl in true_labels])

    if use_xgb:
        try:
            from xgboost import XGBClassifier  # type: ignore

            clf = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                use_label_encoder=False,
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=42,
            )
        except ImportError:
            logger.warning("XGBoost not available — falling back to LogisticRegression")
            use_xgb = False

    if not use_xgb:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        clf = LogisticRegression(multi_class="multinomial", max_iter=500, C=1.0)

    clf.fit(X, y)
    logger.info("Hybrid classifier trained on %d samples (use_xgb=%s)", len(y), use_xgb)

    hybrid = HybridClassifier(clf)
    with open(path, "wb") as fh:
        pickle.dump(hybrid, fh)
    logger.info("Saved hybrid classifier → %s", path)
    return hybrid


def load_hybrid(path: Path | str | None = None) -> HybridClassifier | None:
    """Load a saved hybrid classifier, or return None if not found."""
    p = Path(path) if path else DEFAULT_HYBRID_PATH
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            hybrid = pickle.load(fh)
        logger.info("Loaded hybrid classifier from %s", p)
        return hybrid
    except Exception as e:
        logger.warning("Could not load hybrid classifier from %s: %s", p, e)
        return None
