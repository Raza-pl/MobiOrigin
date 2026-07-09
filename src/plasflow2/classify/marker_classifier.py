"""Marker-based second-stage classifier — XGBoost over biological features.

This module implements a geNomad-inspired second classification stage that
operates on biological marker features rather than raw k-mer frequencies.

How it fits into the pipeline
------------------------------
Stage 1 — MLP (k-mer):
    Input  : raw nucleotide sequence
    Output : plasmid / chromosome / phage scores (softmax)
    Strength: fast, no databases required, good on long contigs

Stage 2 — Marker XGBoost (biological evidence):
    Input  : per-contig annotation features derived from DIAMOND hits
    Output : plasmid / chromosome / phage scores
    Strength: uses protein-level evidence; decisive when hallmark genes present

Aggregation (attention-weighted):
    final_score = α × marker_score + (1-α) × mlp_score
    where α = marker_gene_fraction  (increases as more genes get marker hits)
    When no genes match markers → α=0, MLP score is used unchanged.

Features (15 total)
-------------------
From the MLP (3):
    mlp_plasmid_score, mlp_chromosome_score, mlp_phage_score

Biological markers (8):
    is_conjugative     — MOB DIAMOND: relaxase + MPF hit
    is_mobilizable     — MOB DIAMOND: relaxase hit only
    has_replicon       — replicon type assigned (IncF, IncP, …)
    has_plsdb_match    — PLSDB / RefSeq plasmid DB match
    has_phage_marker   — ICE / MGE hit of phage origin
    n_arg_normalized   — ARG hits per kb
    n_mge_normalized   — MGE hits per kb
    n_ice_normalized   — ICE hits per kb

Sequence features (4):
    log10_length       — log10(contig_length)
    gc_content         — fraction G+C
    coding_density     — fraction of contig covered by ORFs
    n_orfs_per_kb      — ORF density

Training
--------
Run: python scripts/train_marker_model.py
     --features data/marker_features.npz
     --out      data/models/

The training data (marker_features.npz) is built by
scripts/build_marker_dataset.py, which runs DIAMOND against the
mob_proteins / phage hallmark databases on a labelled set and extracts
the features above.

Reference
---------
Camargo et al. geNomad (2023) — hybrid marker + NN classification with
attention-weighted score aggregation.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names (order matters — must match training)
# ---------------------------------------------------------------------------

MARKER_FEATURE_NAMES = [
    # MLP scores (3)
    "mlp_plasmid_score",
    "mlp_chromosome_score",
    "mlp_phage_score",
    # Mobility / plasmid markers (6) — note: has_plsdb_match is intentionally
    # excluded from the model. A PLSDB match is used as a hard rule-based
    # override AFTER the XGBoost scores, not as a learned feature.
    "is_conjugative",
    "is_mobilizable",
    "has_replicon",
    "has_ice",
    "has_rep_protein",  # replication protein hit (RepA/RepB/RepC) — non-mobile plasmids
    # ARG / MGE / rep density (per kb) (4)
    "n_arg_per_kb",
    "n_mge_per_kb",
    "n_ice_per_kb",
    "n_rep_per_kb",  # rep protein hits per kb — density signal
    # Sequence properties (4) — computed from ORF prediction + raw sequence
    "log10_length",
    "gc_content",
    "coding_density",
    "n_orfs_per_kb",
    # geNomad SPM features (12) — from genomad annotate *_genes.tsv
    "p_marker_freq",  # fraction of genes with plasmid-dominant marker hit
    "c_marker_freq",  # fraction of genes with chromosome-dominant marker hit
    "v_marker_freq",  # fraction of genes with virus-dominant marker hit
    "pp_marker_freq",  # fraction of genes with plasmid SPM > 0.5
    "median_p_spm",  # median plasmid SPM across all genes
    "median_c_spm",  # median chromosome SPM
    "median_v_spm",  # median virus SPM
    "p_vs_c_logistic",  # sigmoid(mean(p_spm - c_spm))
    "strand_switch_rate",  # fraction of consecutive gene pairs with strand flips
    "no_rbs_freq",  # fraction of genes with no RBS motif
    "canonical_sd_freq",  # fraction with canonical Shine-Dalgarno motif
    "n_plasmid_markers",  # raw count of plasmid-marker gene hits
]

N_MARKER_FEATURES = len(MARKER_FEATURE_NAMES)
# Mobility feature indices — used to compute marker_gene_fraction
_MOBILITY_FEATURE_INDICES = [3, 4, 5, 6, 7]  # conjugative, mobilizable, replicon, ice, rep_protein


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------


@dataclass
class ContigMarkerFeatures:
    """Marker features for a single contig."""

    contig_id: str

    # MLP scores
    mlp_plasmid_score: float = 0.5
    mlp_chromosome_score: float = 0.25
    mlp_phage_score: float = 0.25

    # Mobility / plasmid markers (binary)
    is_conjugative: float = 0.0
    is_mobilizable: float = 0.0
    has_replicon: float = 0.0
    has_ice: float = 0.0
    has_rep_protein: float = 0.0  # replication protein (RepA/B/C) — key non-mobile plasmid marker
    # Note: has_plsdb_match is NOT a model feature — used as a hard override rule.

    # ARG / MGE / rep density
    n_arg_per_kb: float = 0.0
    n_mge_per_kb: float = 0.0
    n_ice_per_kb: float = 0.0
    n_rep_per_kb: float = 0.0

    # Sequence properties
    log10_length: float = 3.0  # default: 1 kb
    gc_content: float = 0.5
    coding_density: float = 0.85
    n_orfs_per_kb: float = 1.0

    # geNomad SPM features (zero when --genomad-genes not provided)
    p_marker_freq: float = 0.0
    c_marker_freq: float = 0.0
    v_marker_freq: float = 0.0
    pp_marker_freq: float = 0.0
    median_p_spm: float = 0.0
    median_c_spm: float = 0.0
    median_v_spm: float = 0.0
    p_vs_c_logistic: float = 0.5  # neutral sigmoid value
    strand_switch_rate: float = 0.0
    no_rbs_freq: float = 0.0
    canonical_sd_freq: float = 0.0
    n_plasmid_markers: float = 0.0

    def to_array(self) -> NDArray[np.float32]:
        """Return feature vector as float32 array (28 features)."""
        return np.array(
            [
                self.mlp_plasmid_score,
                self.mlp_chromosome_score,
                self.mlp_phage_score,
                self.is_conjugative,
                self.is_mobilizable,
                self.has_replicon,
                self.has_ice,
                self.has_rep_protein,
                self.n_arg_per_kb,
                self.n_mge_per_kb,
                self.n_ice_per_kb,
                self.n_rep_per_kb,
                self.log10_length,
                self.gc_content,
                self.coding_density,
                self.n_orfs_per_kb,
                # geNomad SPM features
                self.p_marker_freq,
                self.c_marker_freq,
                self.v_marker_freq,
                self.pp_marker_freq,
                self.median_p_spm,
                self.median_c_spm,
                self.median_v_spm,
                self.p_vs_c_logistic,
                self.strand_switch_rate,
                self.no_rbs_freq,
                self.canonical_sd_freq,
                self.n_plasmid_markers,
            ],
            dtype=np.float32,
        )

    @property
    def marker_gene_fraction(self) -> float:
        """Fraction of biological marker features that are positive.

        Used as the attention weight α: high when markers present →
        trust XGBoost more. Low when absent → trust MLP more.
        """
        vals = [
            self.is_conjugative,
            self.is_mobilizable,
            self.has_replicon,
            self.has_ice,
            self.has_rep_protein,
        ]
        return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Feature extraction from pipeline annotation results
# ---------------------------------------------------------------------------


def extract_marker_features(
    contig_id: str,
    sequence: str,
    mlp_scores: dict[str, float],
    mobility=None,  # MobilityResult | None
    arg_hits: list = None,  # list[ARGHit]
    mge_hits: list = None,  # list[MGEHit]
    ice_hits: list = None,  # list[ICEHit]
    orfs: list = None,  # list[ORF]
    has_rep_protein: bool = False,  # from rep_protein_hits set in pipeline
    n_rep_hits: int = 0,  # raw rep protein hit count for this contig
    genomad_spm: dict[str, float] | None = None,  # geNomad SPM feature dict
) -> ContigMarkerFeatures:
    """Build a ContigMarkerFeatures from pipeline annotation objects.

    All annotation arguments are optional — absent annotations produce
    zero-valued features, so the XGBoost falls back to the MLP scores.

    Args:
        contig_id: Contig identifier.
        sequence: Nucleotide sequence string.
        mlp_scores: Dict of class → probability from MLP (e.g. {'plasmid': 0.9, ...}).
        mobility: MobilityResult from mobility_diamond.
        arg_hits: ARGHit list for this contig.
        mge_hits: MGEHit list for this contig.
        ice_hits: ICEHit list for this contig.
        orfs: ORF list for this contig (for coding density).
        has_rep_protein: Whether a replication protein was detected for this contig.
        n_rep_hits: Raw count of rep protein hits.
        genomad_spm: Dict of geNomad SPM feature name → value, produced by
            ``scripts/extract_genomad_features.py::compute_features()``.
            When provided, populates all 12 SPM fields in ContigMarkerFeatures;
            when None (geNomad not available), those fields stay at 0.0.

    Returns:
        ContigMarkerFeatures ready for inference.
    """
    if arg_hits is None:
        arg_hits = []
    if mge_hits is None:
        mge_hits = []
    if ice_hits is None:
        ice_hits = []
    if orfs is None:
        orfs = []

    length_bp = max(len(sequence), 1)
    length_kb = length_bp / 1000.0

    # MLP scores
    plas_score = float(mlp_scores.get("plasmid", 0.0))
    chrom_score = float(mlp_scores.get("chromosome", 0.0))
    phage_score = float(mlp_scores.get("phage", 0.0))

    # Normalise to sum=1 in case of floating-point drift
    total = plas_score + chrom_score + phage_score or 1.0
    plas_score /= total
    chrom_score /= total
    phage_score /= total

    # Mobility features
    is_conj = 0.0
    is_mob = 0.0
    has_rep = 0.0
    if mobility is not None:
        mc = getattr(mobility, "mobility_class", "non-mobilizable")
        is_conj = 1.0 if mc == "conjugative" else 0.0
        is_mob = 1.0 if mc == "mobilizable" else 0.0
        _rep_type = getattr(mobility, "replicon_type", None)
        has_rep = 1.0 if _rep_type and _rep_type not in ("unknown", "-", "") else 0.0

    has_ice = 1.0 if ice_hits else 0.0

    # Density features
    n_arg = len(arg_hits) / length_kb
    n_mge = len(mge_hits) / length_kb
    n_ice = len(ice_hits) / length_kb

    # Sequence properties
    log10_len = float(np.log10(length_bp))
    seq_upper = sequence.upper()
    gc = (seq_upper.count("G") + seq_upper.count("C")) / length_bp

    # Coding density from ORFs
    if orfs:
        covered = sum(abs(getattr(o, "end", 0) - getattr(o, "start", 0)) for o in orfs)
        cod_density = min(covered / length_bp, 1.0)
        n_orfs_kb = len(orfs) / length_kb
    else:
        cod_density = 0.85  # typical bacterial coding density as prior
        n_orfs_kb = 1.0

    # geNomad SPM features — populated when geNomad ran automatically
    _gn = genomad_spm or {}
    return ContigMarkerFeatures(
        contig_id=contig_id,
        mlp_plasmid_score=plas_score,
        mlp_chromosome_score=chrom_score,
        mlp_phage_score=phage_score,
        is_conjugative=is_conj,
        is_mobilizable=is_mob,
        has_replicon=has_rep,
        has_ice=has_ice,
        has_rep_protein=1.0 if has_rep_protein else 0.0,
        n_arg_per_kb=n_arg,
        n_mge_per_kb=n_mge,
        n_ice_per_kb=n_ice,
        n_rep_per_kb=n_rep_hits / max(length_kb, 0.001),
        log10_length=log10_len,
        gc_content=gc,
        coding_density=cod_density,
        n_orfs_per_kb=n_orfs_kb,
        # geNomad SPM (12 features; 0.0 / 0.5 defaults when geNomad not available)
        p_marker_freq=float(_gn.get("p_marker_freq", 0.0)),
        c_marker_freq=float(_gn.get("c_marker_freq", 0.0)),
        v_marker_freq=float(_gn.get("v_marker_freq", 0.0)),
        pp_marker_freq=float(_gn.get("pp_marker_freq", 0.0)),
        median_p_spm=float(_gn.get("median_p_spm", 0.0)),
        median_c_spm=float(_gn.get("median_c_spm", 0.0)),
        median_v_spm=float(_gn.get("median_v_spm", 0.0)),
        p_vs_c_logistic=float(_gn.get("p_vs_c_logistic", 0.5)),
        strand_switch_rate=float(_gn.get("strand_switch_rate", 0.0)),
        no_rbs_freq=float(_gn.get("no_rbs_freq", 0.0)),
        canonical_sd_freq=float(_gn.get("canonical_sd_freq", 0.0)),
        n_plasmid_markers=float(_gn.get("n_plasmid_markers", 0.0)),
    )


# ---------------------------------------------------------------------------
# Score aggregation (attention-weighted)
# ---------------------------------------------------------------------------


def aggregate_scores(
    mlp_scores: dict[str, float],
    marker_scores: dict[str, float] | None,
    marker_gene_fraction: float,
) -> dict[str, float]:
    """Attention-weighted combination of MLP and marker scores.

    α = marker_gene_fraction (0–1).
    final = α × marker_score + (1-α) × mlp_score

    When no marker genes are present (α=0), the MLP scores are returned
    unchanged.  When strong marker evidence exists (α→1), the marker
    XGBoost dominates.

    Args:
        mlp_scores: Dict from MLP inference (class → probability).
        marker_scores: Dict from XGBoost inference, or None if unavailable.
        marker_gene_fraction: Fraction of biological markers that fired.

    Returns:
        Aggregated score dict (same keys as mlp_scores).
    """
    if marker_scores is None or marker_gene_fraction == 0.0:
        return mlp_scores

    alpha = min(marker_gene_fraction, 1.0)
    classes = list(mlp_scores.keys())
    combined = {
        c: alpha * marker_scores.get(c, 0.0) + (1.0 - alpha) * mlp_scores.get(c, 0.0)
        for c in classes
    }
    # Re-normalise
    total = sum(combined.values()) or 1.0
    return {c: v / total for c, v in combined.items()}


# ---------------------------------------------------------------------------
# XGBoost model wrapper
# ---------------------------------------------------------------------------


class MarkerClassifier:
    """XGBoost wrapper for marker-based 3-class classification.

    Usage:
        clf = MarkerClassifier()
        clf.train(X, y)               # X: (N, N_MARKER_FEATURES), y: int labels
        clf.save("data/models/marker_xgb.pkl")

        clf2 = MarkerClassifier.load("data/models/marker_xgb.pkl")
        probs = clf2.predict_proba(X)  # (N, 3) plasmid/chromosome/phage
    """

    def __init__(self) -> None:
        self._model = None
        self._classes = ["plasmid", "chromosome", "phage"]

    def train(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int64],
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        eval_fraction: float = 0.1,
        random_state: int = 42,
    ) -> dict:
        """Train the XGBoost on marker features.

        Args:
            X: Feature matrix (N, N_MARKER_FEATURES).
            y: Integer class labels (0=plasmid, 1=chromosome, 2=phage).
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: XGBoost eta.
            subsample: Row subsampling per tree.
            colsample_bytree: Column subsampling per tree.
            eval_fraction: Fraction of data held out for early stopping.
            random_state: RNG seed.

        Returns:
            Dict with 'val_accuracy' and 'feature_importances'.
        """
        try:
            from xgboost import XGBClassifier  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "xgboost is required for the marker classifier. "
                "Install with: pip install xgboost"
            ) from e

        from sklearn.metrics import accuracy_score  # type: ignore[import]
        from sklearn.model_selection import train_test_split  # type: ignore[import]

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=eval_fraction, stratify=y, random_state=random_state
        )

        # Always force 3-class multiclass — even when a class is absent from
        # this training batch (e.g. no phage in benchmark data).  Without
        # explicit num_class XGBoost auto-detects binary mode when only two
        # label values are present, then throws "num_class=1 but found 1".
        n_classes = 3
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=20,
        )
        self._model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        val_preds_raw = self._model.predict(X_va)
        # With objective='multi:softprob', predict() returns (N, n_class) probs
        # rather than class indices — convert to argmax labels.
        if val_preds_raw.ndim == 2:
            val_preds = np.argmax(val_preds_raw, axis=1)
        else:
            val_preds = val_preds_raw
        val_acc = accuracy_score(y_va, val_preds)
        logger.info("MarkerClassifier val accuracy: %.4f", val_acc)

        # Feature names: use those stored in the NPZ when the model has more
        # features than the legacy MARKER_FEATURE_NAMES list.
        n_feat = len(self._model.feature_importances_)
        feat_names = (
            MARKER_FEATURE_NAMES
            if n_feat == len(MARKER_FEATURE_NAMES)
            else [f"f{i}" for i in range(n_feat)]
        )
        importances = dict(zip(feat_names, self._model.feature_importances_))
        top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top-5 marker features: %s", top)

        return {"val_accuracy": val_acc, "feature_importances": importances}

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return (N, 3) probability array [plasmid, chromosome, phage]."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return self._model.predict_proba(X).astype(np.float32)

    def predict_scores(self, features: ContigMarkerFeatures) -> dict[str, float]:
        """Predict class scores for a single contig's marker features.

        Args:
            features: ContigMarkerFeatures object.

        Returns:
            Dict mapping class name → probability.
        """
        x = features.to_array().reshape(1, -1)
        proba = self.predict_proba(x)[0]
        return {c: float(proba[i]) for i, c in enumerate(self._classes)}

    def save(self, path: Path | str) -> None:
        """Pickle the fitted model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._model, fh)
        logger.info("MarkerClassifier saved → %s", path)

    @classmethod
    def load(cls, path: Path | str) -> MarkerClassifier:
        """Load a saved MarkerClassifier."""
        with open(path, "rb") as fh:
            model = pickle.load(fh)  # noqa: S301
        obj = cls()
        obj._model = model
        logger.info("MarkerClassifier loaded from %s", path)
        return obj


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def marker_classifier_available() -> bool:
    """Return True if xgboost is installed."""
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False
