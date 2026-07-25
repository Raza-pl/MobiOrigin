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

Features (28 total)
-------------------
MLP scores (3):
    plasmid, chromosome, and phage probabilities

Mobility and replication evidence (5):
    conjugative, mobilizable, replicon, ICE, and replication-protein evidence

Evidence densities (4):
    ARG, MGE, ICE, and replication-protein hits per kb

Sequence properties (4):
    length, GC content, coding density, and ORF density

geNomad-derived gene features (12):
    plasmid/chromosome/virus marker frequencies, SPM summaries,
    strand-switch and RBS statistics, and plasmid-marker counts

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

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
_MOBILITY_FEATURE_INDICES = [
    3,
    4,
    5,
    6,
    7,
]  # conjugative, mobilizable, replicon, ice, rep_protein


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
# Model-file resolution helpers (JSON/UBJ preferred, legacy .pkl supported)
# ---------------------------------------------------------------------------


def resolve_marker_model_path(path: Path) -> Path | None:
    """Resolve a marker model to native XGBoost JSON or UBJ only.

    Historical callers may still pass ``marker_xgb.pkl`` as the base name.
    In that case a JSON or UBJ sibling is accepted, but the pickle itself is
    never returned or deserialized.
    """
    path = Path(path)
    candidates = [path.with_suffix(".json"), path.with_suffix(".ubj")]

    if path.suffix in {".json", ".ubj"}:
        candidates.insert(0, path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _load_xgboost_native(path: Path):  # type: ignore[no-untyped-def]
    """Load an XGBClassifier from its native JSON/UBJ serialization.

    Raises the native XGBoost parser error if the artifact is malformed.
    Pickle deserialization is deliberately unsupported.
    """
    from xgboost import XGBClassifier  # type: ignore[import]

    model = XGBClassifier()
    model.load_model(str(path))
    return model


def marker_model_safety_issues(metadata: dict) -> list[str]:
    """Return reasons a marker model is unsafe for production fusion.

    A production marker model must be a genuine three-class model trained
    with a leakage-resistant source-group split and the exact runtime feature
    schema. Returning a list makes pipeline warnings actionable and keeps
    validation independently testable.
    """
    issues: list[str] = []

    class_counts = metadata.get("class_counts")
    if not isinstance(class_counts, dict):
        issues.append("model card has no class_counts mapping")
    else:
        for class_name in ("plasmid", "chromosome", "phage"):
            count = class_counts.get(class_name)
            if not isinstance(count, (int, float)) or count <= 0:
                issues.append(f"class {class_name!r} has no positive training rows")

    if metadata.get("split_type") != "grouped_by_source_genome":
        issues.append("training split was not grouped by source genome")

    n_groups = metadata.get("n_distinct_groups")
    if not isinstance(n_groups, int) or n_groups < 3:
        issues.append("model card has no valid distinct source-group count")

    if metadata.get("feature_names") != MARKER_FEATURE_NAMES:
        issues.append("model feature schema does not match the runtime schema")

    if metadata.get("feature_schema_version") != "marker-v2":
        issues.append("model card has no supported marker feature-schema version")

    training_hash = metadata.get("training_data_sha256")
    if not (
        isinstance(training_hash, str)
        and len(training_hash) == 64
        and all(char in "0123456789abcdefABCDEF" for char in training_hash)
    ):
        issues.append("model card has no valid training-data SHA-256")

    if metadata.get("benchmark_lockout_verified") is not True:
        issues.append("benchmark lockout was not verified during training")

    lockout_hash = metadata.get("benchmark_lockout_sha256")
    if not (
        isinstance(lockout_hash, str)
        and len(lockout_hash) == 64
        and all(char in "0123456789abcdefABCDEF" for char in lockout_hash)
    ):
        issues.append("model card has no valid benchmark-lockout SHA-256")

    return issues


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
        self.metadata: dict = {}

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
        groups: NDArray | None = None,
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
            groups: Optional (N,) array of source-genome IDs, one per row.
                The training set (build_marker_dataset.py) slices each source
                genome into multiple overlapping windows (2kb/5kb/10kb, 50%
                step) as separate rows -- adjacent windows of the same genome
                share most of their sequence and are near-duplicates in
                feature space (GC content, coding density, etc. barely
                differ). A plain random split lets siblings of the same
                genome land in both train and validation, so the model can
                effectively validate on data it already trained on,
                inflating val_accuracy. When *groups* is provided, every row
                is assigned to train OR val by its group (never split across
                both) -- done per-class so the held-out fraction still comes
                out close to *eval_fraction* per class, since groups never
                span classes (each genome belongs to exactly one class in
                this dataset design). When *groups* is None (e.g. legacy
                NPZ files built before this was tracked), falls back to the
                old per-row random split, with a warning -- val_accuracy in
                that case may be optimistic.

        Returns:
            Dict with 'val_accuracy' and 'feature_importances'.
        """
        expected_classes = {0, 1, 2}
        observed_classes = {int(value) for value in np.unique(y)}
        if observed_classes != expected_classes:
            raise ValueError(
                "MarkerClassifier requires plasmid, chromosome, and phage "
                f"training rows with labels {sorted(expected_classes)}; "
                f"observed {sorted(observed_classes)}"
            )

        try:
            from xgboost import XGBClassifier  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "xgboost is required for the marker classifier. "
                "Install with: pip install xgboost"
            ) from e

        from sklearn.metrics import accuracy_score  # type: ignore[import]

        if groups is not None:
            from sklearn.model_selection import GroupShuffleSplit  # type: ignore[import]

            groups = np.asarray(groups)
            train_idx_parts: list[NDArray] = []
            val_idx_parts: list[NDArray] = []
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0]
                n_cls_groups = len(np.unique(groups[cls_idx]))
                if n_cls_groups < 2:
                    # Can't hold out a whole group without losing the class
                    # entirely from one side -- fall back to keeping all
                    # rows of this (tiny/ungrouped) class in train.
                    logger.warning(
                        "Class %s has only %d distinct group(s) — all rows kept in train.",
                        cls,
                        n_cls_groups,
                    )
                    train_idx_parts.append(cls_idx)
                    continue
                splitter = GroupShuffleSplit(
                    n_splits=1, test_size=eval_fraction, random_state=random_state
                )
                tr_rel, va_rel = next(splitter.split(cls_idx, groups=groups[cls_idx]))
                train_idx_parts.append(cls_idx[tr_rel])
                val_idx_parts.append(cls_idx[va_rel])
            train_idx = np.concatenate(train_idx_parts)
            val_idx = np.concatenate(val_idx_parts) if val_idx_parts else np.array([], dtype=int)
            X_tr, X_va, y_tr, y_va = X[train_idx], X[val_idx], y[train_idx], y[val_idx]
            n_shared_groups = len(
                set(np.unique(groups[train_idx])) & set(np.unique(groups[val_idx]))
            )
            logger.info(
                "Grouped split: %d train rows / %d val rows, %d groups shared "
                "between train and val (should be 0).",
                len(train_idx),
                len(val_idx),
                n_shared_groups,
            )
        else:
            from sklearn.model_selection import train_test_split  # type: ignore[import]

            logger.warning(
                "No groups provided — falling back to a random per-row split. "
                "If the training data has multiple overlapping windows per "
                "source genome, val_accuracy may be inflated by leakage."
            )
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=eval_fraction, stratify=y, random_state=random_state
            )

        # Class completeness is validated above; never manufacture a nominal
        # three-class model from a dataset with an absent biological class.
        n_classes = len(expected_classes)
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
        scores = {
            class_name: float(proba[i]) for i, class_name in enumerate(self._classes[: len(proba)])
        }
        # Legacy production marker checkpoints are binary
        # (plasmid/chromosome). Keep them usable beside a 3-class MLP by
        # exposing an explicit zero phage score instead of indexing past the
        # two-column probability array.
        for class_name in self._classes:
            scores.setdefault(class_name, 0.0)
        return scores

    def save(self, path: Path | str, metadata: dict | None = None) -> None:
        """Save via XGBoost's native JSON serialization, plus a model card.

        Previously this pickled the fitted model. Pickle can execute
        arbitrary code on load, which is an unnecessary supply-chain risk for
        a file that isn't always built locally — this project's marker model
        is distributed via a GitHub Release download, not committed to git.
        XGBoost's own JSON/UBJ format has no such risk and loads just as
        fast. If *path* has a ``.pkl`` extension (every existing call site
        passes one, e.g. ``data/models/marker_xgb.pkl``), the artifact is
        written to the ``.json`` sibling instead — no ``.pkl`` file is
        created by this method anymore. ``load()`` below accepts either
        extension and resolves to whichever file actually exists, preferring
        ``.json``/``.ubj`` over a legacy ``.pkl``.

        The model card (``<json_path>.meta.json``) carries the provenance a
        raw model file never can — there was previously no way to tell, after
        the fact, what data or code produced a given ``marker_xgb.pkl`` (this
        bit the project once already: has_replicon and every geNomad marker
        feature were silently 0 in the training set that produced a deployed
        model, and it went unnoticed because nothing recorded what that model
        was actually trained on). Callers should pass at least the training
        data path/hash and feature names via *metadata*.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        path = Path(path)
        json_path = path if path.suffix in (".json", ".ubj") else path.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(json_path))
        logger.info("MarkerClassifier saved (XGBoost native format) → %s", json_path)

        meta = dict(metadata or {})
        meta.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        meta.setdefault("format", "xgboost-json")
        meta.setdefault("n_features", int(self._model.n_features_in_))
        meta_path = json_path.with_suffix(json_path.suffix + ".meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        logger.info("Model card saved → %s", meta_path)

    @classmethod
    def load(cls, path: Path | str) -> MarkerClassifier:
        """Load a native XGBoost JSON or UBJ marker model.

        ``.pkl`` may be supplied as a historical base name only when a native
        ``.json`` or ``.ubj`` sibling exists. Pickle artifacts are rejected
        before their contents are opened because deserialization can execute
        arbitrary code.
        """
        path = Path(path)
        resolved = resolve_marker_model_path(path)

        if resolved is None:
            if path.suffix == ".pkl" and path.exists():
                raise ValueError(
                    f"Refusing legacy pickle marker model: {path}. "
                    "Only native XGBoost JSON/UBJ artifacts are supported."
                )
            raise FileNotFoundError(
                f"No native marker model found at {path} "
                f"(checked {path.with_suffix('.json')} and "
                f"{path.with_suffix('.ubj')})"
            )

        if resolved.suffix not in {".json", ".ubj"}:
            raise ValueError(
                f"Unsupported marker model format: {resolved.suffix}. "
                "Only native XGBoost JSON/UBJ artifacts are supported."
            )

        model = _load_xgboost_native(resolved)

        obj = cls()
        obj._model = model
        logger.info("MarkerClassifier loaded from %s", resolved)

        meta_path = resolved.with_suffix(resolved.suffix + ".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as fh:
                    obj.metadata = json.load(fh)
                logger.info("Model card: %s", obj.metadata)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read model card %s: %s", meta_path, exc)
        else:
            logger.warning(
                "No model card at %s — provenance (training data, feature schema, "
                "hyperparameters) for this checkpoint is unknown.",
                meta_path,
            )

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
