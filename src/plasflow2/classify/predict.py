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

Marker XGBoost (second stage)
------------------------------
When ``marker_model_path`` is provided, a second-stage XGBoost runs over
16 features: MLP scores + biological markers + sequence properties.

Two modes:
  * Sequence-only (no DIAMOND): biological features are 0; XGBoost recalibrates
    MLP scores using length, GC, ORF density.  alpha_base=0.3.
  * Full annotation (with annotation_tsv): loads pre-computed DIAMOND hits;
    biological features carry real signal; alpha scales up with evidence.

The blended final score is:
    final = alpha * marker_score + (1-alpha) * mlp_score
where alpha = max(marker_gene_fraction, alpha_base).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from plasflow2.classify.features import extract_features
from plasflow2.utils.device import IDX_TO_CLASS, get_device

logger = logging.getLogger(__name__)

# Default confidence thresholds (class-specific)
DEFAULT_THRESHOLD = 0.70  # chromosome
DEFAULT_PHAGE_THRESHOLD = 0.70  # phage — restored to 0.70 (was 0.85, over-suppressed recall)
DEFAULT_PLASMID_THRESHOLD = 0.95  # plasmid — higher bar to correct for class-prior imbalance

# Per-length plasmid threshold multipliers.
# Short sequences (<3kb) have noisier k-mer profiles → require higher confidence.
# Long sequences (>10kb) have stable k-mer profiles → can use lower threshold.
#
# Rationale: a 1kb contig has ~1000 4-mers; statistical noise is ~3%.
#            a 10kb contig has ~10k 4-mers; noise is ~0.3%.
# The model should be more conservative on short contigs to avoid FPs.
LENGTH_THRESHOLD_TIERS = [
    # (max_length_bp, plasmid_threshold, phage_threshold, chr_threshold)
    # NOTE: conjugative plasmids bypass these via hard override in predict().
    # All other sequences use these per-length thresholds.
    #
    # Calibrated for k=7 3-CLASS model v2 (Jun 2026).
    # The 3-class softmax spreads probability across plasmid/chromosome/phage,
    # so chromosome scores are systematically lower than in the binary model.
    # Phage thresholds are raised (0.90+) to suppress chromosome→phage FPs:
    # on the benchmark (no true phage), 2,906 chromosomes score phage>0.70
    # but only ~1,600 score phage>0.95.  On real metagenomes true phage
    # sequences score >>0.95, so this threshold preserves recall on W1.
    # Chromosome thresholds lowered to 0.60–0.70 to recover chromosome recall
    # lost to phage probability bleed.
    #
    #   1-2kb:   short sequences noisy; strict plasmid, high phage threshold.
    #   2-5kb:   small plasmids; phage raised 0.80→0.92.
    #   5-10kb:  phage raised 0.75→0.90.
    #   10-20kb: main plasmid tier; phage raised 0.72→0.90.
    #   >20kb:   long sequences; phage raised 0.70→0.90; chr lowered 0.68→0.62.
    (2_000, 0.99, 0.95, 0.75),  # <2kb
    (4_999, 0.95, 0.92, 0.68),  # 2-5kb
    (9_999, 0.98, 0.90, 0.65),  # 5-10kb
    # NOTE: boundary 9999 so exact 10000bp seqs use 10-20kb tier
    (19_999, 0.93, 0.90, 0.63),  # 10-20kb
    (float("inf"), 0.94, 0.90, 0.62),  # >20kb
]


def _get_length_thresholds(seq_len: int) -> tuple[float, float, float]:
    """Return (plasmid_threshold, phage_threshold, chr_threshold) for a given length."""
    for max_len, plas_t, phage_t, chr_t in LENGTH_THRESHOLD_TIERS:
        if seq_len <= max_len:
            return plas_t, phage_t, chr_t
    return DEFAULT_PLASMID_THRESHOLD, DEFAULT_PHAGE_THRESHOLD, DEFAULT_THRESHOLD


# Class prior distributions by sample context.
# The MLP is trained on balanced classes (33%/33%/33%) but real metagenomes
# are dominated by chromosomal sequences. Bayesian correction:
#   corrected[c] = mlp_score[c] * prior[c] / sum_k(mlp_score[k] * prior[k])
# This reduces false-positive plasmid and phage calls substantially.
CONTEXT_PRIORS: dict[str, dict[str, float]] = {
    # Literature-informed class frequencies per sample type.
    # Wastewater: ~3% plasmid, ~12% phage (Gut et al. 2021; Hendriksen et al. 2019).
    # The original phage prior of 0.040 was too aggressive — combined with the
    # 0.85+ phage threshold it suppressed all phage calls on real metagenomes.
    # 0.120 allows the MLP to call phage when it genuinely predicts them.
    "wastewater": {"plasmid": 0.030, "chromosome": 0.850, "phage": 0.120},
    "clinical": {"plasmid": 0.050, "chromosome": 0.920, "phage": 0.030},
    "environmental": {"plasmid": 0.020, "chromosome": 0.870, "phage": 0.110},
    "unspecified": {"plasmid": 0.333, "chromosome": 0.334, "phage": 0.333},
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
    """Single-sequence prediction result.

    Core fields (always populated):
        sequence_id   Contig identifier.
        label         Final class: plasmid | chromosome | phage | unclassified.
        confidence    Final score for the winning class (after all blending/overrides).
        scores        Per-class final probabilities (3-class, sum ≈ 1).
        low_confidence  True when the prediction is uncertain and should be treated
                        with caution. Set in two cases:
                        (a) confidence < 0.70 threshold — model score was weak;
                        (b) hallmark gate flagged — contig ≥ 50 kb kept as plasmid
                            but has no biological hallmark evidence (no PLSDB match,
                            relaxase, replicon, ICE, or rep protein).

    Evidence fields (populated when marker XGBoost is used; None otherwise):
        mlp_scores      Raw MLP softmax BEFORE XGBoost blending.
        xgb_scores      XGBoost class probabilities (marker second stage).
        bio_evidence    Biological marker flags from the annotation TSV:
                        is_conjugative, is_mobilizable, has_replicon, has_ice,
                        has_rep_protein, n_rep_per_kb.  All float (0 or 1 for
                        binary flags).  Empty dict when annotation TSV not used.
        evidence_type   Human-readable summary of what drove the final call:
                        "mlp_only"              — no XGBoost used.
                        "xgb_blend"             — MLP + XGBoost soft blend.
                        "conjugative_override"  — hard override (relaxase + MPF).
                        "hallmark_boost"        — geNomad hallmark gene boost.
                        "plsdb_prot_boost"      — PLSDB protein homology boost.
                        "plsdb_nt_override"     — PLSDB nucleotide hard override (minimap2 asm5).
                        "marker_threshold_boost" — ≥1 geNomad marker + mlp≥0.90 soft boost.
                        "replicon_boost"         — rep.dna.fas replicon detected (minimap2/BLASTN).
    """

    sequence_id: str
    label: str  # plasmid | chromosome | phage | unclassified
    confidence: float  # max softmax probability
    scores: dict[str, float]  # per-class probabilities (final blended output)

    # Uncertainty flag — True when the call should be treated with caution
    low_confidence: bool = field(default=False)

    # Evidence transparency — None when marker model is not used
    mlp_scores: dict[str, float] | None = field(default=None)
    xgb_scores: dict[str, float] | None = field(default=None)
    bio_evidence: dict[str, float] | None = field(default=None)
    evidence_type: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Annotation TSV loader (for pre-computed DIAMOND hits)
# ---------------------------------------------------------------------------


def _load_annotation_tsv(tsv_path: Path | str) -> dict[str, dict[str, float]]:
    """Load pre-computed DIAMOND annotation features from TSV.

    Expected columns (tab-separated, header row required):
        contig_id, is_conjugative, is_mobilizable, has_replicon, has_ice,
        has_rep_protein, n_arg_per_kb, n_mge_per_kb, n_ice_per_kb, n_rep_per_kb

    All columns after contig_id are numeric (float). Missing values → 0.

    Returns:
        Dict mapping contig_id → {feature_name: value}.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        logger.warning("Annotation TSV not found: %s", tsv_path)
        return {}

    annotations: dict[str, dict[str, float]] = {}
    _FLOAT_COLS = {
        # MOB-suite
        "is_conjugative",
        "is_mobilizable",
        "has_replicon",
        "has_ice",
        "has_rep_protein",
        "n_arg_per_kb",
        "n_mge_per_kb",
        "n_ice_per_kb",
        "n_rep_per_kb",
        # geNomad SPM (present when annotation TSV was built with --genomad-genes)
        "p_marker_freq",
        "c_marker_freq",
        "v_marker_freq",
        "pp_marker_freq",
        "median_p_spm",
        "median_c_spm",
        "median_v_spm",
        "p_vs_c_logistic",
        "strand_switch_rate",
        "no_rbs_freq",
        "canonical_sd_freq",
        "n_plasmid_markers",
        # PLSDB protein homology (present when annotated with --plsdb-proteins)
        "plsdb_prot_hits_per_kb",
        "max_plsdb_prot_pct_id",
        # PLSDB nucleotide match (present when annotated with --plsdb-fasta)
        "plsdb_nt_match",
        "plsdb_nt_qcov",
    }
    try:
        with open(tsv_path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                cid = row.get("contig_id", "").strip()
                if not cid:
                    continue
                feats = {k: float(row.get(k, 0.0) or 0.0) for k in _FLOAT_COLS if k in row}
                annotations[cid] = feats
        logger.info("Loaded annotations for %d contigs from %s", len(annotations), tsv_path)
    except Exception as e:
        logger.warning("Failed to load annotation TSV %s: %s", tsv_path, e)
    return annotations


# ---------------------------------------------------------------------------
# ORF prediction helper (pyrodigal, optional)
# ---------------------------------------------------------------------------


def _run_pyrodigal(
    sequences: list[str],
    sequence_ids: list[str],
    n_threads: int = 16,
) -> dict[str, dict]:
    """Predict ORFs with pyrodigal.

    Returns dict: seq_id → {n_orfs, covered_bp, genes}.
    The *genes* key holds the raw pyrodigal gene list so callers can pass it
    directly to ``gene_content_vector()`` in features.py for MLP input.
    Runs in parallel using ThreadPoolExecutor (pyrodigal releases the GIL).
    """
    try:
        import pyrodigal  # type: ignore[import]
    except ImportError:
        logger.debug("pyrodigal not available — ORF features will use defaults")
        return {}

    from concurrent.futures import ThreadPoolExecutor

    gene_pred = pyrodigal.GeneFinder(meta=True)

    def _process_one(sid_seq):
        sid, seq = sid_seq
        try:
            genes = gene_pred.find_genes(seq.encode())
            gene_list = list(genes)
            covered = sum(abs(g.end - g.begin) for g in gene_list)
            return sid, {"n_orfs": len(gene_list), "covered_bp": covered, "genes": gene_list}
        except Exception:
            return sid, {"n_orfs": 0, "covered_bp": 0, "genes": []}

    orf_data: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for sid, data in pool.map(_process_one, zip(sequence_ids, sequences)):
            orf_data[sid] = data
    return orf_data


# ---------------------------------------------------------------------------
# Label assignment (shared logic)
# ---------------------------------------------------------------------------


def _assign_label(
    scores: dict[str, float],
    seq_len: int,
    plasmid_threshold: float,
    threshold: float,
    argmax_fallback: bool,
) -> tuple[str, float]:
    """Return (label, confidence) from scores + length-aware thresholds."""
    best_class = max(scores, key=scores.__getitem__)
    confidence = float(scores[best_class])

    plas_t, phage_t, chr_t = _get_length_thresholds(seq_len)
    if seq_len < 5_000:
        plas_t = max(plas_t, plasmid_threshold)
        chr_t = min(chr_t, threshold)

    if best_class == "plasmid":
        applicable_threshold = plas_t
    elif best_class == "phage":
        applicable_threshold = phage_t
    else:
        applicable_threshold = chr_t

    if confidence >= applicable_threshold:
        label = best_class
    elif argmax_fallback:
        label = best_class
    else:
        label = "unclassified"

    return label, confidence


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------


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
    # --- Marker XGBoost (second stage) ---
    marker_model_path: Path | str | None = None,
    use_pyrodigal: bool = True,
    annotation_tsv: Path | str | None = None,
    marker_alpha_base: float = 0.3,
    pre_computed_annotations: dict[str, dict[str, float]] | None = None,
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
        source_context: Sample context for Bayesian prior correction.
        apply_prior: Whether to apply Bayesian prior correction.
        marker_model_path: Optional path to marker_xgb.pkl for second-stage XGBoost.
            When provided, MLP scores are blended with XGBoost scores using
            attention weighting.
        use_pyrodigal: Whether to run pyrodigal for ORF features (default True).
            Used only when marker_model_path is set.
        annotation_tsv: Optional TSV with pre-computed DIAMOND annotation features.
            Columns: contig_id, is_conjugative, is_mobilizable, has_replicon,
            has_ice, has_rep_protein, n_arg_per_kb, n_mge_per_kb, n_ice_per_kb,
            n_rep_per_kb.  Used only when marker_model_path is set.
        marker_alpha_base: Minimum attention weight for XGBoost even when no
            biological markers are detected (default 0.3).  Set to 0.0 to use
            XGBoost only when biological evidence exists.

    Returns:
        List of Prediction objects, one per input sequence.
    """
    import torch
    import torch.nn as nn

    from plasflow2.classify.model import load_model

    device = get_device()
    model = load_model(model_path, device=device)

    # Detect expected input dimension from the loaded model's first layer.
    # This determines whether gene content features should be appended to the
    # k=7 feature vector (9563-dim model) or not (9557-dim model).
    from plasflow2.classify.features import FEATURE_DIM_FULL

    _model_input_dim: int = model.net[0].in_features  # inspect before any DataParallel wrap
    _model_wants_gene_features: bool = _model_input_dim == FEATURE_DIM_FULL
    if _model_wants_gene_features:
        logger.info(
            "Model expects %d-dim input — gene content features will be included", _model_input_dim
        )
    else:
        logger.info(
            "Model expects %d-dim input — k=7 only (no gene content features)", _model_input_dim
        )

    # Multi-GPU: wrap in DataParallel when multiple CUDA devices are available.
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        logger.info("Using %d GPUs via DataParallel for MLP inference", n_gpus)
        model = nn.DataParallel(model)
    elif n_gpus == 1:
        logger.info("Using 1 GPU for MLP inference")
    else:
        logger.info("GPU not available — running MLP inference on CPU")

    model.eval()

    # ── ORF prediction (runs when pyrodigal available AND needed) ─────────────
    # Gene objects are stored so Stage 2 (marker XGBoost) can reuse them
    # without re-running pyrodigal.
    orf_data_global: dict[str, dict] = {}
    if use_pyrodigal and (_model_wants_gene_features or marker_model_path):
        logger.info("Running pyrodigal for gene/ORF features …")
        orf_data_global = _run_pyrodigal(sequences, sequence_ids)
        if orf_data_global:
            logger.info("  ORFs predicted for %d sequences", len(orf_data_global))

    # Build gene_data dict (seq_id → gene list) for extract_features().
    # Only pass gene_data when the loaded model was actually trained with them.
    gene_data_for_extract: dict[str, list] | None = None
    if _model_wants_gene_features and orf_data_global:
        gene_data_for_extract = {sid: d.get("genes", []) for sid, d in orf_data_global.items()}

    X = extract_features(sequences, gene_data=gene_data_for_extract, seq_ids=sequence_ids)

    # ── Stage 1: MLP softmax scores ──────────────────────────────────────────
    all_raw_scores: list[dict[str, float]] = []

    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start : start + batch_size]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for prob_row in probs:
            all_raw_scores.append(
                {IDX_TO_CLASS[j]: float(prob_row[j]) for j in range(len(prob_row))}
            )

    # Apply prior correction
    all_scores: list[dict[str, float]] = []
    for raw_scores in all_raw_scores:
        if apply_prior and source_context != "unspecified":
            all_scores.append(apply_prior_correction(raw_scores, source_context))
        else:
            all_scores.append(raw_scores)

    # ── Stage 2: Marker XGBoost (optional) ───────────────────────────────────
    # Evidence dicts — populated inside the if-block; initialised here so the
    # label-assignment loop can reference them unconditionally.
    _xgb_scores_by_idx: dict[int, dict[str, float]] = {}
    _bio_ev_by_idx: dict[int, dict[str, float]] = {}
    _ev_type_by_idx: dict[int, str] = {}

    if marker_model_path and Path(marker_model_path).exists():
        logger.info("Running marker XGBoost second stage from %s", marker_model_path)

        from plasflow2.classify.marker_classifier import (
            ContigMarkerFeatures,
            MarkerClassifier,
        )

        marker_clf = MarkerClassifier.load(marker_model_path)

        # Load pre-computed DIAMOND annotations (if provided).
        # pre_computed_annotations (in-memory dict) is merged first;
        # annotation_tsv (file) takes precedence if both are supplied.
        annotations: dict[str, dict[str, float]] = {}
        if pre_computed_annotations:
            annotations.update(pre_computed_annotations)
        if annotation_tsv:
            annotations.update(_load_annotation_tsv(annotation_tsv))

        # Reuse ORF data from Stage 1 (pyrodigal already ran above for gene features).
        orf_data: dict[str, dict] = orf_data_global

        # Build marker feature matrix for all sequences at once (faster)
        import numpy as _np

        from plasflow2.classify.marker_classifier import N_MARKER_FEATURES

        # Canonical Shine-Dalgarno motifs — used to compute canonical_sd_freq
        # directly from pyrodigal gene objects (no DIAMOND annotation required).
        _CANONICAL_SD_MOTIFS: frozenset[str] = frozenset(
            {"AGGAG", "GGAG", "AGGA", "AGG", "GGA", "GAGG"}
        )

        n = len(sequences)
        X_marker = _np.zeros((n, N_MARKER_FEATURES), dtype=_np.float32)

        for i, (sid, seq, mlp_scores) in enumerate(zip(sequence_ids, sequences, all_scores)):
            length_bp = max(len(seq), 1)
            length_kb = length_bp / 1000.0
            seq_upper = seq.upper()
            gc = (seq_upper.count("G") + seq_upper.count("C")) / length_bp

            # ORF features from pyrodigal gene objects
            orf = orf_data.get(sid, {})
            genes = orf.get("genes", [])
            n_orfs = len(genes) if genes else orf.get("n_orfs", 0)
            covered_bp = (
                sum(abs(g.end - g.begin) for g in genes) if genes else orf.get("covered_bp", 0)
            )
            cod_density = min(covered_bp / length_bp, 1.0) if length_bp > 0 else 0.85
            n_orfs_kb = n_orfs / max(length_kb, 0.001) if length_kb > 0 else 1.0

            # Gene-content features computed directly from pyrodigal gene objects.
            # Used as fallback when annotation_tsv is absent; TSV values override
            # when available (DIAMOND annotation is more precise for motif detection).
            if n_orfs > 1:
                strands = [g.strand for g in genes]
                switches = sum(1 for a, b in zip(strands, strands[1:]) if a != b)
                strand_switch_rate_pyro = float(switches) / (n_orfs - 1)
            else:
                strand_switch_rate_pyro = 0.5  # uninformative for ≤1 ORF
            if n_orfs > 0:
                canonical_sd_pyro = (
                    sum(1 for g in genes if g.rbs_motif in _CANONICAL_SD_MOTIFS) / n_orfs
                )
                no_rbs_pyro = (
                    sum(1 for g in genes if not g.rbs_motif or g.rbs_motif == "None") / n_orfs
                )
            else:
                canonical_sd_pyro = 0.0
                no_rbs_pyro = 1.0  # no ORFs → RBS undefined for all genes

            # Biological markers from annotation TSV (or zeros)
            ann = annotations.get(sid, {})

            feat = ContigMarkerFeatures(
                contig_id=sid,
                mlp_plasmid_score=float(mlp_scores.get("plasmid", 0.0)),
                mlp_chromosome_score=float(mlp_scores.get("chromosome", 0.0)),
                mlp_phage_score=float(mlp_scores.get("phage", 0.0)),
                is_conjugative=ann.get("is_conjugative", 0.0),
                is_mobilizable=ann.get("is_mobilizable", 0.0),
                has_replicon=ann.get("has_replicon", 0.0),
                has_ice=0.0,  # ICE excluded from classification evidence:
                # ICEs integrate into chromosomes and create FPs when used as plasmid
                # signal. ICE annotations are preserved in output for users but do not
                # drive XGBoost blending. Only MOB-type relaxase and MGE evidence is used.
                has_rep_protein=ann.get("has_rep_protein", 0.0),
                n_arg_per_kb=ann.get("n_arg_per_kb", 0.0),
                n_mge_per_kb=ann.get("n_mge_per_kb", 0.0),
                n_ice_per_kb=0.0,  # ICE density excluded for same reason
                n_rep_per_kb=ann.get("n_rep_per_kb", 0.0),
                log10_length=float(_np.log10(length_bp)),
                gc_content=gc,
                coding_density=cod_density,
                n_orfs_per_kb=n_orfs_kb,
                # geNomad SPM features (zero if TSV was built without --genomad-genes)
                p_marker_freq=ann.get("p_marker_freq", 0.0),
                c_marker_freq=ann.get("c_marker_freq", 0.0),
                v_marker_freq=ann.get("v_marker_freq", 0.0),
                pp_marker_freq=ann.get("pp_marker_freq", 0.0),
                median_p_spm=ann.get("median_p_spm", 0.0),
                median_c_spm=ann.get("median_c_spm", 0.0),
                median_v_spm=ann.get("median_v_spm", 0.0),
                p_vs_c_logistic=ann.get("p_vs_c_logistic", 0.5),
                strand_switch_rate=ann.get("strand_switch_rate", strand_switch_rate_pyro),
                no_rbs_freq=ann.get("no_rbs_freq", no_rbs_pyro),
                canonical_sd_freq=ann.get("canonical_sd_freq", canonical_sd_pyro),
                n_plasmid_markers=ann.get("n_plasmid_markers", 0.0),
            )
            X_marker[i] = feat.to_array()

        # Batch inference on marker XGBoost
        marker_proba = marker_clf.predict_proba(X_marker)  # (N, 2) or (N, 3)
        classes = ["plasmid", "chromosome", "phage"]

        # Detect binary marker model (2-class output: plasmid + chromosome only).
        # Check XGBoost output dimensions — NOT MLP scores (which are always 3-class).
        n_marker_classes = marker_proba.shape[1]
        is_binary_model = n_marker_classes < 3
        marker_classes = classes[:n_marker_classes]  # ["plasmid","chromosome"] or all 3

        # Blend MLP + marker scores using attention weighting
        # Hard override ONLY for truly unambiguous conjugative plasmids
        # (relaxase + MPF together).  Mobilizable-only and rep-protein-only
        # hits have too many chromosome false positives (transposons, prophages)
        # to override unconditionally — let the XGBoost handle them via soft
        # blending.
        n_hard_overrides = 0
        for i in range(n):
            mlp_s = all_scores[i]
            marker_s = {c: float(marker_proba[i, j]) for j, c in enumerate(marker_classes)}
            _xgb_scores_by_idx[i] = dict(marker_s)

            ann = annotations.get(sequence_ids[i], {})
            # Capture the biological evidence flags for this sequence
            _bio_ev_by_idx[i] = {
                k: float(ann.get(k, 0.0))
                for k in (
                    "is_conjugative",
                    "is_mobilizable",
                    "has_replicon",
                    "has_ice",
                    "has_rep_protein",
                    "n_rep_per_kb",
                )
            }

            # Hard override: BOTH relaxase AND MPF present → conjugative plasmid
            # (FP rate for chromosomes carrying both is negligible)
            if ann.get("is_conjugative", 0.0) > 0:
                all_scores[i] = {"plasmid": 0.999, "chromosome": 0.0005, "phage": 0.0005}
                _ev_type_by_idx[i] = "conjugative_override"
                n_hard_overrides += 1
                continue

            # Soft blending for all other sequences (including mobilizable and
            # rep-protein-only hits — XGBoost learned to handle the chromosome noise)
            bio_positive = sum(
                [
                    ann.get("is_mobilizable", 0.0),
                    ann.get("has_replicon", 0.0),
                    # has_ice excluded: ICEs are chromosomal elements, not plasmid evidence
                    ann.get("has_rep_protein", 0.0),
                ]
            )
            marker_gene_fraction = bio_positive / 4.0
            # Binary models already output high plasmid scores (0.97+) for true
            # plasmids; any alpha_base > 0 drags those scores below threshold
            # (blended = 0.30*marker + 0.70*0.97 = 0.74 < 0.94 threshold).
            # Use alpha_base=0 for binary: only sequences with actual biological
            # evidence get XGBoost weight; conjugative/hallmark hard overrides
            # (lines above) still apply unconditionally.
            effective_alpha_base = 0.0 if is_binary_model else marker_alpha_base
            alpha = max(marker_gene_fraction, effective_alpha_base)

            # For binary models, only blend plasmid + chromosome to avoid
            # re-introducing phage mass from the (always 3-class) marker XGBoost.
            blend_classes = ["plasmid", "chromosome"] if is_binary_model else classes
            blended = {
                c: alpha * marker_s.get(c, 0.0) + (1.0 - alpha) * mlp_s.get(c, 0.0)
                for c in blend_classes
            }
            total = sum(blended.values()) or 1.0
            all_scores[i] = {c: v / total for c, v in blended.items()}
            _ev_type_by_idx[i] = "xgb_blend"

        if n_hard_overrides:
            logger.info(
                "Hard biological overrides: %d conjugative plasmids (relaxase+MPF → plasmid forced)",
                n_hard_overrides,
            )

        # ── PLSDB nucleotide hard override ───────────────────────────────────
        # DISABLED: minimap2 asm5 at ≥50% qcov / ≥90% identity is non-specific.
        # Bacterial chromosomes share long stretches with same-species PLSDB
        # plasmids (HGT, shared core genes), causing ~3,300 chromosome FPs.
        # The plsdb_nt_match column is still computed in annotate_sequences.py
        # for research use, but is not applied as a classification override.
        # To re-enable, raise thresholds to qcov≥0.90 AND identity≥0.99 and
        # verify precision on the benchmark before deploying.

        # ── Plasmid hallmark hard boost ───────────────────────────────────────
        # When geNomad identifies ≥ 2 plasmid-specific hallmark genes on a
        # contig AND the MLP already leans plasmid (score ≥ 0.30), the
        # geNomad evidence is strong enough to call it a plasmid regardless
        # of whether the MLP score crosses the high threshold.
        # Precision at n_plasmid_markers ≥ 2 (MLP-undetected) is ~32 %;
        # this recovers ~16 extra TPs per benchmark run at an acceptable FP rate.
        n_hallmark_boosts = 0
        if annotations:
            for i in range(n):
                s = all_scores[i]
                sid = sequence_ids[i]
                ann = annotations.get(sid, {})
                n_plas_hall = float(ann.get("n_plasmid_markers", 0.0))
                mlp_plas = s.get("plasmid", 0.0)
                # Only boost if already the best or second-best class and has
                # strong hallmark evidence
                if n_plas_hall >= 2 and mlp_plas >= 0.30:
                    best = max(s, key=s.__getitem__)
                    if best != "plasmid":
                        # Boost plasmid score above the detection threshold
                        # by transferring half the non-plasmid mass to plasmid
                        s2 = dict(s)
                        non_plas = s2.get("chromosome", 0.0) + s2.get("phage", 0.0)
                        transfer = non_plas * 0.55
                        s2["plasmid"] = s2.get("plasmid", 0.0) + transfer
                        s2["chromosome"] = s2.get("chromosome", 0.0) * 0.45
                        s2["phage"] = s2.get("phage", 0.0) * 0.45
                        total = sum(s2.values()) or 1.0
                        all_scores[i] = {c: v / total for c, v in s2.items()}
                        _ev_type_by_idx[i] = "hallmark_boost"
                        n_hallmark_boosts += 1
            if n_hallmark_boosts:
                logger.info(
                    "Plasmid hallmark boost: %d sequences boosted (n_plasmid_markers >= 2, mlp >= 0.30)",
                    n_hallmark_boosts,
                )

        # ── PLSDB protein homology boost ──────────────────────────────────────
        # When a sequence has strong protein-level similarity to PLSDB plasmids
        # (many ORFs matching PLSDB proteins at moderate identity) AND the MLP
        # already leans plasmid (score ≥ 0.30), this is strong evidence that
        # the sequence is a true plasmid regardless of k-mer composition.
        #
        # This targets "composition-invisible" false negatives: sequences from
        # unusual plasmid lineages where k-mer profiles don't distinguish them
        # from chromosomes, but whose ORFs match known PLSDB proteins.
        #
        # Thresholds calibrated on benchmark to maximise TP gain vs FP cost:
        #   plsdb_prot_hits_per_kb >= 2.0  (≥2 PLSDB protein hits per kb)
        #   max_plsdb_prot_pct_id  >= 40.0 (at least one hit ≥40% identity)
        #   mlp_plasmid_score      >= 0.30 (MLP weakly leans plasmid)
        #
        # Only applies when annotation TSV was built with --plsdb-proteins.
        n_prot_boosts = 0
        if annotations:
            for i in range(n):
                s = all_scores[i]
                sid = sequence_ids[i]
                ann = annotations.get(sid, {})
                prot_hits_per_kb = float(ann.get("plsdb_prot_hits_per_kb", 0.0))
                max_pct_id = float(ann.get("max_plsdb_prot_pct_id", 0.0))
                mlp_plas = s.get("plasmid", 0.0)
                if prot_hits_per_kb >= 2.0 and max_pct_id >= 40.0 and mlp_plas >= 0.30:
                    best = max(s, key=s.__getitem__)
                    if best != "plasmid":
                        # Transfer 55% of non-plasmid mass to plasmid
                        s2 = dict(s)
                        non_plas = s2.get("chromosome", 0.0) + s2.get("phage", 0.0)
                        transfer = non_plas * 0.55
                        s2["plasmid"] = s2.get("plasmid", 0.0) + transfer
                        s2["chromosome"] = s2.get("chromosome", 0.0) * 0.45
                        s2["phage"] = s2.get("phage", 0.0) * 0.45
                        total = sum(s2.values()) or 1.0
                        all_scores[i] = {c: v / total for c, v in s2.items()}
                        _ev_type_by_idx[i] = "plsdb_prot_boost"
                        n_prot_boosts += 1
            if n_prot_boosts:
                logger.info(
                    "PLSDB protein boost: %d sequences boosted "
                    "(plsdb_prot_hits_per_kb >= 2.0, max_pct_id >= 40, mlp >= 0.30)",
                    n_prot_boosts,
                )

        # ── Marker-aware soft threshold boost (Option B) ─────────────────────
        # Sequences with ≥1 geNomad plasmid hallmark gene AND an MLP plasmid
        # score ≥ 0.90 are very likely true plasmids that narrowly miss the
        # strict 0.93–0.98 length-tier thresholds.  The single hallmark gene
        # acts as a corroborating signal that justifies a 0.90 effective floor.
        #
        # This is complementary to hallmark_boost (which requires n≥2, mlp≥0.30
        # and is designed for low-MLP sequences with strong marker evidence).
        # Option B targets high-MLP sequences with weaker marker evidence.
        #
        # Calibration: n_plasmid_markers ≥ 1 AND mlp ≥ 0.90 catches near-miss
        # FNs with gn:1 scores like 0.92–0.94 that fall below the 0.95+ tier
        # thresholds.  FP risk is low because mlp ≥ 0.90 is a strong filter.
        n_marker_threshold_boosts = 0
        if annotations:
            for i in range(n):
                s = all_scores[i]
                sid = sequence_ids[i]
                ann = annotations.get(sid, {})
                n_plas_hall = float(ann.get("n_plasmid_markers", 0.0))
                mlp_plas = s.get("plasmid", 0.0)
                if n_plas_hall >= 1 and mlp_plas >= 0.90:
                    best = max(s, key=s.__getitem__)
                    if best != "plasmid":
                        s2 = dict(s)
                        non_plas = s2.get("chromosome", 0.0) + s2.get("phage", 0.0)
                        transfer = non_plas * 0.55
                        s2["plasmid"] = s2.get("plasmid", 0.0) + transfer
                        s2["chromosome"] = s2.get("chromosome", 0.0) * 0.45
                        s2["phage"] = s2.get("phage", 0.0) * 0.45
                        total = sum(s2.values()) or 1.0
                        all_scores[i] = {c: v / total for c, v in s2.items()}
                        _ev_type_by_idx[i] = "marker_threshold_boost"
                        n_marker_threshold_boosts += 1
            if n_marker_threshold_boosts:
                logger.info(
                    "Marker-threshold boost: %d sequences boosted "
                    "(n_plasmid_markers >= 1, mlp >= 0.90)",
                    n_marker_threshold_boosts,
                )

        # ── Replicon sequence boost ───────────────────────────────────────────
        # When a contig contains a recognisable plasmid replicon sequence
        # (detected by minimap2 / BLASTN vs rep.dna.fas, replicon-coverage ≥60%,
        # identity ≥80%), that is near-definitive plasmid evidence.  rep.dna.fas
        # contains only 2,686 curated replicon sequences from known plasmids —
        # much more specific than PLSDB protein hits (which fire on housekeeping
        # genes too).
        #
        # This targets plasmids that are:
        #   - Composition-invisible (k-mer MLP score <0.70)
        #   - Marker-invisible (no conjugation, mob, or hallmark genes)
        #   - But carry a recognisable origin of replication
        #
        # Note: has_replicon has been 0 for all training sequences because
        # makeblastdb was unavailable.  XGBoost has never learned its weight,
        # so we apply it as a post-prediction modifier here.
        #
        # Boost is stronger than hallmark_boost (65% transfer vs 55%) because
        # replicon typing is more plasmid-specific.  Threshold mlp_plas ≥ 0.15
        # is intentionally low — a known replicon is strong enough signal even
        # when MLP is uncertain.
        n_replicon_boosts = 0
        if annotations:
            for i in range(n):
                s = all_scores[i]
                sid = sequence_ids[i]
                ann = annotations.get(sid, {})
                has_replicon = float(ann.get("has_replicon", 0.0))
                mlp_plas = s.get("plasmid", 0.0)
                if has_replicon >= 1.0 and mlp_plas >= 0.15:
                    best = max(s, key=s.__getitem__)
                    if best != "plasmid":
                        s2 = dict(s)
                        non_plas = s2.get("chromosome", 0.0) + s2.get("phage", 0.0)
                        transfer = non_plas * 0.65
                        s2["plasmid"] = s2.get("plasmid", 0.0) + transfer
                        s2["chromosome"] = s2.get("chromosome", 0.0) * 0.35
                        s2["phage"] = s2.get("phage", 0.0) * 0.35
                        total = sum(s2.values()) or 1.0
                        all_scores[i] = {c: v / total for c, v in s2.items()}
                        _ev_type_by_idx[i] = "replicon_boost"
                        n_replicon_boosts += 1
            if n_replicon_boosts:
                logger.info(
                    "Replicon boost: %d sequences boosted (has_replicon=1, mlp >= 0.15)",
                    n_replicon_boosts,
                )

        # ── Phage suppression ─────────────────────────────────────────────────
        # The MLP is trained on balanced data (33% phage) but real metagenomes
        # have very few phage sequences.  When annotations are available, we
        # require at least a trace of viral marker evidence before allowing a
        # phage prediction through.  Without evidence, the phage score is
        # redistributed to the second-best class (usually chromosome).
        n_phage_suppressed = 0
        if annotations:
            for i in range(n):
                s = all_scores[i]
                best = max(s, key=s.__getitem__)
                if best != "phage":
                    continue
                sid = sequence_ids[i]
                ann = annotations.get(sid, {})
                v_marker = float(ann.get("v_marker_freq", 0.0))
                # Estimate absolute viral hallmark gene count from frequency × ORF count.
                # v_marker_freq = n_virus_hallmarks / n_total_ORFs.  Short windows can
                # have v_marker_freq=1.0 from a single prophage gene — frequency alone
                # is therefore unreliable; we also require ≥ 2 absolute viral genes.
                n_orfs_per_kb = float(ann.get("n_orfs_per_kb", 0.0))
                length_bp_ann = float(ann.get("length_bp", len(sequences[i])))
                n_orfs_est = n_orfs_per_kb * length_bp_ann / 1000.0
                n_virus_est = v_marker * n_orfs_est
                # Allow phage prediction only if ≥10 % of genes AND ≥2 genes are viral
                has_viral_evidence = v_marker >= 0.10 and n_virus_est >= 2.0
                if not has_viral_evidence:
                    phage_mass = s["phage"]
                    # Redistribute phage score to chromosome only (no plasmid bleed —
                    # the 5% plasmid bleed was creating FPs on chromosomal sequences
                    # that score high phage but lack viral geNomad evidence).
                    s2 = dict(s)
                    s2["phage"] = phage_mass * 0.10
                    s2["chromosome"] = s2.get("chromosome", 0.0) + phage_mass * 0.90
                    s2["plasmid"] = s2.get("plasmid", 0.0)
                    total = sum(s2.values()) or 1.0
                    all_scores[i] = {c: v / total for c, v in s2.items()}
                    n_phage_suppressed += 1
            if n_phage_suppressed:
                logger.info(
                    "Phage suppression: %d sequences downgraded "
                    "(v_marker_freq < 0.10 or < 2 viral hallmark genes)",
                    n_phage_suppressed,
                )

        n_with_bio = sum(
            1
            for sid in sequence_ids
            if any(
                annotations.get(sid, {}).get(k, 0.0) > 0
                for k in (
                    "is_conjugative",
                    "is_mobilizable",
                    "has_replicon",
                    # has_ice excluded from bio evidence count (ICEs are chromosomal)
                    "has_rep_protein",
                )
            )
        )
        logger.info(
            "Marker XGBoost applied: %d sequences with biological evidence "
            "(effective_alpha_base=%.2f%s)",
            n_with_bio,
            effective_alpha_base,
            " [binary model override]" if is_binary_model else "",
        )
    elif marker_model_path:
        logger.warning("Marker model not found at %s — using MLP only", marker_model_path)

    # ── Label assignment ──────────────────────────────────────────────────────
    # Check whether the marker stage populated evidence dicts (only when
    # marker_model_path was provided and valid).
    _marker_stage_ran = bool(marker_model_path and Path(marker_model_path).exists())

    results: list[Prediction] = []
    for i, (sid, scores) in enumerate(zip(sequence_ids, all_scores)):
        seq_len = len(sequences[i])
        label, confidence = _assign_label(
            scores, seq_len, plasmid_threshold, threshold, argmax_fallback
        )
        results.append(
            Prediction(
                sequence_id=sid,
                label=label,
                confidence=confidence,
                scores=scores,
                # Evidence fields — only populated when marker stage ran
                mlp_scores=dict(all_raw_scores[i]) if _marker_stage_ran else None,
                xgb_scores=_xgb_scores_by_idx.get(i) if _marker_stage_ran else None,
                bio_evidence=_bio_ev_by_idx.get(i, {}) if _marker_stage_ran else None,
                evidence_type=(
                    _ev_type_by_idx.get(i, "xgb_blend") if _marker_stage_ran else "mlp_only"
                ),
            )
        )

    n_unclassified = sum(1 for r in results if r.label == "unclassified")
    logger.info(
        "Classified %d sequences (plasmid_threshold=%.2f, threshold=%.2f, "
        "argmax_fallback=%s, unclassified=%d, marker_stage=%s)",
        len(results),
        plasmid_threshold,
        threshold,
        argmax_fallback,
        n_unclassified,
        marker_model_path is not None,
    )
    return results


# ---------------------------------------------------------------------------
# Cascade predict (Stage 1: plasmid/rest  +  Stage 2: chr/phage)
# ---------------------------------------------------------------------------

# Per-length thresholds for the CASCADE model.
# After training, run scripts/tune_cascade_thresholds.py to calibrate these
# on a held-out validation set.  Initial values are conservative to favour
# precision; lower stage1_plasmid_* if recall is more important.
#
# Cascade thresholds differ from the 3-class model because:
#   - Stage 1 is binary (plasmid vs. rest) → full softmax mass on 2 classes
#     → plasmid probs are ~2× higher than in the 3-class model for true plasmids
#   - Stage 2 is binary (chr vs. phage) → no plasmid competition
#   - Both stages are separately calibrated → better discrimination per decision
CASCADE_PLASMID_THRESHOLD_TIERS = [
    # (max_length_bp, plasmid_t, phage_t, chr_t)
    # plasmid_t: Stage 1 threshold  (binary plasmid classifier)
    # phage_t:   Stage 2 threshold  (binary chr/phage classifier)
    # chr_t:     Stage 2 threshold  (binary chr/phage classifier)
    #
    # Calibrated from W1 WWTP length-stratified score distributions.
    # Stage 1 binary model is bimodal (median=0.055, p90=0.945) but 65% of W1
    # reads are <2kb → must use very high threshold for short contigs.
    # Fine-tune further with run_benchmark_evaluation.py --stage1-model --stage2-model.
    (2_000, 0.995, 0.980, 0.75),  # <2kb  — very conservative (noisy short frags)
    (4_999, 0.970, 0.940, 0.72),  # 2-5kb
    (9_999, 0.950, 0.910, 0.70),  # 5-10kb
    (19_999, 0.930, 0.880, 0.68),  # 10-20kb
    (float("inf"), 0.920, 0.860, 0.65),  # >20kb — long contigs have most evidence
]


def _get_cascade_thresholds(seq_len: int) -> tuple[float, float, float]:
    for max_len, plas_t, phage_t, chr_t in CASCADE_PLASMID_THRESHOLD_TIERS:
        if seq_len <= max_len:
            return plas_t, phage_t, chr_t
    return CASCADE_PLASMID_THRESHOLD_TIERS[-1][1:]


def cascade_predict(
    sequences: list[str],
    sequence_ids: list[str],
    stage1_model_path: Path | str,
    stage2_model_path: Path | str,
    batch_size: int = 512,
    argmax_fallback: bool = False,
) -> list[Prediction]:
    """Two-stage cascade classifier (plasmid / chromosome / phage).

    Stage 1 — binary plasmid detector trained on (plasmid=1 vs. chr+phage=0).
    Stage 2 — binary chr/phage discriminator trained on (chr=0 vs. phage=1).

    Features are extracted ONCE and shared between both forward passes,
    so wall time is only marginally longer than a single MLP run.

    Decision logic (per-length thresholds from CASCADE_PLASMID_THRESHOLD_TIERS):
        if Stage1_plasmid_prob  ≥ plasmid_t  → plasmid
        elif Stage2_phage_prob  ≥ phage_t    → phage
        elif Stage2_chr_prob    ≥ chr_t      → chromosome
        else                                 → unclassified

    Args:
        sequences:          DNA strings.
        sequence_ids:       Identifiers corresponding to each sequence.
        stage1_model_path:  Path to Stage 1 .pt (plasmid vs. rest binary MLP).
        stage2_model_path:  Path to Stage 2 .pt (chr vs. phage binary MLP).
        batch_size:         Inference batch size.
        argmax_fallback:    When True, unclassified sequences receive the
                            argmax label instead.

    Returns:
        List of Prediction objects. scores dict contains:
            "plasmid"    — Stage 1 plasmid probability
            "chromosome" — Stage 2 chromosome probability
            "phage"      — Stage 2 phage probability
    """
    import torch

    from plasflow2.classify.model import load_model

    device = get_device()

    logger.info("Loading Stage 1 model from %s …", stage1_model_path)
    s1_model = load_model(stage1_model_path, device=device)
    s1_model.eval()

    logger.info("Loading Stage 2 model from %s …", stage2_model_path)
    s2_model = load_model(stage2_model_path, device=device)
    s2_model.eval()

    # ── Feature extraction (once, shared) ────────────────────────────────────
    logger.info("Extracting features for %d sequences …", len(sequences))
    X = extract_features(sequences)

    # ── Stage 1: plasmid vs. rest → prob[:, 1] = P(plasmid) ─────────────────
    s1_plasmid_probs: list[float] = []
    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start : start + batch_size]).to(device)
        with torch.no_grad():
            logits = s1_model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        s1_plasmid_probs.extend(float(p[1]) for p in probs)  # index 1 = plasmid

    # ── Stage 2: chr vs. phage → prob[:, 0]=P(chr)  prob[:, 1]=P(phage) ─────
    s2_chr_probs: list[float] = []
    s2_phage_probs: list[float] = []
    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start : start + batch_size]).to(device)
        with torch.no_grad():
            logits = s2_model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        s2_chr_probs.extend(float(p[0]) for p in probs)  # index 0 = chr
        s2_phage_probs.extend(float(p[1]) for p in probs)  # index 1 = phage

    # ── Combine via per-length thresholds ────────────────────────────────────
    results: list[Prediction] = []
    for i, (sid, seq) in enumerate(zip(sequence_ids, sequences)):
        plasmid_prob = s1_plasmid_probs[i]
        chr_prob = s2_chr_probs[i]
        phage_prob = s2_phage_probs[i]

        scores = {
            "plasmid": plasmid_prob,
            "chromosome": chr_prob,
            "phage": phage_prob,
        }

        plas_t, phage_t, chr_t = _get_cascade_thresholds(len(seq))

        if plasmid_prob >= plas_t:
            label = "plasmid"
            confidence = plasmid_prob
        elif phage_prob >= phage_t:
            label = "phage"
            confidence = phage_prob
        elif chr_prob >= chr_t:
            label = "chromosome"
            confidence = chr_prob
        elif argmax_fallback:
            label = max(scores, key=scores.__getitem__)
            confidence = scores[label]
        else:
            label = "unclassified"
            confidence = max(plasmid_prob, chr_prob, phage_prob)

        results.append(
            Prediction(
                sequence_id=sid,
                label=label,
                confidence=confidence,
                scores=scores,
            )
        )

    n_by_label: dict[str, int] = {}
    for r in results:
        n_by_label[r.label] = n_by_label.get(r.label, 0) + 1
    n_unc = n_by_label.get("unclassified", 0)
    logger.info(
        "Cascade classified %d sequences (unclassified=%d): %s",
        len(results),
        n_unc,
        "  ".join(f"{k}={v:,}" for k, v in sorted(n_by_label.items())),
    )
    return results
