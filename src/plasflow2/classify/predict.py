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
from dataclasses import dataclass
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
    (2_000, 0.99, 0.95, 0.75),   # <2kb
    (4_999, 0.95, 0.92, 0.68),   # 2-5kb
    (9_999, 0.98, 0.90, 0.65),   # 5-10kb
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
    """Single-sequence prediction result."""

    sequence_id: str
    label: str  # plasmid | chromosome | phage | archaea | unclassified
    # Note: 'archaea' is assigned post-classification by the pipeline,
    # not by the MLP itself.
    confidence: float  # max softmax probability
    scores: dict[str, float]  # per-class probabilities (3-class MLP output)


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
) -> dict[str, dict]:
    """Predict ORFs with pyrodigal. Returns dict: seq_id → {n_orfs, covered_bp}."""
    try:
        import pyrodigal  # type: ignore[import]
    except ImportError:
        logger.debug("pyrodigal not available — ORF features will use defaults")
        return {}

    gene_pred = pyrodigal.GeneFinder(meta=True)
    orf_data: dict[str, dict] = {}
    for sid, seq in zip(sequence_ids, sequences):
        try:
            genes = gene_pred.find_genes(seq.encode())
            covered = sum(abs(g.end - g.begin) for g in genes)
            orf_data[sid] = {"n_orfs": len(genes), "covered_bp": covered}
        except Exception:
            orf_data[sid] = {"n_orfs": 0, "covered_bp": 0}
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

    X = extract_features(sequences)

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
    if marker_model_path and Path(marker_model_path).exists():
        logger.info("Running marker XGBoost second stage from %s", marker_model_path)

        from plasflow2.classify.marker_classifier import (
            ContigMarkerFeatures,
            MarkerClassifier,
        )

        marker_clf = MarkerClassifier.load(marker_model_path)

        # Load pre-computed DIAMOND annotations (if provided)
        annotations: dict[str, dict[str, float]] = {}
        if annotation_tsv:
            annotations = _load_annotation_tsv(annotation_tsv)

        # ORF prediction via pyrodigal (optional, for coding density / ORF count)
        orf_data: dict[str, dict] = {}
        if use_pyrodigal:
            logger.info("  Predicting ORFs with pyrodigal …")
            orf_data = _run_pyrodigal(sequences, sequence_ids)
            if orf_data:
                logger.info("  ORFs predicted for %d sequences", len(orf_data))

        # Build marker feature matrix for all sequences at once (faster)
        import numpy as _np

        from plasflow2.classify.marker_classifier import N_MARKER_FEATURES

        n = len(sequences)
        X_marker = _np.zeros((n, N_MARKER_FEATURES), dtype=_np.float32)

        for i, (sid, seq, mlp_scores) in enumerate(zip(sequence_ids, sequences, all_scores)):
            length_bp = max(len(seq), 1)
            length_kb = length_bp / 1000.0
            seq_upper = seq.upper()
            gc = (seq_upper.count("G") + seq_upper.count("C")) / length_bp

            # ORF features
            orf = orf_data.get(sid, {})
            n_orfs = orf.get("n_orfs", 0)
            covered_bp = orf.get("covered_bp", 0)
            cod_density = min(covered_bp / length_bp, 1.0) if covered_bp > 0 else 0.85
            n_orfs_kb = n_orfs / max(length_kb, 0.001) if n_orfs > 0 else 1.0

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
                has_ice=ann.get("has_ice", 0.0),
                has_rep_protein=ann.get("has_rep_protein", 0.0),
                n_arg_per_kb=ann.get("n_arg_per_kb", 0.0),
                n_mge_per_kb=ann.get("n_mge_per_kb", 0.0),
                n_ice_per_kb=ann.get("n_ice_per_kb", 0.0),
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
                strand_switch_rate=ann.get("strand_switch_rate", 0.0),
                no_rbs_freq=ann.get("no_rbs_freq", 0.0),
                canonical_sd_freq=ann.get("canonical_sd_freq", 0.0),
                n_plasmid_markers=ann.get("n_plasmid_markers", 0.0),
            )
            X_marker[i] = feat.to_array()

        # Batch inference on marker XGBoost
        marker_proba = marker_clf.predict_proba(X_marker)  # (N, 3)
        classes = ["plasmid", "chromosome", "phage"]

        # Detect binary model (2-class: plasmid + chromosome, no phage).
        # Binary models lack "phage" in their score dicts; we must not let
        # the 3-class marker XGBoost re-introduce spurious phage mass.
        is_binary_model = "phage" not in (all_scores[0] if all_scores else {})

        # Blend MLP + marker scores using attention weighting
        # Hard override ONLY for truly unambiguous conjugative plasmids
        # (relaxase + MPF together).  Mobilizable-only and rep-protein-only
        # hits have too many chromosome false positives (transposons, prophages)
        # to override unconditionally — let the XGBoost handle them via soft
        # blending.
        n_hard_overrides = 0
        for i in range(n):
            mlp_s = all_scores[i]
            marker_s = {c: float(marker_proba[i, j]) for j, c in enumerate(classes)}

            ann = annotations.get(sequence_ids[i], {})

            # Hard override: BOTH relaxase AND MPF present → conjugative plasmid
            # (FP rate for chromosomes carrying both is negligible)
            if ann.get("is_conjugative", 0.0) > 0:
                all_scores[i] = {"plasmid": 0.999, "chromosome": 0.0005, "phage": 0.0005}
                n_hard_overrides += 1
                continue

            # Soft blending for all other sequences (including mobilizable and
            # rep-protein-only hits — XGBoost learned to handle the chromosome noise)
            bio_positive = sum(
                [
                    ann.get("is_mobilizable", 0.0),
                    ann.get("has_replicon", 0.0),
                    ann.get("has_ice", 0.0),
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

        if n_hard_overrides:
            logger.info(
                "Hard biological overrides: %d conjugative plasmids (relaxase+MPF → plasmid forced)",
                n_hard_overrides,
            )

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
                        n_hallmark_boosts += 1
            if n_hallmark_boosts:
                logger.info(
                    "Plasmid hallmark boost: %d sequences boosted (n_plasmid_markers >= 2, mlp >= 0.30)",
                    n_hallmark_boosts,
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
                    # Redistribute phage score to chromosome (85%) and plasmid (5%)
                    s2 = dict(s)
                    s2["phage"] = phage_mass * 0.10
                    s2["chromosome"] = s2.get("chromosome", 0.0) + phage_mass * 0.85
                    s2["plasmid"] = s2.get("plasmid", 0.0) + phage_mass * 0.05
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
                    "has_ice",
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
    (2_000,        0.995, 0.980, 0.75),  # <2kb  — very conservative (noisy short frags)
    (4_999,        0.970, 0.940, 0.72),  # 2-5kb
    (9_999,        0.950, 0.910, 0.70),  # 5-10kb
    (19_999,       0.930, 0.880, 0.68),  # 10-20kb
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
        batch = torch.tensor(X[start: start + batch_size]).to(device)
        with torch.no_grad():
            logits = s1_model(batch)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        s1_plasmid_probs.extend(float(p[1]) for p in probs)  # index 1 = plasmid

    # ── Stage 2: chr vs. phage → prob[:, 0]=P(chr)  prob[:, 1]=P(phage) ─────
    s2_chr_probs:   list[float] = []
    s2_phage_probs: list[float] = []
    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start: start + batch_size]).to(device)
        with torch.no_grad():
            logits = s2_model(batch)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        s2_chr_probs.extend(float(p[0]) for p in probs)    # index 0 = chr
        s2_phage_probs.extend(float(p[1]) for p in probs)  # index 1 = phage

    # ── Combine via per-length thresholds ────────────────────────────────────
    results: list[Prediction] = []
    for i, (sid, seq) in enumerate(zip(sequence_ids, sequences)):
        plasmid_prob = s1_plasmid_probs[i]
        chr_prob     = s2_chr_probs[i]
        phage_prob   = s2_phage_probs[i]

        scores = {
            "plasmid":    plasmid_prob,
            "chromosome": chr_prob,
            "phage":      phage_prob,
        }

        plas_t, phage_t, chr_t = _get_cascade_thresholds(len(seq))

        if plasmid_prob >= plas_t:
            label      = "plasmid"
            confidence = plasmid_prob
        elif phage_prob >= phage_t:
            label      = "phage"
            confidence = phage_prob
        elif chr_prob >= chr_t:
            label      = "chromosome"
            confidence = chr_prob
        elif argmax_fallback:
            label      = max(scores, key=scores.__getitem__)
            confidence = scores[label]
        else:
            label      = "unclassified"
            confidence = max(plasmid_prob, chr_prob, phage_prob)

        results.append(Prediction(
            sequence_id=sid,
            label=label,
            confidence=confidence,
            scores=scores,
        ))

    n_by_label: dict[str, int] = {}
    for r in results:
        n_by_label[r.label] = n_by_label.get(r.label, 0) + 1
    n_unc = n_by_label.get("unclassified", 0)
    logger.info(
        "Cascade classified %d sequences (unclassified=%d): %s",
        len(results), n_unc,
        "  ".join(f"{k}={v:,}" for k, v in sorted(n_by_label.items())),
    )
    return results
