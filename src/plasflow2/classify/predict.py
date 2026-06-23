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
DEFAULT_PHAGE_THRESHOLD = 0.85  # phage — raised from 0.70 to suppress FPs on chr fragments
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
    # Calibrated for k=7 BINARY model v2 (Jun 2026) with FP hard-negative
    # retrain (29 difficult-chromosome genomes added as hard negatives).
    # Sweep via tune_thresholds_binary.py on fresh benchmark predictions
    # (--no-marker-model, 60k hard-negative budget, 50 epochs, val_acc=0.957):
    #
    #   1-2kb:   0 plasmids in benchmark → 0.99 unchanged
    #   2-5kb:   5 plasmids; opt=0.97 (TP=3, FP=526, P=0.006) — poor precision.
    #            Kept at 0.95 (TP=4, FP higher but recall better). Unchanged.
    #   5-10kb:  4 plasmids; opt=0.97 (TP=1, FP=116) — still bad.
    #            Kept at 0.98 (TP=0, FP=0) to avoid flood of FPs for 1 TP.
    #   10-20kb: 379 plasmids; opt=0.93 (TP=293, FP=57, F1=0.804).
    #            Hard-negative training cut FPs from 223→57. Lowered 0.94→0.93.
    #   >20kb:   6 plasmids; opt=0.94 (TP=4, FP=12, F1=0.364).
    #            Lowered 0.97→0.94 for better recall on long plasmids.
    (2_000, 0.99, 0.95, 0.80),  # <2kb:   very strict (no benchmark plasmids here)
    (4_999, 0.95, 0.90, 0.75),  # 2-5kb:  chr p99=0.978 makes any threshold costly
    (9_999, 0.98, 0.87, 0.72),  # 5-10kb: keep strict; 0.97 adds 116 FPs for 1 TP
    # NOTE: boundary 9999 so exact 10000bp seqs use 10-20kb tier
    (19_999, 0.93, 0.85, 0.70),  # 10-20kb: 0.94→0.93 (hard-neg retrain, FP: 223→57)
    (float("inf"), 0.94, 0.82, 0.68),  # >20kb: 0.97→0.94 (TP: 3→4, F1: 0.286→0.364)
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
    "wastewater": {"plasmid": 0.030, "chromosome": 0.930, "phage": 0.040},
    "clinical": {"plasmid": 0.050, "chromosome": 0.900, "phage": 0.050},
    "environmental": {"plasmid": 0.020, "chromosome": 0.950, "phage": 0.030},
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
