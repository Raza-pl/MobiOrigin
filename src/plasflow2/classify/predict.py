"""Inference: run 3-class MLP classifier on sequences.

Classes: plasmid | chromosome | phage
Archaea is NOT predicted here — it is detected post-classification in the
pipeline by comparing archaeal vs bacterial ORF hits from DIAMOND taxonomy
(paper criteria: archaea_hits > bacteria_hits AND archaea_hits >= 5).

Class-specific thresholds
-------------------------
The MLP is trained on a balanced dataset (~33 % per class), but real
metagenome assemblies contain only ~2–5 % plasmid contigs.  We apply
*class-specific, length-tiered* thresholds to compensate (see
``LENGTH_THRESHOLD_TIERS``).

``plasmid_threshold`` and ``threshold`` are ``None`` by default. That does
NOT mean "use the tier profile at every length" — it means "reproduce
today's actually-shipped default behavior," which is a legacy quirk: the
CLI's historical default (0.95 / 0.70) only ever applied below 5 kb; ≥5 kb
sequences always used the tier profile untouched. That quirk is preserved
on purpose (see ``_assign_label`` docstring for why — short version: the
tier profile's <5kb values were never validated as standalone thresholds,
and applying them "correctly" by default was measured to collapse plasmid
precision from 0.777 to 0.224 on the Tier 1 benchmark). Passing an
*explicit* float (e.g. via ``--plasmid-threshold`` / ``--lenient``) overrides
the tier value at ALL lengths, in whichever direction the caller asks for —
that part WAS a real bug (an explicit lower value was silently discarded by
a ``max()`` against the tier default, since tier defaults ~0.81–0.86 are
always higher than any value meant to loosen the classifier) and is fixed
unconditionally.

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
from plasflow2.classify.model_contract import (
    ModelContract,
    validate_model_profile_pair,
)
from plasflow2.classify.threshold_policy import (
    ThresholdPolicy,
    ThresholdPolicyError,
    default_threshold_policy_for_profile,
    validate_profile_threshold_policy,
)
from plasflow2.utils.device import IDX_TO_CLASS, get_device

logger = logging.getLogger(__name__)

# Default confidence thresholds (class-specific)
DEFAULT_THRESHOLD = 0.70  # chromosome
DEFAULT_PHAGE_THRESHOLD = 0.70  # phage — restored to 0.70 (was 0.85, over-suppressed recall)
DEFAULT_PLASMID_THRESHOLD = 0.80  # plasmid — optimal with COMPASS post-filter (sweep: Tier 1 F1=0.7332 at t=0.80+per-length COMPASS tiers, Jul 2026)

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
    # Rev5 calibrated (Jul 2026) — temperature T=1.547 baked into model weights.
    # Plasmid thresholds derived from precision-recall sweep on Tier 1 benchmark
    # (max-F1 threshold per length tier from calibration.json).
    # Previous thresholds (0.93–0.99) were over-strict, suppressing recall to ~0.46.
    # New thresholds (~0.86) recover recall to ~0.73 at comparable precision (0.42).
    #
    # Phage thresholds recalibrated Jul 2026 on 4,898 locked phage-development
    # positives and 299,589 Tier-1 plasmid/chromosome negatives. The previous
    # 0.90–0.95 thresholds suppressed locked-final phage recall to 0.146.
    # Per-tier max-F1 thresholds are used below. The 2–5kb tier had no positive
    # development examples, so 0.850 is interpolated between adjacent tiers.
    # Frozen final validation: precision=0.728, recall=0.807, F1=0.765.
    # Chromosome thresholds remain at 0.60–0.70.
    #
    #   <2kb:    F1=0.432  P=0.337  R=0.602 at t=0.862
    #   2-5kb:   F1=0.542  P=0.424  R=0.752 at t=0.864
    #   5-10kb:  F1=0.615  P=0.489  R=0.828 at t=0.859
    #   10-50kb: F1=0.735  P=0.631  R=0.880 at t=0.857
    #   >50kb:   F1=1.000  P=1.000  R=1.000 at t=0.809 (only 3 plasmids in benchmark)
    #
    # Attempted re-sweep against the has_replicon-fixed marker_xgb.pkl (commit
    # 6946b66) on Jul 19 2026 using data/benchmark/benchmark.fna (60,394
    # contigs) + annotations_with_replicons.tsv. Not applied: the sweep script
    # used round-number bin edges (5000/10000/20000) instead of this table's
    # actual boundaries (4999/9999/19999), and this benchmark's window lengths
    # cluster hard at exactly 1000/2000/5000/10000/20000bp -- so "5-10kb" as
    # swept and "5-10kb" as coded here are almost disjoint sets (the coded
    # tier has 4 sequences in this benchmark; the swept sample of 379
    # positives was actually the 10-20kb tier's population). Re-running with
    # correct boundaries also didn't reproduce consistent results end to end
    # (a full-cascade rerun showed FP +141 that a 4-sequence tier change can't
    # explain), which isn't understood yet -- so no threshold values were
    # changed here. What IS validated on the same benchmark: the has_replicon
    # model fix alone (no threshold changes) moves full-cascade plasmid recall
    # 0.284->0.538, precision 0.806->0.777, F1 0.420->0.636. Re-attempting
    # this recalibration is a distinct follow-up, not done in this pass.
    (2_000, 0.862, 0.855, 0.75),  # <=2kb
    (4_999, 0.864, 0.850, 0.68),  # 2-5kb
    (9_999, 0.859, 0.845, 0.65),  # 5-10kb
    # NOTE: boundary 9999 so exact 10000bp seqs use 10-20kb tier
    (19_999, 0.857, 0.835, 0.63),  # 10-20kb
    (float("inf"), 0.809, 0.750, 0.62),  # >20kb
]

# Per-length COMPASS containment thresholds.
# Longer sequences require higher containment to be called plasmids because
# long chromosomal HGT fragments match the plasmid database at low containment
# (a single 5kb HGT hotspot contributes ~0.002 in a 100kb contig).
# True plasmids in longer length bins have distinctly higher containment.
#
# Calibrated on Tier 1 benchmark (299,589 sequences, Jul 2026).
# Optimal thresholds discovered by 2D sweep (plasmid_threshold × compass_threshold
# per length tier):
#   <5kb:    ct=0.002  → unchanged (short TPs and FPs have overlapping scores)
#   5-10kb:  ct=0.003  → FP p25=0.0028 vs TP p10=0.0036
#   10-20kb: ct=0.006  → FP p50=0.0056 vs TP p25=0.0090
#   20-50kb: ct=0.010  → FP p50=0.0072 vs TP p25=0.0200
#   ≥50kb:   ct=0.001  → essentially no FPs at this length; keep low to maximise TP
#
# Net effect vs uniform ct=0.002: F1 0.7247 → 0.7332 (+0.0085).
# Tiers scale proportionally when --compass-threshold overrides the 0.002 base.
COMPASS_LENGTH_TIERS = [
    # (max_length_bp, compass_threshold)
    (4_999, 0.002),  # <5kb  — base threshold
    (9_999, 0.003),  # 5-10kb
    (19_999, 0.006),  # 10-20kb
    (49_999, 0.010),  # 20-50kb
    (float("inf"), 0.001),  # ≥50kb — very few FPs; keep low to retain TPs
]

_COMPASS_CALIBRATED_BASE = 0.002  # base threshold used when calibrating COMPASS_LENGTH_TIERS


def _get_compass_threshold(seq_len: int, base_threshold: float) -> float:
    """Return the per-length COMPASS containment threshold for a given sequence length.

    Uses COMPASS_LENGTH_TIERS scaled by (base_threshold / 0.002) so that
    custom --compass-threshold values preserve the relative tier ratios.
    """
    scale = base_threshold / _COMPASS_CALIBRATED_BASE if _COMPASS_CALIBRATED_BASE > 0 else 1.0
    for max_len, tier_ct in COMPASS_LENGTH_TIERS:
        if seq_len <= max_len:
            return tier_ct * scale
    return base_threshold


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
                        "replicon_boost"         — rep.dna.fas replicon detected (minimap2/BLASTN).
                        (hallmark_boost, plsdb_prot_boost, marker_threshold_boost
                        removed 2026-07-21 — proved functionally dead under
                        default thresholds; see docs/CODE_REVIEW_FINDINGS_2026-07.md,
                        Round 5.)
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
            return sid, {
                "n_orfs": len(gene_list),
                "covered_bp": covered,
                "genes": gene_list,
            }
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


_LEGACY_DEFAULT_SHORT_PLASMID_FLOOR = 0.95  # historical CLI default for --plasmid-threshold
_LEGACY_DEFAULT_SHORT_CHR_CEILING = 0.70  # historical CLI default for --threshold


def _assign_label(
    scores: dict[str, float],
    seq_len: int,
    plasmid_threshold: float | None,
    threshold: float | None,
    argmax_fallback: bool,
    threshold_policy: ThresholdPolicy | None = None,
) -> tuple[str, float]:
    """Return (label, confidence) from scores + length-aware thresholds.

    ``plasmid_threshold`` / ``threshold`` of ``None`` mean "no explicit CLI
    override" — this reproduces today's actually-shipped default behavior
    exactly (see below), NOT a clean "always use the tier profile" default.
    An explicit float (e.g. from ``--plasmid-threshold`` or ``--lenient``)
    overrides the tier value at every length, as a direct replacement rather
    than a max()/min() against the tier default.

    Why the ``None`` case isn't just "use the tier profile everywhere":
    the CLI has always defaulted ``--plasmid-threshold`` to 0.95, but the
    previous implementation only applied that default below 5kb (via
    ``max(tier_default, 0.95)``); sequences >=5kb always used the tier
    profile untouched, regardless of any CLI setting. That is what every
    validated benchmark number for this project actually measured. A
    benchmark re-run while fixing this (docs/CODE_REVIEW_FINDINGS_2026-07.md,
    item 2) showed that applying the tier profile below 5kb "correctly" by
    default collapses plasmid precision from 0.777 to 0.224 on the Tier 1
    benchmark — the <5kb tier values in LENGTH_THRESHOLD_TIERS were
    calibrated by a sweep that (unintentionally) always ran with the 0.95
    floor already in place, so they were never actually validated as
    standalone thresholds. Until someone reruns that calibration sweep with
    the floor genuinely removed, the *default* (no explicit CLI value)
    reproduces the legacy floor exactly, so default runs don't silently
    regress. An *explicit* override (a real ask from the caller) is honored
    literally, at every length — that part of the old behavior was simply
    broken (an explicit lower value was discarded by max()) and is fixed
    here unconditionally.
    """
    best_class = max(scores, key=scores.__getitem__)
    confidence = float(scores[best_class])

    if threshold_policy is None:
        plas_t, phage_t, chr_t = _get_length_thresholds(seq_len)
    else:
        policy_tier = threshold_policy.thresholds_for_length(seq_len)
        plas_t = policy_tier.plasmid
        phage_t = policy_tier.phage
        chr_t = policy_tier.chromosome

    if plasmid_threshold is not None:
        plas_t = plasmid_threshold
    elif threshold_policy is None and seq_len < 5_000:
        plas_t = max(plas_t, _LEGACY_DEFAULT_SHORT_PLASMID_FLOOR)

    if threshold is not None:
        chr_t = threshold
    elif threshold_policy is None and seq_len < 5_000:
        chr_t = min(chr_t, _LEGACY_DEFAULT_SHORT_CHR_CEILING)

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


def _mlp_scores_chunked(
    sequences: list[str],
    sequence_ids: list[str],
    model,
    device,
    *,
    batch_size: int,
    gene_data: dict[str, list] | None = None,
) -> list[dict[str, float]]:
    """Extract features and infer scores in bounded-memory chunks."""

    import torch

    if len(sequences) != len(sequence_ids):
        raise ValueError("sequences and sequence_ids must have the same length")

    all_raw_scores: list[dict[str, float]] = []
    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        feature_chunk = extract_features(
            sequences[start:end],
            gene_data=gene_data,
            seq_ids=sequence_ids[start:end],
        )
        batch = torch.from_numpy(feature_chunk).to(device)
        with torch.no_grad():
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        all_raw_scores.extend(
            {IDX_TO_CLASS[j]: float(probability[j]) for j in range(len(probability))}
            for probability in probabilities
        )
        if end == len(sequences) or end % 10_000 == 0:
            logger.info("MLP inference: %d / %d sequences", end, len(sequences))
    return all_raw_scores


def _resolve_prediction_policy(
    model_path: Path | str,
    profile: str,
    *,
    allow_unverified_custom_model: bool,
) -> tuple[ModelContract | None, ThresholdPolicy]:
    """Verify the model/profile pairing and resolve its immutable policy."""
    contract = validate_model_profile_pair(
        model_path,
        profile,
        allow_unverified_custom_model=allow_unverified_custom_model,
    )

    if contract is None:
        return None, default_threshold_policy_for_profile(profile)

    policy = validate_profile_threshold_policy(
        profile,
        contract.threshold_policy_for(profile),
    )
    return contract, policy


def _validate_prediction_policy_options(
    policy: ThresholdPolicy,
    *,
    threshold: float | None,
    plasmid_threshold: float | None,
    argmax_fallback: bool,
    marker_model_path: Path | str | None,
    compass_sketch_path: Path | str | None,
    apply_prior: bool | None,
) -> bool:
    """Reject runtime options that violate a frozen threshold policy."""
    if not policy.allow_threshold_overrides and (
        threshold is not None or plasmid_threshold is not None
    ):
        raise ThresholdPolicyError(f"Profile {policy.profile!r} forbids threshold overrides.")

    if policy.requires_compass and compass_sketch_path is None:
        raise ThresholdPolicyError(f"Profile {policy.profile!r} requires a COMPASS sketch.")

    if not policy.allow_compass and compass_sketch_path is not None:
        raise ThresholdPolicyError(f"Profile {policy.profile!r} forbids COMPASS post-processing.")

    if not policy.allow_marker_fusion and marker_model_path is not None:
        raise ThresholdPolicyError(f"Profile {policy.profile!r} forbids marker-model fusion.")

    if not policy.allow_argmax_fallback and argmax_fallback:
        raise ThresholdPolicyError(f"Profile {policy.profile!r} forbids argmax fallback.")

    if apply_prior is None:
        return policy.apply_prior_correction

    if not policy.allow_threshold_overrides and apply_prior != policy.apply_prior_correction:
        raise ThresholdPolicyError(
            f"Profile {policy.profile!r} requires " f"apply_prior={policy.apply_prior_correction}."
        )

    return apply_prior


def predict(
    sequences: list[str],
    sequence_ids: list[str],
    model_path: Path | str,
    threshold: float | None = None,
    plasmid_threshold: float | None = None,
    batch_size: int = 512,
    argmax_fallback: bool = False,
    source_context: str = "unspecified",
    apply_prior: bool | None = None,
    # --- Marker XGBoost (second stage) ---
    marker_model_path: Path | str | None = None,
    use_pyrodigal: bool = True,
    annotation_tsv: Path | str | None = None,
    marker_alpha_base: float = 0.3,
    pre_computed_annotations: dict[str, dict[str, float]] | None = None,
    precomputed_orf_data: dict[str, dict] | None = None,
    # --- COMPASS containment filter (post-processing) ---
    compass_sketch_path: Path | str | None = None,
    compass_threshold: float = 0.002,
    # --- Verified model/profile contract ---
    profile: str = "sequence-only",
    allow_unverified_custom_model: bool = False,
) -> list[Prediction]:
    """Classify sequences using the 3-class MLP (plasmid / chromosome / phage).

    Archaea is not a model output — it is assigned post-classification by
    the pipeline using DIAMOND taxonomy ORF voting.

    Args:
        sequences: DNA strings.
        sequence_ids: Identifiers corresponding to each sequence.
        model_path: Path to saved .pt weights.
        threshold: Minimum confidence for chromosome calls. ``None`` (default)
            reproduces the historical CLI default (0.70, applied only below
            5kb; ≥5kb uses the tier profile) — see module docstring. An
            explicit value overrides the tier value at every length.
        plasmid_threshold: Minimum confidence for plasmid calls. ``None``
            (default) reproduces the historical CLI default (0.95, applied
            only below 5kb; ≥5kb uses the tier profile) — see module
            docstring for why this isn't simply "use the tier profile." An
            explicit value overrides the tier value at every length (this is
            what ``--lenient`` / ``--plasmid-threshold`` set).
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
        precomputed_orf_data: Pre-computed pyrodigal orf_data dict
            (seq_id → {n_orfs, covered_bp, genes}).  When provided, the internal
            pyrodigal call is skipped entirely — useful when the caller has
            already run pyrodigal for another purpose (e.g. mob DIAMOND).
        compass_sketch_path: Optional path to the COMPASS MinHash sketch .npy
            file (k=21, sorted uint64 bottom-S hashes).  When provided, sequences
            predicted as plasmid are validated against the sketch: those with
            containment < compass_threshold are reclassified as chromosome.
            Improves Tier 1 plasmid F1 from 0.534 → 0.733 at plasmid_threshold=0.80,
            compass_threshold=0.002 with per-length COMPASS tiers
            (Tier 1 benchmark, 299,589 sequences, Jul 2026).
            Hard biological overrides (conjugative_override, replicon_boost)
            are exempt — biological evidence supersedes containment.
        compass_threshold: Base containment score for COMPASS filtering (default 0.002).
            Per-length thresholds are applied automatically (see COMPASS_LENGTH_TIERS):
            longer sequences require higher containment because chromosomal HGT
            fragments can match the plasmid database at low containment.
            This threshold is the baseline for <5kb sequences; longer tiers
            scale proportionally if you override this value.

    Returns:
        List of Prediction objects, one per input sequence.
    """
    import torch
    import torch.nn as nn

    from plasflow2.classify.model import load_model

    model_contract, threshold_policy = _resolve_prediction_policy(
        model_path,
        profile,
        allow_unverified_custom_model=allow_unverified_custom_model,
    )
    effective_apply_prior = _validate_prediction_policy_options(
        threshold_policy,
        threshold=threshold,
        plasmid_threshold=plasmid_threshold,
        argmax_fallback=argmax_fallback,
        marker_model_path=marker_model_path,
        compass_sketch_path=compass_sketch_path,
        apply_prior=apply_prior,
    )

    logger.info(
        "Classification contract: profile=%s policy=%s model=%s verified=%s prior=%s",
        profile,
        threshold_policy.policy_id,
        model_contract.model_id if model_contract is not None else Path(model_path).name,
        model_contract is not None,
        effective_apply_prior,
    )

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
            "Model expects %d-dim input — gene content features will be included",
            _model_input_dim,
        )
    else:
        logger.info(
            "Model expects %d-dim input — k=7 only (no gene content features)",
            _model_input_dim,
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
    # When the caller passes precomputed_orf_data (e.g. already ran pyrodigal
    # for mob DIAMOND), we skip the internal call entirely — saving ~15 min on
    # 200k contigs.
    orf_data_global: dict[str, dict] = {}
    if precomputed_orf_data:
        orf_data_global = precomputed_orf_data
        logger.info(
            "Reusing pre-computed pyrodigal ORF data (%d sequences)",
            len(orf_data_global),
        )
    elif use_pyrodigal and (_model_wants_gene_features or marker_model_path):
        logger.info("Running pyrodigal for gene/ORF features …")
        orf_data_global = _run_pyrodigal(sequences, sequence_ids)
        if orf_data_global:
            logger.info("  ORFs predicted for %d sequences", len(orf_data_global))

    # Build gene_data dict (seq_id → gene list) for extract_features().
    # Only pass gene_data when the loaded model was actually trained with them.
    gene_data_for_extract: dict[str, list] | None = None
    if _model_wants_gene_features:
        gene_data_for_extract = {sid: d.get("genes", []) for sid, d in orf_data_global.items()}

    # ── Stage 1: MLP softmax scores ──────────────────────────────────────────
    all_raw_scores = _mlp_scores_chunked(
        sequences,
        sequence_ids,
        model,
        device,
        batch_size=batch_size,
        gene_data=gene_data_for_extract,
    )

    # Apply prior correction
    all_scores: list[dict[str, float]] = []
    for raw_scores in all_raw_scores:
        if effective_apply_prior and source_context != "unspecified":
            all_scores.append(apply_prior_correction(raw_scores, source_context))
        else:
            all_scores.append(raw_scores)

    # ── Stage 2: Marker XGBoost (optional) ───────────────────────────────────
    # Evidence dicts — populated inside the if-block; initialised here so the
    # label-assignment loop can reference them unconditionally.
    _xgb_scores_by_idx: dict[int, dict[str, float]] = {}
    _bio_ev_by_idx: dict[int, dict[str, float]] = {}
    _ev_type_by_idx: dict[int, str] = {}

    from plasflow2.classify.marker_classifier import resolve_marker_model_path

    _resolved_marker_model_path = (
        resolve_marker_model_path(Path(marker_model_path)) if marker_model_path else None
    )
    if _resolved_marker_model_path is not None:
        from plasflow2.classify.marker_classifier import (
            MARKER_PROFILE_QUICK,
            MarkerClassifier,
            marker_model_safety_issues,
        )

        _candidate_marker_clf = MarkerClassifier.load(_resolved_marker_model_path)
        _marker_safety_issues = marker_model_safety_issues(
            _candidate_marker_clf.metadata,
            required_feature_profile=MARKER_PROFILE_QUICK,
        )
        if _marker_safety_issues:
            logger.warning(
                "Unsafe marker model disabled: %s — using MLP scores only.",
                "; ".join(_marker_safety_issues),
            )
            _resolved_marker_model_path = None

    if _resolved_marker_model_path is not None:
        logger.info("Running marker XGBoost second stage from %s", _resolved_marker_model_path)

        from plasflow2.classify.marker_classifier import (
            ContigMarkerFeatures,
            MarkerClassifier,
        )

        marker_clf = MarkerClassifier.load(_resolved_marker_model_path)

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

        # ── Post hoc rule map: trained-feature redundancy vs. genuinely new
        # evidence ─────────────────────────────────────────────────────────
        # Everything from here down is a hand-coded override/boost applied
        # AFTER the marker XGBoost has already produced marker_s above.
        #
        #   conjugative_override   -- is_conjugative     -- TRAINED feature (redundant,
        #                               but kept as a hard override deliberately --
        #                               see comment below).
        #   replicon_boost         -- has_replicon        -- TRAINED as of the
        #                               fix in fix_has_replicon_feature.py, but the
        #                               *deployed* marker_xgb.pkl predates that fix
        #                               (see commit 6946b66) -- currently still the
        #                               only place this evidence has any effect.
        #   plsdb_nt_override      -- direct PLSDB/COMPASS nucleotide match --
        #                               deliberately kept as a hard override rather
        #                               than a feature (see has_plsdb_match comment
        #                               in marker_classifier.py): a database
        #                               membership match is closer to ground truth
        #                               than a probabilistic signal, by design, not
        #                               an oversight.
        #   phage suppression      -- v_marker_freq       -- TRAINED feature (redundant)
        #
        #   hallmark_boost, plsdb_prot_boost, marker_threshold_boost -- REMOVED
        #   2026-07-21. All three were flagged here as redundant with the
        #   trained n_plasmid_markers feature; investigating that led to a
        #   bigger finding -- they were functionally dead code regardless of
        #   redundancy (see the removed-code comment a few lines down for the
        #   full proof). Removing dead code doesn't require the benchmark
        #   validation dance the rest of this session needed, since a no-op
        #   removal can't regress anything by construction -- verified with a
        #   bit-identical before/after run instead.
        #
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
                all_scores[i] = {
                    "plasmid": 0.999,
                    "chromosome": 0.0005,
                    "phage": 0.0005,
                }
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

        # ── REMOVED: hallmark_boost, plsdb_prot_boost, marker_threshold_boost ──
        # (2026-07-21) All three used the same "transfer 55% of non-plasmid
        # probability mass to plasmid, then renormalize" mechanism, gated on
        # mlp_plas >= 0.30 (0.90 for marker_threshold_boost) and best != "plasmid".
        # Removed after proving they were functionally dead code under the
        # default (non-lenient) threshold regime:
        #
        #   marker_threshold_boost: PROVABLY dead under any settings. Its own
        #   trigger requires mlp_plas >= 0.90 AND best != "plasmid" -- but
        #   scores sum to 1.0, so mlp_plas >= 0.90 forces the other class(es)
        #   to <= 0.10 combined, which makes best == "plasmid" unconditionally.
        #   The two conditions are mutually exclusive; the boost body could
        #   never execute regardless of annotation data, model, or thresholds.
        #
        #   hallmark_boost / plsdb_prot_boost: near-dead under default
        #   thresholds. The 55% transfer caps the post-boost plasmid score at
        #   0.45*mlp_plas + 0.55 -- and since best != "plasmid" bounds
        #   mlp_plas below ~0.5 (it can't be the argmax by definition), the
        #   ceiling is ~0.775. Every non-lenient tier threshold is >= 0.809,
        #   so the boosted score can never cross it. Confirmed empirically on
        #   real annotation data (data/benchmark/annotations_with_plsdb_prot.tsv,
        #   full 394-true-plasmid coverage): plsdb_prot_boost fired on 441/16394
        #   contigs, hallmark_boost on 3 -- zero of either ever produced a final
        #   "plasmid" label. Only in --lenient mode (threshold ~0.70) could the
        #   ~0.775 ceiling theoretically clear the bar, and even there the
        #   observed firing rate is too low to matter in practice.
        #
        #   hallmark_boost and marker_threshold_boost were also flagged
        #   redundant with n_plasmid_markers, a trained XGBoost feature --
        #   double-counting concern from the original review, on top of being
        #   inert. plsdb_prot_boost is NOT redundant (no trained feature
        #   covers plsdb_prot_hits_per_kb) but was equally non-functional as
        #   currently wired; if that evidence signal is worth using, it needs
        #   a mechanism that can actually clear the threshold (e.g. a trained
        #   feature or a harder override), not a bigger version of this one.
        #
        # See docs/CODE_REVIEW_FINDINGS_2026-07.md, Round 5, for the full
        # derivation and the (bit-identical) before/after verification.
        #
        # replicon_boost below is NOT removed -- its stronger 65% transfer
        # (vs. 55%) and lower floor (mlp_plas >= 0.15) give it a ceiling of
        # ~0.35*0.5+0.65 = 0.8215, which DOES exceed the lowest tier
        # threshold (0.809), so it is not provably dead the same way.

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
        # Note: has_replicon was 0 for all 90,000 rows in the committed marker
        # training set (data/marker_features_balanced_28_genomad.npz) because
        # the minimap2-vs-rep.dna.fas step never ran successfully when that
        # dataset was built — see scripts/fix_has_replicon_feature.py, which
        # fixed the column in place (777/90,000 rows now correctly flagged;
        # feature importance goes from 0.0 to ~0.03 in a fresh XGBoost train).
        # This post-prediction boost is kept for now because the *deployed*
        # data/models/marker_xgb.pkl binary (a separate, untracked build
        # artifact — see install.sh) has not yet been retrained on the fixed
        # data and redistributed. Once that happens, re-evaluate whether this
        # heuristic still adds value on top of the model's own learned
        # has_replicon weight, or whether it now double-counts the same
        # evidence.
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
        logger.warning(
            "Marker model not found at %s (also checked .json/.ubj siblings) — using MLP only",
            marker_model_path,
        )

    # ── Label assignment ──────────────────────────────────────────────────────
    # Check whether the marker stage populated evidence dicts (only when
    # marker_model_path was provided and resolved to a real file).
    _marker_stage_ran = _resolved_marker_model_path is not None

    results: list[Prediction] = []
    for i, (sid, scores) in enumerate(zip(sequence_ids, all_scores)):
        seq_len = len(sequences[i])
        label, confidence = _assign_label(
            scores,
            seq_len,
            plasmid_threshold,
            threshold,
            argmax_fallback,
            threshold_policy=threshold_policy,
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

    # ── COMPASS containment filter (post-processing) ──────────────────────────
    # Applied AFTER label assignment so that biological hard overrides
    # (conjugative_override, hallmark_boost, replicon_boost) are honoured even
    # when COMPASS containment is low (e.g. novel conjugative plasmid not yet in
    # the database).
    if compass_sketch_path is not None:
        from plasflow2.classify.containment import CompassFilter

        _EXEMPT_EVIDENCE = {"conjugative_override", "replicon_boost"}
        try:
            _compass = CompassFilter.load(compass_sketch_path, threshold=compass_threshold)
            n_reclassified = 0
            for i, result in enumerate(results):
                if result.label != "plasmid":
                    continue
                # Exempt hard biological overrides
                if result.evidence_type in _EXEMPT_EVIDENCE:
                    continue
                c_score = _compass.score(sequences[i])
                # Per-length threshold: longer sequences need higher containment
                # (chromosomal HGT fragments match at low containment; true
                # plasmids of the same length score distinctly higher).
                eff_threshold = _get_compass_threshold(len(sequences[i]), compass_threshold)
                if c_score < eff_threshold:
                    results[i] = Prediction(
                        sequence_id=result.sequence_id,
                        label="chromosome",
                        confidence=result.confidence,
                        scores=result.scores,
                        low_confidence=True,  # flagged: containment-rejected plasmid
                        mlp_scores=result.mlp_scores,
                        xgb_scores=result.xgb_scores,
                        bio_evidence=result.bio_evidence,
                        evidence_type="compass_rejected",
                    )
                    n_reclassified += 1
            logger.info(
                "COMPASS containment filter: %d plasmid predictions reclassified as chromosome "
                "(compass_threshold=%.4f base, per-length tiers active, sketch=%s)",
                n_reclassified,
                compass_threshold,
                Path(compass_sketch_path).name,
            )
        except FileNotFoundError as e:
            logger.warning("COMPASS filter skipped — sketch not found: %s", e)

    n_unclassified = sum(1 for r in results if r.label == "unclassified")
    _plas_t_desc = f"{plasmid_threshold:.2f}" if plasmid_threshold is not None else "tiered"
    _chr_t_desc = f"{threshold:.2f}" if threshold is not None else "tiered"
    logger.info(
        "Classified %d sequences (plasmid_threshold=%s, threshold=%s, "
        "argmax_fallback=%s, unclassified=%d, marker_stage=%s)",
        len(results),
        _plas_t_desc,
        _chr_t_desc,
        argmax_fallback,
        n_unclassified,
        _marker_stage_ran,
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
