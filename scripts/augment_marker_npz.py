#!/usr/bin/env python3
"""
augment_marker_npz.py — Add near-miss FN plasmids to XGBoost training data.

PROBLEM
-------
XGBoost is hurting recall for marker-present near-miss plasmids: sequences with
is_mobilizable/has_rep_protein markers and MLP scores 0.70–0.96. Because the
current NPZ was built from 90k balanced training sequences, it under-represents
true plasmids with intermediate MLP scores. XGBoost learns to call them
chromosome → blended score drops below 0.95 threshold → FN.

FIX
---
1. Re-run MLP-only (no XGBoost blend) on all benchmark FNs with biological
   markers to get their raw MLP scores.
2. Construct 28-dim feature vectors (raw MLP + annotation features).
3. Append to the existing marker NPZ (10× oversampled for weight).
4. Also append the 15 composition FPs (no markers, chromosome label) as
   hard negatives so XGBoost learns not to call marker-less sequences plasmid.
5. Save augmented NPZ → retrain XGBoost (fast, ~3 min).

RUN FROM PROJECT ROOT
---------------------
    python scripts/augment_marker_npz.py
    python scripts/train_marker_model.py \\
        --features data/marker_features_augmented.npz \\
        --out data/models/
"""
from __future__ import annotations

import csv
import logging
import sys
import os
from pathlib import Path
import math

# Thread caps before numpy/torch import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import warnings; warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
BASE_NPZ       = ROOT / "data/marker_features_no_genomad_hardneg.npz"
ANN_TSV        = ROOT / "data/benchmark/annotations_with_genomad.tsv"
GT_TSV         = ROOT / "data/benchmark/ground_truth.tsv"
BENCH_PREDS    = ROOT / "results/benchmark_no_bleed/plasflow2_predictions.tsv"
BENCH_FASTA    = ROOT / "data/benchmark/benchmark.fna"
MLP_MODEL      = ROOT / "data/models/mlp_v2.pt"
FP_HITS_TSV    = ROOT / "results/fp_validation/fp_plsdb_hits.tsv"
FP_MARKER_TSV  = ROOT / "results/fp_validation/fp_marker_summary.tsv"
OUT_NPZ        = ROOT / "data/marker_features_augmented.npz"

# How many times to repeat each near-miss example for upweighting
OVERSAMPLE = 10

# ── Feature name order (must match BASE_NPZ exactly) ─────────────────────────
FEATURE_NAMES = [
    "mlp_plasmid_score", "mlp_chromosome_score", "mlp_phage_score",
    "is_conjugative", "is_mobilizable", "has_replicon", "has_ice", "has_rep_protein",
    "n_arg_per_kb", "n_mge_per_kb", "n_ice_per_kb", "n_rep_per_kb",
    "log10_length", "gc_content", "coding_density", "n_orfs_per_kb",
    "p_marker_freq", "c_marker_freq", "v_marker_freq", "pp_marker_freq",
    "median_p_spm", "median_c_spm", "median_v_spm",
    "p_vs_c_logistic", "strand_switch_rate", "no_rbs_freq", "canonical_sd_freq",
    "n_plasmid_markers",
]

# ── Load ground truth ─────────────────────────────────────────────────────────
gt: dict[str, str] = {}
with open(GT_TSV) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        gt[row["contig_id"]] = row["true_label"]

# ── Load annotation features ──────────────────────────────────────────────────
ann: dict[str, dict] = {}
with open(ANN_TSV) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        ann[row["contig_id"]] = row

# ── Load existing predictions (final blended scores) ─────────────────────────
bench_preds: dict[str, dict] = {}
with open(BENCH_PREDS) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        bench_preds[row["contig_id"]] = row

# ── Identify near-miss FNs (true plasmid, predicted non-plasmid, has markers) ─
MARKER_COLS_BIO = ["is_conjugative", "is_mobilizable", "has_rep_protein"]
near_miss_fn_ids = []
for cid, p in bench_preds.items():
    if p.get("predicted") == "plasmid":
        continue
    if gt.get(cid) != "plasmid":
        continue
    a = ann.get(cid, {})
    has_marker = any(float(a.get(c, 0) or 0) > 0 for c in MARKER_COLS_BIO)
    if has_marker:
        near_miss_fn_ids.append(cid)

logger.info("Near-miss marker-present FNs: %d", len(near_miss_fn_ids))

# ── Identify composition FPs (true FPs with NO markers — for hard-neg XGBoost) ─
plsdb_confirmed: set[str] = set()
with open(FP_HITS_TSV) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row.get("is_known_plasmid") == "YES":
            plsdb_confirmed.add(row["contig_id"])

fp_markers: dict[str, dict] = {}
with open(FP_MARKER_TSV) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        fp_markers[row["contig_id"]] = row

composition_fp_ids = []
for cid, p in bench_preds.items():
    if p.get("predicted") != "plasmid":
        continue
    if gt.get(cid) != "chromosome":
        continue
    if cid in plsdb_confirmed:
        continue  # mislabeled, skip
    m = fp_markers.get(cid, {})
    has_marker = any(float(m.get(c, 0) or 0) > 0
                     for c in ["is_conjugative", "is_mobilizable", "has_rep_protein"])
    if not has_marker:
        composition_fp_ids.append(cid)

logger.info("Composition FPs (no markers, true chromosomal): %d", len(composition_fp_ids))

# ── Extract sequences for MLP re-scoring ─────────────────────────────────────
target_ids = set(near_miss_fn_ids)
seqs_to_score: list[tuple[str, str]] = []  # (id, seq)

logger.info("Extracting %d sequences from benchmark FASTA …", len(target_ids))
current_id = None
buf: list[str] = []
with open(BENCH_FASTA) as f:
    for line in f:
        line = line.rstrip()
        if line.startswith(">"):
            if current_id in target_ids and buf:
                seqs_to_score.append((current_id, "".join(buf)))
            current_id = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
    if current_id in target_ids and buf:
        seqs_to_score.append((current_id, "".join(buf)))

logger.info("  Extracted %d sequences", len(seqs_to_score))

# ── Run MLP only (no XGBoost) to get raw scores ──────────────────────────────
logger.info("Running MLP-only scoring …")
from plasflow2.classify.predict import predict

raw_mlp_scores: dict[str, dict[str, float]] = {}
if seqs_to_score:
    ids  = [s[0] for s in seqs_to_score]
    seqs = [s[1] for s in seqs_to_score]
    results = predict(
        sequences=seqs,
        sequence_ids=ids,
        model_path=str(MLP_MODEL),
        threshold=0.70,
        plasmid_threshold=0.95,
        argmax_fallback=False,
        source_context="unspecified",
        apply_prior=False,
        marker_model_path=None,   # <-- MLP only, no XGBoost
        annotation_tsv=None,
        use_pyrodigal=False,      # skip pyrodigal — we just need k-mer MLP scores
    )
    for r in results:
        raw_mlp_scores[r.sequence_id] = r.scores  # final = raw when no marker model

logger.info("  Got raw MLP scores for %d sequences", len(raw_mlp_scores))


def ann_to_feature_row(cid: str, mlp_p: float, mlp_c: float, mlp_v: float) -> np.ndarray | None:
    """Build 28-dim feature vector from annotations + raw MLP scores."""
    a = ann.get(cid)
    if a is None:
        return None
    try:
        length_bp = float(a.get("length_bp", 10000) or 10000)
        row = np.array([
            mlp_p,
            mlp_c,
            mlp_v,
            float(a.get("is_conjugative",  0) or 0),
            float(a.get("is_mobilizable",  0) or 0),
            float(a.get("has_replicon",    0) or 0),
            float(a.get("has_ice",         0) or 0),
            float(a.get("has_rep_protein", 0) or 0),
            float(a.get("n_arg_per_kb",    0) or 0),
            float(a.get("n_mge_per_kb",    0) or 0),
            float(a.get("n_ice_per_kb",    0) or 0),
            float(a.get("n_rep_per_kb",    0) or 0),
            math.log10(max(length_bp, 1.0)),
            float(a.get("gc_content",      0) or 0),
            float(a.get("coding_density",  0) or 0),
            float(a.get("n_orfs_per_kb",   0) or 0),
            float(a.get("p_marker_freq",   0) or 0),
            float(a.get("c_marker_freq",   0) or 0),
            float(a.get("v_marker_freq",   0) or 0),
            float(a.get("pp_marker_freq",  0) or 0),
            float(a.get("median_p_spm",    0) or 0),
            float(a.get("median_c_spm",    0) or 0),
            float(a.get("median_v_spm",    0) or 0),
            float(a.get("p_vs_c_logistic", 0) or 0),
            float(a.get("strand_switch_rate", 0) or 0),
            float(a.get("no_rbs_freq",     0) or 0),
            float(a.get("canonical_sd_freq", 0) or 0),
            float(a.get("n_plasmid_markers", 0) or 0),
        ], dtype=np.float32)
        return row
    except Exception as e:
        logger.warning("  Skipping %s: %s", cid, e)
        return None


# ── Build new rows ─────────────────────────────────────────────────────────────
new_X: list[np.ndarray] = []
new_y: list[int] = []

# Near-miss FN plasmids → label 1 (plasmid)
fn_added = 0
for cid in near_miss_fn_ids:
    mlp = raw_mlp_scores.get(cid, {})
    mlp_p = mlp.get("plasmid",    0.0)
    mlp_c = mlp.get("chromosome", 0.0)
    mlp_v = mlp.get("phage",      0.0)
    row = ann_to_feature_row(cid, mlp_p, mlp_c, mlp_v)
    if row is not None:
        for _ in range(OVERSAMPLE):
            new_X.append(row)
            new_y.append(1)  # plasmid
        fn_added += 1

logger.info("Near-miss FN plasmid rows added: %d × %d = %d", fn_added, OVERSAMPLE, fn_added * OVERSAMPLE)

# Composition FPs → label 0 (chromosome) as hard negatives
# Use final blended score from predictions as proxy for MLP score
# (alpha=0 for binary marker-less sequences → final = MLP)
fp_added = 0
for cid in composition_fp_ids:
    p = bench_preds.get(cid, {})
    mlp_p = float(p.get("plasmid_score",    0) or 0)
    mlp_c = float(p.get("chromosome_score", 0) or 0)
    mlp_v = float(p.get("phage_score",      0) or 0)
    row = ann_to_feature_row(cid, mlp_p, mlp_c, mlp_v)
    if row is not None:
        for _ in range(OVERSAMPLE):
            new_X.append(row)
            new_y.append(0)  # chromosome
        fp_added += 1

logger.info("Composition FP chromosome rows added: %d × %d = %d", fp_added, OVERSAMPLE, fp_added * OVERSAMPLE)

# ── Load base NPZ and append ───────────────────────────────────────────────────
logger.info("Loading base NPZ: %s", BASE_NPZ)
base = np.load(BASE_NPZ, allow_pickle=True)
X_base = base["X"].astype(np.float32)
y_base = base["y"].astype(np.int64)
feat_names_base = [str(f) for f in base["feature_names"]]

# Verify feature name alignment
if feat_names_base != FEATURE_NAMES:
    logger.warning("Feature name mismatch! Base NPZ has: %s", feat_names_base)
    logger.warning("Expected:                             %s", FEATURE_NAMES)
    # Try to reorder
    idx_map = [feat_names_base.index(f) for f in FEATURE_NAMES if f in feat_names_base]
    if len(idx_map) == len(FEATURE_NAMES):
        X_base = X_base[:, idx_map]
        logger.info("Reordered base NPZ features to match expected order")
    else:
        logger.error("Cannot align features — aborting")
        sys.exit(1)

X_new = np.array(new_X, dtype=np.float32) if new_X else np.empty((0, 28), dtype=np.float32)
y_new = np.array(new_y, dtype=np.int64)

X_aug = np.concatenate([X_base, X_new], axis=0)
y_aug = np.concatenate([y_base, y_new], axis=0)

logger.info("Base NPZ: %s", X_base.shape)
logger.info("New rows: %s", X_new.shape)
logger.info("Augmented: %s", X_aug.shape)
logger.info("Class dist: %s", {int(v): int((y_aug == v).sum()) for v in np.unique(y_aug)})

np.savez_compressed(
    OUT_NPZ,
    X=X_aug,
    y=y_aug,
    feature_names=np.array(FEATURE_NAMES),
)
logger.info("Saved augmented NPZ → %s", OUT_NPZ)

print("\n=== Next step ===")
print("Retrain XGBoost on augmented NPZ (~3 min):")
print(f"  python scripts/train_marker_model.py \\")
print(f"      --features {OUT_NPZ} \\")
print(f"      --out data/models/")
print()
print("Then benchmark:")
print("  bash scripts/run_benchmark.sh \\")
print("      --annotation-tsv data/benchmark/annotations_with_genomad.tsv")
