#!/usr/bin/env bash
# retrain_hard_neg.sh — Retrain with hard-negative chromosome injection.
#
# MOTIVATION
# ----------
# F1=0.688 baseline has 148 FPs concentrated in the 10-20kb bin:
#   GCF_000017645.1  Burkholderia vietnamiensis  68 FPs
#   GCF_001593285.1  Enterobacter cloacae        66 FPs
#
# Their secondary chromosomes and genomic islands were never in the
# chromosome training pool, so the MLP learned to score them as plasmid.
# This script adds 10k guaranteed chromosome windows from these organisms
# (and related multi-chromosome species) into the training set.
#
# STEPS
# -----
#   0. Download hard-negative genomes → data/hard_negatives/
#   1. Build dataset (5kb+10kb, 128-dim PCA) with hard negatives injected
#   2. Retrain MLP (input_dim=1493)
#   3. Recalibrate XGBoost
#   4. Benchmark → compare with F1=0.688 baseline
#
# RUNTIME
#   Step 0  : ~5-10 min (12 genomes, ~50 MB each)
#   Step 1  : ~90 min
#   Step 2  : ~20 min
#   Step 3  : ~30 min
#   Step 4  : ~10 min
#   Total   : ~3 hr
#
# USAGE
#   nohup bash scripts/retrain_hard_neg.sh > data/retrain_hard_neg.log 2>&1 &
#   tail -f data/retrain_hard_neg.log
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$ROOT/data"
SCRIPTS_DIR="$ROOT/scripts"
SRC_DIR="$ROOT/src"

HARD_NEG_DIR="$DATA_DIR/hard_negatives"
EXPERIMENT_DIR="$DATA_DIR/hard_neg_experiment"
EXPERIMENT_MODELS="$EXPERIMENT_DIR/models"
MLP_OUT="$EXPERIMENT_MODELS/mlp_v2.pt"
XGB_OUT="$EXPERIMENT_MODELS/marker_xgb.pkl"
PCA_128="$DATA_DIR/models/k6_pca.pkl"   # existing 128-dim PCA

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"

echo "============================================================"
echo "  PlasFlow v2 — Hard-negative chromosome injection"
echo "  Start : $(date)"
echo "  Output: $EXPERIMENT_DIR"
echo "============================================================"
echo ""

mkdir -p "$HARD_NEG_DIR" "$EXPERIMENT_DIR" "$EXPERIMENT_MODELS"

# ── Step 0: Download hard-negative genomes ────────────────────────────────
echo "[0/4] Downloading hard-negative genomes …"
echo "  Output: $HARD_NEG_DIR"
echo "  Start : $(date)"
echo ""

python3 -u "$SCRIPTS_DIR/download_hard_negatives.py" \
    --out "$HARD_NEG_DIR"

echo ""
N_FNA=$(find "$HARD_NEG_DIR" -name "*.fna" | wc -l | tr -d ' ')
echo "[0/4] Download complete: $N_FNA FNA files in $HARD_NEG_DIR  ($(date))"
echo ""

# ── Step 1: Build dataset with hard negatives injected ───────────────────
echo "[1/4] Building dataset (5kb+10kb, hard negatives injected) …"
echo "  hard-negative-dir : $HARD_NEG_DIR"
echo "  hard-negative-max : 10000"
echo "  Start             : $(date)"
echo ""

python3 -u "$SCRIPTS_DIR/build_dataset.py" \
    --plasmid-files  "$DATA_DIR/databases/plasmids/plsdb.fasta,$DATA_DIR/databases/plasmids/COMPASS.fna" \
    --gtdb-dir       "$DATA_DIR/gtdb_genomes/bacteria" \
    --gtdb-n-genomes 500 \
    --data-dir       "$DATA_DIR/databases" \
    --window-sizes   5000,10000 \
    --min-length     4500 \
    --max-per-class  50000 \
    --skip-download \
    --hard-negative-dir "$HARD_NEG_DIR" \
    --hard-negative-max 10000 \
    --out            "$EXPERIMENT_DIR" \
    --seed           42

echo ""
echo "[1/4] Dataset build complete: $(date)"
echo ""

# ── Step 2: Retrain MLP (input_dim=1493, same as baseline) ───────────────
echo "[2/4] Retraining MLP on dataset with hard negatives …"
echo "  Output: $MLP_OUT"
echo "  Start : $(date)"
echo ""

python3 -u "$SCRIPTS_DIR/train_model.py" \
    --data   "$EXPERIMENT_DIR/features.npy" \
    --labels "$EXPERIMENT_DIR/labels.npy" \
    --out    "$EXPERIMENT_MODELS" \
    --mlp    \
    --epochs 50

echo "  MLP saved → $MLP_OUT"
echo "[2/4] MLP training complete: $(date)"
echo ""

# ── Step 3: Recalibrate XGBoost ──────────────────────────────────────────
echo "[3/4] Recalibrating XGBoost for new MLP …"
echo "  Start: $(date)"
echo ""

cp "$DATA_DIR/marker_features_balanced_28_genomad.npz" \
   "$EXPERIMENT_DIR/marker_features_hard_neg.npz"

python3 -u "$SCRIPTS_DIR/update_marker_mlp_scores.py" \
    --npz   "$EXPERIMENT_DIR/marker_features_hard_neg.npz" \
    --model "$MLP_OUT"

python3 -u "$SCRIPTS_DIR/train_marker_model.py" \
    --features "$EXPERIMENT_DIR/marker_features_hard_neg.npz" \
    --out      "$EXPERIMENT_MODELS"

echo "  XGBoost saved → $XGB_OUT"
echo "[3/4] XGBoost recalibration complete: $(date)"
echo ""

# ── Step 4: Benchmark ─────────────────────────────────────────────────────
echo "[4/4] Running benchmark …"
echo "  MLP    : $MLP_OUT"
echo "  XGBoost: $XGB_OUT"
echo "  Start  : $(date)"
echo ""

bash "$SCRIPTS_DIR/run_benchmark.sh" \
    --model          "$MLP_OUT" \
    --marker-model   "$XGB_OUT" \
    --annotation-tsv "$DATA_DIR/benchmark/annotations_with_genomad.tsv" \
    2>&1 | tee "$EXPERIMENT_DIR/benchmark_hard_neg.log"

echo ""
echo "------------------------------------------------------------"
echo "  Experiment complete: $(date)"
echo ""
echo "  Results:"
grep -E "PlasFlow v2 plasmid:|Per-length|10-20kb.*F1|5-10kb.*F1|>20kb.*F1" \
    "$EXPERIMENT_DIR/benchmark_hard_neg.log" 2>/dev/null | tail -12
echo ""
echo "  Comparison:"
echo "    5k+10k, 128-dim PCA, no hard negs (baseline) : P=0.637 R=0.749 F1=0.688"
echo "    5k+10k, 128-dim PCA, with hard negs  (this)  : see benchmark_hard_neg.log"
echo "    PlasFlow v1 (target)                         : P=0.963 R=0.917 F1=0.939"
echo "============================================================"
