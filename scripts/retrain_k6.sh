#!/usr/bin/env bash
# retrain_k6.sh — End-to-end pipeline to add k=6 features and retrain the MLP.
#
# Steps:
#   1. Back up existing models and features
#   2. Fit k=6 PCA from current sequences
#   3. Rebuild dataset (auto-detects PCA → produces 1493-dim features)
#   4. Retrain MLP on new features
#
# Expected runtime (Apple M-series, CPU only):
#   Step 1 (backup)    :   <1 min
#   Step 2 (PCA fit)   :  ~15–30 min
#   Step 3 (dataset)   :  ~2–4 hrs  (900k sequences × k=1–6)
#   Step 4 (MLP train) :  ~2–3 hrs  (50 epochs, mmap-based)
#   Total              :  ~5–8 hrs
#
# Run from the project root:
#   bash scripts/retrain_k6.sh 2>&1 | tee data/retrain_k6.log
#
# To run in the background:
#   nohup bash scripts/retrain_k6.sh > data/retrain_k6.log 2>&1 &
#   tail -f data/retrain_k6.log
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR="$ROOT/data"
MODELS_DIR="$DATA_DIR/models"
SCRIPTS_DIR="$ROOT/scripts"
SRC_DIR="$ROOT/src"

FEATURES_NPY="$DATA_DIR/features.npy"
LABELS_NPY="$DATA_DIR/labels.npy"
SEQ_IDS_TXT="$DATA_DIR/seq_ids.txt"
MLP_MODEL="$MODELS_DIR/mlp_v2.pt"
K6_PCA="$MODELS_DIR/k6_pca.pkl"

TS=$(date +%Y%m%d_%H%M%S)

# ── Thread caps (prevents macOS ARM BLAS race → segfault) ──────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Python path ────────────────────────────────────────────────────────────
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"

echo "============================================================"
echo "  PlasFlow v2 — k=6 retrain pipeline"
echo "  Start: $(date)"
echo "  Root : $ROOT"
echo "============================================================"
echo ""

# ── Step 1: Backup ─────────────────────────────────────────────────────────
echo "[1/4] Backing up existing features and models..."

BACKUP_DIR="$DATA_DIR/backup_before_k6_$TS"
mkdir -p "$BACKUP_DIR"

for f in "$FEATURES_NPY" "$LABELS_NPY" "$SEQ_IDS_TXT" "$MLP_MODEL"; do
    if [ -f "$f" ]; then
        cp "$f" "$BACKUP_DIR/"
        echo "  Backed up: $(basename $f)"
    fi
done

echo "  Backup saved to: $BACKUP_DIR"
echo ""

# ── Step 2: Fit k=6 PCA ────────────────────────────────────────────────────
echo "[2/4] Fitting k=6 PCA..."
echo "  Output: $K6_PCA"
echo "  Start: $(date)"
echo ""

python3 "$SCRIPTS_DIR/fit_k6_pca.py" \
    --plasmid-dir "$DATA_DIR/databases/plasmids" \
    --chrom-fna   "$DATA_DIR/databases/chromosomes.fna" \
    --chrom-dir   "$DATA_DIR/chromosomes/bacteria" \
    --inphared-gz "$DATA_DIR/databases/inphared/inphared_phages.fa.gz" \
    --out         "$K6_PCA" \
    --n-per-class 33000 \
    --n-components 128

echo ""
echo "  PCA fitting done: $(date)"
echo ""

# ── Step 3: Rebuild dataset ────────────────────────────────────────────────
echo "[3/4] Rebuilding dataset with k=6 features..."
echo "  The PCA at $K6_PCA will be auto-detected by extract_features()"
echo "  Output features will be 1493-dim (was 1365)"
echo "  Start: $(date)"
echo ""

python3 "$SCRIPTS_DIR/build_dataset.py" \
    --plasmid-files "$DATA_DIR/databases/plasmids/plsdb.fasta,$DATA_DIR/databases/plasmids/COMPASS.fna" \
    --data-dir      "$DATA_DIR/databases" \
    --chrom-dir     "$DATA_DIR/chromosomes/bacteria" \
    --max-per-class 300000 \
    --out           "$DATA_DIR" \
    --skip-download \
    --seed          42

echo ""
echo "  Dataset rebuild done: $(date)"

# Verify feature dimensions
python3 - <<'PYEOF'
import numpy as np
X = np.load("data/features.npy", mmap_mode="r")
y = np.load("data/labels.npy")
print(f"  features.npy: {X.shape}  (expected dim=1493)")
print(f"  labels.npy:   {y.shape}")
assert X.shape[1] == 1493, f"Expected 1493 features, got {X.shape[1]} — PCA may not have been loaded"
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  label {u}: {c:,} ({100*c/len(y):.1f}%)")
PYEOF

echo ""

# ── Step 4: Retrain MLP ────────────────────────────────────────────────────
echo "[4/4] Retraining MLP on k=6 features..."
echo "  Input dim: 1493   Output: $MLP_MODEL"
echo "  Start: $(date)"
echo ""

python3 "$SCRIPTS_DIR/train_model.py" \
    --data   "$FEATURES_NPY" \
    --labels "$LABELS_NPY" \
    --out    "$MODELS_DIR" \
    --mlp    \
    --epochs 50

echo ""
echo "============================================================"
echo "  Pipeline complete: $(date)"
echo "  Model saved: $MLP_MODEL"
echo ""
echo "  Next steps:"
echo "    1. Run the benchmark:"
echo "       bash scripts/run_benchmark.sh"
echo "    2. Compare vs PlasFlow v1 and geNomad"
echo "============================================================"
