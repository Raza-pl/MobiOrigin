#!/usr/bin/env bash
# retrain_marker_3class.sh — Rebuild marker features with 3-class MLP and retrain XGBoost
#
# The old marker_xgb.pkl was trained on binary labels (plasmid/chromosome only)
# because build_marker_dataset.py was run before phage was added to the MLP.
# This script:
#   1. Rebuilds marker features using the current 3-class MLP
#      (plasmid=0, chromosome=1, phage=2 — INPHARED phages included)
#   2. Retrains the XGBoost on the 3-class features
#   3. Verifies 3-class output and backs up old model
#
# PREREQUISITES (all present):
#   - data/models/mlp_v2.pt              (3-class MLP — net.11.weight [3, 128])
#   - data/databases/plasmids/           (PLSDB + COMPASS)
#   - data/gtdb_genomes/bacteria/        (GTDB genomes)
#   - data/databases/inphared/inphared_phages.fa.gz   (INPHARED phages)
#   - data/databases/mob_suite/          (MOB-suite DBs for marker features)
#
# USAGE (from project root, plasflow2 conda env):
#   bash scripts/retrain_marker_3class.sh [--max-per-class N] [--threads N]
#
#   Defaults: --max-per-class 30000  --threads 8
#
# ESTIMATED TIME on Apple Silicon (8 threads):
#   Feature build : ~20–40 min  (DIAMOND searches for 3×30k sequences)
#   XGBoost train : ~2–5 min
#   Total         : ~25–45 min

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

THREADS=8
MAX_PER_CLASS=30000

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-per-class) MAX_PER_CLASS="$2"; shift 2 ;;
        --threads)       THREADS="$2";       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DATA_DIR="$ROOT/data"
MODEL="$DATA_DIR/models/mlp_v2.pt"
OUT_NPZ="$DATA_DIR/marker_features_3class.npz"

echo "============================================================"
echo " PlasFlow v2 — 3-class marker XGBoost retrain"
echo " Start      : $(date)"
echo " Max/class  : $MAX_PER_CLASS"
echo " Threads    : $THREADS"
echo " MLP model  : $MODEL"
echo "============================================================"
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────
for req in \
    "$MODEL" \
    "$DATA_DIR/databases/plasmids/plsdb.fasta" \
    "$DATA_DIR/databases/inphared/inphared_phages.fa.gz" \
    "$DATA_DIR/gtdb_genomes/bacteria"
do
    if [ ! -e "$req" ]; then
        echo "ERROR: missing prerequisite: $req"
        exit 1
    fi
done

# Verify MLP is 3-class
python3 - <<PYCHECK
import torch
state = torch.load("$MODEL", map_location="cpu", weights_only=False)
if hasattr(state, "state_dict"): state = state.state_dict()
layers = [(k, v.shape) for k, v in state.items() if "weight" in k]
last = layers[-1][1]
assert last[0] == 3, f"MLP last layer is {last[0]}-class, expected 3. Wrong model?"
print(f"OK — MLP is 3-class: {last}")
PYCHECK

echo ""
echo "[1/3] Building 3-class marker features …"
echo "  Note: INPHARED phages will be included as class 2"
echo "  Start: $(date)"
echo ""

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 -u "$SCRIPT_DIR/build_marker_dataset.py" \
    --plasmid-dir  "$DATA_DIR/databases/plasmids" \
    --chrom-dir    "$DATA_DIR/gtdb_genomes/bacteria" \
    --data-dir     "$DATA_DIR/databases" \
    --model        "$MODEL" \
    --max-per-class "$MAX_PER_CLASS" \
    --threads      "$THREADS" \
    --out          "$OUT_NPZ" \
    --seed         42 \
    2>&1 | tee "$DATA_DIR/marker_build_3class.log"

echo ""
echo "[1/3] Feature build complete: $(date)"

# Verify 3 classes
python3 - <<PYCHECK2
import numpy as np, collections
f = np.load("$OUT_NPZ", allow_pickle=True)
X, y = f['X'], f['y']
dist = dict(collections.Counter(y.tolist()).most_common())
print(f"X shape: {X.shape}")
print(f"Label distribution: {dist}")
assert len(dist) == 3, f"Expected 3 classes, got {len(dist)}"
for cls in [0, 1, 2]:
    assert cls in dist, f"Missing class {cls}"
print("OK — 3-class marker features confirmed (plasmid=0, chromosome=1, phage=2)")
PYCHECK2

echo ""
echo "[2/3] Training XGBoost on 3-class features …"
echo "  Start: $(date)"
echo ""

# Backup existing marker model
if [ -f "$DATA_DIR/models/marker_xgb.pkl" ]; then
    BACKUP="$DATA_DIR/models/marker_xgb_binary_backup_$(date +%Y%m%d_%H%M%S).pkl"
    cp "$DATA_DIR/models/marker_xgb.pkl" "$BACKUP"
    echo "  Backed up binary model → $BACKUP"
fi

python3 -u "$SCRIPT_DIR/train_marker_model.py" \
    --features "$OUT_NPZ" \
    --out      "$DATA_DIR/models" \
    2>&1 | tee "$DATA_DIR/marker_train_3class.log"

echo ""
echo "[2/3] Training complete: $(date)"

# Verify output
echo ""
echo "[3/3] Verifying trained model …"
python3 - <<PYVERIFY
import sys
sys.path.insert(0, "src")
from plasflow2.classify.marker_classifier import MarkerClassifier
import numpy as np

clf = MarkerClassifier()
clf.load("data/models/marker_xgb.pkl")

# Quick smoke test: random 5-row input, 28 features
np.random.seed(0)
X_test = np.random.rand(5, 28).astype(np.float32)
probs = clf.predict_proba(X_test)
print(f"predict_proba output shape: {probs.shape}  (expected [5, 3])")
assert probs.shape == (5, 3), f"Wrong shape: {probs.shape}"
assert abs(probs.sum(axis=1).mean() - 1.0) < 1e-4, "Probabilities don't sum to 1"
print("PASS — marker_xgb.pkl is 3-class and predicts valid probabilities")
PYVERIFY

echo ""
echo "============================================================"
echo " All done: $(date)"
echo " New marker model : data/models/marker_xgb.pkl  (3-class)"
echo " Features NPZ     : $OUT_NPZ"
echo " Build log        : data/marker_build_3class.log"
echo " Train log        : data/marker_train_3class.log"
echo ""
echo " Next steps:"
echo "  1. Re-run W1 WITHOUT --no-marker-model to test marker blending:"
echo "     python scripts/predict_sequences.py \\"
echo "         --input  data/test/W1.contigs.fa \\"
echo "         --model  data/models/mlp_v2.pt \\"
echo "         --out    results/W1_plasflow2/all_predictions_with_marker.tsv"
echo "  2. Compare phage/plasmid counts with and without marker model"
echo "============================================================"
