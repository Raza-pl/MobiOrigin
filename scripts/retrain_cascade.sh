#!/usr/bin/env bash
# retrain_cascade.sh — Train the two-stage cascade MLP classifier
#
# Architecture:
#   Stage 1: Binary plasmid detector  (plasmid=1 vs. chromosome+phage=0)
#            Input: 9557-dim k=7 features  Output: [N, 2]
#   Stage 2: Binary chr/phage discriminator  (chromosome=0 vs. phage=1)
#            Input: same 9557-dim features  Output: [N, 2]
#            Trained only on chr+phage samples (plasmid rows excluded)
#
# Benefits over 3-class softmax:
#   - Stage 1 has NO phage competition → plasmid probabilities are higher
#     → recall improvement expected (plasmid F1 v2: 0.774 → target 0.85+)
#   - Stage 2 is a clean binary decision, no plasmid interference
#   - Features extracted once at inference → wall time nearly identical
#
# PREREQUISITES:
#   data/k7_3class_experiment/features.npy  (300k × 9557, exists from 3-class run)
#   data/k7_3class_experiment/labels.npy    (300k entries, 0=plasmid 1=chr 2=phage)
#
# USAGE (from project root, plasflow2 conda env):
#   bash scripts/retrain_cascade.sh [--threads N] [--epochs N]
#
#   Defaults: --threads 16  --epochs 50
#
# ESTIMATED TIME on Apple Silicon (16 threads):
#   Stage 1 training : ~45–70 min  (300k samples, binary)
#   Stage 2 training : ~30–50 min  (200k samples filtered, binary)
#   Total            : ~75–120 min
#
# OUTPUTS:
#   data/k7_cascade_experiment/stage1_labels.npy
#   data/k7_cascade_experiment/stage1_models/mlp_v2.pt  → stage1 binary MLP
#   data/k7_cascade_experiment/stage2_models/mlp_v2.pt  → stage2 binary MLP
#   data/models/mlp_cascade_stage1.pt  (deployed, auto-detected by predict_sequences.py)
#   data/models/mlp_cascade_stage2.pt  (deployed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

THREADS=16
EPOCHS=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --threads) THREADS="$2"; shift 2 ;;
        --epochs)  EPOCHS="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

EXPERIMENT_DIR="data/k7_cascade_experiment"
S1_MODELS="$EXPERIMENT_DIR/stage1_models"
S2_MODELS="$EXPERIMENT_DIR/stage2_models"
FEATURES="data/k7_3class_experiment/features.npy"
LABELS="data/k7_3class_experiment/labels.npy"

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export VECLIB_MAXIMUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

echo "============================================================"
echo " PlasFlow v2 — Cascade MLP training"
echo " Start   : $(date)"
echo " Threads : $THREADS"
echo " Epochs  : $EPOCHS"
echo " Features: $FEATURES"
echo "============================================================"
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────
for req in "$FEATURES" "$LABELS"; do
    if [ ! -f "$req" ]; then
        echo "ERROR: missing $req"
        echo "Run scripts/retrain_3class.sh first to build the 3-class dataset."
        exit 1
    fi
done

python3 - <<PYCHECK
import numpy as np, collections
labels = np.load("$LABELS")
feat   = np.load("$FEATURES", mmap_mode="r")
dist   = dict(collections.Counter(labels.tolist()).most_common())
print(f"Features: {feat.shape}  Labels: {dist}")
assert feat.shape[1] == 9557, f"Expected 9557 features, got {feat.shape[1]}"
assert set(dist.keys()) == {0,1,2}, f"Expected 3 classes, got {set(dist.keys())}"
print("OK")
PYCHECK

mkdir -p "$EXPERIMENT_DIR" "$S1_MODELS" "$S2_MODELS"

# ── Step 1: Build cascade label arrays ────────────────────────────────────
echo "[1/4] Building cascade label arrays …"
python3 "$SCRIPT_DIR/build_cascade_labels.py" \
    --labels "$LABELS" \
    --out    "$EXPERIMENT_DIR"

echo ""

# ── Step 2: Train Stage 1 — plasmid vs. rest ─────────────────────────────
echo "[2/4] Training Stage 1 (plasmid vs. rest) …"
echo "  Labels: $EXPERIMENT_DIR/stage1_labels.npy (0=non-plasmid, 1=plasmid)"
echo "  Samples: 300k  |  Imbalance 2:1 — class weighting applied"
echo "  Start: $(date)"
echo ""

python3 - <<PYRUN1
import sys, torch
torch.set_num_threads($THREADS)
torch.set_num_interop_threads(4)
print(f"torch threads: {torch.get_num_threads()}")
sys.argv = [
    'train_model.py',
    '--data',   '$EXPERIMENT_DIR/features.npy',
    '--labels', '$EXPERIMENT_DIR/stage1_labels.npy',
    '--out',    '$S1_MODELS',
    '--mlp',
    '--epochs', '$EPOCHS',
]
exec(open('$SCRIPT_DIR/train_model.py').read())
PYRUN1

echo ""
echo "[2/4] Stage 1 training complete: $(date)"
echo ""

# Verify Stage 1 output
python3 - <<'PYVERIFY1'
import torch
state = torch.load("data/k7_cascade_experiment/stage1_models/mlp_v2.pt",
                   map_location="cpu", weights_only=False)
if hasattr(state, "state_dict"): state = state.state_dict()
layers = [(k, v.shape) for k, v in state.items() if "weight" in k]
first, last = layers[0][1], layers[-1][1]
print(f"Stage 1: input={first[1]}  output={last[0]}")
assert first[1] == 9557, f"Expected input 9557, got {first[1]}"
assert last[0] == 2, f"Expected binary output (2), got {last[0]}"
print("PASS — Stage 1: (9557 → 2048 → 512 → 128 → 2) binary plasmid detector")
PYVERIFY1

# ── Step 3: Train Stage 2 — chromosome vs. phage ─────────────────────────
echo ""
echo "[3/4] Training Stage 2 (chromosome vs. phage) …"
echo "  Labels: $LABELS with --class-filter 1,2 (chr→0, phage→1)"
echo "  Samples: ~200k chr+phage (plasmid rows excluded at runtime)"
echo "  Start: $(date)"
echo ""

python3 - <<PYRUN2
import sys, torch
torch.set_num_threads($THREADS)
torch.set_num_interop_threads(4)
print(f"torch threads: {torch.get_num_threads()}")
sys.argv = [
    'train_model.py',
    '--data',         '$EXPERIMENT_DIR/features.npy',
    '--labels',       '$LABELS',
    '--class-filter', '1,2',
    '--out',          '$S2_MODELS',
    '--mlp',
    '--epochs',       '$EPOCHS',
]
exec(open('$SCRIPT_DIR/train_model.py').read())
PYRUN2

echo ""
echo "[3/4] Stage 2 training complete: $(date)"
echo ""

# Verify Stage 2 output
python3 - <<'PYVERIFY2'
import torch
state = torch.load("data/k7_cascade_experiment/stage2_models/mlp_v2.pt",
                   map_location="cpu", weights_only=False)
if hasattr(state, "state_dict"): state = state.state_dict()
layers = [(k, v.shape) for k, v in state.items() if "weight" in k]
first, last = layers[0][1], layers[-1][1]
print(f"Stage 2: input={first[1]}  output={last[0]}")
assert first[1] == 9557, f"Expected input 9557, got {first[1]}"
assert last[0] == 2, f"Expected binary output (2), got {last[0]}"
print("PASS — Stage 2: (9557 → 2048 → 512 → 128 → 2) chr/phage discriminator")
PYVERIFY2

# ── Step 4: Deploy ────────────────────────────────────────────────────────
echo ""
echo "[4/4] Deploying cascade models …"
mkdir -p data/models

cp "data/k7_cascade_experiment/stage1_models/mlp_v2.pt" \
   "data/models/mlp_cascade_stage1.pt"
echo "  Stage 1 → data/models/mlp_cascade_stage1.pt"

cp "data/k7_cascade_experiment/stage2_models/mlp_v2.pt" \
   "data/models/mlp_cascade_stage2.pt"
echo "  Stage 2 → data/models/mlp_cascade_stage2.pt"

echo ""
echo "============================================================"
echo " All done: $(date)"
echo ""
echo " Deployed models:"
echo "   data/models/mlp_cascade_stage1.pt  (plasmid vs. rest)"
echo "   data/models/mlp_cascade_stage2.pt  (chr vs. phage)"
echo ""
echo " Test on W1 metagenome:"
echo "   python scripts/predict_sequences.py \\"
echo "     --input data/test/W1.contigs.fa \\"
echo "     --stage1-model data/models/mlp_cascade_stage1.pt \\"
echo "     --stage2-model data/models/mlp_cascade_stage2.pt \\"
echo "     --no-marker-model \\"
echo "     --out results/W1_plasflow2/cascade_predictions.tsv"
echo ""
echo " Benchmark (delete cache first):"
echo "   rm -f data/benchmark/results/plasflow2_predictions.tsv"
echo "   python scripts/run_benchmark_evaluation.py \\"
echo "     --model data/models/mlp_cascade_stage1.pt \\"   # placeholder — benchmark needs cascade mode
echo "     --benchmark-dir data/benchmark"
echo ""
echo " NOTE: After benchmarking, run threshold calibration:"
echo "   python scripts/tune_cascade_thresholds.py \\"
echo "     --plasflow results/W1_plasflow2/cascade_predictions.tsv \\"
echo "     --benchmark data/benchmark/results/"
echo "============================================================"
