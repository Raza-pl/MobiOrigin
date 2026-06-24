#!/usr/bin/env bash
# retrain_3class.sh — Rebuild dataset and retrain MLP as 3-class
#                     (plasmid / chromosome / phage)
#
# This script mirrors retrain_k7_binary.sh exactly, but WITHOUT --binary-plasmid,
# so phage sequences are kept as class 2 instead of being remapped to chromosome.
# The resulting model has shape (9557 → 2048 → 512 → 128 → 3), compatible with
# the production predict() pipeline.
#
# PREREQUISITES (same as retrain_k7_binary.sh):
#   - data/gtdb_genomes/bacteria/      (GTDB genomes — already present)
#   - data/databases/plasmids/         (PLSDB + COMPASS — already present)
#   - data/databases/inphared/         (INPHARED phages — already present)
#   - data/hard_negatives/             (hard-negative chromosomes — already present)
#   - data/databases/chromids/         (chromids — already present)
#
# USAGE (from project root, plasflow2 conda env):
#   bash scripts/retrain_3class.sh [--max-per-class N] [--threads N]
#
#   Defaults: --max-per-class 100000  --threads 16
#   For full dataset (slower): --max-per-class 300000
#
# ESTIMATED TIME on Apple Silicon (16 threads):
#   Dataset build : ~60–90 min  (GTDB windowing, k=7 feature extraction)
#   MLP training  : ~45–90 min  (50 epochs, 100k/class)
#   Total         : ~2–3 h
#
# OUTPUTS:
#   data/k7_3class_experiment/features.npy   (9557-dim, 3-class)
#   data/k7_3class_experiment/labels.npy
#   data/k7_3class_experiment/models/mlp_v2.pt  (3-class model)
#   data/models/mlp_v2.pt                    (replaced with 3-class model)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

THREADS=16
MAX_PER_CLASS=100000

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-per-class) MAX_PER_CLASS="$2"; shift 2 ;;
        --threads)       THREADS="$2";       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DATA_DIR="$ROOT/data"
SCRIPTS_DIR="$ROOT/scripts"
EXPERIMENT_DIR="$DATA_DIR/k7_3class_experiment"
EXPERIMENT_MODELS="$EXPERIMENT_DIR/models"
HARD_NEG_DIR="$DATA_DIR/hard_negatives"
CHROMID_DIR="$DATA_DIR/databases/chromids"
CHROMID_FASTA="$CHROMID_DIR/chromids.fna"

mkdir -p "$EXPERIMENT_DIR" "$EXPERIMENT_MODELS"

echo "============================================================"
echo " PlasFlow v2 — 3-class MLP retrain (plasmid/chromosome/phage)"
echo " Start      : $(date)"
echo " Max/class  : $MAX_PER_CLASS"
echo " Threads    : $THREADS"
echo " Experiment : $EXPERIMENT_DIR"
echo "============================================================"
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────
for req in \
    "$HARD_NEG_DIR" \
    "$DATA_DIR/databases/plasmids/plsdb.fasta" \
    "$DATA_DIR/databases/inphared/inphared_phages.fa.gz" \
    "$DATA_DIR/gtdb_genomes/bacteria"
do
    if [ ! -e "$req" ]; then
        echo "ERROR: missing prerequisite: $req"
        exit 1
    fi
done

N_FNA=$(find "$HARD_NEG_DIR" -name "*.fna" | wc -l | tr -d ' ')
echo "  Hard-negative genomes: $N_FNA .fna files in $HARD_NEG_DIR"

# ── Chromid FASTA (same as binary retrain) ─────────────────────────────────
if [ ! -f "$CHROMID_FASTA" ] || [ ! -s "$CHROMID_FASTA" ]; then
    echo "Building chromid FASTA from $CHROMID_DIR ..."
    python3 "$SCRIPTS_DIR/download_chromid_sequences.py" \
        --out "$CHROMID_FASTA" \
        2>&1 | tail -5
else
    echo "  Chromid FASTA already exists: $CHROMID_FASTA"
fi

# ── Step 1: Build 3-class dataset (9557-dim features) ─────────────────────
echo ""
echo "[1/3] Building 3-class dataset (plasmid=0, chromosome=1, phage=2) …"
echo "  Note: NO --binary-plasmid flag — phage kept as class 2"
echo "  Start: $(date)"
echo ""

python3 -u "$SCRIPTS_DIR/build_dataset.py" \
    --plasmid-files  "$DATA_DIR/databases/plasmids/plsdb.fasta,$DATA_DIR/databases/plasmids/COMPASS.fna" \
    --gtdb-dir       "$DATA_DIR/gtdb_genomes/bacteria" \
    --gtdb-n-genomes 500 \
    --data-dir       "$DATA_DIR/databases" \
    --window-sizes   5000,10000 \
    --min-length     4500 \
    --max-per-class  "$MAX_PER_CLASS" \
    --skip-download \
    --hard-negative-dir          "$HARD_NEG_DIR" \
    --hard-negative-max          30000 \
    --hard-negative-window-sizes 10000 \
    --out  "$EXPERIMENT_DIR" \
    --seed 42 \
    2>&1 | tee "$EXPERIMENT_DIR/build_dataset.log"

echo ""
echo "[1/3] Dataset build complete: $(date)"
echo ""

# Verify 3 classes
python3 - <<PYCHECK
import numpy as np, collections
labels = np.load("$EXPERIMENT_DIR/labels.npy")
counts = collections.Counter(labels.tolist())
feat = np.load("$EXPERIMENT_DIR/features.npy", mmap_mode="r")
print("Features shape:", feat.shape)
print("Label distribution:")
for k, v in sorted(counts.items()):
    print(f"  class {k}: {v:,}")
assert feat.shape[1] == 9557, f"Expected 9557 features, got {feat.shape[1]}"
assert len(counts) == 3, f"Expected 3 classes, got {len(counts)}"
print("OK — 9557-dim 3-class dataset confirmed")
PYCHECK

# ── Step 2: Train 3-class MLP ──────────────────────────────────────────────
echo ""
echo "[2/3] Training 3-class MLP (9557→2048→512→128→3) …"
echo "  Start: $(date)"
echo ""

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export VECLIB_MAXIMUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

python3 - <<PYRUN
import os, sys, torch
torch.set_num_threads($THREADS)
torch.set_num_interop_threads(4)
print(f"torch.get_num_threads() = {torch.get_num_threads()}")

sys.argv = [
    'train_model.py',
    '--data',   '$EXPERIMENT_DIR/features.npy',
    '--labels', '$EXPERIMENT_DIR/labels.npy',
    '--out',    '$EXPERIMENT_MODELS',
    '--mlp',
    '--epochs', '50',
]
exec(open('$SCRIPTS_DIR/train_model.py').read())
PYRUN

echo ""
echo "[2/3] Training complete: $(date)"

# ── Step 3: Verify and deploy ──────────────────────────────────────────────
python3 - <<'PYVERIFY'
import torch
state = torch.load("data/k7_3class_experiment/models/mlp_v2.pt",
                   map_location="cpu", weights_only=False)
if hasattr(state, "state_dict"): state = state.state_dict()
layers = [(k, v.shape) for k, v in state.items() if "weight" in k]
first, last = layers[0][1], layers[-1][1]
print(f"First layer: {first}  Last layer: {last}")
assert first[1] == 9557, f"Expected input_dim=9557, got {first[1]}"
assert last[0]  == 3,    f"Expected 3 output classes, got {last[0]}"
print("PASS — model shape (9557→…→3) verified")
PYVERIFY

# Backup existing model and deploy
BACKUP="$DATA_DIR/models/mlp_v2_binary_backup_$(date +%Y%m%d_%H%M%S).pt"
cp "$DATA_DIR/models/mlp_v2.pt" "$BACKUP"
echo "Backed up current model → $BACKUP"

cp "$EXPERIMENT_MODELS/mlp_v2.pt" "$DATA_DIR/models/mlp_v2.pt"
echo "Deployed new 3-class model → data/models/mlp_v2.pt"

# ── Step 3: Quick benchmark ────────────────────────────────────────────────
echo ""
echo "[3/3] Running benchmark with new 3-class model …"

# Delete cached predictions to force fresh benchmark run
rm -f "$DATA_DIR/benchmark/results/plasflow2_predictions.tsv"

python3 "$SCRIPTS_DIR/run_benchmark_evaluation.py" \
    --model "$DATA_DIR/models/mlp_v2.pt" \
    --benchmark-dir "$DATA_DIR/benchmark" \
    2>&1 | tail -30

echo ""
echo "============================================================"
echo " All done: $(date)"
echo " 3-class model: data/models/mlp_v2.pt"
echo " Binary backup: $BACKUP"
echo "============================================================"
