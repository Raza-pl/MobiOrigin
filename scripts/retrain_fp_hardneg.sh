#!/usr/bin/env bash
# retrain_fp_hardneg.sh — Retrain binary MLP with FP-genome hard negatives.
#
# WHAT THIS DOES
# --------------
# Adds the 29 chromosome sequences that PlasFlow v2 currently false-positives
# (high-confidence mis-classifications) as hard negatives in the training set.
# These "plasmid-like" chromosomes (Acinetobacter baumannii ACICU, etc.) were
# not in previous hard-negative sets and are the dominant source of FP errors.
#
# After retraining, runs a threshold sweep to pick the optimal per-length
# plasmid threshold (tune_thresholds_binary.py).
#
# PREREQUISITES
# -------------
#   python scripts/collect_fp_hard_negatives.py   # collect FP chr FASTs first
#
# USAGE
#   conda activate plasflow2
#   python scripts/collect_fp_hard_negatives.py
#   nohup bash scripts/retrain_fp_hardneg.sh > data/retrain_fp_hardneg.log 2>&1 &
#   tail -f data/retrain_fp_hardneg.log
#
# TIMING (MacBook Pro M-series, ~8 threads)
#   Step 0 — merge hard-neg dirs : < 1 min
#   Step 1 — build dataset       : 30–60 min
#   Step 2 — train MLP           : 45–60 min
#   Step 3 — benchmark           : 30 min
#   Step 4 — threshold sweep     : 2 min
#   Total                        : ~2–3 hours

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$ROOT/data"
SCRIPTS_DIR="$ROOT/scripts"

EXPERIMENT_DIR="$DATA_DIR/fp_hardneg_experiment"
EXPERIMENT_MODELS="$EXPERIMENT_DIR/models"
MLP_OUT="$EXPERIMENT_MODELS/mlp_v2.pt"

# Hard negatives: original set + new FP-genome chromosomes
ORIG_HN_DIR="$DATA_DIR/hard_negatives"
FP_HN_DIR="$DATA_DIR/fp_hard_negatives"
COMBINED_HN_DIR="$DATA_DIR/combined_hard_negatives"

CHROMID_DIR="$DATA_DIR/databases/chromids"
CHROMID_FASTA="$CHROMID_DIR/chromids.fna"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "  PlasFlow v2 — FP hard-negative retrain"
echo "  Start: $(date)"
echo "  Experiment dir: $EXPERIMENT_DIR"
echo "============================================================"
echo ""

# ── Guard: FP hard negatives must exist ───────────────────────────────────────
if [ ! -d "$FP_HN_DIR" ] || [ "$(find "$FP_HN_DIR" -name '*.fna' 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
    echo "ERROR: FP hard negatives not found in $FP_HN_DIR"
    echo "  Run first: python scripts/collect_fp_hard_negatives.py"
    exit 1
fi
N_FP=$(find "$FP_HN_DIR" -name "*.fna" | wc -l | tr -d ' ')
echo "Found $N_FP FP hard-negative genomes in $FP_HN_DIR"

mkdir -p "$EXPERIMENT_DIR" "$EXPERIMENT_MODELS" "$CHROMID_DIR"

# ── Step 0a: Merge original + FP hard-negative dirs ───────────────────────────
echo ""
echo "[0a/4] Merging hard-negative directories …"
mkdir -p "$COMBINED_HN_DIR"
# Copy originals (symlink to save disk)
for f in "$ORIG_HN_DIR"/*.fna; do
    fname="$(basename "$f")"
    [ ! -e "$COMBINED_HN_DIR/$fname" ] && cp -n "$f" "$COMBINED_HN_DIR/$fname" && echo "  + $fname"
done
# Copy FP genomes
for f in "$FP_HN_DIR"/*.fna; do
    fname="$(basename "$f")"
    [ ! -e "$COMBINED_HN_DIR/$fname" ] && cp -n "$f" "$COMBINED_HN_DIR/$fname" && echo "  + $fname"
done
N_COMBINED=$(find "$COMBINED_HN_DIR" -name "*.fna" | wc -l | tr -d ' ')
echo "Combined hard-negative dir: $COMBINED_HN_DIR ($N_COMBINED files)"

# ── Step 0b: Chromid sequences ────────────────────────────────────────────────
echo ""
echo "[0b/4] Chromid sequences …"
if [ -f "$CHROMID_FASTA" ]; then
    N_CHROMIDS=$(grep -c "^>" "$CHROMID_FASTA" 2>/dev/null || echo "0")
    echo "  Already exists: $CHROMID_FASTA ($N_CHROMIDS seqs) — skipping"
else
    python3 -u "$SCRIPTS_DIR/extract_chromid_sequences.py" \
        --out "$CHROMID_FASTA" \
        --min-length 100000
    N_CHROMIDS=$(grep -c "^>" "$CHROMID_FASTA" 2>/dev/null || echo "0")
    echo "  Extracted $N_CHROMIDS chromid sequences → $CHROMID_FASTA"
fi

# ── Step 1: Build dataset ─────────────────────────────────────────────────────
echo ""
echo "[1/4] Building binary dataset …"
echo "  Hard-negative dir : $COMBINED_HN_DIR ($N_COMBINED genomes)"
echo "  Hard-negative max : 60000 windows (raised from 30000 to cover all FP genomes)"
echo "  Start: $(date)"
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
    --hard-negative-dir          "$COMBINED_HN_DIR" \
    --hard-negative-max          60000 \
    --hard-negative-window-sizes 10000 \
    --binary-plasmid \
    --chromid-dir    "$CHROMID_DIR" \
    --out  "$EXPERIMENT_DIR" \
    --seed 42

echo ""
echo "[1/4] Dataset build complete: $(date)"

# ── Step 2: Train binary MLP ──────────────────────────────────────────────────
echo ""
echo "[2/4] Training binary MLP …"
echo "  Start: $(date)"
echo ""

python3 -u "$SCRIPTS_DIR/train_model.py" \
    --data             "$EXPERIMENT_DIR/features.npy" \
    --labels           "$EXPERIMENT_DIR/labels.npy" \
    --out              "$EXPERIMENT_MODELS" \
    --mlp              \
    --epochs           50 \
    --no-class-weights

echo ""
echo "[2/4] MLP training complete: $(date)"

# ── Step 3: Benchmark ─────────────────────────────────────────────────────────
echo ""
echo "[3/4] Running benchmark …"
echo "  Start: $(date)"
echo ""

bash "$SCRIPTS_DIR/run_benchmark.sh" \
    --no-marker-model \
    --model          "$MLP_OUT" \
    --annotation-tsv "$DATA_DIR/benchmark/annotations_with_genomad.tsv" \
    2>&1 | tee "$EXPERIMENT_DIR/benchmark_fp_hardneg.log"

echo ""
echo "[3/4] Benchmark complete: $(date)"

# ── Step 4: Threshold sweep ───────────────────────────────────────────────────
echo ""
echo "[4/4] Running threshold sweep to find optimal per-length plasmid thresholds …"
echo "  Start: $(date)"
echo ""

# Get the predictions file written by run_benchmark.sh
PRED_TSV=$(find "$DATA_DIR/benchmark" -name "plasflow2_predictions.tsv" -newer "$MLP_OUT" | head -1)
if [ -z "$PRED_TSV" ]; then
    echo "  WARNING: could not find fresh predictions TSV — skipping sweep"
else
    python3 -u "$SCRIPTS_DIR/tune_thresholds_binary.py" \
        --predictions "$PRED_TSV" \
        --ground-truth "$DATA_DIR/benchmark/ground_truth.tsv" \
        --out "$EXPERIMENT_DIR/threshold_sweep.json"
    echo "  Sweep saved to $EXPERIMENT_DIR/threshold_sweep.json"
fi

echo ""
echo "============================================================"
echo "  Retrain complete: $(date)"
echo ""
echo "  If plasmid F1 improved, deploy the new model:"
echo "    cp $MLP_OUT $DATA_DIR/models/mlp_v2.pt"
echo "    # Update LENGTH_THRESHOLD_TIERS in src/plasflow2/classify/predict.py"
echo "    # using values from $EXPERIMENT_DIR/threshold_sweep.json"
echo "============================================================"
