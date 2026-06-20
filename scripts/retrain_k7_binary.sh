#!/usr/bin/env bash
# retrain_k7_binary.sh — Binary plasmid/chromosome classifier.
#
# WHY BINARY?
# -----------
# The 3-class softmax (plasmid / chromosome / phage) is structurally flawed
# for plasmid detection:
#   - Chromosome class averages only 0.638 mean score (barely above random 0.333)
#     because phage probability bleeds in via softmax competition.
#   - PR ceiling at any single threshold: F1 ≤ 0.267 (k=7 v2 run).
#   - PlasFlow v1 achieves F1=0.939 as a binary classifier with the same k=7
#     features — the binary formulation is the right architecture.
#
# THIS APPROACH (Option B — Phage-as-Chromosome)
# -----------------------------------------------
# - Phage training windows are relabeled as "chromosome" via --binary-plasmid.
# - The MLP trains a clean 2-class boundary: plasmid vs not-plasmid.
# - At inference, existing phage suppression in predict.py (v_marker_freq,
#   viral hallmark genes) handles the phage case — those 6,657 sequences are
#   already correctly suppressed in benchmarks.
# - No marker XGBoost in this run (step 3 skipped); binary MLP alone is tested.
#
# PREREQUISITES
# -------------
#   python scripts/extract_chr2_hard_neg.py
#   (Hard-negative genomes already downloaded)
#
# USAGE
#   nohup bash scripts/retrain_k7_binary.sh > data/retrain_k7_binary.log 2>&1 &
#   tail -f data/retrain_k7_binary.log
#
# TIMING (approximate, MacBook Pro M-series)
#   Step 1 — build dataset  : 30–60 min
#   Step 2 — train MLP      : 45–60 min  (2-class, faster than 3-class)
#   Step 3 — benchmark      : 30 min
#   Total                   : ~2–3 hours

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$ROOT/data"
SCRIPTS_DIR="$ROOT/scripts"
SRC_DIR="$ROOT/src"

EXPERIMENT_DIR="$DATA_DIR/k7_binary_experiment"
EXPERIMENT_MODELS="$EXPERIMENT_DIR/models"
MLP_OUT="$EXPERIMENT_MODELS/mlp_v2.pt"

HARD_NEG_DIR="$DATA_DIR/hard_negatives"
CHR2_FILE="$HARD_NEG_DIR/GCF_000017645.1_chr2_chromid.fna"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"

# ── Checks ────────────────────────────────────────────────────────────────────
if [ ! -d "$HARD_NEG_DIR" ] || [ "$(find "$HARD_NEG_DIR" -name '*.fna' 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
    echo "ERROR: Hard-negative genomes not found in $HARD_NEG_DIR"
    echo "  Run: python scripts/download_hard_negatives.py"
    exit 1
fi

if [ ! -f "$CHR2_FILE" ]; then
    echo "ERROR: Chr2 standalone file not found: $CHR2_FILE"
    echo "  Run: python scripts/extract_chr2_hard_neg.py"
    exit 1
fi

N_FNA=$(find "$HARD_NEG_DIR" -name "*.fna" | wc -l | tr -d ' ')

echo "============================================================"
echo "  PlasFlow v2 — k=7 canonical BINARY (plasmid vs chromosome)"
echo "  Start  : $(date)"
echo "  Output : $EXPERIMENT_DIR"
echo "  Hard-neg files: $N_FNA"
echo "  Mode: --binary-plasmid (phage windows → chromosome label)"
echo "============================================================"
echo ""

mkdir -p "$EXPERIMENT_DIR" "$EXPERIMENT_MODELS"

# ── Step 1: Build dataset (binary labels) ─────────────────────────────────────
echo "[1/3] Building binary dataset (plasmid=0, chromosome=1, phage remapped→1) …"
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
    --hard-negative-dir         "$HARD_NEG_DIR" \
    --hard-negative-max         30000 \
    --hard-negative-window-sizes 10000 \
    --binary-plasmid \
    --out  "$EXPERIMENT_DIR" \
    --seed 42

echo ""
echo "[1/3] Dataset build complete: $(date)"
echo ""

# ── Step 2: Train binary MLP ──────────────────────────────────────────────────
echo "[2/3] Training binary MLP (9557→2048→512→128→2) …"
echo "  num_classes will be inferred automatically from unique labels in labels.npy"
echo "  Start: $(date)"
echo ""

python3 -u "$SCRIPTS_DIR/train_model.py" \
    --data             "$EXPERIMENT_DIR/features.npy" \
    --labels           "$EXPERIMENT_DIR/labels.npy" \
    --out              "$EXPERIMENT_MODELS" \
    --mlp              \
    --epochs           50 \
    --no-class-weights

echo "[2/3] MLP training complete: $(date)"
echo ""

# ── Step 3: Benchmark (MLP only — no marker XGBoost for binary model) ─────────
echo "[3/3] Running benchmark (binary MLP, no marker XGBoost) …"
echo "  NOTE: marker XGBoost is not used here because it was trained on 3-class"
echo "        labels and would require full retraining to support binary output."
echo "        The binary MLP is evaluated standalone."
echo "  Start: $(date)"
echo ""

bash "$SCRIPTS_DIR/run_benchmark.sh" \
    --no-marker-model \
    --model          "$MLP_OUT" \
    --annotation-tsv "$DATA_DIR/benchmark/annotations_with_genomad.tsv" \
    2>&1 | tee "$EXPERIMENT_DIR/benchmark_k7_binary.log"

echo ""
echo "------------------------------------------------------------"
echo "  Experiment complete: $(date)"
echo ""
echo "  Comparison:"
echo "    k=5 + k=6-PCA (baseline)               : F1=0.688"
echo "    k=7 canonical v2 (3-class, tuned thresh): F1≈0.691"
echo "    k=7 canonical binary (this run)         : see benchmark_k7_binary.log"
echo "    PlasFlow v1 (target)                    : F1=0.939"
echo ""
echo "  If binary F1 > 0.85:"
echo "    cp $MLP_OUT $DATA_DIR/models/mlp_v2.pt"
echo "    # Then retrain marker XGBoost with binary MLP scores:"
echo "    # python scripts/update_marker_mlp_scores.py --npz ... --model $MLP_OUT"
echo "    # python scripts/train_marker_model.py --features ... --out $EXPERIMENT_MODELS"
echo "============================================================"
