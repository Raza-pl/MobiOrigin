#!/usr/bin/env bash
# =============================================================================
# retrain_with_genomad.sh
#
# Full pipeline to retrain the XGBoost marker classifier with real geNomad SPM
# features for the 30k balanced training sequences.
#
# Steps
# -----
#   1. Reconstruct per-class training FASTAs from source databases
#   2. Run geNomad annotate on each training FASTA (--splits 4 for speed)
#   3. Rebuild marker NPZ with real geNomad SPM values
#   4. Retrain XGBoost marker classifier
#
# Usage
# -----
#   bash scripts/retrain_with_genomad.sh
#   bash scripts/retrain_with_genomad.sh --threads 8   # default: 16
#
# Prerequisites
# -------------
#   conda activate plasflow2
#   genomad download-database data/databases/genomad_db/   # if not done
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

THREADS=16
GENOMAD_DB="data/databases/genomad_db"
MARKER_WORK="data/marker_work"
SCRIPTS="scripts"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --threads) THREADS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "================================================================="
echo "STEP 1: Build per-class training FASTAs from source databases"
echo "================================================================="
python "${SCRIPTS}/build_training_fasta.py"

echo ""
echo "================================================================="
echo "STEP 2: Run geNomad annotate on each training FASTA"
echo "================================================================="
for CLS in plasmid chromosome phage; do
  FASTA="${MARKER_WORK}/${CLS}_training.fna"
  ANN_DIR="${MARKER_WORK}/${CLS}_genomad_ann"

  if [[ ! -f "$FASTA" ]]; then
    echo "WARNING: $FASTA not found — skipping geNomad annotate for $CLS"
    continue
  fi

  if [[ -d "$ANN_DIR" ]] && ls "${ANN_DIR}"/*_genes.tsv 1>/dev/null 2>&1; then
    echo "[$CLS] geNomad output already exists in $ANN_DIR — skipping."
    echo "       Delete the directory to re-run."
    continue
  fi

  echo "[$CLS] Running genomad annotate (threads=$THREADS, splits=4) …"
  genomad annotate \
    "$FASTA" \
    "$ANN_DIR" \
    "$GENOMAD_DB" \
    --threads "${THREADS}" \
    --splits 4 \
    --cleanup
  echo "[$CLS] geNomad annotate done."
done

echo ""
echo "================================================================="
echo "STEP 3: Rebuild marker NPZ with real geNomad SPM features"
echo "================================================================="
python "${SCRIPTS}/rebuild_marker_npz_with_genomad.py" \
  --base-npz    data/marker_features_balanced_28.npz \
  --ann-dir     "${MARKER_WORK}" \
  --proteins-dir "${MARKER_WORK}" \
  --out         data/marker_features_balanced_28_genomad.npz

echo ""
echo "================================================================="
echo "STEP 4: Retrain XGBoost marker classifier"
echo "================================================================="
python "${SCRIPTS}/train_marker_model.py" \
  --features data/marker_features_balanced_28_genomad.npz \
  --out      data/models/

echo ""
echo "================================================================="
echo "Done! New model saved to data/models/marker_xgb.pkl"
echo ""
echo "Run benchmark to evaluate:"
echo "  python scripts/benchmark.py \\"
echo "      --test-fasta  data/benchmark/sequences.fna \\"
echo "      --ground-truth data/benchmark/ground_truth.tsv \\"
echo "      --model       data/models/mlp_v2.pt \\"
echo "      --xgb-model   data/models/marker_xgb.pkl \\"
echo "      --annotations data/benchmark/annotations_with_genomad.tsv \\"
echo "      --out-dir     data/benchmark/results/"
echo "================================================================="
