#!/usr/bin/env bash
# run_w1_predictions.sh — Re-run PlasFlow v2 on W1 with 3-class MLP
#
# Saves predictions to results/W1_plasflow2/all_predictions_no_marker.tsv
# which is then used by tune_phage_threshold.py for threshold analysis.
#
# USAGE (from project root, plasflow2 conda env):
#   bash scripts/run_w1_predictions.sh [--with-marker]
#
#   Default: MLP-only (required for threshold tuning analysis)
#   --with-marker: also run with new 3-class marker_xgb.pkl (after task 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

WITH_MARKER=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-marker) WITH_MARKER=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# macOS ARM segfault fix
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p results/W1_plasflow2

echo "============================================================"
echo " PlasFlow v2 — W1 predictions (3-class MLP)"
echo " Start : $(date)"
echo "============================================================"
echo ""

# ── MLP-only run (for threshold tuning) ────────────────────────────────────
echo "[1/1] MLP-only predictions …"
python3 scripts/predict_sequences.py \
    --input  data/test/W1.contigs.fa \
    --model  data/models/mlp_v2.pt \
    --no-marker-model \
    --out    results/W1_plasflow2/all_predictions_no_marker.tsv

echo ""
echo "MLP-only predictions written to results/W1_plasflow2/all_predictions_no_marker.tsv"
echo ""

# ── With marker (optional, after task 1 complete) ──────────────────────────
if [ "$WITH_MARKER" -eq 1 ]; then
    if [ ! -f "data/models/marker_xgb.pkl" ]; then
        echo "WARNING: marker_xgb.pkl not found — skipping marker run"
    else
        echo "[2/2] MLP + marker XGBoost predictions …"
        python3 scripts/predict_sequences.py \
            --input  data/test/W1.contigs.fa \
            --model  data/models/mlp_v2.pt \
            --marker-model data/models/marker_xgb.pkl \
            --out    results/W1_plasflow2/all_predictions_with_marker.tsv
        echo "Marker predictions written to results/W1_plasflow2/all_predictions_with_marker.tsv"
    fi
fi

echo ""
echo "============================================================"
echo " Done: $(date)"
echo ""
echo " Next — run phage threshold analysis:"
echo "   python scripts/tune_phage_threshold.py \\"
echo "     --plasflow results/W1_plasflow2/all_predictions_no_marker.tsv \\"
echo "     --out      results/phage_threshold_analysis/"
echo "============================================================"
