#!/usr/bin/env bash
# run_benchmark_3class.sh — Re-run benchmark with the 3-class MLP
#
# Deletes the cached predictions TSV so the benchmark runs fresh inference
# instead of reusing the old binary-model results.
#
# Run this AFTER retrain_marker_3class.sh — the benchmark will auto-detect
# and use the new marker_xgb.pkl.  If you want MLP-only numbers first,
# pass --no-marker.
#
# USAGE (from project root, plasflow2 conda env):
#   bash scripts/run_benchmark_3class.sh [--no-marker]
#
# OUTPUTS (data/benchmark/results/):
#   plasflow2_metrics.json              — precision/recall/F1 all seqs
#   plasflow2_classified_only_metrics.json — classified-only F1
#   plasflow2_by_length.csv             — F1 by contig length bucket
#   comparison_table.csv                — v1 / v2 / geNomad side-by-side

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

USE_MARKER=1
MARKER_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-marker) USE_MARKER=0; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# macOS ARM segfault fix
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CACHE="data/benchmark/results/plasflow2_predictions.tsv"
if [ -f "$CACHE" ]; then
    echo "Deleting cached predictions: $CACHE"
    rm -f "$CACHE"
fi

echo "============================================================"
echo " PlasFlow v2 — 3-class benchmark re-run"
echo " Start : $(date)"
if [ "$USE_MARKER" -eq 0 ]; then
    echo " Mode  : MLP-only (--no-marker)"
else
    if [ -f "data/models/marker_xgb.pkl" ]; then
        echo " Mode  : MLP + marker XGBoost (data/models/marker_xgb.pkl)"
        MARKER_ARGS="--marker-model data/models/marker_xgb.pkl"
    else
        echo " Mode  : MLP-only (marker_xgb.pkl not found)"
    fi
fi
echo "============================================================"
echo ""

python3 scripts/run_benchmark_evaluation.py \
    --model data/models/mlp_v2.pt \
    --benchmark-dir data/benchmark \
    $MARKER_ARGS \
    2>&1 | tee data/benchmark/results/benchmark_3class_run_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "============================================================"
echo " Done: $(date)"
echo " Results: data/benchmark/results/"
echo "  plasflow2_metrics.json"
echo "  plasflow2_classified_only_metrics.json"
echo "============================================================"
