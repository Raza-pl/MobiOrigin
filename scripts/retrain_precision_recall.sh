#!/usr/bin/env bash
# retrain_precision_recall.sh
#
# Two-phase retrain to improve PlasFlow v2 precision and recall:
#
# PHASE 1 — XGBoost retrain (fast, ~5 min total):
#   - Augment marker NPZ with near-miss FN plasmids (×10 oversampled)
#     and composition FP chromosomes as hard negatives
#   - Retrain XGBoost on augmented NPZ
#   - Benchmark to verify improvement
#   Expected: +6-8 TP recovered (recall), −5 FP (composition precision)
#
# PHASE 2 — MLP hard-negative retrain (slow, ~2-3 hrs):
#   - Extract exact 15 composition FP windows at their exact offsets
#   - Add to fp_hard_negatives/ for MLP to learn these specific sequences
#   - Run full MLP retrain (retrain_fp_hardneg.sh)
#   Expected: 15 composition FPs eliminated from MLP output
#
# Usage:
#   conda activate plasflow2
#   bash scripts/retrain_precision_recall.sh [--phase1-only] [--phase2-only]
#
# Default: runs phase 1 automatically, phase 2 as background nohup job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

PHASE1=true
PHASE2=true
for arg in "$@"; do
    case $arg in
        --phase1-only) PHASE2=false ;;
        --phase2-only) PHASE1=false ;;
    esac
done

echo "============================================================"
echo "  PlasFlow v2 — Precision + Recall retrain"
echo "  Start: $(date)"
echo "  Phase 1 (XGBoost): $PHASE1"
echo "  Phase 2 (MLP):     $PHASE2"
echo "============================================================"
echo ""

# ── PHASE 1: XGBoost augment + retrain ───────────────────────────────────────
if $PHASE1; then
    echo "[Phase 1a] Augmenting marker NPZ with near-miss FNs and composition FPs …"
    echo "  Start: $(date)"
    python3 "$SCRIPT_DIR/augment_marker_npz.py"
    echo "  Done: $(date)"
    echo ""

    echo "[Phase 1b] Retraining XGBoost on augmented NPZ …"
    echo "  Start: $(date)"
    python3 "$SCRIPT_DIR/train_marker_model.py" \
        --features "$ROOT/data/marker_features_augmented.npz" \
        --out       "$ROOT/data/models/"
    echo "  Done: $(date)"
    echo "  Model saved: data/models/marker_xgb.pkl"
    echo ""

    echo "[Phase 1c] Benchmarking with new XGBoost …"
    echo "  Start: $(date)"
    bash "$SCRIPT_DIR/run_benchmark.sh" \
        --annotation-tsv "$ROOT/data/benchmark/annotations_with_genomad.tsv" \
        2>&1 | tee "$ROOT/data/retrain_precision_recall_phase1.log"
    echo "  Done: $(date)"
    echo ""
    echo "=== Phase 1 complete. Check benchmark above for recall/precision changes. ==="
    echo ""
fi

# ── PHASE 2: MLP hard-negative retrain ────────────────────────────────────────
if $PHASE2; then
    echo "[Phase 2a] Extracting exact composition FP windows for MLP hard-negative set …"
    python3 "$SCRIPT_DIR/collect_composition_fp_hardnegs.py"
    echo ""

    echo "[Phase 2b] Launching MLP retrain in background (nohup) …"
    LOG="$ROOT/data/retrain_composition_fp_hardneg.log"
    nohup bash "$SCRIPT_DIR/retrain_fp_hardneg.sh" > "$LOG" 2>&1 &
    BGPID=$!
    echo "  MLP retrain PID: $BGPID"
    echo "  Log: $LOG"
    echo "  Monitor: tail -f $LOG"
    echo ""
    echo "  The MLP retrain will take 2-3 hours."
    echo "  When it finishes, deploy the new model:"
    echo "    cp data/fp_hardneg_experiment/models/mlp_v2.pt data/models/mlp_v2.pt"
    echo "  Then re-run augment_marker_npz.py + train_marker_model.py"
    echo "  (to update XGBoost MLP score columns for the new MLP)."
fi

echo ""
echo "============================================================"
echo "  retrain_precision_recall.sh complete: $(date)"
echo "============================================================"
