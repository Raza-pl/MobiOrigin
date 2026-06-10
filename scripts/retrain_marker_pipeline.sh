#!/usr/bin/env bash
# retrain_marker_pipeline.sh
#
# End-to-end pipeline to retrain the marker XGBoost second-stage classifier:
#
#   Step 1. Update MLP score columns in existing marker_features.npz
#           (preserves real DIAMOND biological features from the original run)
#
#   Step 2. Retrain marker XGBoost on updated features
#
#   Step 3. (Optional) Annotate benchmark sequences with DIAMOND
#           Requires: diamond (conda install -c bioconda diamond)
#
#   Step 4. (Optional) Re-run benchmark with marker XGBoost
#
# Usage
# -----
#   bash scripts/retrain_marker_pipeline.sh 2>&1 | tee data/retrain_marker.log
#
# Prerequisites
# -------------
#   pip install pyrodigal xgboost scikit-learn
#   (DIAMOND only needed for step 3 — conda install -c bioconda diamond)
#
# Expected runtime (Apple M-series):
#   Step 1: ~30 min  (90k sequences × 1493 features)
#   Step 2: ~2 min   (XGBoost training)
#   Step 3: ~15 min  (pyrodigal + DIAMOND on benchmark)
#   Step 4: ~5 min   (prediction + benchmark eval)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Thread caps (prevents macOS ARM BLAS race → segfault) ──────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

DATA_DIR="$ROOT/data"
MODELS_DIR="$DATA_DIR/models"
BENCH_DIR="$DATA_DIR/benchmark"

echo "============================================================"
echo "  PlasFlow v2 — Marker XGBoost retrain pipeline"
echo "  Start: $(date)"
echo "  Root : $ROOT"
echo "============================================================"
echo ""

# ── Step 1: Update MLP score columns ───────────────────────────────────────
echo "[1/4] Updating MLP scores in marker_features.npz …"
echo "  This re-runs the new 1493-dim MLP on re-sampled sequences."
echo "  Biological features (DIAMOND hits) are preserved."
echo "  Start: $(date)"
echo ""

python3 "$ROOT/scripts/update_marker_mlp_scores.py" \
    --npz   "$DATA_DIR/marker_features.npz" \
    --model "$MODELS_DIR/mlp_v2.pt"

echo ""
echo "  MLP score update done: $(date)"
echo ""

# ── Step 2: Retrain XGBoost ─────────────────────────────────────────────────
echo "[2/4] Training marker XGBoost …"

python3 "$ROOT/scripts/train_marker_model.py" \
    --features "$DATA_DIR/marker_features.npz" \
    --out      "$MODELS_DIR"

echo ""
echo "  XGBoost training done: $(date)"
echo "  Model saved: $MODELS_DIR/marker_xgb.pkl"
echo ""

# ── Step 3: Annotate benchmark sequences (optional, requires DIAMOND) ───────
echo "[3/4] Annotating benchmark sequences with DIAMOND …"
echo "  Checking if DIAMOND is available …"

if ! command -v diamond &> /dev/null; then
    echo "  WARNING: 'diamond' not found in PATH."
    echo "  Install with: conda install -c bioconda diamond"
    echo "  OR: brew install diamond"
    echo "  Skipping benchmark annotation — will use MLP-only marker features."
    echo ""
    ANNOTATION_TSV=""
else
    echo "  DIAMOND found: $(diamond version 2>&1 | head -1)"

    # Locate benchmark FASTA
    BENCH_FASTA=""
    for candidate in \
        "$BENCH_DIR/benchmark.fna" \
        "$BENCH_DIR/benchmark.fasta" \
        "$BENCH_DIR/benchmark.fa" \
        "$BENCH_DIR/test_sequences.fasta" \
        "$BENCH_DIR/sequences.fasta"
    do
        if [ -f "$candidate" ]; then
            BENCH_FASTA="$candidate"
            break
        fi
    done

    if [ -z "$BENCH_FASTA" ]; then
        echo "  WARNING: Benchmark FASTA not found (expected at $BENCH_DIR/benchmark.fasta)"
        echo "  Skipping benchmark annotation."
        ANNOTATION_TSV=""
    else
        echo "  Benchmark FASTA: $BENCH_FASTA"
        ANNOTATION_TSV="$BENCH_DIR/annotations.tsv"
        WORK_DIR="$BENCH_DIR/ann_work"
        mkdir -p "$WORK_DIR"

        python3 "$ROOT/scripts/annotate_sequences.py" \
            --fasta    "$BENCH_FASTA" \
            --mob-db   "$DATA_DIR/databases/mob_suite/mob_proteins.dmnd" \
            --mpf-db   "$DATA_DIR/databases/mob_suite/mpf_proteins.dmnd" \
            --rep-db   "$DATA_DIR/databases/mob_suite/rep_proteins.dmnd" \
            --out      "$ANNOTATION_TSV" \
            --work-dir "$WORK_DIR" \
            --threads  8

        echo "  Annotation TSV: $ANNOTATION_TSV"
        echo "  Conjugative plasmids found: $(awk -F'\t' 'NR>1 && $2==1' "$ANNOTATION_TSV" | wc -l)"
        echo "  Rep protein hits: $(awk -F'\t' 'NR>1 && $6==1' "$ANNOTATION_TSV" | wc -l)"
    fi
    echo ""
    echo "  Benchmark annotation done: $(date)"
    echo ""
fi

# ── Step 4: Re-run benchmark with marker XGBoost ────────────────────────────
echo "[4/4] Re-running benchmark …"

# Remove cached predictions to force re-run
CACHE_FILE="$BENCH_DIR/results/plasflow2_predictions.tsv"
if [ -f "$CACHE_FILE" ]; then
    echo "  Removing cached predictions: $CACHE_FILE"
    rm "$CACHE_FILE"
fi

# Build extra args for marker model
MARKER_ARGS="--marker-model $MODELS_DIR/marker_xgb.pkl"
if [ -n "${ANNOTATION_TSV:-}" ] && [ -f "${ANNOTATION_TSV:-}" ]; then
    MARKER_ARGS="$MARKER_ARGS --annotation-tsv $ANNOTATION_TSV"
fi

echo "  Running benchmark with marker XGBoost …"
bash "$ROOT/scripts/run_benchmark.sh" $MARKER_ARGS 2>&1

echo ""
echo "============================================================"
echo "  Pipeline complete: $(date)"
echo ""
echo "  Files:"
echo "    marker_features.npz (updated MLP scores) : $DATA_DIR/marker_features.npz"
echo "    marker_xgb.pkl (retrained XGBoost)        : $MODELS_DIR/marker_xgb.pkl"
if [ -n "${ANNOTATION_TSV:-}" ]; then
    echo "    annotations.tsv (benchmark DIAMOND hits)  : $ANNOTATION_TSV"
fi
echo "============================================================"
