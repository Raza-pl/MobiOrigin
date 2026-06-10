#!/usr/bin/env bash
# run_benchmark.sh — Full PlasFlow v2 benchmark pipeline (3 steps)
#
# Usage:
#   bash scripts/run_benchmark.sh                        # normal run (skips completed steps)
#   bash scripts/run_benchmark.sh --force                # clear all cached data and re-run
#   bash scripts/run_benchmark.sh --genomad-db /path/to/genomad_db
#   bash scripts/run_benchmark.sh --threads 16 --benchmark-dir data/benchmark
#
# Prerequisites:
#   pip install biopython --break-system-packages    # for genome download
#   pip install matplotlib --break-system-packages   # for figures (optional)
#
# For geNomad comparison (optional):
#   conda install -c conda-forge -c bioconda genomad -y
#   genomad download-database genomad_db/
#   bash scripts/run_benchmark.sh --genomad-db genomad_db/
#
# For PlasFlow v1 comparison (optional):
#   conda create -n plasflow1 python=3.7 -y
#   conda activate plasflow1 && conda install -c bioconda plasflow -y
#   conda run -n plasflow1 python scripts/run_plasflow1_benchmark.py \
#       --benchmark-dir data/benchmark --out data/benchmark/results/
#   # Then re-run this script — it will pick up plasflow1_metrics.json automatically.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
BENCHMARK_DIR="data/benchmark"
MODEL="data/models/mlp_v2.pt"
THREADS=8
GENOMAD_DB=""
SEED=42
FORCE=0
MIN_GENOMES=20   # re-download if fewer than this many genomes in genome_list.tsv

# ── Argument parsing ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --threads)       THREADS="$2";       shift 2 ;;
        --genomad-db)    GENOMAD_DB="$2";    shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --force)         FORCE=1;            shift   ;;
        --marker-model)  MARKER_MODEL="$2"; shift 2 ;;
        --annotation-tsv) ANNOTATION_TSV="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RESULTS_DIR="${BENCHMARK_DIR}/results"
MARKER_MODEL=""
ANNOTATION_TSV=""

echo "======================================================================"
echo " PlasFlow v2 Benchmark Pipeline"
echo "======================================================================"
echo "  Benchmark dir : ${BENCHMARK_DIR}"
echo "  Model         : ${MODEL}"
echo "  Threads       : ${THREADS}"
echo "  Results dir   : ${RESULTS_DIR}"
[[ -n "${GENOMAD_DB}" ]] && echo "  geNomad DB    : ${GENOMAD_DB}"
[[ -n "${MARKER_MODEL}" ]] && echo "  Marker XGBoost: ${MARKER_MODEL}"
[[ -n "${ANNOTATION_TSV}" ]] && echo "  Annotation TSV: ${ANNOTATION_TSV}"
echo "======================================================================"
echo ""

# ── Check Python dependencies ─────────────────────────────────────────────
echo "[check] Python dependencies …"
python -c "from Bio import SeqIO" 2>/dev/null || {
    echo "[install] BioPython not found — installing …"
    pip install biopython -q
}
python -c "import torch" 2>/dev/null || {
    echo "[install] PyTorch not found — installing (CPU build) …"
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q \
        || pip install torch -q  # fallback for macOS/ARM
}
echo "[check] PlasFlow v2 importable …"
python -c "import sys; sys.path.insert(0,'src'); import plasflow2" || {
    echo "ERROR: plasflow2 not importable. Run from the project root directory."
    exit 1
}

# ── Step 1: Download benchmark genomes ────────────────────────────────────
GENOME_LIST="${BENCHMARK_DIR}/genome_list.tsv"
GENOME_COUNT=0
[[ -f "${GENOME_LIST}" ]] && GENOME_COUNT=$(( $(wc -l < "${GENOME_LIST}") - 1 ))  # subtract header

NEEDS_DOWNLOAD=0
if [[ ${FORCE} -eq 1 ]]; then
    echo "[step 1/3] --force: clearing old benchmark data …"
    rm -f "${BENCHMARK_DIR}/genome_list.tsv" \
          "${BENCHMARK_DIR}/benchmark.fna" \
          "${BENCHMARK_DIR}/benchmark_stats.txt" \
          "${BENCHMARK_DIR}/ground_truth.tsv" \
          "${RESULTS_DIR}/plasflow2_predictions.tsv"
    NEEDS_DOWNLOAD=1
elif [[ ${GENOME_COUNT} -lt ${MIN_GENOMES} ]]; then
    echo "[step 1/3] Only ${GENOME_COUNT} genomes found (need >= ${MIN_GENOMES}) — downloading …"
    # Also clear stale benchmark so it gets rebuilt
    rm -f "${BENCHMARK_DIR}/benchmark.fna" \
          "${BENCHMARK_DIR}/benchmark_stats.txt" \
          "${BENCHMARK_DIR}/ground_truth.tsv" \
          "${RESULTS_DIR}/plasflow2_predictions.tsv"
    NEEDS_DOWNLOAD=1
else
    echo "[step 1/3] Genome download — SKIPPED (${GENOME_COUNT} genomes in ${GENOME_LIST})"
fi

if [[ ${NEEDS_DOWNLOAD} -eq 1 ]]; then
    python scripts/download_benchmark_genomes.py \
        --out     "${BENCHMARK_DIR}" \
        --threads "${THREADS}"
    echo "[step 1/3] Done"
fi
echo ""

# ── Step 2: Build benchmark FASTA + ground truth ──────────────────────────
BENCHMARK_FASTA="${BENCHMARK_DIR}/benchmark.fna"
GT_TSV="${BENCHMARK_DIR}/ground_truth.tsv"
if [[ -f "${BENCHMARK_FASTA}" ]] && [[ -f "${GT_TSV}" ]]; then
    CONTIG_COUNT=$(grep -c "^>" "${BENCHMARK_FASTA}" 2>/dev/null || echo 0)
    echo "[step 2/3] Build benchmark — SKIPPED (${CONTIG_COUNT} contigs in benchmark.fna)"
else
    echo "[step 2/3] Building benchmark FASTA and ground truth …"
    python scripts/build_benchmark.py \
        --benchmark-dir "${BENCHMARK_DIR}" \
        --seed          "${SEED}"
    echo "[step 2/3] Done"
fi
echo ""

# ── Step 3: Run PlasFlow v2 evaluation ────────────────────────────────────
echo "[step 3/3] Running PlasFlow v2 evaluation …"

# Optional: pick up PlasFlow v1 metrics if they were pre-computed
PF1_METRICS=""
if [[ -f "${RESULTS_DIR}/plasflow1_metrics.json" ]]; then
    PF1_METRICS="--plasflow1-metrics ${RESULTS_DIR}/plasflow1_metrics.json"
    echo "           (incorporating PlasFlow v1 actual metrics)"
fi

# Build geNomad flag
GENOMAD_FLAG=""
[[ -n "${GENOMAD_DB}" ]] && GENOMAD_FLAG="--genomad-db ${GENOMAD_DB}"

# Auto-detect marker model and annotation TSV at standard paths if not
# specified on the command line. This means plain `bash run_benchmark.sh`
# will automatically use the marker XGBoost if it has been trained.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${MARKER_MODEL}" ]] && [[ -f "${ROOT_DIR}/data/models/marker_xgb.pkl" ]]; then
    MARKER_MODEL="${ROOT_DIR}/data/models/marker_xgb.pkl"
    echo "  [auto] Marker XGBoost detected: ${MARKER_MODEL}"
fi
if [[ -z "${ANNOTATION_TSV}" ]] && [[ -f "${BENCHMARK_DIR}/annotations.tsv" ]]; then
    ANNOTATION_TSV="${BENCHMARK_DIR}/annotations.tsv"
    echo "  [auto] Annotation TSV detected: ${ANNOTATION_TSV}"
fi

# Build marker XGBoost flags
MARKER_FLAG=""
[[ -n "${MARKER_MODEL}" ]] && MARKER_FLAG="--marker-model ${MARKER_MODEL}"
ANNOTATION_FLAG=""
[[ -n "${ANNOTATION_TSV}" ]] && ANNOTATION_FLAG="--annotation-tsv ${ANNOTATION_TSV}"

# Auto-clear prediction cache when marker model/annotations are active —
# cached predictions are MLP-only and won't have XGBoost scores applied.
if [[ -n "${MARKER_MODEL}" ]] && [[ -f "${RESULTS_DIR}/plasflow2_predictions.tsv" ]]; then
    echo "  [cache] Clearing MLP-only cache (marker model active)"
    rm -f "${RESULTS_DIR}/plasflow2_predictions.tsv"
fi

python scripts/run_benchmark_evaluation.py \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --model         "${MODEL}" \
    --threads       "${THREADS}" \
    --out           "${RESULTS_DIR}" \
    ${PF1_METRICS} \
    ${GENOMAD_FLAG} \
    ${MARKER_FLAG} \
    ${ANNOTATION_FLAG}

echo ""
echo "[step 3/3] Done"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────
echo "======================================================================"
echo " Results written to: ${RESULTS_DIR}/"
echo ""
echo " Key files:"
echo "   ${RESULTS_DIR}/comparison_table.csv       ← paper Table 1"
echo "   ${RESULTS_DIR}/plasflow2_metrics.json      ← P/R/F1 per class"
echo "   ${RESULTS_DIR}/plasflow2_pr_curve.csv      ← PR curve data"
echo "   ${RESULTS_DIR}/plasflow2_by_length.csv     ← length-stratified metrics"
echo "   ${RESULTS_DIR}/plasflow2_threshold_sweep.csv"
[[ -n "${GENOMAD_DB}" ]] && \
echo "   ${RESULTS_DIR}/genomad_metrics.json        ← geNomad comparison"
echo "   ${RESULTS_DIR}/figures/                    ← PDF/PNG plots"
echo ""
echo " To add PlasFlow v1 comparison:"
echo "   conda create -n plasflow1 python=3.7 -y"
echo "   conda activate plasflow1"
echo "   conda install -c bioconda plasflow -y"
echo "   conda run -n plasflow1 python scripts/run_plasflow1_benchmark.py \\"
echo "       --benchmark-dir ${BENCHMARK_DIR} --out ${RESULTS_DIR}/"
echo "   bash scripts/run_benchmark.sh  # re-run to merge into comparison table"
echo "======================================================================"
