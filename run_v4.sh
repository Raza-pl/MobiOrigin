#!/usr/bin/env bash
# PlasFlow v2 — full pipeline run on GCA_054405655 wastewater assembly
# Optimised for 16-thread CPU + multi-GPU (CUDA) machine.
#
# GPU note:
#   DIAMOND, minimap2, pyrodigal, mob_typer → CPU-only tools, use THREADS below
#   PyTorch MLP inference                   → auto-detects all CUDA GPUs via DataParallel
#
# Speed optimisations applied:
#   1. --threads 16           → all DIAMOND / minimap2 steps use 16 threads
#   2. Taxonomy uses blastp   → reuses pre-predicted ORFs from ARG step (~6× faster than blastx)
#   3. DIAMOND block_size=4.0 → loads more DB into RAM per thread chunk (~4× faster vs 0.5)
#   4. DIAMOND --faster flag  → taxonomy uses faster preset (sufficient for LCA; ~3× faster vs --sensitive)
#   5. DataParallel MLP       → inference splits across all available CUDA GPUs
#
# Measured wall-clock on Apple Silicon (CPU only, 16 threads, first run):
#   Step 1  Feature extraction + MLP classify     ~15 sec    (CPU, GPU would be <5 sec)
#   Step 2  Topology detection (DTR)              ~1 sec
#   Step 3  ORF prediction (pyrodigal, 24k ctg)  ~5 min
#   Step 4a DIAMOND CARD ARG (2.4 MB DB)          ~14 sec
#   Step 4b DIAMOND SARG (57 MB DB)               ~5 min
#   Step 4c DIAMOND MGE (0.8 MB DB)               ~9 sec
#   Step 4d Plasmid-DB minimap2 (13 GB combined)  ~10 min    (split-prefix, no OOM)
#   Step 5  mob_typer (2559 contigs, per-contig)  ~1 min
#   Step 6  Taxonomy DIAMOND blastp (897 MB DB)   ~20–40 min (16 threads, block_size=4)
#   Step 7  Risk scoring + HTML report            ~30 sec
#   ─────────────────────────────────────────────────────────────────────
#   TOTAL estimated with taxonomy                 ~45–65 min
#   TOTAL without taxonomy (--skip-taxonomy)      ~10–15 min
#
# Known fixes applied vs first run:
#   - Removed --faster flag (not supported on all DIAMOND versions)
#   - minimap2 now uses --split-prefix (handles 13 GB combined plasmid FASTA)
#   - mob_typer now runs per-contig (was returning 1 aggregate row for all 2559 contigs)

set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
# Use the work-dir copy as input — always present, survives source cleanup
INPUT="$PROJ/results/GCA_054405655_v4_allfeatures/work/all_contigs.fasta"
OUTPUT="$PROJ/results/GCA_054405655_v4_allfeatures"
THREADS=16
LOG="$OUTPUT/run.log"

echo "=== PlasFlow v2 — optimised full run ==="
echo "Input  : $INPUT"
echo "Output : $OUTPUT"
echo "Threads: $THREADS"
echo "Started: $(date)"
echo ""

mkdir -p "$OUTPUT"
cd "$PROJ"

# Prefer the installed entry-point; fall back to python -m
if command -v plasflow2 &>/dev/null; then
    CMD="plasflow2"
elif command -v poetry &>/dev/null; then
    CMD="poetry run plasflow2"
else
    CMD="python -m plasflow2.cli"
fi

# Time the full run
START_TS=$(date +%s)

$CMD run \
    --input   "$INPUT" \
    --output  "$OUTPUT" \
    --threads "$THREADS" \
    --context wastewater \
    --plasmid-threshold 0.95 \
    --min-confidence 0.70 \
    2>&1 | tee "$LOG"

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "=== Done: $(date) ==="
echo "=== Wall-clock time: ${MINS}m ${SECS}s ==="
echo ""
echo "Key outputs:"
echo "  $OUTPUT/all_predictions.tsv          (all contigs, all columns)"
echo "  $OUTPUT/annotated_predictions.tsv    (annotated contigs only)"
echo "  $OUTPUT/genes.tsv"
echo "  $OUTPUT/report_plasmid.html"
echo "  $OUTPUT/report_chromosome.html"
echo "  $OUTPUT/report_phage.html"
