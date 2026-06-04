#!/usr/bin/env bash
set -euo pipefail
PROJ="$(cd "$(dirname "$0")" && pwd)"
INPUT="$PROJ/data/test/GCA_054405655.1_ASM5440565v1_genomic.fna"
OUTPUT="$PROJ/results/GCA_054405655_newmodel"
THREADS=16
LOG="$PROJ/GCA_newmodel.log"

# Try the test directory first, fall back to work-dir copy
[ -f "$INPUT" ] || INPUT="$PROJ/results/GCA_054405655_v4_allfeatures/work/all_contigs.fasta"

echo "=== PlasFlow v2 — GCA_054405655 (new model) ==="
echo "Input  : $INPUT"
echo "Output : $OUTPUT"
echo "Started: $(date)"

rm -rf "$OUTPUT" && mkdir -p "$OUTPUT"

if command -v plasflow2 &>/dev/null; then CMD="plasflow2"
elif command -v poetry &>/dev/null; then CMD="poetry run plasflow2"
else CMD="python -m plasflow2.cli"; fi

$CMD run \
    --input   "$INPUT" \
    --output  "$OUTPUT" \
    --threads "$THREADS" \
    --context wastewater \
    --plasmid-threshold 0.98 \
    --min-confidence 0.70 \
    2>&1 | tee "$LOG"

echo "=== Done: $(date) ==="
