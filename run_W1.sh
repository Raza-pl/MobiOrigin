#!/usr/bin/env bash
# PlasFlow v2 — full pipeline run on W1.contigs.fa.gz
set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
INPUT="$PROJ/data/test/W1.contigs.fa.gz"
OUTPUT="$PROJ/results/W1"
THREADS=16
LOG="$PROJ/w1.log"

# ── sanity check ──────────────────────────────────────────────────────────────
if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: Input not found at $INPUT"
    echo "Edit the INPUT variable in this script to point to W1.contigs.fa.gz"
    exit 1
fi

echo "=== PlasFlow v2 — W1.contigs.fa.gz ==="
echo "Input  : $INPUT"
echo "Output : $OUTPUT"
echo "Threads: $THREADS"
echo "Started: $(date)"
echo ""

# Fresh run — wipe previous results
if [ -d "$OUTPUT" ]; then
    echo "Removing previous results at $OUTPUT …"
    rm -rf "$OUTPUT"
fi
mkdir -p "$OUTPUT"
cd "$PROJ"

if command -v plasflow2 &>/dev/null; then
    CMD="plasflow2"
elif command -v poetry &>/dev/null; then
    CMD="poetry run plasflow2"
else
    CMD="python -m plasflow2.cli"
fi

START_TS=$(date +%s)

$CMD run \
    --input   "$INPUT" \
    --output  "$OUTPUT" \
    --threads "$THREADS" \
    --context wastewater \
    --plasmid-threshold 0.98 \
    --min-confidence 0.70 \
    2>&1 | tee "$LOG"

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
echo ""
echo "=== Done: $(date) ==="
echo "=== Wall-clock time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s ==="
echo ""
echo "Key outputs:"
echo "  $OUTPUT/all_predictions.tsv          (all contigs, all columns)"
echo "  $OUTPUT/annotated_predictions.tsv    (annotated contigs only)"
echo "  $OUTPUT/genes.tsv"
echo "  $OUTPUT/report_plasmid.html"
echo "  $OUTPUT/report_chromosome.html"
echo "  $OUTPUT/report_phage.html"
