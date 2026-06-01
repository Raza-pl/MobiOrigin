#!/usr/bin/env bash
# Build a DIAMOND database from the AMRFinderPlus AMRProt FASTA.
#
# Usage (auto-detects data/databases/AMRfinder.fasta):
#   bash scripts/setup_amrprot_diamond.sh
#
# Usage with explicit path:
#   bash scripts/setup_amrprot_diamond.sh /path/to/AMRfinder.fasta
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$PROJ/data/databases/amrfinder"

# ── locate FASTA ──────────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    AMRPROT_FASTA="$1"
else
    # Check the default location the user placed it
    AMRPROT_FASTA="$PROJ/data/databases/AMRfinder.fasta"
fi

if [[ ! -f "$AMRPROT_FASTA" ]]; then
    echo "ERROR: AMRFinder FASTA not found at $AMRPROT_FASTA"
    echo "Usage: bash scripts/setup_amrprot_diamond.sh [/path/to/AMRfinder.fasta]"
    exit 1
fi

echo "=== AMRProt DIAMOND setup ==="
echo "Source : $AMRPROT_FASTA"
echo "Output : $OUT_DIR/"
mkdir -p "$OUT_DIR"

# Build DIAMOND database
echo "Building DIAMOND database …"
diamond makedb --in "$AMRPROT_FASTA" -d "$OUT_DIR/amrprot" --quiet
echo "  → $OUT_DIR/amrprot.dmnd"

echo ""
echo "Done. PlasFlow will auto-detect data/databases/amrfinder/amrprot.dmnd on next run."
echo "Log will show: Annotating ARGs on ALL N contigs (CARD + SARG + AMRProt)"
