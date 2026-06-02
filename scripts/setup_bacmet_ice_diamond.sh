#!/usr/bin/env bash
# Build DIAMOND databases for BacMet2 and ICEberg3.
# Run once after downloading the FASTA files.
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== BacMet2 DIAMOND setup ==="
mkdir -p "$PROJ/data/databases/bacmet"
diamond makedb \
    --in  "$PROJ/data/databases/bacmet/BacMet2_EXP.fasta" \
    -d    "$PROJ/data/databases/bacmet/bacmet" \
    --quiet
echo "  → data/databases/bacmet/bacmet.dmnd"

echo "=== ICEberg3 DIAMOND setup ==="
mkdir -p "$PROJ/data/databases/ice"
# Accept either filename the user may have saved
ICE_FA="$PROJ/data/databases/ice/ICEberg3_experimental.fasta"
[ -f "$ICE_FA" ] || ICE_FA="$PROJ/data/databases/ICE_aa_experimental.fas"
diamond makedb \
    --in  "$ICE_FA" \
    -d    "$PROJ/data/databases/ice/ice" \
    --quiet
echo "  → data/databases/ice/ice.dmnd"

echo ""
echo "Done. Both databases will be auto-detected on the next pipeline run."
