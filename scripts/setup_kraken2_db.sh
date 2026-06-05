#!/usr/bin/env bash
# PlasFlow v2 — Kraken2 database setup
#
# Downloads the pre-built Kraken2 Standard database (~8 GB) for fast nucleotide-
# level taxonomy of all contigs, including short ones with no detectable ORFs.
#
# Kraken2 classifies 170k contigs in ~30 seconds vs DIAMOND's 20-40 minutes,
# and covers contigs that DIAMOND misses (no proteins → no blastp hits).
# Used as a FALLBACK — DIAMOND result takes priority when available.
#
# Requirements:
#   conda install -c bioconda kraken2
#
# Usage:
#   bash scripts/setup_kraken2_db.sh [--db-dir data/databases/kraken2] [--threads 16]
#
# Database options:
#   standard (~8 GB, recommended): bacteria + archaea + viruses + human
#   standard-8  (~8 GB, pre-indexed, fastest download)
#   minikraken  (~4 GB, lower sensitivity, good for RAM-constrained systems)

set -euo pipefail

DB_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/databases/kraken2"
THREADS=8
DB_TYPE="standard-8"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --db-dir)   DB_DIR="$2";   shift 2 ;;
        --threads)  THREADS="$2";  shift 2 ;;
        --type)     DB_TYPE="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$DB_DIR"

echo "=== Kraken2 database setup ==="
echo "DB dir  : $DB_DIR"
echo "Threads : $THREADS"
echo "Type    : $DB_TYPE"
echo ""

# Check if DB already built
if [[ -f "$DB_DIR/hash.k2d" && -f "$DB_DIR/taxo.k2d" && -f "$DB_DIR/opts.k2d" ]]; then
    echo "[skip] Kraken2 database already present at $DB_DIR"
    exit 0
fi

if ! command -v kraken2-build &>/dev/null; then
    echo "ERROR: kraken2 not found. Install with:"
    echo "  conda install -c bioconda kraken2"
    exit 1
fi

echo "[1] Downloading Kraken2 $DB_TYPE database (~8 GB) …"
echo "    This may take 10-30 min depending on connection speed."
echo ""

# Download pre-built database (faster than building from scratch)
KRAKEN2_URL="https://genome-idx.s3.amazonaws.com/kraken/k2_${DB_TYPE}_$(date +%Y)0101.tar.gz"

# Try current year, fall back to 2024
if ! wget -q --spider "$KRAKEN2_URL" 2>/dev/null; then
    KRAKEN2_URL="https://genome-idx.s3.amazonaws.com/kraken/k2_standard_08gb_20240904.tar.gz"
    echo "    Using 2024-09 standard-8 release …"
fi

cd "$DB_DIR"
wget -q --show-progress "$KRAKEN2_URL" -O kraken2_db.tar.gz
echo "[2] Extracting …"
tar xf kraken2_db.tar.gz
rm -f kraken2_db.tar.gz

echo ""
echo "=== Done ==="
echo "DB path: $DB_DIR"
echo ""
echo "PlasFlow auto-detects Kraken2 DB at:"
echo "  data/databases/kraken2/"
echo ""
echo "Kraken2 provides fallback taxonomy for contigs where DIAMOND finds no hits."
echo "DIAMOND result always takes priority (higher sensitivity in protein space)."
