#!/usr/bin/env bash
# PlasFlow v2 — Kaiju taxonomy database setup
#
# Kaiju is 20–50× faster than DIAMOND for contig taxonomy annotation.
# This script sets up the Kaiju FM-index database + NCBI taxonomy files.
#
# Requirements:
#   conda install -c bioconda kaiju   (or pip install kaiju-py, or binary download)
#   ~25 GB disk space for the RefSeq nr_euk+nr_prok database
#   ~50 MB for NCBI taxonomy files
#
# Usage:
#   bash scripts/setup_kaiju_db.sh [--threads 16] [--db refseq|nr|progenomes]
#
# After this script, run:
#   plasflow2 run --input assembly.fasta --output results/ --taxonomy-engine kaiju
# or just:
#   plasflow2 run --input assembly.fasta --output results/
# (kaiju is auto-detected when data/databases/kaiju/*.fmi exists)

set -euo pipefail

THREADS=8
DB_TYPE="refseq"   # refseq | nr | progenomes (refseq is best for environmental metagenomics)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --db)      DB_TYPE="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
KAIJU_DIR="$PROJ/data/databases/kaiju"
mkdir -p "$KAIJU_DIR"

echo "=== Kaiju database setup ==="
echo "Output dir: $KAIJU_DIR"
echo "DB type   : $DB_TYPE"
echo "Threads   : $THREADS"
echo ""

# ── Step 1: Download NCBI taxonomy (nodes.dmp + names.dmp) ──────────────────
NODES="$KAIJU_DIR/nodes.dmp"
NAMES="$KAIJU_DIR/names.dmp"

if [[ -f "$NODES" && -f "$NAMES" ]]; then
    echo "[skip] NCBI taxonomy files already present."
else
    echo "[1/2] Downloading NCBI taxonomy (~50 MB)…"
    cd "$KAIJU_DIR"
    wget -q --show-progress \
        https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz \
        -O taxdump.tar.gz
    tar xf taxdump.tar.gz nodes.dmp names.dmp
    rm -f taxdump.tar.gz
    echo "      → nodes.dmp + names.dmp written."
fi

# ── Step 2: Build Kaiju FM-index ─────────────────────────────────────────────
FMI="$KAIJU_DIR/kaiju_db_${DB_TYPE}.fmi"

if [[ -f "$FMI" ]]; then
    echo "[skip] Kaiju FM-index already present: $FMI"
else
    echo "[2/2] Building Kaiju $DB_TYPE database (~10–60 min depending on DB size)…"
    echo "      This downloads protein sequences and builds the BWT/FM-index."
    cd "$KAIJU_DIR"
    kaiju-makedb \
        --threads "$THREADS" \
        --database "$DB_TYPE" \
        --out "$KAIJU_DIR/kaiju_db_${DB_TYPE}"
    echo "      → FM-index written: $FMI"
fi

echo ""
echo "=== Done ==="
echo "Kaiju DB   : $FMI"
echo "nodes.dmp  : $NODES"
echo "names.dmp  : $NAMES"
echo ""
echo "PlasFlow v2 will auto-detect these files."
echo "Next run:  plasflow2 run --input assembly.fasta --output results/ --threads $THREADS"
echo "(or add --taxonomy-engine kaiju to force Kaiju even if DIAMOND DB also present)"

# ── Alternative: build from existing taxonomy_proteins.faa ───────────────────
# If you already have data/databases/taxonomy/taxonomy_proteins.faa and don't
# want to download the full RefSeq, you can build a smaller custom FM-index:
#
#   PROT="$PROJ/data/databases/taxonomy/taxonomy_proteins.faa"
#   if [[ -f "$PROT" ]]; then
#       kaiju-mkbwt -n "$THREADS" -a ACDEFGHIKLMNPQRSTVWY \
#           -o "$KAIJU_DIR/kaiju_custom" "$PROT"
#       kaiju-mkfmi "$KAIJU_DIR/kaiju_custom"
#   fi
#
# Note: the custom DB will have GTDB-style headers, not NCBI taxids.
# In that case, use --taxonomy-engine diamond (the GTDB DIAMOND DB is better
# for GTDB-style lineages).  Kaiju works best with NCBI taxonomy.
