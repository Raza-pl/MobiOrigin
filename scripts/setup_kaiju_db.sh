#!/usr/bin/env bash
# PlasFlow v2 — Kaiju taxonomy database setup
#
# Builds TWO Kaiju FM-index databases:
#   1. plasmids   (4 GB RAM)  — used for plasmid contig taxonomy (fast, targeted)
#   2. refseq_ref (54 GB RAM) — used for chromosome/phage/archaea taxonomy
#
# Requirements:
#   conda install -c bioconda kaiju
#   ~10 GB disk space for plasmids DB
#   ~60 GB disk space for refseq_ref DB
#   ~50 MB for NCBI taxonomy files
#
# Usage:
#   bash scripts/setup_kaiju_db.sh [--threads 16] [--plasmids-only] [--refseq-only]
#
# After this script, PlasFlow auto-detects and uses both DBs:
#   plasmid contigs   → kaiju with plasmids DB
#   all other contigs → kaiju with refseq_ref DB (falls back to DIAMOND)

set -euo pipefail

THREADS=8
BUILD_PLASMIDS=true
BUILD_REFSEQ=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads)      THREADS="$2";        shift 2 ;;
        --plasmids-only) BUILD_REFSEQ=false;  shift 1 ;;
        --refseq-only)  BUILD_PLASMIDS=false; shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
KAIJU_DIR="$PROJ/data/databases/kaiju"
mkdir -p "$KAIJU_DIR"

echo "=== Kaiju database setup ==="
echo "Output dir : $KAIJU_DIR"
echo "Threads    : $THREADS"
echo "Build      : plasmids=$BUILD_PLASMIDS  refseq_ref=$BUILD_REFSEQ"
echo ""

# ── Step 1: Download NCBI taxonomy (shared by both DBs) ───────────────────────
NODES="$KAIJU_DIR/nodes.dmp"
NAMES="$KAIJU_DIR/names.dmp"

if [[ -f "$NODES" && -f "$NAMES" ]]; then
    echo "[skip] NCBI taxonomy already present."
else
    echo "[1] Downloading NCBI taxonomy (~50 MB)…"
    cd "$KAIJU_DIR"
    wget -q --show-progress \
        https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz \
        -O taxdump.tar.gz
    tar xf taxdump.tar.gz nodes.dmp names.dmp
    rm -f taxdump.tar.gz
    echo "    → nodes.dmp + names.dmp written."
fi

# ── Step 2: plasmids DB (~4 GB RAM, ~10 min) ─────────────────────────────────
if [[ "$BUILD_PLASMIDS" == "true" ]]; then
    FMI_PLASMIDS="$KAIJU_DIR/kaiju_db_plasmids.fmi"
    if [[ -f "$FMI_PLASMIDS" ]]; then
        echo "[skip] Plasmids FM-index already present: $FMI_PLASMIDS"
    else
        echo "[2] Building plasmids DB (~10 min, ~4 GB RAM)…"
        cd "$KAIJU_DIR"
        kaiju-makedb \
            -s plasmids \
            -t "$THREADS"
        echo "    → $FMI_PLASMIDS"
    fi
fi

# ── Step 3: refseq_ref DB (~54 GB RAM, ~30–60 min) ───────────────────────────
if [[ "$BUILD_REFSEQ" == "true" ]]; then
    FMI_REFSEQ="$KAIJU_DIR/kaiju_db_refseq_ref.fmi"
    if [[ -f "$FMI_REFSEQ" ]]; then
        echo "[skip] refseq_ref FM-index already present: $FMI_REFSEQ"
    else
        echo "[3] Building refseq_ref DB (~30–60 min, ~54 GB RAM)…"
        echo "    NOTE: requires 54 GB free RAM. Skip with --plasmids-only if insufficient."
        cd "$KAIJU_DIR"
        kaiju-makedb \
            -s refseq_ref \
            -t "$THREADS"
        echo "    → $FMI_REFSEQ"
    fi
fi

echo ""
echo "=== Done ==="
echo "nodes.dmp        : $NODES"
echo "names.dmp        : $NAMES"
[[ "$BUILD_PLASMIDS" == "true" ]] && echo "plasmids DB      : $KAIJU_DIR/kaiju_db_plasmids.fmi"
[[ "$BUILD_REFSEQ"   == "true" ]] && echo "refseq_ref DB    : $KAIJU_DIR/kaiju_db_refseq_ref.fmi"
echo ""
echo "PlasFlow will auto-detect both DBs on next run:"
echo "  plasmid contigs   → Kaiju plasmids DB"
echo "  all other contigs → Kaiju refseq_ref DB (falls back to DIAMOND)"
