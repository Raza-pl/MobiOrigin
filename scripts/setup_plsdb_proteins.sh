#!/usr/bin/env bash
# setup_plsdb_proteins.sh
#
# One-time setup: translate all PLSDB plasmids into proteins, then build a
# DIAMOND blastp database.  This enables the --plsdb-proteins feature in
# annotate_sequences.py which adds plsdb_prot_hits_per_kb and
# max_plsdb_prot_pct_id to the annotation TSV — used by predict.py to boost
# recall for composition-invisible FN plasmids.
#
# Requirements:
#   conda activate plasflow2   (needs pyrodigal, diamond, biopython)
#
# Runtime:
#   Translation (pyrodigal): ~3-5 min
#   diamond makedb:           ~2-5 min
#   Total:                    ~5-10 min
#
# Usage:
#   bash scripts/setup_plsdb_proteins.sh
#   bash scripts/setup_plsdb_proteins.sh /path/to/plsdb.fasta  # custom PLSDB path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PLSDB="${1:-$ROOT/data/databases/plasmids/plsdb.fasta}"
OUT_FAA="$ROOT/data/databases/plasmids/plsdb_proteins.faa"
OUT_DMND="$ROOT/data/databases/plasmids/plsdb_proteins.dmnd"
THREADS="${2:-8}"

echo "============================================================"
echo "  PlasFlow v2 — Build PLSDB protein DIAMOND DB"
echo "  Start: $(date)"
echo "  PLSDB:   $PLSDB"
echo "  Threads: $THREADS"
echo "============================================================"
echo ""

if [[ ! -f "$PLSDB" ]]; then
    echo "ERROR: PLSDB FASTA not found: $PLSDB"
    exit 1
fi

# ── Step 1: Translate PLSDB plasmids with pyrodigal ──────────────────────────
if [[ -f "$OUT_FAA" ]]; then
    echo "[1/2] $OUT_FAA already exists — skipping translation."
    echo "      (Delete it and re-run to regenerate.)"
else
    echo "[1/2] Translating PLSDB plasmids with pyrodigal …"
    echo "  Input:  $PLSDB"
    echo "  Output: $OUT_FAA"
    echo "  Start:  $(date)"
    python3 - "$PLSDB" "$OUT_FAA" "$THREADS" <<'PYEOF'
import sys, os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import pyrodigal
from pathlib import Path

fasta_in  = Path(sys.argv[1])
faa_out   = Path(sys.argv[2])
threads   = int(sys.argv[3])

# ── Parse FASTA ───────────────────────────────────────────────────────────────
sequences: list[tuple[str, str]] = []
sid, buf = None, []
with open(fasta_in) as f:
    for line in f:
        line = line.rstrip()
        if line.startswith(">"):
            if sid and buf:
                sequences.append((sid, "".join(buf)))
            sid = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
if sid and buf:
    sequences.append((sid, "".join(buf)))

print(f"  Loaded {len(sequences):,} sequences", flush=True)

# ── Translate in meta mode ────────────────────────────────────────────────────
orf_finder = pyrodigal.GeneFinder(meta=True)

written = 0
with open(faa_out, "w") as out:
    for i, (seq_id, seq) in enumerate(sequences):
        if i % 5000 == 0 and i > 0:
            print(f"  Progress: {i:,}/{len(sequences):,} sequences …", flush=True)
        try:
            genes = orf_finder.find_genes(seq.encode())
            for j, gene in enumerate(genes, 1):
                prot = gene.translate()
                out.write(f">{seq_id}_{j}\n{prot}\n")
                written += 1
        except Exception as e:
            print(f"  WARN: pyrodigal failed on {seq_id}: {e}", flush=True)

print(f"  Wrote {written:,} protein sequences → {faa_out}", flush=True)
PYEOF
    echo "  Done:  $(date)"
fi

echo ""

# ── Step 2: Build DIAMOND DB ──────────────────────────────────────────────────
if [[ -f "$OUT_DMND" ]]; then
    echo "[2/2] $OUT_DMND already exists — skipping diamond makedb."
    echo "      (Delete it and re-run to regenerate.)"
else
    echo "[2/2] Building DIAMOND database …"
    echo "  Input:  $OUT_FAA"
    echo "  Output: $OUT_DMND"
    echo "  Start:  $(date)"
    diamond makedb \
        --in    "$OUT_FAA" \
        --db    "${OUT_DMND%.dmnd}" \
        --threads "$THREADS" \
        --quiet
    echo "  Done:  $(date)"
fi

echo ""
echo "============================================================"
echo "  PLSDB protein DB ready."
echo "  FAA:  $OUT_FAA"
echo "  DMND: $OUT_DMND"
echo ""
echo "  Next — re-annotate benchmark sequences with protein features:"
echo ""
echo "  python scripts/annotate_sequences.py \\"
echo "      --fasta data/benchmark/benchmark.fna \\"
echo "      --plsdb-proteins $OUT_DMND \\"
echo "      --genomad-genes data/benchmark/genomad_full/benchmark_annotate/benchmark_genes.tsv \\"
echo "      --out data/benchmark/annotations_with_plsdb_prot.tsv \\"
echo "      --threads $THREADS"
echo ""
echo "  Then run the recall improvement benchmark:"
echo "  bash scripts/retrain_with_plsdb_prot.sh"
echo "============================================================"
