#!/usr/bin/env bash
# PlasFlow v2 — build DIAMOND rep protein database from mob_suite rep.dna.fas
#
# mob_suite ships rep.dna.fas: nucleotide sequences of plasmid replication genes
# (RepA, RepB, RepC, etc. from IncF, IncI, IncP, IncQ, IncW, ColE1, …).
# Translating in all 6 frames and building a DIAMOND blastp DB lets us detect
# rep proteins via protein search, which is more sensitive than minimap2 on
# short or divergent fragments.
#
# Usage:
#   bash scripts/setup_rep_diamond.sh
#
# Output:
#   data/databases/mob_suite/rep_proteins.dmnd
#   data/databases/mob_suite/rep_proteins.faa   (translated sequences, kept for inspection)

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
MOB_DIR="$PROJ/data/databases/mob_suite"
REP_DNA="$MOB_DIR/rep.dna.fas"
REP_FAA="$MOB_DIR/rep_proteins.faa"
REP_DMND="$MOB_DIR/rep_proteins.dmnd"

echo "=== Rep protein DIAMOND database setup ==="
echo "Source : $REP_DNA"
echo "Output : $REP_DMND"
echo ""

if [[ ! -f "$REP_DNA" ]]; then
    echo "ERROR: $REP_DNA not found."
    echo "Run: bash scripts/setup_mob_diamond.sh  to populate mob_suite databases first."
    exit 1
fi

if [[ -f "$REP_DMND" ]]; then
    echo "[skip] rep_proteins.dmnd already present."
    exit 0
fi

# Translate rep.dna.fas in all 6 reading frames using Python (no EMBOSS needed)
echo "[1] Translating rep.dna.fas in all 6 frames …"
python3 - <<'PYEOF'
import sys, gzip
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq

rep_dna = Path("data/databases/mob_suite/rep.dna.fas")
rep_faa = Path("data/databases/mob_suite/rep_proteins.faa")

STOP = "*"
MIN_AA = 30  # minimum ORF length to keep (amino acids)

written = 0
with open(rep_faa, "w") as fout:
    for rec in SeqIO.parse(str(rep_dna), "fasta"):
        seq = str(rec.seq).upper().replace("-", "N")
        for strand, nuc in [(+1, seq), (-1, str(Seq(seq).reverse_complement()))]:
            for frame in range(3):
                trans = str(Seq(nuc[frame:]).translate(stop_symbol="*"))
                # Split on stop codons → keep all ORFs >= MIN_AA aa
                parts = trans.split("*")
                for j, part in enumerate(parts):
                    if len(part) >= MIN_AA:
                        fout.write(f">{rec.id}_s{strand}_f{frame}_o{j}\n{part}\n")
                        written += 1

print(f"  Written {written:,} translated ORFs → {rep_faa}")
PYEOF

echo "[2] Building DIAMOND database …"
diamond makedb \
    --in "$REP_FAA" \
    --db "$MOB_DIR/rep_proteins" \
    --quiet

echo ""
echo "=== Done ==="
echo "rep_proteins.dmnd : $REP_DMND"
echo ""
echo "PlasFlow auto-detects this DB on next run:"
echo "  data/databases/mob_suite/rep_proteins.dmnd"
