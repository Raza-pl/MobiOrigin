#!/usr/bin/env bash
# run_w1_full_pipeline.sh — Run PlasFlow v2 full pipeline on W1.contigs.fa
#
# USAGE
#   conda activate plasflow2
#   nohup bash scripts/run_w1_full_pipeline.sh > results/W1_full_pipeline/pipeline.log 2>&1 &
#   tail -f results/W1_full_pipeline/pipeline.log
#
# EXPECTED RUNTIME (Apple Silicon, 8 threads, 205k contigs)
#   MLP classify         : ~20 min
#   ORF prediction       :  ~5 min
#   ARG (CARD+SARG+AMR)  :  ~2 min
#   VF, MGE, BacMet, ICE :  ~3 min
#   Mobility (mob_suite) :  ~5 min
#   Plasmid-DB match     : ~10 min
#   Taxonomy (DIAMOND)   : ~30 min
#   Report generation    :  ~2 min
#   TOTAL                : ~80 min
#
# OUTPUTS (results/W1_full_pipeline/)
#   all_predictions.tsv       — all 205k contigs, per-sequence label + scores
#   annotated_predictions.tsv — subset with ARGs / MGEs / VFs / mobility
#   plasmids.fasta            — predicted plasmid sequences
#   chromosome.fasta          — predicted chromosome sequences
#   genes.tsv                 — all ORFs with ARG/VF/MGE flags
#   annotations.json          — full annotation per plasmid contig
#   report_plasmid.html       — interactive plasmid report
#   report_chromosome.html    — chromosome report
#   report_phage.html         — phage report
#   report_unclassified.html  — unclassified report
#   report_circular_plasmids.html — SVG circular maps

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

W1="$ROOT/data/test/W1.contigs.fa"
OUT="$ROOT/results/W1_full_pipeline"
DB="$ROOT/data/databases"

mkdir -p "$OUT"

echo "======================================================"
echo "  PlasFlow v2 — Full Pipeline on W1.contigs.fa"
echo "  Start: $(date)"
echo "  Input: $W1  ($(grep -c '^>' "$W1") contigs)"
echo "  Output: $OUT"
echo "======================================================"

t0=$SECONDS

plasflow2 run \
    --input       "$W1" \
    --output      "$OUT" \
    --context     wastewater \
    --threads     8 \
    --min-length  1000 \
    --card-db     "$DB/card/card.dmnd" \
    --aro-index   "$DB/card/aro_index.tsv" \
    --sarg-db     "$DB/sarg/sarg.dmnd" \
    --amrprot-db  "$DB/amrfinder/amrprot.dmnd" \
    --vfdb        "$DB/vfdb/vfdb_setA.dmnd" \
    --mge-db      "$DB/mge/isfinder.dmnd" \
    --taxonomy-db "$DB/taxonomy/refseq_taxonomy.dmnd" \
    --min-identity 80 \
    --min-confidence 0.5

elapsed=$((SECONDS - t0))
wall=$(printf '%02d:%02d:%02d' $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)))

echo ""
echo "======================================================"
echo "  Done: $(date)   Wall time: $wall"
echo ""
echo "  Key outputs:"
ls -lh "$OUT"/*.tsv "$OUT"/*.html "$OUT"/*.fasta "$OUT"/*.json 2>/dev/null
echo ""

echo "  Summary:"
echo "    Total contigs : $(grep -c '^>' "$W1")"
if [ -f "$OUT/all_predictions.tsv" ]; then
    echo "    Plasmid       : $(awk -F'\t' 'NR>1 && $3=="plasmid" {n++} END {print n+0}' "$OUT/all_predictions.tsv")"
    echo "    Chromosome    : $(awk -F'\t' 'NR>1 && $3=="chromosome" {n++} END {print n+0}' "$OUT/all_predictions.tsv")"
    echo "    Phage         : $(awk -F'\t' 'NR>1 && $3=="phage" {n++} END {print n+0}' "$OUT/all_predictions.tsv")"
    echo "    Unclassified  : $(awk -F'\t' 'NR>1 && $3=="unclassified" {n++} END {print n+0}' "$OUT/all_predictions.tsv")"
fi
if [ -f "$OUT/annotated_predictions.tsv" ]; then
    echo "    Annotated (ARG/VF/MGE/mobility) : $(wc -l < "$OUT/annotated_predictions.tsv") contigs"
fi
echo "======================================================"
