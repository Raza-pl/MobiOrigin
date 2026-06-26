#!/usr/bin/env bash
# Validate PlasFlow v2 benchmark false-positives against PLSDB and COMPASS.
#
# Runs minimap2 (asm5 preset — for high-identity plasmid matches) then filters
# hits by query coverage ≥ 50% and identity ≥ 90%.
#
# Prerequisites:
#   minimap2  (brew install minimap2 / conda install -c bioconda minimap2)
#
# Usage:
#   bash scripts/run_fp_minimap2.sh [plsdb_fasta] [compass_fasta]
#
# Defaults:
#   plsdb_fasta   = data/databases/plasmids/plsdb.fasta
#   compass_fasta = data/databases/plasmids/COMPASS.fna

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
FPS="$PROJ/results/fp_validation/benchmark_fps.fna"
PLSDB="${1:-$PROJ/data/databases/plasmids/plsdb.fasta}"
COMPASS="${2:-$PROJ/data/databases/plasmids/COMPASS.fna}"
OUT_DIR="$PROJ/results/fp_validation"

if [[ ! -f "$FPS" ]]; then
    echo "ERROR: FP sequences not found at $FPS"
    echo "Run:  python3 scripts/validate_fps_plsdb.py  first, or extract manually:"
    echo "  python3 - <<'EOF'"
    echo "  # (see validate_fps_plsdb.py for extraction code)"
    echo "  EOF"
    exit 1
fi

echo "=== PlasFlow v2 FP validation via minimap2 ==="
echo "Query  : $FPS ($(grep -c '^>' "$FPS") sequences)"
echo "PLSDB  : $PLSDB"
echo "COMPASS: $COMPASS"
echo "Out    : $OUT_DIR"
echo ""

mkdir -p "$OUT_DIR"

# ─── Helper: filter PAF for high-confidence hits ────────────────────────────
# PAF columns: qname qlen qstart qend strand tname tlen tstart tend nmatch alen mapq
filter_paf() {
    local paf="$1"
    local min_cov="${2:-0.50}"   # query coverage ≥ 50%
    local min_id="${3:-0.90}"    # sequence identity ≥ 90%
    awk -v min_cov="$min_cov" -v min_id="$min_id" '
    {
        qlen=$2; qstart=$3; qend=$4; alen=$10; nmatch=$10; blen=$11
        qcov = (qend - qstart) / qlen
        identity = nmatch / blen
        if (qcov >= min_cov && identity >= min_id) print
    }' "$paf"
}

# ─── COMPASS search ──────────────────────────────────────────────────────────
echo "[1/2] Searching against COMPASS ($(grep -c '^>' "$COMPASS") plasmids)..."
minimap2 -c -x asm5 --secondary=no -t 4 \
    "$COMPASS" "$FPS" \
    > "$OUT_DIR/fps_vs_compass.paf" 2>/dev/null

echo "  Raw hits: $(wc -l < "$OUT_DIR/fps_vs_compass.paf")"
filter_paf "$OUT_DIR/fps_vs_compass.paf" > "$OUT_DIR/fps_vs_compass_filtered.paf"
echo "  Filtered (cov≥50% id≥90%): $(wc -l < "$OUT_DIR/fps_vs_compass_filtered.paf")"

# ─── PLSDB search ────────────────────────────────────────────────────────────
echo ""
echo "[2/2] Searching against PLSDB ($(grep -c '^>' "$PLSDB") plasmids)..."
minimap2 -c -x asm5 --secondary=no -t 4 \
    "$PLSDB" "$FPS" \
    > "$OUT_DIR/fps_vs_plsdb.paf" 2>/dev/null

echo "  Raw hits: $(wc -l < "$OUT_DIR/fps_vs_plsdb.paf")"
filter_paf "$OUT_DIR/fps_vs_plsdb.paf" > "$OUT_DIR/fps_vs_plsdb_filtered.paf"
echo "  Filtered (cov≥50% id≥90%): $(wc -l < "$OUT_DIR/fps_vs_plsdb_filtered.paf")"

# ─── Summarize ───────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
python3 - <<'PYEOF'
import sys
from pathlib import Path

out = Path("results/fp_validation")

def parse_paf_hits(paf_file):
    """Return set of query IDs with high-confidence hits."""
    hits = {}
    try:
        with open(paf_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 12: continue
                qname, qlen, qstart, qend = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                tname = parts[5]
                nmatch, blen = int(parts[9]), int(parts[10])
                qcov = (qend - qstart) / qlen
                identity = nmatch / blen if blen > 0 else 0
                if qname not in hits or qcov > hits[qname]["qcov"]:
                    hits[qname] = {"ref": tname, "qcov": qcov, "identity": identity}
    except FileNotFoundError:
        pass
    return hits

compass_hits = parse_paf_hits(out / "fps_vs_compass_filtered.paf")
plsdb_hits   = parse_paf_hits(out / "fps_vs_plsdb_filtered.paf")

all_fps = set()
try:
    with open(out / "benchmark_fps.fna") as f:
        for line in f:
            if line.startswith(">"):
                all_fps.add(line[1:].strip())
except: pass

confirmed = all_fps & (set(compass_hits) | set(plsdb_hits))
true_fps = all_fps - confirmed

print(f"  Total FPs analysed:          {len(all_fps)}")
print(f"  Match known plasmid:         {len(confirmed)}  ← these may be mislabeled in benchmark")
print(f"  True novel FPs (no hit):     {len(true_fps)}")
print()

if confirmed:
    print("  Confirmed plasmid-matching FPs:")
    for fp in sorted(confirmed):
        ch = compass_hits.get(fp)
        ph = plsdb_hits.get(fp)
        if ch:
            print(f"    {fp[-55:]}")
            print(f"      COMPASS: {ch['ref']}  cov={ch['qcov']:.2f}  id={ch['identity']:.3f}")
        if ph:
            print(f"      PLSDB:   {ph['ref']}  cov={ph['qcov']:.2f}  id={ph['identity']:.3f}")

# Write final TSV
import csv
rows = []
for fp in sorted(all_fps):
    ch = compass_hits.get(fp, {})
    ph = plsdb_hits.get(fp, {})
    rows.append({
        "contig_id": fp,
        "compass_hit": ch.get("ref", ""),
        "compass_qcov": f"{ch['qcov']:.3f}" if ch else "",
        "compass_identity": f"{ch['identity']:.3f}" if ch else "",
        "plsdb_hit": ph.get("ref", ""),
        "plsdb_qcov": f"{ph['qcov']:.3f}" if ph else "",
        "plsdb_identity": f"{ph['identity']:.3f}" if ph else "",
        "is_known_plasmid": "YES" if fp in confirmed else "no",
    })

with open(out / "fp_plsdb_hits.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader()
    w.writerows(rows)

print(f"\nFull results: {out}/fp_plsdb_hits.tsv")
PYEOF

echo ""
echo "Done. See results/fp_validation/ for all output files."
