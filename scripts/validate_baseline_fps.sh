#!/usr/bin/env bash
# Validate baseline-model FPs against PLSDB/COMPASS, then report corrected F1.
#
# Usage:
#   bash scripts/validate_baseline_fps.sh [plsdb_fasta] [compass_fasta]

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
FP_IDS="$PROJ/results/baseline_plsdb_validation/fp_ids.txt"
BENCHMARK="$PROJ/data/benchmark/benchmark.fna"
PLSDB="${1:-$PROJ/data/databases/plasmids/plsdb.fasta}"
COMPASS="${2:-$PROJ/data/databases/plasmids/COMPASS.fna}"
OUT="$PROJ/results/baseline_plsdb_validation"

mkdir -p "$OUT"

FP_COUNT=$(wc -l < "$FP_IDS")
echo "=== Baseline FP PLSDB validation ==="
echo "FPs      : $FP_COUNT"
echo "Benchmark: $BENCHMARK"
echo "PLSDB    : $PLSDB"
echo "COMPASS  : $COMPASS"
echo ""

# ── 1. Extract FP sequences ───────────────────────────────────────────────────
echo "[1/3] Extracting FP sequences..."
python3 - <<PYEOF
fp_ids = set(open("$FP_IDS").read().splitlines())
written = 0
current_id, buf = None, []
with open("$BENCHMARK") as fin, open("$OUT/baseline_fps.fna", "w") as fout:
    for line in fin:
        line = line.rstrip()
        if line.startswith(">"):
            if current_id in fp_ids and buf:
                fout.write(f">{current_id}\n{''.join(buf)}\n")
                written += 1
            current_id = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
    if current_id in fp_ids and buf:
        fout.write(f">{current_id}\n{''.join(buf)}\n")
        written += 1
print(f"  Wrote {written} sequences → $OUT/baseline_fps.fna")
PYEOF

echo ""

# ── 2. minimap2 vs PLSDB and COMPASS ─────────────────────────────────────────
filter_paf() {
    awk '{qlen=$2; qstart=$3; qend=$4; nmatch=$10; blen=$11;
          qcov=(qend-qstart)/qlen; id=nmatch/blen;
          if(qcov>=0.50 && id>=0.90) print}' "$1"
}

echo "[2/3] Searching COMPASS..."
minimap2 -c -x asm5 --secondary=no -t 4 \
    "$COMPASS" "$OUT/baseline_fps.fna" \
    > "$OUT/baseline_fps_vs_compass.paf" 2>/dev/null
filter_paf "$OUT/baseline_fps_vs_compass.paf" > "$OUT/baseline_fps_vs_compass_filtered.paf"
echo "  Filtered hits (cov≥50% id≥90%): $(wc -l < "$OUT/baseline_fps_vs_compass_filtered.paf")"

echo ""
echo "[3/3] Searching PLSDB..."
minimap2 -c -x asm5 --secondary=no -t 4 \
    "$PLSDB" "$OUT/baseline_fps.fna" \
    > "$OUT/baseline_fps_vs_plsdb.paf" 2>/dev/null
filter_paf "$OUT/baseline_fps_vs_plsdb.paf" > "$OUT/baseline_fps_vs_plsdb_filtered.paf"
echo "  Filtered hits (cov≥50% id≥90%): $(wc -l < "$OUT/baseline_fps_vs_plsdb_filtered.paf")"

echo ""
echo "=== Corrected metrics ==="
python3 - <<'PYEOF'
import csv
from pathlib import Path

out = Path("results/baseline_plsdb_validation")

def parse_hits(paf):
    hits = {}
    try:
        for line in open(paf):
            p = line.strip().split("\t")
            if len(p) < 12: continue
            qname = p[0]; qlen = int(p[1])
            qstart, qend = int(p[2]), int(p[3])
            nmatch, blen = int(p[9]), int(p[10])
            qcov = (qend - qstart) / qlen
            identity = nmatch / blen if blen else 0
            if qname not in hits or qcov > hits[qname]["qcov"]:
                hits[qname] = {"ref": p[5], "qcov": qcov, "identity": identity}
    except FileNotFoundError:
        pass
    return hits

compass_hits = parse_hits(out / "baseline_fps_vs_compass_filtered.paf")
plsdb_hits   = parse_hits(out / "baseline_fps_vs_plsdb_filtered.paf")

fp_ids = set(open(out / "fp_ids.txt").read().splitlines())
mislabeled = fp_ids & (set(compass_hits) | set(plsdb_hits))
true_fps   = fp_ids - mislabeled

# Read raw TP/FN from predictions
gt = {}
with open("data/benchmark/ground_truth.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        gt[row["contig_id"]] = row["true_label"]

tp = fn = 0
with open(out / "predictions.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        pred = row["predicted"]
        true = gt.get(row["contig_id"], "unknown")
        if pred not in ("plasmid", "chromosome"): continue
        if pred == "plasmid" and true == "plasmid": tp += 1
        elif pred == "chromosome" and true == "plasmid": fn += 1

raw_fp    = len(fp_ids)
adj_fp    = len(true_fps)
mislabels = len(mislabeled)

prec_raw = tp / (tp + raw_fp) if tp + raw_fp else 0
rec_raw  = tp / (tp + fn)     if tp + fn     else 0
f1_raw   = 2*prec_raw*rec_raw / (prec_raw+rec_raw) if prec_raw+rec_raw else 0

prec_adj = tp / (tp + adj_fp) if tp + adj_fp else 0
rec_adj  = tp / (tp + fn)     if tp + fn     else 0
f1_adj   = 2*prec_adj*rec_adj / (prec_adj+rec_adj) if prec_adj+rec_adj else 0

print(f"  Mislabeled FPs (match PLSDB/COMPASS): {mislabels}/{raw_fp}")
print()
print(f"  {'Metric':<10} {'Raw':>8} {'Corrected':>10}")
print(f"  {'-'*30}")
print(f"  {'TP':<10} {tp:>8} {tp:>10}")
print(f"  {'FP':<10} {raw_fp:>8} {adj_fp:>10}")
print(f"  {'FN':<10} {fn:>8} {fn:>10}")
print(f"  {'Precision':<10} {prec_raw:>8.4f} {prec_adj:>10.4f}")
print(f"  {'Recall':<10} {rec_raw:>8.4f} {rec_adj:>10.4f}")
print(f"  {'F1':<10} {f1_raw:>8.4f} {f1_adj:>10.4f}")

# Write mislabeled list
with open(out / "baseline_mislabeled_fps.txt", "w") as f:
    for cid in sorted(mislabeled):
        ch = compass_hits.get(cid, {})
        ph = plsdb_hits.get(cid, {})
        ref  = ph.get("ref", ch.get("ref", ""))
        qcov = ph.get("qcov", ch.get("qcov", 0))
        ident = ph.get("identity", ch.get("identity", 0))
        f.write(f"{cid}\t{ref}\t{qcov:.3f}\t{ident:.3f}\n")
print(f"\n  Mislabeled FP list → {out}/baseline_mislabeled_fps.txt")
PYEOF

echo ""
echo "Done."
