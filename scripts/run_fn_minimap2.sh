#!/usr/bin/env bash
# Validate PlasFlow v2 benchmark false negatives against PLSDB and COMPASS.
#
# For each true plasmid sequence that was missed (predicted chromosome or
# unclassified), this script checks whether the sequence has any significant
# match in PLSDB/COMPASS. Sequences with no hit are "dark plasmids" —
# sequences the classifier cannot recover regardless of tuning because they
# carry no detectable plasmid-like signal (no k-mer similarity to known
# plasmids, no MOB-suite markers, no geNomad hallmarks).
#
# Prerequisites:
#   minimap2  (brew install minimap2 / conda install -c bioconda minimap2)
#
# Usage:
#   bash scripts/run_fn_minimap2.sh [predictions_tsv] [plsdb_fasta] [compass_fasta]
#
# Defaults:
#   predictions_tsv = results/benchmark_no_bleed/plasflow2_predictions.tsv
#   plsdb_fasta     = data/databases/plasmids/plsdb.fasta
#   compass_fasta   = data/databases/plasmids/COMPASS.fna

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PREDS="${1:-$PROJ/results/benchmark_no_bleed/plasflow2_predictions.tsv}"
BENCHMARK="$PROJ/data/benchmark/benchmark.fna"
GT="$PROJ/data/benchmark/ground_truth.tsv"
PLSDB="${2:-$PROJ/data/databases/plasmids/plsdb.fasta}"
COMPASS="${3:-$PROJ/data/databases/plasmids/COMPASS.fna}"
OUT_DIR="$PROJ/results/fn_validation"

mkdir -p "$OUT_DIR"

echo "=== PlasFlow v2 FN validation via minimap2 ==="
echo "Predictions: $PREDS"
echo "Benchmark  : $BENCHMARK"
echo "PLSDB      : $PLSDB"
echo "COMPASS    : $COMPASS"
echo "Out        : $OUT_DIR"
echo ""

# ─── Extract FN contig IDs ─────────────────────────────────────────────────
echo "[1/4] Extracting false negative IDs..."
python3 - <<PYEOF
import csv

gt = {}
with open("$GT") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        gt[row["contig_id"]] = row["true_label"]

fn_ids = []
with open("$PREDS") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        cid = row["contig_id"]
        if row["predicted"] != "plasmid" and gt.get(cid) == "plasmid":
            fn_ids.append(cid)

print(f"  FNs: {len(fn_ids)}")

# Also note unclassified true plasmids (above threshold check)
all_plasmids = [cid for cid, lbl in gt.items() if lbl == "plasmid"]
predicted_ids = set()
with open("$PREDS") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        predicted_ids.add(row["contig_id"])
unclassified = [cid for cid in all_plasmids if cid not in predicted_ids]
print(f"  True plasmids not in predictions (below threshold/unclassified): {len(unclassified)}")

with open("$OUT_DIR/fn_ids.txt", "w") as f:
    f.write("\n".join(fn_ids + unclassified))
print(f"  Written: $OUT_DIR/fn_ids.txt ({len(fn_ids)+len(unclassified)} total)")
PYEOF

FN_COUNT=$(wc -l < "$OUT_DIR/fn_ids.txt")
echo ""

# ─── Extract FN sequences from benchmark FASTA ────────────────────────────
echo "[2/4] Extracting FN sequences from benchmark FASTA..."
python3 - <<PYEOF
fn_ids = set(open("$OUT_DIR/fn_ids.txt").read().splitlines())
written = 0
current_id, buf = None, []
with open("$BENCHMARK") as fin, open("$OUT_DIR/benchmark_fns.fna", "w") as fout:
    for line in fin:
        line = line.rstrip()
        if line.startswith(">"):
            if current_id in fn_ids and buf:
                fout.write(f">{current_id}\n{''.join(buf)}\n")
                written += 1
            current_id = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
    if current_id in fn_ids and buf:
        fout.write(f">{current_id}\n{''.join(buf)}\n")
        written += 1
print(f"  Written: $OUT_DIR/benchmark_fns.fna ({written} sequences)")
PYEOF

echo ""

# ─── COMPASS search ──────────────────────────────────────────────────────────
echo "[3/4] Searching against COMPASS ($(grep -c '^>' "$COMPASS") plasmids)..."
minimap2 -c -x asm5 --secondary=no -t 4 \
    "$COMPASS" "$OUT_DIR/benchmark_fns.fna" \
    > "$OUT_DIR/fns_vs_compass.paf" 2>/dev/null

echo "  Raw hits: $(wc -l < "$OUT_DIR/fns_vs_compass.paf")"
awk '{qlen=$2; qstart=$3; qend=$4; nmatch=$10; blen=$11;
      qcov=(qend-qstart)/qlen; id=nmatch/blen;
      if(qcov>=0.30 && id>=0.85) print}' \
    "$OUT_DIR/fns_vs_compass.paf" > "$OUT_DIR/fns_vs_compass_filtered.paf"
echo "  Filtered (cov≥30% id≥85%): $(wc -l < "$OUT_DIR/fns_vs_compass_filtered.paf")"

echo ""

# ─── PLSDB search ────────────────────────────────────────────────────────────
echo "[4/4] Searching against PLSDB ($(grep -c '^>' "$PLSDB") plasmids)..."
minimap2 -c -x asm5 --secondary=no -t 4 \
    "$PLSDB" "$OUT_DIR/benchmark_fns.fna" \
    > "$OUT_DIR/fns_vs_plsdb.paf" 2>/dev/null

echo "  Raw hits: $(wc -l < "$OUT_DIR/fns_vs_plsdb.paf")"
awk '{qlen=$2; qstart=$3; qend=$4; nmatch=$10; blen=$11;
      qcov=(qend-qstart)/qlen; id=nmatch/blen;
      if(qcov>=0.30 && id>=0.85) print}' \
    "$OUT_DIR/fns_vs_plsdb.paf" > "$OUT_DIR/fns_vs_plsdb_filtered.paf"
echo "  Filtered (cov≥30% id≥85%): $(wc -l < "$OUT_DIR/fns_vs_plsdb_filtered.paf")"

# ─── Summarize ───────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
python3 - <<'PYEOF'
import csv
from pathlib import Path
from collections import defaultdict

out = Path("results/fn_validation")

fn_ids = set(open(out / "fn_ids.txt").read().splitlines())

def parse_hits(paf_file, min_cov=0.30, min_id=0.85):
    hits = {}
    try:
        with open(paf_file) as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) < 12: continue
                qname = p[0]; qlen = int(p[1])
                qstart, qend = int(p[2]), int(p[3])
                tname = p[5]
                nmatch, blen = int(p[9]), int(p[10])
                qcov = (qend - qstart) / qlen
                identity = nmatch / blen if blen else 0
                if qcov >= min_cov and identity >= min_id:
                    if qname not in hits or qcov > hits[qname]["qcov"]:
                        hits[qname] = {"ref": tname, "qcov": qcov, "identity": identity}
    except FileNotFoundError:
        pass
    return hits

compass_hits = parse_hits(out / "fns_vs_compass_filtered.paf")
plsdb_hits   = parse_hits(out / "fns_vs_plsdb_filtered.paf")

matched = fn_ids & (set(compass_hits) | set(plsdb_hits))
dark    = fn_ids - matched

print(f"  Total FNs analysed:                {len(fn_ids)}")
print(f"  Match PLSDB/COMPASS (cov≥30% id≥85%): {len(matched)}")
print(f"  'Dark plasmids' (no hit):          {len(dark)}")
print()

# Length distribution of dark plasmids
import re
dark_lens = []
current_id, buf = None, []
try:
    with open(out / "benchmark_fns.fna") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id in dark:
                    dark_lens.append(len("".join(buf)))
                current_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if current_id in dark:
            dark_lens.append(len("".join(buf)))
except: pass

if dark_lens:
    dark_lens.sort()
    print(f"  Dark plasmid lengths:")
    print(f"    Median: {dark_lens[len(dark_lens)//2]:,} bp")
    print(f"    Min:    {dark_lens[0]:,} bp")
    print(f"    Max:    {dark_lens[-1]:,} bp")
    bins = {"<2kb":0,"2-5kb":0,"5-10kb":0,"10-20kb":0,">20kb":0}
    for l in dark_lens:
        if l < 2000: bins["<2kb"] += 1
        elif l < 5000: bins["2-5kb"] += 1
        elif l < 10000: bins["5-10kb"] += 1
        elif l < 20000: bins["10-20kb"] += 1
        else: bins[">20kb"] += 1
    print(f"    By length: {bins}")

# Write TSV
rows = []
for fn_id in sorted(fn_ids):
    ch = compass_hits.get(fn_id, {})
    ph = plsdb_hits.get(fn_id, {})
    rows.append({
        "contig_id": fn_id,
        "compass_hit": ch.get("ref",""),
        "compass_qcov": f"{ch['qcov']:.3f}" if ch else "",
        "compass_identity": f"{ch['identity']:.3f}" if ch else "",
        "plsdb_hit": ph.get("ref",""),
        "plsdb_qcov": f"{ph['qcov']:.3f}" if ph else "",
        "plsdb_identity": f"{ph['identity']:.3f}" if ph else "",
        "is_dark_plasmid": "YES" if fn_id in dark else "no",
    })

with open(out / "fn_plsdb_hits.tsv", "w", newline="") as f:
    import csv as csv_mod
    w = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader()
    w.writerows(rows)
print(f"\nFull results: {out}/fn_plsdb_hits.tsv")
PYEOF

echo ""
echo "Done. See results/fn_validation/ for all output files."
echo ""
echo "Interpretation:"
echo "  - FNs that match PLSDB/COMPASS (even at lower thresholds) are recoverable"
echo "    plasmids — the model should be able to learn these with more training data."
echo "  - 'Dark plasmids' with no PLSDB hit represent the true detection ceiling:"
echo "    novel sequences with no k-mer or protein similarity to known plasmids."
