#!/usr/bin/env bash
# retrain_with_replicon.sh
#
# Re-annotate the benchmark with minimap2-based replicon detection and
# benchmark the replicon_boost modifier added to predict.py.
#
# Background:
#   has_replicon has been 0 for ALL benchmark sequences because makeblastdb
#   was unavailable. The minimap2 fallback added to run_blastn_replicon()
#   now detects replicons by mapping contigs vs rep.dna.fas (2,686 curated
#   replicon sequences), filtering: replicon-coverage ≥60% AND identity ≥80%.
#
# Usage:
#   conda activate plasflow2
#   bash scripts/retrain_with_replicon.sh
#
# Flags:
#   --skip-annotate     Use existing annotations_with_replicons.tsv
#   --threads N         Number of threads (default: 8)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

THREADS=8
SKIP_ANNOTATE=false
for arg in "$@"; do
    case $arg in
        --skip-annotate) SKIP_ANNOTATE=true ;;
        --threads=*) THREADS="${arg#*=}" ;;
        --threads)   shift; THREADS=$1 ;;
    esac
done

REP_DNA="$ROOT/data/databases/mob_suite/rep.dna.fas"
ANN_IN="$ROOT/data/benchmark/annotations_with_genomad.tsv"
ANN_OUT="$ROOT/data/benchmark/annotations_with_replicons.tsv"
BENCH_FASTA="$ROOT/data/benchmark/benchmark.fna"
GENOMAD_GENES="$ROOT/data/benchmark/genomad_full/benchmark_annotate/benchmark_genes.tsv"
RESULTS_DIR="$ROOT/results/replicon_benchmark"
BENCHMARK_DIR="$ROOT/data/benchmark"

echo "============================================================"
echo "  PlasFlow v2 — Replicon-based recall improvement"
echo "  Start: $(date)"
echo "  rep.dna.fas: $REP_DNA ($(grep -c '^>' "$REP_DNA" 2>/dev/null || echo '?') sequences)"
echo "  Threads: $THREADS"
echo "============================================================"
echo ""

# ── Preflight ────────────────────────────────────────────────────────────────
if [[ ! -f "$REP_DNA" ]]; then
    echo "ERROR: rep.dna.fas not found at $REP_DNA"
    exit 1
fi
if [[ ! -f "$BENCH_FASTA" ]]; then
    echo "ERROR: Benchmark FASTA not found: $BENCH_FASTA"
    exit 1
fi
if ! command -v minimap2 &>/dev/null; then
    echo "ERROR: minimap2 not in PATH. Run: conda activate plasflow2"
    exit 1
fi
echo "  minimap2 found: $(minimap2 --version)"
echo ""

# ── Step 1: Re-annotate ───────────────────────────────────────────────────────
if $SKIP_ANNOTATE; then
    echo "[Step 1] Skipping re-annotation (--skip-annotate)"
    if [[ ! -f "$ANN_OUT" ]]; then
        echo "ERROR: $ANN_OUT not found. Run without --skip-annotate first."
        exit 1
    fi
else
    echo "[Step 1] Re-annotating benchmark with replicon detection …"
    echo "  The minimap2 fallback in run_blastn_replicon() will run automatically"
    echo "  since makeblastdb is unavailable."
    echo "  Mapping $(grep -c '^>' "$BENCH_FASTA") contigs vs ${REP_DNA##*/} …"
    echo "  Start: $(date)"

    GENOMAD_ARG=""
    if [[ -f "$GENOMAD_GENES" ]]; then
        GENOMAD_ARG="--genomad-genes $GENOMAD_GENES"
    else
        echo "  WARN: geNomad genes not found — skipping SPM features"
    fi

    python3 "$SCRIPT_DIR/annotate_sequences.py" \
        --fasta   "$BENCH_FASTA" \
        $GENOMAD_ARG \
        --out     "$ANN_OUT" \
        --threads "$THREADS"

    echo "  Done: $(date)"
fi

echo ""

# ── Step 2: Diagnostic ───────────────────────────────────────────────────────
echo "[Step 2] Replicon hit diagnostics …"
python3 - <<PYEOF
import csv
from pathlib import Path

ROOT = Path("$ROOT")
ANN  = Path("$ANN_OUT")
GT   = ROOT / "data/benchmark/ground_truth.tsv"

# Load ground truth
gt = {}
with open(GT) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        gt[row["contig_id"]] = row["true_label"]

# Load annotations
ann = {}
with open(ANN) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        ann[row["contig_id"]] = row

# Previous predictions (to identify FNs/FPs)
prev_candidates = [
    ROOT / "results/new_model_fp_validation/plasflow2_predictions.tsv",
    ROOT / "results/benchmark_no_bleed/plasflow2_predictions.tsv",
]
prev_pred = None
for c in prev_candidates:
    if c.exists():
        prev_pred = c
        break

# Count has_replicon by label
n_rep_by_class = {"plasmid": 0, "chromosome": 0, "phage": 0, "other": 0}
n_total_by_class = {"plasmid": 0, "chromosome": 0, "phage": 0, "other": 0}
for cid, a in ann.items():
    label = gt.get(cid, "other")
    has_rep = float(a.get("has_replicon", 0) or 0)
    cls = label if label in n_total_by_class else "other"
    n_total_by_class[cls] += 1
    if has_rep > 0:
        n_rep_by_class[cls] += 1

print("\n  has_replicon=1 by true label:")
for cls in ("plasmid", "chromosome", "phage"):
    n = n_total_by_class[cls]
    r = n_rep_by_class[cls]
    pct = 100*r/n if n else 0
    print(f"    {cls:12s}: {r:4d} / {n:5d} ({pct:.1f}%)")

# If we have previous predictions, show FN breakdown
if prev_pred:
    preds = {}
    with open(prev_pred) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            preds[row["contig_id"]] = row

    fn_ids = [cid for cid, p in preds.items()
               if p.get("predicted") not in ("plasmid",) and gt.get(cid) == "plasmid"]
    fp_ids = [cid for cid, p in preds.items()
               if p.get("predicted") == "plasmid" and gt.get(cid) == "chromosome"]

    fn_rep = sum(1 for cid in fn_ids if float(ann.get(cid, {}).get("has_replicon", 0) or 0) > 0)
    fp_rep = sum(1 for cid in fp_ids if float(ann.get(cid, {}).get("has_replicon", 0) or 0) > 0)

    print(f"\n  FN plasmids with replicon hit: {fn_rep} / {len(fn_ids)}")
    print(f"  FP chromosomes with replicon hit: {fp_rep} / {len(fp_ids)}")
    print(f"\n  FN detail with has_replicon=1:")
    for cid in fn_ids:
        a = ann.get(cid, {})
        if float(a.get("has_replicon", 0) or 0) > 0:
            mlp_s = preds.get(cid, {}).get("plasmid_score", "?")
            marker = ""
            if float(a.get("is_conjugative", 0)): marker += "conj "
            if float(a.get("is_mobilizable", 0)): marker += "mob "
            if float(a.get("has_rep_protein", 0)): marker += "rep "
            if float(a.get("n_plasmid_markers", 0)) > 0: marker += f"gn:{a['n_plasmid_markers']}"
            print(f"    {cid[:65]:<65}  mlp={mlp_s}  {marker}")
else:
    # Fallback: show all replicon-positive plasmids
    rep_plasmids = [(cid, a) for cid, a in ann.items()
                    if gt.get(cid) == "plasmid"
                    and float(a.get("has_replicon", 0) or 0) > 0]
    print(f"\n  All replicon-positive true plasmids: {len(rep_plasmids)}")
    for cid, a in rep_plasmids[:20]:
        print(f"    {cid}")
PYEOF

echo ""

# ── Step 3: Benchmark with replicon boost ─────────────────────────────────────
echo "[Step 3] Running benchmark with replicon_boost …"
mkdir -p "$RESULTS_DIR"
echo "  Results dir: $RESULTS_DIR"

bash "$SCRIPT_DIR/run_benchmark.sh" \
    --annotation-tsv "$ANN_OUT" \
    --benchmark-dir  "$BENCHMARK_DIR" \
    2>&1 | tee "$RESULTS_DIR/benchmark.log"

# Copy results
cp "$BENCHMARK_DIR/results/plasflow2_predictions.tsv" \
   "$RESULTS_DIR/plasflow2_predictions.tsv" 2>/dev/null || {
    echo "WARN: could not copy predictions from $BENCHMARK_DIR/results/"
}

echo "  Done: $(date)"
echo ""

# ── Step 4: Compare ───────────────────────────────────────────────────────────
echo "[Step 4] Comparing vs previous benchmark …"
python3 - <<PYEOF
import csv
from pathlib import Path

ROOT     = Path("$ROOT")
GT       = ROOT / "data/benchmark/ground_truth.tsv"
NEW_PRED = Path("$RESULTS_DIR/plasflow2_predictions.tsv")

OLD_CANDIDATES = [
    ROOT / "results/new_model_fp_validation/plasflow2_predictions.tsv",
    ROOT / "results/benchmark_no_bleed/plasflow2_predictions.tsv",
]
old_pred = next((c for c in OLD_CANDIDATES if c.exists()), None)

gt = {}
with open(GT) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        gt[row["contig_id"]] = row["true_label"]

def score(pred_path):
    tp = fp = fn = 0
    n_replicon = 0
    with open(pred_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pred = row.get("predicted", "")
            true = gt.get(row["contig_id"], "unknown")
            if pred not in ("plasmid", "chromosome"):
                continue
            if pred == "plasmid"      and true == "plasmid":    tp += 1
            elif pred == "plasmid"    and true == "chromosome":  fp += 1
            elif pred == "chromosome" and true == "plasmid":     fn += 1
            if row.get("evidence_type", "") == "replicon_boost":
                n_replicon += 1
    p  = tp / (tp+fp) if tp+fp else 0
    r  = tp / (tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return tp, fp, fn, p, r, f1, n_replicon

if not NEW_PRED.exists():
    print("  New predictions not found:", NEW_PRED)
else:
    ntp, nfp, nfn, np_, nr, nf1, nrep = score(NEW_PRED)
    print(f"\n  New results (with replicon_boost):")
    print(f"    TP={ntp}  FP={nfp}  FN={nfn}  P={np_:.4f}  R={nr:.4f}  F1={nf1:.4f}")
    print(f"    replicon_boost fired: {nrep} sequences")

    if old_pred:
        otp, ofp, ofn, op, or_, of1, _ = score(old_pred)
        print(f"\n  Previous results ({old_pred.parent.name}):")
        print(f"    TP={otp}  FP={ofp}  FN={ofn}  P={op:.4f}  R={or_:.4f}  F1={of1:.4f}")
        dtp = ntp-otp; dfp = nfp-ofp; df1 = nf1-of1
        print(f"\n  Delta:")
        print(f"    ΔTP={dtp:+d}  ΔFP={dfp:+d}  ΔF1={df1:+.4f}")
        if df1 > 0.001:
            print(f"\n  ✓ Improvement: F1 {of1:.4f} → {nf1:.4f} (+{df1:.4f})")
        elif abs(df1) <= 0.001:
            print(f"\n  — No change (replicon boost fired {nrep} times).")
        else:
            print(f"\n  ✗ Regression. Check FP breakdown.")
PYEOF

echo ""
echo "============================================================"
echo "  retrain_with_replicon.sh complete: $(date)"
echo "============================================================"
