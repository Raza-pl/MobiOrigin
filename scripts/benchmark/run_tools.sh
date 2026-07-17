#!/usr/bin/env bash
# Run all benchmark tools on a PlasFlow v2 benchmark dataset.
#
# Usage:
#   bash scripts/benchmark/run_tools.sh \
#       --input  data/benchmark/tier1/all_species/input.fasta \
#       --out    data/benchmark/results/tier1_all \
#       --threads 8
#
# Prerequisites (install each tool before running):
#   PlasFlow v2  : already installed (this repo)
#   geNomad      : conda install -c conda-forge -c bioconda genomad
#   PlasClass    : pip install plasclass
#   RFPlasmid    : pip install rfplasmid
#   MOB-recon    : pip install mob-suite && mob_init
#
# Outputs (one subdirectory per tool):
#   {out}/plasflow2/    all_predictions.tsv
#   {out}/genomad/      {prefix}_plasmid.tsv, {prefix}_virus.tsv
#   {out}/plasclass/    plasclass_scores.csv
#   {out}/rfplasmid/    outputRFPlasmid.txt
#   {out}/mobrecon/     contig_report.txt
#   {out}/timing.tsv    wall-clock seconds per tool

set -euo pipefail

# ── Argument parsing ───────────────────────────────────────────────────────────

INPUT=""
OUTDIR=""
THREADS=4
GENOMAD_DB="${PLASFLOW_GENOMAD_DB:-data/databases/genomad_db}"
SKIP=""  # comma-separated list of tools to skip, e.g. "mobrecon,rfplasmid"

while [[ $# -gt 0 ]]; do
  case $1 in
    --input)    INPUT="$2";      shift 2 ;;
    --out)      OUTDIR="$2";     shift 2 ;;
    --threads)  THREADS="$2";    shift 2 ;;
    --genomad-db) GENOMAD_DB="$2"; shift 2 ;;
    --skip)     SKIP="$2";       shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$INPUT"  ]] && { echo "Error: --input is required" >&2; exit 1; }
[[ -z "$OUTDIR" ]] && { echo "Error: --out is required"   >&2; exit 1; }
[[ -f "$INPUT"  ]] || { echo "Error: $INPUT does not exist" >&2; exit 1; }

mkdir -p "$OUTDIR"
TIMING_TSV="$OUTDIR/timing.tsv"
echo -e "tool\twallclock_sec\tstatus" > "$TIMING_TSV"

_skip() { echo "$SKIP" | tr ',' '\n' | grep -qx "$1"; }
_time_tool() {
  local tool="$1"; shift
  local outd="$1"; shift
  if _skip "$tool"; then
    echo "  [skip] $tool"
    echo -e "$tool\t-1\tskipped" >> "$TIMING_TSV"
    return
  fi
  echo "── $tool ────────────────────────────"
  mkdir -p "$outd"
  local start end elapsed
  start=$(date +%s)
  if "$@" 2>&1 | tee "$outd/run.log"; then
    end=$(date +%s); elapsed=$((end - start))
    echo -e "$tool\t$elapsed\tok" >> "$TIMING_TSV"
    echo "  ✓ $tool completed in ${elapsed}s"
  else
    end=$(date +%s); elapsed=$((end - start))
    echo -e "$tool\t$elapsed\tfailed" >> "$TIMING_TSV"
    echo "  ✗ $tool FAILED after ${elapsed}s (see $outd/run.log)" >&2
  fi
}

# ── PlasFlow v2 ────────────────────────────────────────────────────────────────

_time_tool "plasflow2" "$OUTDIR/plasflow2" \
  plasflow2 run \
    --input "$INPUT" \
    --output "$OUTDIR/plasflow2" \
    --threads "$THREADS" \
    --skip-genomad \
    --skip-plasmid-db

# ── geNomad ────────────────────────────────────────────────────────────────────

_time_tool "genomad" "$OUTDIR/genomad" \
  genomad end-to-end \
    "$INPUT" \
    "$OUTDIR/genomad" \
    "$GENOMAD_DB" \
    --threads "$THREADS" \
    --cleanup

# ── PlasClass ─────────────────────────────────────────────────────────────────

_time_tool "plasclass" "$OUTDIR/plasclass" \
  python -u scripts/benchmark/run_plasclass_streaming.py \
    --input "$INPUT" \
    --output "$OUTDIR/plasclass/plasclass_scores.csv" \
    --processes "$THREADS"

# ── RFPlasmid ─────────────────────────────────────────────────────────────────

_time_tool "rfplasmid" "$OUTDIR/rfplasmid" bash -c "
  cd '$OUTDIR/rfplasmid' && \
  RFPlasmid \
    --input '$INPUT' \
    --threads $THREADS \
    --output outputRFPlasmid.txt 2>&1
"

# ── MOB-recon ─────────────────────────────────────────────────────────────────

_time_tool "mobrecon" "$OUTDIR/mobrecon" \
  mob_recon \
    --infile "$INPUT" \
    --outdir "$OUTDIR/mobrecon" \
    --num_threads "$THREADS" \
    --force

echo ""
echo "All tools finished. Timing summary:"
cat "$TIMING_TSV"
echo ""
echo "Next: python scripts/benchmark/evaluate.py --results $OUTDIR --labels <path/to/labels.tsv>"
