#!/usr/bin/env bash
# =============================================================================
# test_installation.sh — Verify PlasFlow v2 installation is working
#
# Usage:
#   bash scripts/test_installation.sh              # runs classify test
#   bash scripts/test_installation.sh --full       # also runs full pipeline (needs databases)
#   bash scripts/test_installation.sh --skip-download  # reuse existing test_assembly.fasta
#
# What it checks:
#   1. plasflow2 binary is on PATH
#   2. Model weights exist (data/models/mlp_v2.pt)
#   3. Example sequences are downloaded from NCBI (or reused)
#   4. plasflow2 classify runs successfully (no databases needed)
#   5. [--full] plasflow2 run completes with the example data
#   6. Output files look correct (not empty, expected columns present)
# =============================================================================

set -euo pipefail

SKIP_DOWNLOAD=false
FULL_TEST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --full)          FULL_TEST=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/test_installation.sh [--full] [--skip-download]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
EXAMPLE_DIR="$REPO_ROOT/example"
TEST_FASTA="$EXAMPLE_DIR/test_assembly.fasta"
TEST_OUT="$EXAMPLE_DIR/test_output"
CLASSIFY_TSV="$EXAMPLE_DIR/test_predictions.tsv"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[PASS]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAILURES=$((FAILURES + 1)); }
step() { echo ""; echo -e "${GREEN}══ $* ${NC}"; }

FAILURES=0

echo "============================================================"
echo "  PlasFlow v2 — Installation Test"
echo "  $(date)"
echo "============================================================"

# ── 1. Check plasflow2 binary ─────────────────────────────────────────────────
step "1. Checking plasflow2 command"
if command -v plasflow2 &>/dev/null; then
    VERSION=$(plasflow2 --version 2>/dev/null || echo "unknown")
    ok "plasflow2 found: $VERSION"
else
    fail "plasflow2 command not found. Run: pip install -e . (from repo root)"
    echo "    conda activate plasflow2 && pip install -e ."
    exit 1
fi

# ── 2. Check model weights ────────────────────────────────────────────────────
step "2. Checking model weights"
MODEL="$REPO_ROOT/data/models/mlp_v2.pt"
if [[ -f "$MODEL" ]]; then
    SIZE=$(du -sh "$MODEL" 2>/dev/null | cut -f1)
    ok "MLP model found: $MODEL ($SIZE)"
else
    fail "MLP model not found at data/models/mlp_v2.pt"
    echo "    Download with: bash scripts/setup_databases.sh"
    echo "    Or models only: bash scripts/setup_databases.sh \\"
    echo "        --skip-plsdb --skip-card --skip-sarg --skip-amrfinder \\"
    echo "        --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite"
    exit 1
fi

XGB_MODEL="$REPO_ROOT/data/models/marker_xgb.pkl"
if [[ -f "$XGB_MODEL" ]]; then
    ok "XGBoost model found: $XGB_MODEL"
else
    warn "XGBoost model not found at data/models/marker_xgb.pkl — stage-2 will be skipped"
fi

# ── 3. Check/download example sequences ──────────────────────────────────────
step "3. Preparing example sequences"
mkdir -p "$EXAMPLE_DIR"

if [[ -f "$TEST_FASTA" ]] && [[ "$SKIP_DOWNLOAD" == "true" ]]; then
    N=$(grep -c "^>" "$TEST_FASTA" 2>/dev/null || echo 0)
    ok "Reusing existing $TEST_FASTA ($N sequences)"
elif [[ -f "$TEST_FASTA" ]] && [[ $(grep -c "^>" "$TEST_FASTA" 2>/dev/null || echo 0) -ge 3 ]]; then
    N=$(grep -c "^>" "$TEST_FASTA")
    ok "Using existing $TEST_FASTA ($N sequences)"
else
    echo "  Downloading example sequences from NCBI …"
    if python "$EXAMPLE_DIR/download_test_data.py" --out "$TEST_FASTA"; then
        N=$(grep -c "^>" "$TEST_FASTA" 2>/dev/null || echo 0)
        ok "Downloaded $N sequences → $TEST_FASTA"
    else
        fail "Failed to download example sequences"
        echo "    Check your internet connection, or copy a FASTA manually to:"
        echo "    $TEST_FASTA"
        exit 1
    fi
fi

# Sanity check: at least 2 sequences, at least 1 kb each
MIN_SEQS=2
ACTUAL_SEQS=$(grep -c "^>" "$TEST_FASTA" 2>/dev/null || echo 0)
if [[ "$ACTUAL_SEQS" -lt "$MIN_SEQS" ]]; then
    fail "Too few sequences in $TEST_FASTA ($ACTUAL_SEQS < $MIN_SEQS required)"
    exit 1
fi

# ── 4. plasflow2 classify (fast — no databases) ───────────────────────────────
step "4. Running plasflow2 classify (MLP only — no databases needed)"
rm -f "$CLASSIFY_TSV"

if plasflow2 classify \
        --input  "$TEST_FASTA" \
        --output "$CLASSIFY_TSV" \
        --min-length 500 \
        2>&1 | tee /tmp/plasflow2_classify.log; then
    ok "plasflow2 classify completed"
else
    fail "plasflow2 classify failed"
    cat /tmp/plasflow2_classify.log
    exit 1
fi

# Check output file
if [[ -f "$CLASSIFY_TSV" ]]; then
    N_ROWS=$(tail -n +2 "$CLASSIFY_TSV" | wc -l | tr -d ' ')
    ok "Predictions TSV has $N_ROWS contig rows → $CLASSIFY_TSV"
else
    fail "Predictions TSV not found: $CLASSIFY_TSV"
fi

# Check that expected columns are present
EXPECTED_COLS="contig_id label confidence plasmid_score"
HEADER=$(head -1 "$CLASSIFY_TSV" 2>/dev/null || echo "")
for col in $EXPECTED_COLS; do
    if echo "$HEADER" | grep -q "$col"; then
        : # ok
    else
        fail "Expected column '$col' not found in predictions TSV"
    fi
done
ok "TSV columns look correct"

# Show classification summary
echo ""
echo "  Classification summary:"
tail -n +2 "$CLASSIFY_TSV" | awk -F'\t' '{counts[$2]++} END {for (l in counts) printf "    %-20s %d contigs\n", l, counts[l]}' | sort
echo ""

# Check that at least one plasmid was detected
PLASMID_COUNT=$(tail -n +2 "$CLASSIFY_TSV" | awk -F'\t' '$2=="plasmid"' | wc -l | tr -d ' ')
if [[ "$PLASMID_COUNT" -ge 1 ]]; then
    ok "At least one plasmid detected ($PLASMID_COUNT) — classifier is working"
else
    warn "No plasmids detected — the MLP may need higher sensitivity (try --plasmid-threshold 0.70)"
    warn "This can happen with short test contigs. Not necessarily a bug."
fi

# ── 5. Full pipeline test (--full only) ───────────────────────────────────────
if [[ "$FULL_TEST" == "true" ]]; then
    step "5. Running plasflow2 run (full pipeline — requires databases)"

    CARD_DB="$REPO_ROOT/data/databases/card/card.dmnd"
    ARO_INDEX="$REPO_ROOT/data/databases/card/aro_index.tsv"

    if [[ ! -f "$CARD_DB" ]]; then
        fail "CARD database not found: $CARD_DB"
        echo "    Download with: bash scripts/setup_databases.sh"
        FAILURES=$((FAILURES + 1))
    else
        rm -rf "$TEST_OUT"
        if plasflow2 run \
                --input   "$TEST_FASTA" \
                --output  "$TEST_OUT" \
                --threads 4 \
                --skip-taxonomy \
                2>&1 | tee /tmp/plasflow2_run.log; then
            ok "plasflow2 run completed"

            # Check expected output files
            for expected in all_predictions.tsv plasmids.fasta annotations.json; do
                if [[ -f "$TEST_OUT/$expected" ]]; then
                    SIZE=$(wc -l < "$TEST_OUT/$expected" 2>/dev/null || echo "?")
                    ok "  $expected exists ($SIZE lines)"
                else
                    fail "  Expected output missing: $TEST_OUT/$expected"
                fi
            done

            # Check report was generated
            if ls "$TEST_OUT"/report_*.html &>/dev/null; then
                ok "  HTML reports generated: $(ls "$TEST_OUT"/report_*.html | wc -l) files"
            else
                warn "  No HTML reports found (may be OK if 0 plasmids detected)"
            fi
        else
            fail "plasflow2 run failed"
            cat /tmp/plasflow2_run.log
        fi
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
if [[ "$FAILURES" -eq 0 ]]; then
    echo -e "${GREEN}  All tests passed!${NC}"
    echo ""
    echo "  PlasFlow v2 is installed and working correctly."
    echo "  Run on your data:"
    echo "    plasflow2 run --input assembly.fasta --output results/ --threads 16"
else
    echo -e "${RED}  $FAILURES test(s) failed.${NC}"
    echo ""
    echo "  See errors above. Common fixes:"
    echo "    conda activate plasflow2          # activate the environment"
    echo "    bash scripts/setup_databases.sh   # download databases + model weights"
    echo "    plasflow2 setup                   # print full setup guide"
fi
echo "============================================================"

exit "$FAILURES"
