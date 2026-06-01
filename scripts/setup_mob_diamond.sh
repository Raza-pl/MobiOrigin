#!/usr/bin/env bash
# Extract MOB-suite's internal databases and build DIAMOND indexes from them.
# This enables PlasFlow v2 to do mobility typing in ~10 seconds instead of
# running 2559 sequential mob_typer processes (~8 minutes).
#
# Usage:  bash scripts/setup_mob_diamond.sh
# Run once after mob_init has been called.

set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$PROJ/data/databases/mob_suite"
mkdir -p "$OUT"
THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 8)

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }

echo "=== MOB-suite DIAMOND setup ==="
echo "Output: $OUT"
echo ""

# ── Step 1: Find mob_suite databases ─────────────────────────────────────────
MOB_DB_DIR=""

# Try Python path first (most reliable)
MOB_DB_DIR=$(python3 -c "
import mob_suite, pathlib, sys
p = pathlib.Path(mob_suite.__file__).parent
# mob_suite stores databases either inside package or in user home
candidates = [
    p / 'databases',
    pathlib.Path.home() / '.mob_suite',
    pathlib.Path.home() / '.local/share/mob-suite',
    pathlib.Path('/usr/local/share/mob-suite'),
    pathlib.Path('/opt/conda/share/mob-suite'),
    pathlib.Path('/opt/homebrew/share/mob-suite'),
]
for d in candidates:
    if d.exists() and any(d.iterdir()):
        print(d)
        sys.exit(0)
sys.exit(1)
" 2>/dev/null) || true

if [[ -z "$MOB_DB_DIR" ]]; then
    # Fallback: search common install paths
    for d in \
        "$HOME/.mob_suite" \
        "$HOME/.local/share/mob-suite" \
        "/usr/local/share/mob-suite" \
        "/opt/conda/share/mob-suite"; do
        if [[ -d "$d" ]] && ls "$d"/*.fasta >/dev/null 2>&1; then
            MOB_DB_DIR="$d"; break
        fi
    done
fi

if [[ -z "$MOB_DB_DIR" ]]; then
    warn "Cannot find mob_suite databases automatically."
    warn "Run:  mob_init   (downloads databases to ~/.mob_suite)"
    warn "Then re-run this script."
    exit 1
fi

ok "Found mob_suite databases: $MOB_DB_DIR"
ls "$MOB_DB_DIR"/*.fasta 2>/dev/null | head -10 | sed 's/^/    /'
echo ""

# ── Step 2: Copy key database files ──────────────────────────────────────────
# MOB relaxase proteins → used to detect MOB family + predict mobility class
MOB_PROT_SRC=""
for f in \
    "$MOB_DB_DIR/mob.proteins.faa" \
    "$MOB_DB_DIR/mob.proteins.fasta" \
    "$MOB_DB_DIR/mob_db.fasta"; do
    [[ -f "$f" && ! "$(basename $f)" == ._* ]] && { MOB_PROT_SRC="$f"; break; }
done

# MPF (mating pair formation) proteins → conjugative systems
MPF_PROT_SRC=""
for f in \
    "$MOB_DB_DIR/mpf.proteins.faa" \
    "$MOB_DB_DIR/mpf.proteins.fasta" \
    "$MOB_DB_DIR/mpf_db.fasta"; do
    [[ -f "$f" && ! "$(basename $f)" == ._* ]] && { MPF_PROT_SRC="$f"; break; }
done

# Replicon database (nucleotide) → IncF, IncP, etc.
REP_DB_SRC=""
for f in \
    "$MOB_DB_DIR/rep.dna.fas" \
    "$MOB_DB_DIR/rep_db.fasta" \
    "$MOB_DB_DIR/NCBI_mob_rep_db.fasta" \
    "$MOB_DB_DIR/replicons.fasta"; do
    [[ -f "$f" && ! "$(basename $f)" == ._* ]] && { REP_DB_SRC="$f"; break; }
done

echo "Key files found:"
[[ -n "$MOB_PROT_SRC" ]] && ok "MOB proteins:  $MOB_PROT_SRC" \
                          || warn "MOB proteins NOT FOUND in $MOB_DB_DIR"
[[ -n "$MPF_PROT_SRC" ]] && ok "MPF proteins:  $MPF_PROT_SRC" \
                          || warn "MPF proteins NOT FOUND in $MOB_DB_DIR"
[[ -n "$REP_DB_SRC"   ]] && ok "Replicon DB:   $REP_DB_SRC" \
                          || warn "Replicon DB NOT FOUND in $MOB_DB_DIR"
echo ""

# ── Step 3: Build DIAMOND databases for protein files ────────────────────────
build_dmnd() {
    local src="$1" name="$2"
    local dest="$OUT/${name}.dmnd"
    if [[ -f "$dest" ]]; then
        ok "$name DIAMOND DB already exists: $dest"
        return 0
    fi
    echo "  Building DIAMOND DB: $name ..."
    diamond makedb --in "$src" --db "$OUT/$name" --threads "$THREADS" --quiet
    ok "Built: $dest"
}

[[ -n "$MOB_PROT_SRC" ]] && build_dmnd "$MOB_PROT_SRC" "mob_proteins"
[[ -n "$MPF_PROT_SRC" ]] && build_dmnd "$MPF_PROT_SRC" "mpf_proteins"

# Copy replicon DB (nucleotide — used with minimap2 not DIAMOND)
if [[ -n "$REP_DB_SRC" && ! -f "$OUT/rep_db.fasta" ]]; then
    cp "$REP_DB_SRC" "$OUT/rep_db.fasta"
    ok "Copied replicon DB: $OUT/rep_db.fasta"
fi

# Also copy the full mob_db_dir path for reference
echo "$MOB_DB_DIR" > "$OUT/mob_db_dir.txt"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Done ==="
echo "Files in $OUT:"
ls -lh "$OUT/" | tail -n +2 | awk '{print "  "$NF, $5}'
echo ""
echo "PlasFlow v2 will auto-detect these and use DIAMOND for mobility typing."
echo "Next run will be ~10 seconds for mobility instead of ~8 minutes."
