#!/usr/bin/env bash
# =============================================================================
# setup_databases.sh — PlasFlow v2 one-shot database installer
#
# Downloads and builds all annotation databases:
#   1. CARD         ARG annotation (DIAMOND + ARO index)
#   2. SARG         supplementary ARG annotation (DIAMOND)
#   3. VFDB set A   virulence factor annotation (DIAMOND)
#   4. MGE          IS elements / integrons / transposons (DIAMOND)
#   5. PLSDB        plasmid nucleotide DB for minimap2 matching
#   6. mob-suite    plasmid mobility typing (mob_init)
#
# All databases land at their auto-detected default paths — no flags needed.
#
# Usage:
#   bash scripts/setup_databases.sh              # full setup
#   bash scripts/setup_databases.sh --skip-plsdb # skip the 5 GB PLSDB download
#   bash scripts/setup_databases.sh --threads 16
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB_DIR="$PROJECT_DIR/data/databases"
THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 8)
SKIP_PLSDB=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-plsdb) SKIP_PLSDB=true; shift ;;
        --threads)    THREADS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }
info() { echo "    $*"; }

echo "============================================================"
echo "  PlasFlow v2 — Database Setup"
echo "  DB root : $DB_DIR"
echo "  Threads : $THREADS"
echo "============================================================"
echo ""

# ── Helpers ──────────────────────────────────────────────────────────────────

download() {
    local url="$1" dest="$2"
    [[ -f "$dest" ]] && { ok "Already exists: $(basename "$dest")"; return 0; }
    info "Downloading $(basename "$dest") from $url ..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url" || { rm -f "$dest"; return 1; }
    else
        curl -L --progress-bar -o "$dest" "$url" || { rm -f "$dest"; return 1; }
    fi
    ok "Downloaded: $(basename "$dest") ($(du -sh "$dest" | cut -f1))"
}

build_diamond_db() {
    local fasta="$1" db_prefix="$2"
    info "Building DIAMOND database: $db_prefix ..."
    diamond makedb --in "$fasta" --db "$db_prefix" --threads "$THREADS" --quiet
    ok "DIAMOND database built: ${db_prefix}.dmnd"
}

# ── 0. Tools check ────────────────────────────────────────────────────────────
echo "─── Checking tools ──────────────────────────────────────────"

for tool in diamond minimap2; do
    if command -v "$tool" &>/dev/null; then
        ok "$tool: $(command -v $tool)"
    else
        warn "$tool not found — install with: conda install -c bioconda $tool"
    fi
done

if command -v mob_typer &>/dev/null; then
    ok "mob_typer: $(command -v mob_typer)"
else
    warn "mob_typer not found"
    info "Install: conda install -c bioconda mob-suite  OR  pip install mob-suite"
fi
echo ""

# ── 1. CARD ───────────────────────────────────────────────────────────────────
echo "─── 1. CARD (Antibiotic Resistance) ────────────────────────"
CARD_DIR="$DB_DIR/card"
CARD_DMND="$CARD_DIR/card.dmnd"
CARD_ARO="$CARD_DIR/aro_index.tsv"
mkdir -p "$CARD_DIR"

if [[ -f "$CARD_DMND" && -f "$CARD_ARO" ]]; then
    ok "CARD database: $CARD_DMND"
    ok "ARO index:     $CARD_ARO"
else
    info "CARD database not found — building via plasflow2..."
    python3 -c "
from plasflow2.annotate.args import setup_card_db
import sys
card_dir = sys.argv[1]
try:
    dmnd, aro = setup_card_db(card_dir)
    print(f'CARD database: {dmnd}')
    print(f'ARO index:     {aro}')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" "$CARD_DIR"
    [[ -f "$CARD_DMND" ]] && ok "CARD built: $CARD_DMND" || warn "CARD build failed — check card.tar.bz2 in $CARD_DIR"
fi
echo ""

# ── 2. SARG ───────────────────────────────────────────────────────────────────
echo "─── 2. SARG (Structured ARG Database) ──────────────────────"
SARG_DIR="$DB_DIR/sarg"
SARG_DMND="$SARG_DIR/sarg.dmnd"
mkdir -p "$SARG_DIR"

if [[ -f "$SARG_DMND" ]]; then
    ok "SARG database: $SARG_DMND  ($(du -sh "$SARG_DMND" | cut -f1))"
else
    SARG_FASTA="$SARG_DIR/sarg.fasta"
    # SARG v3 — hosted on HKUST lab server + GitHub mirror
    SARG_URL="https://raw.githubusercontent.com/biofuture/Ublastx_stageone/master/DB/SARG.fasta"
    SARG_URL_ALT="https://smile.hku.hk/SARGs/static/download/SARG.fasta"

    if ! download "$SARG_URL" "$SARG_FASTA"; then
        info "Primary URL failed, trying alternate..."
        download "$SARG_URL_ALT" "$SARG_FASTA" || {
            warn "SARG download failed. Download manually from https://smile.hku.hk/SARGs"
            warn "Save to: $SARG_FASTA  then re-run this script."
        }
    fi

    if [[ -f "$SARG_FASTA" ]]; then
        build_diamond_db "$SARG_FASTA" "$SARG_DIR/sarg"
    fi
fi
echo ""

# ── 3. VFDB set A ─────────────────────────────────────────────────────────────
echo "─── 3. VFDB (Virulence Factors) ────────────────────────────"
VFDB_DIR="$DB_DIR/vfdb"
VFDB_DMND="$VFDB_DIR/vfdb.dmnd"   # NOTE: must match _DEFAULT_VFDB in cli.py
mkdir -p "$VFDB_DIR"

if [[ -f "$VFDB_DMND" ]]; then
    ok "VFDB database: $VFDB_DMND  ($(du -sh "$VFDB_DMND" | cut -f1))"
else
    VFDB_FASTA="$VFDB_DIR/VFDB_setA_pro.fas"
    VFDB_GZ="$VFDB_DIR/VFDB_setA_pro.fas.gz"
    # VFDB set A = experimentally validated virulence factors only
    VFDB_URL="http://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz"

    if download "$VFDB_URL" "$VFDB_GZ"; then
        [[ -f "$VFDB_FASTA" ]] || { info "Decompressing..."; gunzip -k "$VFDB_GZ"; }
        build_diamond_db "$VFDB_FASTA" "$VFDB_DIR/vfdb"
    else
        warn "VFDB download failed (server may be slow)."
        info "Manual download: http://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz"
        info "Save to: $VFDB_GZ  then re-run."
    fi
fi
echo ""

# ── 4. MGE database ───────────────────────────────────────────────────────────
echo "─── 4. MGE database (IS elements / integrons / transposons) ─"
MGE_DIR="$DB_DIR/mge"
MGE_DMND="$MGE_DIR/isfinder.dmnd"
mkdir -p "$MGE_DIR"

if [[ -f "$MGE_DMND" ]]; then
    ok "MGE database: $MGE_DMND  ($(du -sh "$MGE_DMND" | cut -f1))"
else
    MGE_NT_FASTA="$MGE_DIR/MGEs_FINAL_99perc_trim.fasta"
    MGE_AA_FASTA="$MGE_DIR/mge_proteins.faa"
    MGE_TGZ="$MGE_DIR/MGEs_FINAL_99perc_trim.fasta.tar.gz"
    MGE_URL="https://github.com/KatariinaParnanen/MobileGeneticElementDatabase/raw/master/MGEs_FINAL_99perc_trim.fasta.tar.gz"

    if download "$MGE_URL" "$MGE_TGZ"; then
        [[ -f "$MGE_NT_FASTA" ]] || tar -xzf "$MGE_TGZ" -C "$MGE_DIR"

        info "Translating CDS → proteins..."
        python3 - "$MGE_NT_FASTA" "$MGE_AA_FASTA" <<'PYEOF'
import sys
from Bio import SeqIO
from Bio.Seq import Seq
in_fa, out_fa = sys.argv[1], sys.argv[2]
written = 0
with open(out_fa, "w") as fh:
    for rec in SeqIO.parse(in_fa, "fasta"):
        nt = str(rec.seq).upper().replace("-","N")
        if len(nt) % 3: nt += "N" * (3 - len(nt) % 3)
        aa = str(Seq(nt).translate(to_stop=True))
        if len(aa) >= 30:
            fh.write(f">{rec.id} {rec.description[len(rec.id):].strip()}\n{aa}\n")
            written += 1
print(f"    {written} proteins translated")
PYEOF
        build_diamond_db "$MGE_AA_FASTA" "$MGE_DIR/isfinder"
    else
        warn "MGE download failed."
    fi
fi
echo ""

# ── 5. Plasmid databases (for minimap2 matching) ─────────────────────────────
echo "─── 5. Plasmid databases (PLSDB + RefSeq + COMPASS) ────────"
PLAS_DIR="$DB_DIR/plasmids"
mkdir -p "$PLAS_DIR"

if $SKIP_PLSDB; then
    warn "Skipping plasmid DB download (--skip-plsdb)"
else
    # PLSDB — curated plasmid sequence database (~5 GB nucleotide FASTA)
    PLSDB_FILE="$PLAS_DIR/PLSDB.fna"
    PLSDB_URL="https://ccb-microbe.cs.uni-saarland.de/plsdb/plasmids/download/plsdb.fna.bz2"
    PLSDB_BZ2="$PLAS_DIR/plsdb.fna.bz2"

    if [[ -f "$PLSDB_FILE" ]]; then
        ok "PLSDB: $PLSDB_FILE  ($(du -sh "$PLSDB_FILE" | cut -f1))"
    else
        info "Downloading PLSDB (~1 GB compressed, ~5 GB uncompressed)..."
        if download "$PLSDB_URL" "$PLSDB_BZ2"; then
            info "Decompressing PLSDB..."
            bzip2 -dk "$PLSDB_BZ2"
            mv "$PLAS_DIR/plsdb.fna" "$PLSDB_FILE"
            ok "PLSDB: $PLSDB_FILE  ($(du -sh "$PLSDB_FILE" | cut -f1))"
        else
            warn "PLSDB download failed. The plasmid-DB matching step will be skipped."
            info "Manual: https://ccb-microbe.cs.uni-saarland.de/plsdb"
        fi
    fi

    # Check for RefSeq and COMPASS (typically pre-downloaded or user-provided)
    for db in RefSeq COMPASS; do
        f="$PLAS_DIR/${db}.fna"
        [[ -f "$f" ]] && ok "$db: $f  ($(du -sh "$f" | cut -f1))" \
                       || info "$db not found at $f (optional — PLSDB alone is sufficient)"
    done

    # Build combined FASTA for minimap2 (done lazily at runtime if missing)
    COMBINED="$PLAS_DIR/combined.fna"
    [[ -f "$COMBINED" ]] && ok "Combined plasmid FASTA: $COMBINED  ($(du -sh "$COMBINED" | cut -f1))" \
                          || info "combined.fna will be built automatically on first run"
fi
echo ""

# ── 6. MOB-suite databases ────────────────────────────────────────────────────
echo "─── 6. MOB-suite databases ──────────────────────────────────"
if command -v mob_typer &>/dev/null; then
    # Check if mob_init has been run (databases exist somewhere)
    MOB_DB_FOUND=false
    for d in "$HOME/.mob_suite" "$HOME/.local/share/mob-suite" \
              "/usr/local/share/mob-suite" "/opt/conda/share/mob-suite"; do
        [[ -d "$d" ]] && { ok "mob-suite databases: $d"; MOB_DB_FOUND=true; break; }
    done
    if ! $MOB_DB_FOUND; then
        info "Running mob_init (downloads ~500 MB — takes a few minutes)..."
        mob_init && ok "mob-suite databases initialised" \
                 || warn "mob_init failed — run 'mob_init' manually"
    fi
else
    warn "mob_typer not installed — skipping"
    info "Install: conda install -c bioconda -c conda-forge mob-suite"
    info "   OR:   pip install mob-suite && mob_init"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Setup Summary"
echo "============================================================"

check_db() {
    local label="$1" path="$2"
    if [[ -f "$path" ]]; then
        printf "  ${GREEN}✓${NC} %-22s %s\n" "$label:" "$path"
    else
        printf "  ${YELLOW}✗${NC} %-22s NOT FOUND\n" "$label:"
    fi
}

check_db "CARD"        "$DB_DIR/card/card.dmnd"
check_db "ARO index"   "$DB_DIR/card/aro_index.tsv"
check_db "SARG"        "$DB_DIR/sarg/sarg.dmnd"
check_db "VFDB"        "$DB_DIR/vfdb/vfdb.dmnd"
check_db "MGE"         "$DB_DIR/mge/isfinder.dmnd"
check_db "PLSDB"       "$DB_DIR/plasmids/PLSDB.fna"

echo ""
echo "  Run the pipeline (all databases auto-detected):"
echo "    plasflow2 run --input assembly.fasta --output results/ --threads $THREADS"
echo ""
echo "  Or with context:"
echo "    plasflow2 run --input assembly.fasta --output results/ \\"
echo "      --context wastewater --threads $THREADS --plasmid-threshold 0.95"
echo "============================================================"
