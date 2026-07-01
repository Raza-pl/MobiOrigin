#!/usr/bin/env bash
# =============================================================================
# setup_databases.sh — PlasFlow v2 complete database + model installer
#
# Downloads everything PlasFlow v2 needs to run:
#   1. Model weights (MLP + XGBoost)  ~85 MB
#   2. CARD           ARG annotation  ~300 MB
#   3. SARG           ARG annotation  ~50 MB
#   4. AMRFinderPlus  ARG annotation  ~30 MB
#   5. VFDB           virulence factors ~10 MB
#   6. BacMet2        metal/biocide resistance ~5 MB
#   7. MGE / ISfinder mobile elements ~20 MB
#   8. ICEberg3       integrative conjugative elements ~5 MB
#   9. PLSDB          plasmid matching ~5 GB
#  10. MOB-suite      mobility typing  ~500 MB  (via mob_init)
#
# Usage:
#   bash scripts/setup_databases.sh
#   bash scripts/setup_databases.sh --skip-plsdb          # skip 5 GB PLSDB
#   bash scripts/setup_databases.sh --skip-vfdb --skip-mge
#   bash scripts/setup_databases.sh --threads 16
#
# Skip flags:  --skip-models --skip-card --skip-sarg --skip-amrfinder
#              --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg
#              --skip-plsdb --skip-mobsuite
#
# Point to an existing database (bypasses download):
#   bash scripts/setup_databases.sh --card-path /existing/card/card.dmnd
#   bash scripts/setup_databases.sh --plsdb-path /existing/PLSDB.fna
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB_DIR="$PROJECT_DIR/data/databases"
MODEL_DIR="$PROJECT_DIR/data/models"
THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 8)

# ── Defaults ─────────────────────────────────────────────────────────────────
SKIP_MODELS=false
SKIP_CARD=false
SKIP_SARG=false
SKIP_AMRFINDER=false
SKIP_VFDB=false
SKIP_BACMET=false
SKIP_MGE=false
SKIP_ICEBERG=false
SKIP_PLSDB=false
SKIP_MOBSUITE=false

CARD_PATH=""
PLSDB_PATH=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-models)    SKIP_MODELS=true;    shift ;;
        --skip-card)      SKIP_CARD=true;      shift ;;
        --skip-sarg)      SKIP_SARG=true;      shift ;;
        --skip-amrfinder) SKIP_AMRFINDER=true; shift ;;
        --skip-vfdb)      SKIP_VFDB=true;      shift ;;
        --skip-bacmet)    SKIP_BACMET=true;    shift ;;
        --skip-mge)       SKIP_MGE=true;       shift ;;
        --skip-iceberg)   SKIP_ICEBERG=true;   shift ;;
        --skip-plsdb)     SKIP_PLSDB=true;     shift ;;
        --skip-mobsuite)  SKIP_MOBSUITE=true;  shift ;;
        --card-path)      CARD_PATH="$2";      shift 2 ;;
        --plsdb-path)     PLSDB_PATH="$2";     shift 2 ;;
        --threads)        THREADS="$2";        shift 2 ;;
        -h|--help)
            sed -n '/^# Usage/,/^# ====/p' "$0" | grep -v "^# ====" | sed 's/^# //'
            exit 0 ;;
        *) echo "Unknown argument: $1  (run with --help for usage)"; exit 1 ;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
info() { echo "    $*"; }
section() { echo ""; echo -e "${CYAN}─── $* ${NC}"; }

mkdir -p "$DB_DIR" "$MODEL_DIR"

echo "============================================================"
echo "  PlasFlow v2 — Database & Model Setup"
echo "  Project : $PROJECT_DIR"
echo "  DB root : $DB_DIR"
echo "  Models  : $MODEL_DIR"
echo "  Threads : $THREADS"
echo "============================================================"

# ── Helper: download a file if it doesn't already exist ──────────────────────
download() {
    local url="$1" dest="$2" label="${3:-$(basename "$dest")}"
    if [[ -f "$dest" ]]; then
        ok "Already exists: $label  ($(du -sh "$dest" | cut -f1))"
        return 0
    fi
    info "Downloading $label ..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url" || { rm -f "$dest"; return 1; }
    else
        curl -L --progress-bar -o "$dest" "$url" || { rm -f "$dest"; return 1; }
    fi
    ok "Downloaded: $label  ($(du -sh "$dest" | cut -f1))"
}

# ── Helper: build a DIAMOND protein database ─────────────────────────────────
build_diamond() {
    local fasta="$1" prefix="$2"
    if [[ -f "${prefix}.dmnd" ]]; then
        ok "DIAMOND db exists: ${prefix}.dmnd"
        return 0
    fi
    info "Building DIAMOND database: $prefix ..."
    diamond makedb --in "$fasta" --db "$prefix" --threads "$THREADS" --quiet
    ok "DIAMOND database built: ${prefix}.dmnd"
}

# ── 0. External tools check ───────────────────────────────────────────────────
section "Checking external tools"
MISSING_TOOLS=()
for tool in diamond minimap2; do
    if command -v "$tool" &>/dev/null; then
        ok "$tool  ($(command -v "$tool"))"
    else
        err "$tool not found"
        MISSING_TOOLS+=("$tool")
    fi
done
if command -v mob_typer &>/dev/null; then
    ok "mob_typer  ($(command -v mob_typer))"
else
    warn "mob_typer not found — mobility typing will be skipped"
    MISSING_TOOLS+=("mob-suite")
fi

if [[ ${#MISSING_TOOLS[@]} -gt 0 ]]; then
    echo ""
    warn "Missing tools: ${MISSING_TOOLS[*]}"
    info "Install with:"
    info "  conda install -c bioconda -c conda-forge ${MISSING_TOOLS[*]}"
    info ""
    info "Re-run this script after installing."
    # Continue anyway — some databases can still be downloaded
fi

# ── 1. Model weights ─────────────────────────────────────────────────────────
section "1. Model weights (MLP + XGBoost + PCA)"

if $SKIP_MODELS; then
    warn "Skipping model download (--skip-models)"
else
    MLP_PT="$MODEL_DIR/mlp_v2.pt"
    XGB_PKL="$MODEL_DIR/marker_xgb.pkl"
    PCA_PKL="$MODEL_DIR/k6_pca.pkl"

    ALL_MODELS_OK=true
    for f in "$MLP_PT" "$XGB_PKL" "$PCA_PKL"; do
        [[ -f "$f" ]] || { ALL_MODELS_OK=false; break; }
    done

    if $ALL_MODELS_OK; then
        ok "mlp_v2.pt        ($(du -sh "$MLP_PT" | cut -f1))"
        ok "marker_xgb.pkl   ($(du -sh "$XGB_PKL" | cut -f1))"
        ok "k6_pca.pkl       ($(du -sh "$PCA_PKL" | cut -f1))"
    else
        # ── Download from GitHub Releases ─────────────────────────────────────
        # Release v2.0.0 hosts all three model files (~85 MB total).
        # URL format: https://github.com/Raza-pl/plasflow2.0/releases/download/v2.0.0/<file>
        RELEASE_BASE="https://github.com/Raza-pl/plasflow2.0/releases/download/v2.0.0"

        model_ok=true
        for fname in mlp_v2.pt marker_xgb.pkl k6_pca.pkl; do
            dest="$MODEL_DIR/$fname"
            if [[ ! -f "$dest" ]]; then
                if ! download "$RELEASE_BASE/$fname" "$dest" "$fname"; then
                    err "Could not download $fname"
                    model_ok=false
                fi
            else
                ok "$fname  ($(du -sh "$dest" | cut -f1))"
            fi
        done

        if ! $model_ok; then
            echo ""
            warn "Automatic model download failed."
            info "The model files are not yet on a public release, OR the release URL changed."
            info ""
            info "Options:"
            info "  1. Ask the developer for the model files and copy them to:"
            info "       $MODEL_DIR/"
            info "     Required files: mlp_v2.pt  marker_xgb.pkl  k6_pca.pkl"
            info ""
            info "  2. Train from scratch (requires training data):"
            info "       python scripts/train_model.py --help"
            info ""
            info "  PlasFlow v2 will NOT run without mlp_v2.pt."
        fi
    fi
fi

# ── 2. CARD (Comprehensive Antibiotic Resistance Database) ───────────────────
section "2. CARD (Antibiotic Resistance Genes)"
CARD_DIR="$DB_DIR/card"
CARD_DMND="$CARD_DIR/card.dmnd"
CARD_ARO="$CARD_DIR/aro_index.tsv"
mkdir -p "$CARD_DIR"

if $SKIP_CARD; then
    warn "Skipping CARD (--skip-card)"
elif [[ -n "$CARD_PATH" ]]; then
    if [[ -f "$CARD_PATH" ]]; then
        ok "Using existing CARD: $CARD_PATH"
        [[ "$CARD_PATH" != "$CARD_DMND" ]] && ln -sf "$CARD_PATH" "$CARD_DMND"
    else
        err "--card-path file not found: $CARD_PATH"
    fi
elif [[ -f "$CARD_DMND" && -f "$CARD_ARO" ]]; then
    ok "CARD database: $CARD_DMND  ($(du -sh "$CARD_DMND" | cut -f1))"
    ok "ARO index:     $CARD_ARO"
else
    # Download CARD canonical data bundle from McMaster
    CARD_TBZ="$CARD_DIR/card.tar.bz2"
    CARD_URL="https://card.mcmaster.ca/latest/data"

    info "Downloading CARD from card.mcmaster.ca (~300 MB)..."
    if download "$CARD_URL" "$CARD_TBZ" "card.tar.bz2"; then
        info "Extracting CARD archive..."
        tar -xjf "$CARD_TBZ" -C "$CARD_DIR" 2>/dev/null || tar -xf "$CARD_TBZ" -C "$CARD_DIR"

        # Find the protein homolog FASTA (name varies by CARD release)
        CARD_FASTA=$(find "$CARD_DIR" -name "protein_fasta_protein_homolog_model.fasta" | head -1)
        if [[ -z "$CARD_FASTA" ]]; then
            CARD_FASTA=$(find "$CARD_DIR" -name "*.fasta" -not -name "nucleotide*" | head -1)
        fi

        if [[ -n "$CARD_FASTA" ]]; then
            build_diamond "$CARD_FASTA" "$CARD_DIR/card"
        else
            err "Could not find CARD protein FASTA in $CARD_DIR"
        fi

        # Copy ARO index
        ARO_SRC=$(find "$CARD_DIR" -name "aro_index.tsv" | head -1)
        if [[ -n "$ARO_SRC" && "$ARO_SRC" != "$CARD_ARO" ]]; then
            cp "$ARO_SRC" "$CARD_ARO"
            ok "ARO index: $CARD_ARO"
        fi
    else
        err "CARD download failed."
        info "Manual download: https://card.mcmaster.ca/download"
        info "  → Download 'CARD Data' (card-data.tar.bz2)"
        info "  → Extract to: $CARD_DIR/"
        info "  → Re-run this script"
    fi
fi

# ── 3. SARG (Structured ARG Database) ────────────────────────────────────────
section "3. SARG (Structured ARG Database)"
SARG_DIR="$DB_DIR/sarg"
SARG_DMND="$SARG_DIR/sarg.dmnd"
mkdir -p "$SARG_DIR"

if $SKIP_SARG; then
    warn "Skipping SARG (--skip-sarg)"
elif [[ -f "$SARG_DMND" ]]; then
    ok "SARG: $SARG_DMND  ($(du -sh "$SARG_DMND" | cut -f1))"
else
    SARG_FASTA="$SARG_DIR/sarg.fasta"
    # SARG v3 URLs — try multiple mirrors in order of reliability.
    # Note: GitHub raw has a 100 MB limit; the HKU server can be slow from some regions.
    # ARGs-OAP v3 (xinehc) is the actively maintained source.
    SARG_URLS=(
        "https://github.com/xinehc/args_oap/raw/main/src/args_oap/db/SARG.fasta.gz"
        "https://raw.githubusercontent.com/biofuture/Ublastx_stageone/master/DB/SARG.fasta"
        "https://smile.hku.hk/SARGs/static/download/SARG.fasta"
    )

    downloaded=false
    SARG_GZ="$SARG_DIR/sarg.fasta.gz"
    for url in "${SARG_URLS[@]}"; do
        info "Trying: $url"
        if [[ "$url" == *.gz ]]; then
            # Compressed download — decompress after
            if download "$url" "$SARG_GZ" "SARG.fasta.gz"; then
                info "Decompressing SARG.fasta.gz ..."
                gunzip -f "$SARG_GZ" && downloaded=true && break
            fi
        else
            if download "$url" "$SARG_FASTA" "SARG.fasta"; then
                downloaded=true && break
            fi
        fi
        info "Trying next URL..."
    done

    if $downloaded; then
        build_diamond "$SARG_FASTA" "$SARG_DIR/sarg"
    else
        warn "SARG download failed from all URLs. (SARG is optional — AMRFinder + CARD cover most use cases.)"
        info "To add SARG manually:"
        info "  1. Download SARG.fasta from https://smile.hku.hk/SARGs"
        info "     or: conda install -c bioconda args-oap  (includes SARG)"
        info "  2. Copy the FASTA to: $SARG_FASTA"
        info "  3. Re-run: bash scripts/setup_databases.sh --skip-card --skip-amrfinder"
    fi
fi

# ── 4. AMRFinderPlus protein database ────────────────────────────────────────
section "4. AMRFinderPlus (NCBI)"
AMR_DIR="$DB_DIR/amrfinder"
AMR_DMND="$AMR_DIR/amrprot.dmnd"
mkdir -p "$AMR_DIR"

if $SKIP_AMRFINDER; then
    warn "Skipping AMRFinderPlus (--skip-amrfinder)"
elif [[ -f "$AMR_DMND" ]]; then
    ok "AMRFinderPlus: $AMR_DMND  ($(du -sh "$AMR_DMND" | cut -f1))"
else
    AMR_FASTA="$AMR_DIR/AMR_CDS.fa"
    # NCBI AMRFinderPlus CDS protein file (updated quarterly)
    AMR_URL="https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/latest/AMR_CDS.fa"

    if download "$AMR_URL" "$AMR_FASTA" "AMR_CDS.fa"; then
        build_diamond "$AMR_FASTA" "$AMR_DIR/amrprot"
    else
        warn "AMRFinderPlus download failed."
        info "Manual: https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/latest/"
        info "  → Download AMR_CDS.fa to: $AMR_FASTA"
    fi
fi

# ── 5. VFDB set A (Virulence Factors) ────────────────────────────────────────
section "5. VFDB (Virulence Factor Database)"
VFDB_DIR="$DB_DIR/vfdb"
VFDB_DMND="$VFDB_DIR/vfdb.dmnd"
mkdir -p "$VFDB_DIR"

if $SKIP_VFDB; then
    warn "Skipping VFDB (--skip-vfdb)"
elif [[ -f "$VFDB_DMND" ]]; then
    ok "VFDB: $VFDB_DMND  ($(du -sh "$VFDB_DMND" | cut -f1))"
else
    VFDB_GZ="$VFDB_DIR/VFDB_setA_pro.fas.gz"
    VFDB_FASTA="$VFDB_DIR/VFDB_setA_pro.fas"
    VFDB_URL="http://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz"

    if download "$VFDB_URL" "$VFDB_GZ" "VFDB_setA_pro.fas.gz"; then
        [[ -f "$VFDB_FASTA" ]] || gunzip -k "$VFDB_GZ"
        build_diamond "$VFDB_FASTA" "$VFDB_DIR/vfdb"
    else
        warn "VFDB download failed (server at mgc.ac.cn is sometimes slow)."
        info "Manual: http://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz"
        info "  → Save to: $VFDB_GZ  then re-run"
    fi
fi

# ── 6. BacMet2 (biocide & metal resistance) ──────────────────────────────────
section "6. BacMet2 (Biocide & Metal Resistance)"
BACMET_DIR="$DB_DIR/bacmet"
BACMET_DMND="$BACMET_DIR/bacmet.dmnd"
mkdir -p "$BACMET_DIR"

if $SKIP_BACMET; then
    warn "Skipping BacMet2 (--skip-bacmet)"
elif [[ -f "$BACMET_DMND" ]]; then
    ok "BacMet2: $BACMET_DMND  ($(du -sh "$BACMET_DMND" | cut -f1))"
else
    BACMET_GZ="$BACMET_DIR/BacMet2_predicted.fasta.gz"
    BACMET_FASTA="$BACMET_DIR/BacMet2_predicted.fasta"
    BACMET_URL="http://bacmet.biomedicine.gu.se/download/BacMet2_predicted_database.fasta.gz"

    if download "$BACMET_URL" "$BACMET_GZ" "BacMet2_predicted.fasta.gz"; then
        [[ -f "$BACMET_FASTA" ]] || gunzip -k "$BACMET_GZ"
        build_diamond "$BACMET_FASTA" "$BACMET_DIR/bacmet"
    else
        warn "BacMet2 download failed."
        info "Manual: http://bacmet.biomedicine.gu.se/download.html"
        info "  → Download 'Predicted protein sequences' and save to: $BACMET_GZ"
    fi
fi

# ── 7. MGE / ISfinder (mobile genetic elements) ──────────────────────────────
section "7. MGE database (IS elements / integrons / transposons)"
MGE_DIR="$DB_DIR/mge"
MGE_DMND="$MGE_DIR/isfinder.dmnd"
mkdir -p "$MGE_DIR"

if $SKIP_MGE; then
    warn "Skipping MGE (--skip-mge)"
elif [[ -f "$MGE_DMND" ]]; then
    ok "MGE: $MGE_DMND  ($(du -sh "$MGE_DMND" | cut -f1))"
else
    MGE_TGZ="$MGE_DIR/MGEs_FINAL_99perc_trim.fasta.tar.gz"
    MGE_NT_FASTA="$MGE_DIR/MGEs_FINAL_99perc_trim.fasta"
    MGE_AA_FASTA="$MGE_DIR/mge_proteins.faa"
    MGE_URL="https://github.com/KatariinaParnanen/MobileGeneticElementDatabase/raw/master/MGEs_FINAL_99perc_trim.fasta.tar.gz"

    if download "$MGE_URL" "$MGE_TGZ" "MGEs database"; then
        [[ -f "$MGE_NT_FASTA" ]] || tar -xzf "$MGE_TGZ" -C "$MGE_DIR"

        info "Translating MGE nucleotide sequences → proteins..."
        python3 - "$MGE_NT_FASTA" "$MGE_AA_FASTA" <<'PYEOF'
import sys
from Bio import SeqIO
from Bio.Seq import Seq
in_fa, out_fa = sys.argv[1], sys.argv[2]
written = 0
with open(out_fa, "w") as fh:
    for rec in SeqIO.parse(in_fa, "fasta"):
        nt = str(rec.seq).upper().replace("-", "N")
        if len(nt) % 3:
            nt += "N" * (3 - len(nt) % 3)
        aa = str(Seq(nt).translate(to_stop=True))
        if len(aa) >= 30:
            fh.write(f">{rec.id} {rec.description[len(rec.id):].strip()}\n{aa}\n")
            written += 1
print(f"    {written} protein sequences written")
PYEOF
        build_diamond "$MGE_AA_FASTA" "$MGE_DIR/isfinder"
    else
        warn "MGE download failed."
    fi
fi

# ── 8. ICEberg3 (integrative conjugative elements) ───────────────────────────
section "8. ICEberg3 (Integrative Conjugative Elements)"
ICE_DIR="$DB_DIR/ice"
ICE_DMND="$ICE_DIR/ice.dmnd"
mkdir -p "$ICE_DIR"

if $SKIP_ICEBERG; then
    warn "Skipping ICEberg3 (--skip-iceberg)"
elif [[ -f "$ICE_DMND" ]]; then
    ok "ICEberg3: $ICE_DMND  ($(du -sh "$ICE_DMND" | cut -f1))"
else
    ICE_FASTA="$ICE_DIR/ICEberg3_proteins.faa"
    # ICEberg3 protein download (SJTU server — may require manual download)
    ICE_URL="https://bioinfo-mml.sjtu.edu.cn/ICEberg3/download/ICEberg3_protein.faa"

    if download "$ICE_URL" "$ICE_FASTA" "ICEberg3_proteins.faa"; then
        build_diamond "$ICE_FASTA" "$ICE_DIR/ice"
    else
        warn "ICEberg3 download failed (SJTU server may be slow or require VPN)."
        info "Manual: https://bioinfo-mml.sjtu.edu.cn/ICEberg3/  → Download → Protein sequences"
        info "  → Save to: $ICE_FASTA  then re-run"
        info "  → This database is optional. ICE annotation will be skipped if absent."
    fi
fi

# ── 9. PLSDB + plasmid databases (for minimap2 matching) ─────────────────────
section "9. Plasmid databases (PLSDB + RefSeq + COMPASS)"
PLAS_DIR="$DB_DIR/plasmids"
mkdir -p "$PLAS_DIR"

if $SKIP_PLSDB; then
    warn "Skipping plasmid databases (--skip-plsdb)"
else
    PLSDB_FILE="$PLAS_DIR/PLSDB.fna"

    if [[ -n "$PLSDB_PATH" ]]; then
        if [[ -f "$PLSDB_PATH" ]]; then
            ok "Using existing PLSDB: $PLSDB_PATH"
            [[ "$PLSDB_PATH" != "$PLSDB_FILE" ]] && ln -sf "$PLSDB_PATH" "$PLSDB_FILE"
        else
            err "--plsdb-path file not found: $PLSDB_PATH"
        fi
    elif [[ -f "$PLSDB_FILE" ]]; then
        ok "PLSDB: $PLSDB_FILE  ($(du -sh "$PLSDB_FILE" | cut -f1))"
    else
        PLSDB_BZ2="$PLAS_DIR/plsdb.fna.bz2"
        # Try PLSDB 2025 URL first, fall back to legacy URL
        PLSDB_URLS=(
            "https://ccb.uni-saarland.de/plsdb2025/plasmids/download/plsdb.fna.bz2"
            "https://ccb-microbe.cs.uni-saarland.de/plsdb/plasmids/download/plsdb.fna.bz2"
        )
        info "Downloading PLSDB (~1 GB compressed, ~5 GB uncompressed)..."
        downloaded=false
        for url in "${PLSDB_URLS[@]}"; do
            if download "$url" "$PLSDB_BZ2" "PLSDB.fna.bz2"; then
                downloaded=true
                break
            fi
            info "Trying next URL..."
        done

        if $downloaded; then
            info "Decompressing PLSDB (this may take a few minutes)..."
            bzip2 -dk "$PLSDB_BZ2"
            # bzip2 -dk creates plsdb.fna in the same dir
            [[ -f "$PLAS_DIR/plsdb.fna" ]] && mv "$PLAS_DIR/plsdb.fna" "$PLSDB_FILE"
            ok "PLSDB: $PLSDB_FILE  ($(du -sh "$PLSDB_FILE" | cut -f1))"
        else
            warn "PLSDB download failed. Plasmid DB matching will be skipped at runtime."
            info "Manual: https://ccb.uni-saarland.de/plsdb2025/  → Download FASTA"
            info "  → Save decompressed file to: $PLSDB_FILE"
        fi
    fi

    # COMPASS and RefSeq are optional supplements — note their expected paths
    for db in RefSeq COMPASS; do
        f="$PLAS_DIR/${db}.fna"
        [[ -f "$f" ]] && ok "$db: $f  ($(du -sh "$f" | cut -f1))" \
                       || info "$db not found at $f  (optional, PLSDB alone is sufficient)"
    done

    # combined.fna is built automatically at first run
    COMBINED="$PLAS_DIR/combined.fna"
    [[ -f "$COMBINED" ]] && ok "Combined FASTA: $COMBINED" \
                          || info "combined.fna will be built automatically on first run"
fi

# ── 10. MOB-suite databases ───────────────────────────────────────────────────
section "10. MOB-suite databases (mobility typing)"

if $SKIP_MOBSUITE; then
    warn "Skipping MOB-suite init (--skip-mobsuite)"
elif ! command -v mob_typer &>/dev/null; then
    warn "mob_typer not installed — skipping"
    info "Install: conda install -c bioconda -c conda-forge mob_suite"
    info "Then re-run this script (or just run: mob_init)"
else
    # Check if mob_init has already been run
    MOB_DB_FOUND=false
    for d in "$HOME/.mob_suite" "$HOME/.local/share/mob-suite" \
              "/usr/local/share/mob-suite" "/opt/conda/share/mob-suite" \
              "/opt/conda/envs/plasflow2/share/mob-suite"; do
        if [[ -d "$d" ]]; then
            ok "mob-suite databases: $d"
            MOB_DB_FOUND=true
            break
        fi
    done

    if ! $MOB_DB_FOUND; then
        info "Running mob_init (~500 MB download, a few minutes)..."
        if mob_init; then
            ok "mob-suite databases initialised"
        else
            warn "mob_init failed."
            info "Try manually: conda activate plasflow2 && mob_init"
        fi
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup Summary"
echo "============================================================"

check_file() {
    local label="$1" path="$2" optional="${3:-false}"
    if [[ -f "$path" ]]; then
        printf "  ${GREEN}✓${NC} %-26s %s\n" "$label" "$(du -sh "$path" | cut -f1)"
    elif $optional; then
        printf "  ${YELLOW}○${NC} %-26s optional — skipped\n" "$label"
    else
        printf "  ${RED}✗${NC} %-26s NOT FOUND\n" "$label"
    fi
}

echo ""
echo "  Models:"
check_file "mlp_v2.pt"          "$MODEL_DIR/mlp_v2.pt"
check_file "marker_xgb.pkl"     "$MODEL_DIR/marker_xgb.pkl"
check_file "k6_pca.pkl"         "$MODEL_DIR/k6_pca.pkl"
echo ""
echo "  Databases:"
check_file "CARD.dmnd"          "$DB_DIR/card/card.dmnd"
check_file "CARD ARO index"     "$DB_DIR/card/aro_index.tsv"
check_file "SARG.dmnd"          "$DB_DIR/sarg/sarg.dmnd"
check_file "AMRFinderPlus.dmnd" "$DB_DIR/amrfinder/amrprot.dmnd"
check_file "VFDB.dmnd"          "$DB_DIR/vfdb/vfdb.dmnd"
check_file "BacMet2.dmnd"       "$DB_DIR/bacmet/bacmet.dmnd"
check_file "MGE/ISfinder.dmnd"  "$DB_DIR/mge/isfinder.dmnd"
check_file "ICEberg3.dmnd"      "$DB_DIR/ice/ice.dmnd"          true
check_file "PLSDB.fna"          "$DB_DIR/plasmids/PLSDB.fna"

echo ""
echo "  ► Ready to run:"
echo "    plasflow2 run --input assembly.fasta --output results/ --threads $THREADS"
echo ""
echo "  ► Quick classify (no databases needed):"
echo "    plasflow2 classify --input assembly.fasta --output predictions.tsv"
echo "============================================================"
