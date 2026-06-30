#!/usr/bin/env bash
# =============================================================================
# install.sh — PlasFlow v2 one-command installer
#
# Supported platforms:
#   Mac Intel (x86_64), Mac M1/M2/M3/M4/M5 (arm64), Linux x86_64, WSL Ubuntu
#
# What this does:
#   1. Creates conda environment "plasflow2" (Python 3.10 + all tools)
#   2. Installs PlasFlow v2 Python package (editable)
#   3. Downloads model weights + all annotation databases
#
# Usage:
#   bash install.sh              # full install
#   bash install.sh --skip-plsdb # skip the 5 GB PLSDB download
#   bash install.sh --env-only   # create conda env + install pkg; skip databases
#
# After install, activate the environment:
#   conda activate plasflow2
# =============================================================================

set -euo pipefail

SKIP_DATABASES=false
SETUP_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-only)    SKIP_DATABASES=true; shift ;;
        --skip-plsdb)  SETUP_ARGS+=("--skip-plsdb"); shift ;;
        --threads)     SETUP_ARGS+=("--threads" "$2"); shift 2 ;;
        -h|--help)
            echo "Usage: bash install.sh [--env-only] [--skip-plsdb] [--threads N]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }
step() { echo ""; echo -e "${GREEN}══ $* ${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  PlasFlow v2 — Installer"
echo "  Platform: $(uname -s) $(uname -m)"
echo "============================================================"

# ── Detect conda ─────────────────────────────────────────────────────────────
step "1. Checking conda"

if ! command -v conda &>/dev/null; then
    err "conda not found. Install Miniconda first:
  Mac/Linux: https://docs.conda.io/en/latest/miniconda.html
  WSL Ubuntu: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
              bash Miniconda3-latest-Linux-x86_64.sh"
fi

CONDA_VERSION=$(conda --version)
ok "conda found: $CONDA_VERSION"

# ── Create conda environment ──────────────────────────────────────────────────
step "2. Creating conda environment 'plasflow2'"

if conda env list | grep -q "^plasflow2 "; then
    warn "Environment 'plasflow2' already exists."
    read -rp "    Update it? [y/N]: " choice
    if [[ "${choice:-N}" =~ ^[Yy]$ ]]; then
        conda env update -f environment.yml --prune
        ok "Environment updated"
    else
        ok "Using existing environment"
    fi
else
    echo "  Creating from environment.yml (this takes 5–10 min on first run)..."
    conda env create -f environment.yml
    ok "Environment 'plasflow2' created"
fi

# ── Activate env for subsequent commands ─────────────────────────────────────
# shellcheck disable=SC1091
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate plasflow2

# ── Verify plasflow2 CLI is installed ────────────────────────────────────────
step "3. Verifying PlasFlow v2 installation"

if command -v plasflow2 &>/dev/null; then
    ok "plasflow2: $(plasflow2 --version 2>/dev/null || echo installed)"
else
    warn "plasflow2 command not found — installing..."
    pip install -e .
    ok "PlasFlow v2 installed"
fi

# ── Download databases + model weights ───────────────────────────────────────
if $SKIP_DATABASES; then
    warn "Skipping database download (--env-only)"
else
    step "4. Downloading model weights and databases"
    bash scripts/setup_databases.sh "${SETUP_ARGS[@]}"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  Activate the environment:"
echo "    conda activate plasflow2"
echo ""
echo "  Run PlasFlow v2:"
echo "    plasflow2 run --input assembly.fasta --output results/"
echo ""
echo "  For help:"
echo "    plasflow2 --help"
echo "    plasflow2 run --help"
echo "============================================================"
