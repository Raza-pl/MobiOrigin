#!/usr/bin/env bash

# Guided MobiOrigin source installation. Commands are checked explicitly;
# errexit is intentionally not enabled so failures remain readable.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
SOFTWARE_ONLY=false
SKIP_DEMO=false
SKIP_ANNOTATION_DATABASES=false
ACCEPT_THIRD_PARTY_TERMS=false
DATABASE_DIR="${MOBIORIGIN_DATABASE_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/mobiorigin/marker_databases}"
MODEL_DIR="${MOBIORIGIN_MODEL_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/mobiorigin/models/dev1}"
ANNOTATION_DATABASE_DIR="${MOBIORIGIN_ANNOTATION_DATABASE_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/mobiorigin/annotation_databases}"
DEMO_DIR="${PROJECT_DIR}/mobiorigin_demo"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --software-only) SOFTWARE_ONLY=true ;;
    --skip-demo) SKIP_DEMO=true ;;
    --skip-annotation-databases) SKIP_ANNOTATION_DATABASES=true ;;
    --accept-third-party-terms) ACCEPT_THIRD_PARTY_TERMS=true ;;
    --database-dir)
      shift
      DATABASE_DIR="$1"
      ;;
    --model-dir)
      shift
      MODEL_DIR="$1"
      ;;
    --annotation-database-dir)
      shift
      ANNOTATION_DATABASE_DIR="$1"
      ;;
    --demo-dir)
      shift
      DEMO_DIR="$1"
      ;;
    --help|-h)
      echo "Usage: bash install.sh [--software-only] [--skip-demo] [--skip-annotation-databases]"
      echo "                       [--accept-third-party-terms] [--database-dir PATH]"
      echo "                       [--model-dir PATH]"
      echo "                       [--annotation-database-dir PATH] [--demo-dir PATH]"
      exit 0
      ;;
    *) echo "STOP: Unknown installer option: $1" >&2; exit 2 ;;
  esac
  shift
done

if command -v mamba >/dev/null 2>&1; then
  ENV_MANAGER="mamba"
elif command -v conda >/dev/null 2>&1; then
  ENV_MANAGER="conda"
else
  echo "STOP: Conda or Mamba was not found." >&2
  echo "Install Miniforge, reopen the terminal, and rerun: bash install.sh" >&2
  exit 1
fi

if [ ! -f "$PROJECT_DIR/environment.yml" ]; then
  echo "STOP: This is not a complete MobiOrigin source checkout." >&2
  exit 1
fi

cd "$PROJECT_DIR"
project_cd_rc=$?
if [ "$project_cd_rc" -ne 0 ]; then
  echo "STOP: Could not enter the MobiOrigin source directory." >&2
  exit "$project_cd_rc"
fi

echo "===== MOBIORIGIN GUIDED INSTALLATION ====="
echo "Environment manager: $ENV_MANAGER"
echo "Runtime environment: mobiorigin"

"$ENV_MANAGER" run -n mobiorigin python --version >/dev/null 2>&1
environment_exists=$?
if [ "$environment_exists" -eq 0 ]; then
  echo "Updating the existing mobiorigin environment..."
  "$ENV_MANAGER" env update --name mobiorigin --file "$PROJECT_DIR/environment.yml" --prune
else
  echo "Creating the mobiorigin environment..."
  "$ENV_MANAGER" env create --file "$PROJECT_DIR/environment.yml"
fi
environment_rc=$?
if [ "$environment_rc" -ne 0 ]; then
  echo "STOP: The MobiOrigin runtime environment could not be created." >&2
  exit "$environment_rc"
fi

echo "Checking the installed software..."
"$ENV_MANAGER" run -n mobiorigin mobiorigin doctor --software-only
doctor_rc=$?
if [ "$doctor_rc" -ne 0 ]; then
  echo "STOP: The installed runtime failed its software check." >&2
  exit "$doctor_rc"
fi

if [ "$SOFTWARE_ONLY" = true ]; then
  echo "PASS: Software installation is complete. Database setup was skipped by request."
  echo "Next: bash scripts/setup_mobiorigin_databases.sh \"$DATABASE_DIR\""
  exit 0
fi

echo "Preparing the frozen marker databases..."
bash "$PROJECT_DIR/scripts/setup_mobiorigin_databases.sh" "$DATABASE_DIR" "$MODEL_DIR"
database_rc=$?
if [ "$database_rc" -ne 0 ]; then
  echo "STOP: Software is installed, but marker database setup did not complete." >&2
  echo "Resume with: bash scripts/setup_mobiorigin_databases.sh \"$DATABASE_DIR\"" >&2
  exit "$database_rc"
fi

"$ENV_MANAGER" run -n mobiorigin mobiorigin doctor \
  --database-dir "$DATABASE_DIR" \
  --model-dir "$MODEL_DIR"
full_doctor_rc=$?
if [ "$full_doctor_rc" -ne 0 ]; then
  echo "STOP: The final installation check failed." >&2
  exit "$full_doctor_rc"
fi

if [ "$SKIP_ANNOTATION_DATABASES" = false ]; then
  if [ "$ACCEPT_THIRD_PARTY_TERMS" = false ]; then
    echo
    echo "Annotation setup downloads CARD, SARG, AMRFinderPlus, VFDB, mobileOG-db,"
    echo "and BacMet from their publishers. VFDB and SARG restrict some uses."
    echo "Review: docs/MOBIORIGIN_ANNOTATION.md"
    if [ -t 0 ]; then
      printf "Accept the documented third-party terms and continue? [y/N] "
      read -r annotation_terms_answer
      case "$annotation_terms_answer" in
        y|Y|yes|YES) ACCEPT_THIRD_PARTY_TERMS=true ;;
        *) echo "Annotation database setup skipped by user."; SKIP_ANNOTATION_DATABASES=true ;;
      esac
    else
      echo "STOP: Non-interactive comprehensive installation requires explicit acceptance." >&2
      echo "Rerun with --accept-third-party-terms, or --skip-annotation-databases." >&2
      exit 1
    fi
  fi
fi

if [ "$SKIP_ANNOTATION_DATABASES" = false ]; then
  if [ -d "$ANNOTATION_DATABASE_DIR" ]; then
    echo "The annotation database directory exists; verifying it without overwriting."
    "$ENV_MANAGER" run -n mobiorigin mobiorigin setup-databases \
      --component annotation \
      --output-dir "$ANNOTATION_DATABASE_DIR" \
      --profile comprehensive \
      --check
  else
    echo "Downloading and building the comprehensive annotation databases..."
    "$ENV_MANAGER" run -n mobiorigin mobiorigin setup-databases \
      --component annotation \
      --output-dir "$ANNOTATION_DATABASE_DIR" \
      --marker-database-dir "$DATABASE_DIR" \
      --profile comprehensive \
      --accept-third-party-terms
  fi
  annotation_database_rc=$?
  if [ "$annotation_database_rc" -ne 0 ]; then
    echo "STOP: Annotation database setup did not complete." >&2
    echo "Resume with: conda activate mobiorigin" >&2
    echo "  mobiorigin setup-databases --component annotation --profile comprehensive --accept-third-party-terms" >&2
    exit "$annotation_database_rc"
  fi
fi

if [ "$SKIP_DEMO" = false ]; then
  if [ -e "$DEMO_DIR" ]; then
    echo "Demo output already exists and will not be overwritten: $DEMO_DIR"
    echo "Choose a fresh path with: bash install.sh --demo-dir /path/to/new_demo"
  else
    echo "Running the bundled installation test..."
    "$ENV_MANAGER" run -n mobiorigin mobiorigin demo \
      --database-dir "$DATABASE_DIR" \
      --output-dir "$DEMO_DIR"
    demo_rc=$?
    if [ "$demo_rc" -ne 0 ]; then
      echo "STOP: Installation passed, but the bundled end-to-end test failed." >&2
      exit "$demo_rc"
    fi
  fi
fi

echo
echo "===== INSTALLATION COMPLETE ====="
echo "Activate: conda activate mobiorigin"
echo "Analyze:  mobiorigin run --input-fasta assembly.fasta --output-dir results --threads 8"
echo "Annotate: mobiorigin annotate --input-fasta assembly.fasta --output-dir annotations --profile comprehensive --threads 8"
echo "Demo:     $DEMO_DIR"
echo "Open:     $DEMO_DIR/visualization/mobiorigin_dashboard.html"
exit 0
