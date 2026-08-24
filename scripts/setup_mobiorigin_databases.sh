#!/usr/bin/env bash

# Prepare the frozen MobiOrigin marker databases without mixing MOB-suite's
# legacy dependency stack into the MobiOrigin runtime environment.

OUTPUT_DIR="${1:-${HOME}/mobiorigin_databases}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"

if command -v mamba >/dev/null 2>&1; then
  ENV_MANAGER="mamba"
elif command -v conda >/dev/null 2>&1; then
  ENV_MANAGER="conda"
else
  echo "STOP: Conda or Mamba is required." >&2
  echo "Install Miniforge, then rerun this command." >&2
  exit 1
fi

if [ -z "$PROJECT_DIR" ] || [ ! -f "$PROJECT_DIR/environment.mob-database.yml" ]; then
  echo "STOP: Run this helper from a complete MobiOrigin source checkout." >&2
  exit 1
fi

echo "MobiOrigin database setup"
echo "Environment manager: $ENV_MANAGER"
echo "Output directory: $OUTPUT_DIR"

"$ENV_MANAGER" run -n mobiorigin mobiorigin --help >/dev/null 2>&1
runtime_rc=$?
if [ "$runtime_rc" -ne 0 ]; then
  echo "STOP: The 'mobiorigin' runtime environment is unavailable." >&2
  echo "Create it first with: $ENV_MANAGER env create -f environment.yml" >&2
  exit 1
fi

if [ -d "$OUTPUT_DIR" ]; then
  echo "The output directory already exists; verifying it without overwriting."
  "$ENV_MANAGER" run -n mobiorigin mobiorigin setup-databases \
    --check \
    --output-dir "$OUTPUT_DIR"
  check_rc=$?
  if [ "$check_rc" -ne 0 ]; then
    echo "STOP: Existing database directory failed verification: $OUTPUT_DIR" >&2
    echo "Keep it for diagnosis or choose a new output directory." >&2
    exit "$check_rc"
  fi
  echo "PASS: Existing MobiOrigin marker databases are valid."
  exit 0
fi

DATABASE_PLATFORM_ARGS=()
APPLE_SILICON_DATABASE=false
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  if ! arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
    echo "STOP: The MOB-suite database builder requires Apple's Rosetta compatibility layer." >&2
    echo "Install Rosetta once, then rerun this helper:" >&2
    echo "  softwareupdate --install-rosetta" >&2
    exit 1
  fi
  DATABASE_PLATFORM_ARGS=(--platform osx-64)
  APPLE_SILICON_DATABASE=true
  echo "Database-builder platform: osx-64 under Rosetta (MobiOrigin remains native arm64)"
fi

database_architecture="$("$ENV_MANAGER" run -n mobiorigin-db python -c 'import platform; print(platform.machine())' 2>/dev/null)"
database_environment_exists=$?
if [ "$database_environment_exists" -eq 0 ]; then
  if [ "$APPLE_SILICON_DATABASE" = true ] && [ "$database_architecture" != "x86_64" ]; then
    echo "STOP: Existing 'mobiorigin-db' is native $database_architecture, but MOB-suite" >&2
    echo "requires an osx-64 environment on Apple Silicon." >&2
    echo "Remove only that bootstrap environment, then rerun this helper:" >&2
    echo "  $ENV_MANAGER env remove -n mobiorigin-db" >&2
    exit 1
  fi
  "$ENV_MANAGER" env update \
    --name mobiorigin-db \
    --file "$PROJECT_DIR/environment.mob-database.yml" \
    --prune
else
  "$ENV_MANAGER" env create \
    --file "$PROJECT_DIR/environment.mob-database.yml" \
    "${DATABASE_PLATFORM_ARGS[@]}"
fi
environment_rc=$?
if [ "$environment_rc" -ne 0 ]; then
  echo "STOP: The isolated MOB-suite database environment could not be created." >&2
  echo "Do not force-install older NumPy or pandas into the mobiorigin environment." >&2
  exit "$environment_rc"
fi

echo "Initializing the official MOB-suite 3.1.8 database..."
"$ENV_MANAGER" run -n mobiorigin-db mob_init
mob_init_rc=$?
if [ "$mob_init_rc" -ne 0 ]; then
  echo "STOP: MOB-suite database initialization failed." >&2
  echo "The MobiOrigin runtime environment was not modified." >&2
  exit "$mob_init_rc"
fi

MOB_DATA_DIR="$("$ENV_MANAGER" run -n mobiorigin-db python -c 'import pathlib, mob_suite; print(pathlib.Path(mob_suite.__file__).resolve().parent / "data")')"
path_rc=$?
if [ "$path_rc" -ne 0 ] || [ -z "$MOB_DATA_DIR" ]; then
  echo "STOP: Could not locate the initialized MOB-suite data directory." >&2
  exit 1
fi

for database_name in rep_proteins.dmnd mob_proteins.dmnd mpf_proteins.dmnd; do
  if [ ! -f "$MOB_DATA_DIR/$database_name" ]; then
    echo "STOP: MOB-suite initialization did not create $MOB_DATA_DIR/$database_name" >&2
    exit 1
  fi
done

echo "Copying and cryptographically verifying the three required databases..."
"$ENV_MANAGER" run -n mobiorigin mobiorigin setup-databases \
  --source-dir "$MOB_DATA_DIR" \
  --output-dir "$OUTPUT_DIR"
copy_rc=$?
if [ "$copy_rc" -ne 0 ]; then
  echo "STOP: MobiOrigin rejected the retrieved database identities." >&2
  echo "No partial output directory was published." >&2
  exit "$copy_rc"
fi

"$ENV_MANAGER" run -n mobiorigin mobiorigin setup-databases \
  --check \
  --output-dir "$OUTPUT_DIR"
check_rc=$?
if [ "$check_rc" -ne 0 ]; then
  echo "STOP: Final MobiOrigin database verification failed." >&2
  exit "$check_rc"
fi

echo "PASS: MobiOrigin marker databases are ready at $OUTPUT_DIR"
exit 0
