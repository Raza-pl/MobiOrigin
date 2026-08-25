#!/usr/bin/env bash

# Prepare the frozen MobiOrigin marker databases without mixing MOB-suite's
# legacy dependency stack into the MobiOrigin runtime environment.

OUTPUT_DIR="${1:-${HOME}/mobiorigin_databases}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

if command -v mamba >/dev/null 2>&1; then
  ENV_MANAGER="mamba"
elif command -v conda >/dev/null 2>&1; then
  ENV_MANAGER="conda"
else
  echo "STOP: Conda or Mamba is required." >&2
  echo "Install Miniforge, then rerun this command." >&2
  exit 1
fi

if [ -z "$PROJECT_DIR" ] || \
  [ ! -f "$PROJECT_DIR/environment.mob-database.yml" ] || \
  [ ! -f "$PROJECT_DIR/environment.marker-build.yml" ]; then
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
  echo "Reusing the existing isolated MOB-suite environment."
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

marker_build_architecture="$("$ENV_MANAGER" run -n mobiorigin-marker-build python -c 'import platform; print(platform.machine())' 2>/dev/null)"
marker_build_environment_exists=$?
if [ "$marker_build_environment_exists" -eq 0 ]; then
  if [ "$APPLE_SILICON_DATABASE" = true ] && [ "$marker_build_architecture" != "x86_64" ]; then
    echo "STOP: Existing 'mobiorigin-marker-build' has the wrong architecture." >&2
    echo "Remove only that small helper environment, then rerun:" >&2
    echo "  $ENV_MANAGER env remove -n mobiorigin-marker-build" >&2
    exit 1
  fi
  "$ENV_MANAGER" run -n mobiorigin-marker-build diamond version | grep -q 'diamond version 2.0.15'
  marker_version_rc=$?
  if [ "$marker_version_rc" -ne 0 ]; then
    echo "STOP: Existing marker-build helper does not contain DIAMOND 2.0.15." >&2
    echo "Remove only that small helper environment, then rerun:" >&2
    echo "  $ENV_MANAGER env remove -n mobiorigin-marker-build" >&2
    exit 1
  fi
  echo "Reusing the isolated DIAMOND 2.0.15 marker-build environment."
else
  "$ENV_MANAGER" env create \
    --file "$PROJECT_DIR/environment.marker-build.yml" \
    "${DATABASE_PLATFORM_ARGS[@]}"
  marker_environment_rc=$?
  if [ "$marker_environment_rc" -ne 0 ]; then
    echo "STOP: The isolated DIAMOND marker-build environment could not be created." >&2
    exit "$marker_environment_rc"
  fi
fi

"$ENV_MANAGER" run -n mobiorigin-db python -c 'import numpy, pandas, mob_suite; print("Database environment imports: PASS")'
import_rc=$?
if [ "$import_rc" -ne 0 ]; then
  echo "STOP: The isolated database environment has incompatible Python packages." >&2
  echo "User-site packages are disabled by this helper; recreate only mobiorigin-db and rerun." >&2
  echo "  $ENV_MANAGER env remove -n mobiorigin-db" >&2
  exit "$import_rc"
fi

locate_mob_data() {
  "$ENV_MANAGER" run -n mobiorigin-db python -c '
import pathlib
import mob_suite
root = pathlib.Path(mob_suite.__file__).resolve().parent
for candidate in (root / "databases", root / "data"):
    required = ("rep.dna.fas", "mob.proteins.faa", "mpf.proteins.faa")
    if all((candidate / name).is_file() for name in required):
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
' 2>/dev/null
}

MOB_DATA_DIR="$(locate_mob_data)"
path_rc=$?
if [ "$path_rc" -eq 0 ] && [ -n "$MOB_DATA_DIR" ]; then
  echo "Reusing the completed official MOB-suite database download."
else
  echo "Initializing the official MOB-suite 3.1.8 database..."
  "$ENV_MANAGER" run -n mobiorigin-db mob_init
  mob_init_rc=$?
  if [ "$mob_init_rc" -ne 0 ]; then
    echo "STOP: MOB-suite database initialization failed." >&2
    echo "The MobiOrigin runtime environment was not modified." >&2
    exit "$mob_init_rc"
  fi
  MOB_DATA_DIR="$(locate_mob_data)"
  path_rc=$?
  if [ "$path_rc" -ne 0 ] || [ -z "$MOB_DATA_DIR" ]; then
    echo "STOP: Could not locate MOB-suite's initialized raw marker files." >&2
    exit 1
  fi
fi

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mobiorigin-marker-build.XXXXXX")"
build_path_rc=$?
if [ "$build_path_rc" -ne 0 ] || [ -z "$BUILD_DIR" ]; then
  echo "STOP: Could not create a temporary marker build directory." >&2
  exit 1
fi

echo "Reconstructing the three frozen marker databases with DIAMOND 2.0.15..."
"$ENV_MANAGER" run -n mobiorigin-marker-build python \
  "$PROJECT_DIR/src/mobiorigin/marker_database_builder.py" \
  --raw-dir "$MOB_DATA_DIR" \
  --output-dir "$BUILD_DIR/frozen" \
  --diamond diamond
build_rc=$?
if [ "$build_rc" -ne 0 ]; then
  echo "STOP: Frozen marker database reconstruction failed." >&2
  echo "Temporary diagnostics retained at: $BUILD_DIR" >&2
  exit "$build_rc"
fi

echo "Copying and cryptographically verifying the three required databases..."
"$ENV_MANAGER" run -n mobiorigin mobiorigin setup-databases \
  --source-dir "$BUILD_DIR/frozen" \
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

"$ENV_MANAGER" run -n mobiorigin-marker-build python -c \
  'import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$BUILD_DIR"

echo "PASS: MobiOrigin marker databases are ready at $OUTPUT_DIR"
exit 0
