#!/usr/bin/env bash

# Run the bundled eight-contig assembly example. Prediction and visualization
# always run. Comprehensive annotation runs only when ANNOTATION_DATABASE points
# to the user-prepared CARD/SARG/VFDB/MGE/BacMet/MOB database directory.

set +e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MOBIORIGIN_BIN="${MOBIORIGIN_BIN:-$(command -v mobiorigin)}"
INPUT_FASTA="$ROOT/src/mobiorigin/data/examples/annotated_assembly_example.fasta"
MARKER_DATABASE="${MOBIORIGIN_DATABASE_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/mobiorigin/marker_databases}"
ANNOTATION_DATABASE="${ANNOTATION_DATABASE:-}"
AMRFINDER_DATABASE="${AMRFINDER_DATABASE:-}"
THREADS="${THREADS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/mobiorigin_assembly_example_$(date +%Y%m%d_%H%M%S)}"
ANALYSIS_DIR="$OUTPUT_DIR/analysis"
ANNOTATION_DIR="$OUTPUT_DIR/annotation"
INTEGRATED_DIR="$OUTPUT_DIR/integrated_visualization"

echo "MobiOrigin bundled assembly example"
echo "Input: $INPUT_FASTA"
echo "Output: $OUTPUT_DIR"
echo "Expected output classes: chromosome=2, plasmid=2, phage=2, unclassified=2"
echo "Boundary: software demonstration only; not an accuracy or prevalence estimate"

if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "STOP: activate the MobiOrigin environment first; Python was not found." >&2
  exit 1
fi
if [ -z "$MOBIORIGIN_BIN" ] || [ ! -x "$MOBIORIGIN_BIN" ]; then
  echo "STOP: mobiorigin was not found in the active environment." >&2
  exit 1
fi
if [ ! -s "$INPUT_FASTA" ]; then
  echo "STOP: bundled assembly example is missing: $INPUT_FASTA" >&2
  exit 1
fi
if [ ! -d "$MARKER_DATABASE" ]; then
  echo "STOP: marker database directory is missing: $MARKER_DATABASE" >&2
  echo "Run the guided installer or scripts/setup_mobiorigin_databases.sh first." >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "STOP: fresh output directory required: $OUTPUT_DIR" >&2
  exit 1
fi

echo "Running prediction and visualization..."
"$MOBIORIGIN_BIN" run \
  --input-fasta "$INPUT_FASTA" \
  --output-dir "$ANALYSIS_DIR" \
  --database-dir "$MARKER_DATABASE" \
  --threads "$THREADS"
prediction_rc=$?
if [ "$prediction_rc" -ne 0 ]; then
  echo "STOP: bundled assembly prediction failed (status $prediction_rc)." >&2
  exit "$prediction_rc"
fi

"$PYTHON_BIN" - "$ANALYSIS_DIR/predictions/predictions.tsv" "$OUTPUT_DIR/four_class_prediction_check.tsv" <<'PY'
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

prediction_path = Path(sys.argv[1])
check_path = Path(sys.argv[2])
expected = {
    "assembly_example_chromosome_01": "chromosome",
    "assembly_example_chromosome_02": "chromosome",
    "assembly_example_plasmid_01": "plasmid",
    "assembly_example_plasmid_02": "plasmid",
    "assembly_example_phage_01": "phage",
    "assembly_example_phage_02": "phage",
    "assembly_example_unclassified_01": "unclassified",
    "assembly_example_unclassified_02": "unclassified",
}

with prediction_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

observed = {row["sequence_id"]: row["prediction"] for row in rows}
if observed != expected:
    raise RuntimeError(f"Bundled four-class output was not reproduced: {observed}")

fields = (
    "sequence_id",
    "expected_output",
    "observed_output",
    "p_chromosome",
    "p_plasmid",
    "p_phage",
    "plasmid_score",
    "abstention_reason",
)
with check_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "sequence_id": row["sequence_id"],
                "expected_output": expected[row["sequence_id"]],
                "observed_output": row["prediction"],
                "p_chromosome": row["p_chromosome"],
                "p_plasmid": row["p_plasmid"],
                "p_phage": row["p_phage"],
                "plasmid_score": row["plasmid_score"],
                "abstention_reason": row["abstention_reason"],
            }
        )

counts = Counter(observed.values())
print("PASS: " + ", ".join(f"{label}={counts[label]}" for label in sorted(counts)))
PY
check_rc=$?
if [ "$check_rc" -ne 0 ]; then
  echo "STOP: bundled four-class reproduction check failed (status $check_rc)." >&2
  exit "$check_rc"
fi

if [ -z "$ANNOTATION_DATABASE" ]; then
  echo
  echo "PASS: four-class prediction example completed."
  echo "Comprehensive annotation was not requested."
  echo "To add it, rerun with ANNOTATION_DATABASE=/path/to/prepared/databases."
  echo "Predictions: $ANALYSIS_DIR/predictions/predictions.tsv"
  echo "Dashboard: $ANALYSIS_DIR/visualization/mobiorigin_dashboard.html"
  echo "Class check: $OUTPUT_DIR/four_class_prediction_check.tsv"
  exit 0
fi

if [ ! -d "$ANNOTATION_DATABASE" ]; then
  echo "STOP: annotation database directory is missing: $ANNOTATION_DATABASE" >&2
  exit 1
fi

if [ -z "$AMRFINDER_DATABASE" ]; then
  AMRFINDER_DATABASE="$($PYTHON_BIN - <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

prefixes = [Path(sys.prefix)]
if Path(sys.prefix).parent.name == "envs":
    prefixes.append(Path(sys.prefix).parent.parent)
amrfinder = shutil.which("amrfinder")
if amrfinder:
    executable_prefix = Path(amrfinder).resolve().parent.parent
    prefixes.append(executable_prefix)
    if executable_prefix.parent.name == "envs":
        prefixes.append(executable_prefix.parent.parent)

for prefix in dict.fromkeys(prefixes):
    root = prefix / "share" / "amrfinderplus" / "data"
    latest = root / "latest"
    if latest.is_dir():
        print(latest)
        raise SystemExit(0)
    if root.is_dir():
        versions = sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        if versions:
            print(versions[0])
            raise SystemExit(0)
PY
)"
fi

if [ -z "$AMRFINDER_DATABASE" ] || [ ! -d "$AMRFINDER_DATABASE" ]; then
  echo "STOP: official AMRFinderPlus database could not be located." >&2
  echo "Set AMRFINDER_DATABASE=/path/to/amrfinderplus/data/latest and retry." >&2
  exit 1
fi

echo "Running comprehensive independent annotation..."
"$MOBIORIGIN_BIN" annotate \
  --input-fasta "$INPUT_FASTA" \
  --output-dir "$ANNOTATION_DIR" \
  --database-dir "$ANNOTATION_DATABASE" \
  --profile comprehensive \
  --predictions-tsv "$ANALYSIS_DIR/predictions/predictions.tsv" \
  --amrfinder-mode official \
  --amrfinder-bin "$(command -v amrfinder)" \
  --amrfinder-database "$AMRFINDER_DATABASE" \
  --threads "$THREADS"
annotation_rc=$?
if [ "$annotation_rc" -ne 0 ]; then
  echo "STOP: comprehensive annotation failed (status $annotation_rc)." >&2
  exit "$annotation_rc"
fi

"$MOBIORIGIN_BIN" visualize \
  --predictions-tsv "$ANALYSIS_DIR/predictions/predictions.tsv" \
  --annotated-results-tsv "$ANNOTATION_DIR/mobiorigin_annotated_results.tsv" \
  --output-dir "$INTEGRATED_DIR"
visualization_rc=$?
if [ "$visualization_rc" -ne 0 ]; then
  echo "STOP: integrated visualization failed (status $visualization_rc)." >&2
  exit "$visualization_rc"
fi

echo
echo "PASS: bundled four-class prediction and comprehensive annotation completed."
echo "Predictions: $ANALYSIS_DIR/predictions/predictions.tsv"
echo "Annotation table: $ANNOTATION_DIR/mobiorigin_annotated_results.tsv"
echo "Annotation report: $ANNOTATION_DIR/mobiorigin_report.html"
echo "Integrated dashboard: $INTEGRATED_DIR/mobiorigin_dashboard.html"
echo "Interpretation: software demonstration only; not accuracy or prevalence evidence."

exit 0
