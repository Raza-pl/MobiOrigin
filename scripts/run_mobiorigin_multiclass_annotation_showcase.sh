#!/usr/bin/env bash

# Build and run a compact four-class MobiOrigin showcase from the completed W1
# analysis. The records are outcome-selected for demonstration and must not be
# used to estimate classification accuracy.

set +e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MOBIORIGIN_BIN="${MOBIORIGIN_BIN:-$(command -v mobiorigin)}"
W1_FASTA="${W1_FASTA:-$ROOT/data/test/W1.contigs.fa.gz}"
FROZEN_PREDICTIONS="${FROZEN_PREDICTIONS:-$ROOT/results/mobiorigin_full_w1_20260824/prediction/predictions.tsv}"
FROZEN_ANNOTATIONS="${FROZEN_ANNOTATIONS:-$ROOT/results/mobiorigin_full_w1_20260824/annotation/mobiorigin_annotated_results.tsv}"
MARKER_DATABASE="${MOBIORIGIN_DATABASE_DIR:-$HOME/.local/share/mobiorigin/marker_databases}"
ANNOTATION_DATABASE="${ANNOTATION_DATABASE:-$ROOT/data/databases}"
AMRFINDER_DATABASE="${AMRFINDER_DATABASE:-}"
THREADS="${THREADS:-8}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/mobiorigin_multiclass_annotation_showcase_$RUN_STAMP}"
INPUT_DIR="$OUTPUT_DIR/input"
SHOWCASE_FASTA="$INPUT_DIR/w1_four_class_showcase.fasta"
SELECTION_TABLE="$INPUT_DIR/selection_context.tsv"
ANALYSIS_DIR="$OUTPUT_DIR/analysis"
ANNOTATION_DIR="$OUTPUT_DIR/annotation"
INTEGRATED_VISUALIZATION_DIR="$OUTPUT_DIR/integrated_visualization"

typeset -i overall_rc=0

echo "MobiOrigin four-class annotation showcase"
echo "Source assembly: $W1_FASTA"
echo "Output: $OUTPUT_DIR"
echo "Design: two outcome- and evidence-selected W1 contigs per MobiOrigin output class"
echo "Boundary: reproducibility/annotation showcase only; not an accuracy test"

if [[ -z "$PYTHON_BIN" ]] || [[ ! -x "$PYTHON_BIN" ]]; then
  echo "STOP: activate the MobiOrigin environment first; Python was not found."
  exit 1
fi
if [[ -z "$MOBIORIGIN_BIN" ]] || [[ ! -x "$MOBIORIGIN_BIN" ]]; then
  echo "STOP: mobiorigin was not found in the active environment."
  exit 1
fi
for required in "$W1_FASTA" "$FROZEN_PREDICTIONS" "$FROZEN_ANNOTATIONS"; do
  if [[ ! -s "$required" ]]; then
    echo "STOP: required file is missing or empty: $required"
    exit 1
  fi
done

if [[ -z "$AMRFINDER_DATABASE" ]]; then
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

seen: set[Path] = set()
for prefix in prefixes:
    if prefix in seen:
        continue
    seen.add(prefix)
    data_root = prefix / "share" / "amrfinderplus" / "data"
    latest = data_root / "latest"
    if latest.is_dir():
        print(latest)
        raise SystemExit(0)
    if data_root.is_dir():
        versions = sorted(
            (path for path in data_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        if versions:
            print(versions[0])
            raise SystemExit(0)
PY
)"
fi

if [[ -z "$AMRFINDER_DATABASE" ]]; then
  echo "STOP: the official AMRFinderPlus database could not be located."
  echo "Set AMRFINDER_DATABASE=/path/to/amrfinderplus/data/latest and retry."
  exit 1
fi

for required_dir in "$MARKER_DATABASE" "$ANNOTATION_DATABASE" "$AMRFINDER_DATABASE"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "STOP: required database directory is missing: $required_dir"
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "STOP: fresh output directory required: $OUTPUT_DIR"
  exit 1
fi

mkdir -p "$INPUT_DIR"

"$PYTHON_BIN" - "$FROZEN_PREDICTIONS" "$FROZEN_ANNOTATIONS" "$W1_FASTA" "$SHOWCASE_FASTA" "$SELECTION_TABLE" <<'PY'
from __future__ import annotations

import csv
import gzip
import hashlib
import sys
from collections import Counter
from pathlib import Path

predictions_path = Path(sys.argv[1])
annotations_path = Path(sys.argv[2])
fasta_path = Path(sys.argv[3])
output_fasta = Path(sys.argv[4])
selection_path = Path(sys.argv[5])
classes = ("chromosome", "plasmid", "phage", "unclassified")

with predictions_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
with annotations_path.open(encoding="utf-8", newline="") as handle:
    annotations = {row["sequence_id"]: row for row in csv.DictReader(handle, delimiter="\t")}

if {row["sequence_id"] for row in rows} != annotations.keys():
    raise RuntimeError("Frozen W1 prediction and annotation identifiers are not identical")

selected = []
tier_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
evidence_fields = (
    "consensus_arg_orfs",
    "virulence_hits",
    "mge_hits",
    "stress_biocide_metal_hits",
    "mobility_marker_hits",
)
for label in classes:
    eligible = [
        row
        for row in rows
        if row["prediction"] == label and 5_000 <= int(row["length_bp"]) <= 100_000
    ]
    eligible.sort(
        key=lambda row: (
            tier_rank.get(annotations[row["sequence_id"]]["evidence_priority_tier"], 9),
            -sum(int(annotations[row["sequence_id"]][field]) for field in evidence_fields),
            abs(int(row["length_bp"]) - 20_000),
            row["sequence_id"],
        )
    )
    if len(eligible) < 2:
        raise RuntimeError(f"Fewer than two supported-length W1 records are available for {label}")
    selected.extend(eligible[:2])

wanted = {row["sequence_id"] for row in selected}
sequences: dict[str, str] = {}
opener = gzip.open if fasta_path.suffix == ".gz" else open
with opener(fasta_path, "rt", encoding="utf-8") as handle:
    identifier = None
    chunks: list[str] = []
    for raw_line in handle:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier in wanted:
                sequences[identifier] = "".join(chunks).upper()
            identifier = line[1:].split()[0]
            chunks = []
        elif identifier in wanted:
            chunks.append(line)
    if identifier in wanted:
        sequences[identifier] = "".join(chunks).upper()

missing = sorted(wanted - sequences.keys())
if missing:
    raise RuntimeError(f"Selected W1 records were not found in the FASTA: {missing}")

with output_fasta.open("w", encoding="utf-8", newline="\n") as fasta_handle, selection_path.open(
    "w", encoding="utf-8", newline=""
) as table_handle:
    fields = (
        "sequence_id",
        "length_bp",
        "previous_mobiorigin_output",
        "previous_evidence_tier",
        "previous_arg_genes",
        "previous_mobility_class",
        "selection_role",
        "sequence_sha256",
    )
    writer = csv.DictWriter(table_handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in selected:
        identifier = row["sequence_id"]
        sequence = sequences[identifier]
        if len(sequence) != int(row["length_bp"]):
            raise RuntimeError(f"Length mismatch for {identifier}")
        fasta_handle.write(f">{identifier}\n")
        for offset in range(0, len(sequence), 80):
            fasta_handle.write(sequence[offset : offset + 80] + "\n")
        writer.writerow(
            {
                "sequence_id": identifier,
                "length_bp": len(sequence),
                "previous_mobiorigin_output": row["prediction"],
                "previous_evidence_tier": annotations[identifier]["evidence_priority_tier"],
                "previous_arg_genes": annotations[identifier]["arg_genes"],
                "previous_mobility_class": annotations[identifier]["mobility_class"],
                "selection_role": "outcome_and_evidence_selected_reproducibility_showcase",
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            }
        )

counts = Counter(row["prediction"] for row in selected)
if counts != Counter({label: 2 for label in classes}):
    raise RuntimeError(f"Four-class selection accounting failed: {counts}")
print(f"Prepared {len(selected)} records: " + ", ".join(f"{key}={counts[key]}" for key in classes))
PY
prepare_rc=$?
if [[ $prepare_rc -ne 0 ]]; then
  echo "STOP: showcase FASTA preparation failed (status $prepare_rc)."
  exit $prepare_rc
fi

echo "Running MobiOrigin prediction and basic visualization..."
"$MOBIORIGIN_BIN" run \
  --input-fasta "$SHOWCASE_FASTA" \
  --output-dir "$ANALYSIS_DIR" \
  --database-dir "$MARKER_DATABASE" \
  --threads "$THREADS"
prediction_rc=$?
if [[ $prediction_rc -ne 0 ]]; then
  echo "STOP: MobiOrigin prediction failed (status $prediction_rc)."
  exit $prediction_rc
fi

"$PYTHON_BIN" - "$SELECTION_TABLE" "$ANALYSIS_DIR/predictions/predictions.tsv" "$OUTPUT_DIR/four_class_reproduction.tsv" <<'PY'
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

selection_path, prediction_path, output_path = map(Path, sys.argv[1:])
with selection_path.open(encoding="utf-8", newline="") as handle:
    selected = {row["sequence_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
with prediction_path.open(encoding="utf-8", newline="") as handle:
    observed = list(csv.DictReader(handle, delimiter="\t"))

fields = (
    "sequence_id",
    "length_bp",
    "previous_mobiorigin_output",
    "reproduced_prediction",
    "prediction_reproduced",
    "p_chromosome",
    "p_plasmid",
    "p_phage",
    "plasmid_score",
    "abstention_reason",
)
with output_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in observed:
        prior = selected[row["sequence_id"]]
        writer.writerow(
            {
                "sequence_id": row["sequence_id"],
                "length_bp": row["length_bp"],
                "previous_mobiorigin_output": prior["previous_mobiorigin_output"],
                "reproduced_prediction": row["prediction"],
                "prediction_reproduced": str(
                    prior["previous_mobiorigin_output"] == row["prediction"]
                ).lower(),
                "p_chromosome": row["p_chromosome"],
                "p_plasmid": row["p_plasmid"],
                "p_phage": row["p_phage"],
                "plasmid_score": row["plasmid_score"],
                "abstention_reason": row["abstention_reason"],
            }
        )

counts = Counter(row["prediction"] for row in observed)
expected = Counter({label: 2 for label in ("chromosome", "plasmid", "phage", "unclassified")})
if counts != expected:
    raise RuntimeError(f"The rerun did not reproduce two records per output class: {counts}")
if any(selected[row["sequence_id"]]["previous_mobiorigin_output"] != row["prediction"] for row in observed):
    raise RuntimeError("At least one selected record did not reproduce its frozen output label")
print("PASS: two predictions from each of the four MobiOrigin output classes were reproduced.")
PY
reproduction_rc=$?
if [[ $reproduction_rc -ne 0 ]]; then
  echo "STOP: four-class reproduction check failed (status $reproduction_rc)."
  exit $reproduction_rc
fi

echo "Running comprehensive annotation on the same eight sequences..."
"$MOBIORIGIN_BIN" annotate \
  --input-fasta "$SHOWCASE_FASTA" \
  --output-dir "$ANNOTATION_DIR" \
  --database-dir "$ANNOTATION_DATABASE" \
  --threads "$THREADS" \
  --profile comprehensive \
  --amrfinder-mode official \
  --amrfinder-bin "$(command -v amrfinder)" \
  --amrfinder-database "$AMRFINDER_DATABASE" \
  --predictions-tsv "$ANALYSIS_DIR/predictions/predictions.tsv"
annotation_rc=$?
if [[ $annotation_rc -ne 0 ]]; then
  echo "STOP: comprehensive annotation failed (status $annotation_rc)."
  exit $annotation_rc
fi

"$MOBIORIGIN_BIN" visualize \
  --predictions-tsv "$ANALYSIS_DIR/predictions/predictions.tsv" \
  --annotated-results-tsv "$ANNOTATION_DIR/mobiorigin_annotated_results.tsv" \
  --output-dir "$INTEGRATED_VISUALIZATION_DIR"
visualization_rc=$?
if [[ $visualization_rc -ne 0 ]]; then
  echo "STOP: integrated visualization failed (status $visualization_rc)."
  exit $visualization_rc
fi

"$PYTHON_BIN" - "$SELECTION_TABLE" "$ANNOTATION_DIR/mobiorigin_annotated_results.tsv" "$OUTPUT_DIR/showcase_summary.tsv" "$OUTPUT_DIR/showcase_result.json" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

selection_path, annotation_path, output_table, output_json = map(Path, sys.argv[1:])
with selection_path.open(encoding="utf-8", newline="") as handle:
    context = {row["sequence_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
with annotation_path.open(encoding="utf-8", newline="") as handle:
    annotated = list(csv.DictReader(handle, delimiter="\t"))

fields = (
    "sequence_id",
    "length_bp",
    "prediction",
    "p_chromosome",
    "p_plasmid",
    "p_phage",
    "abstention_reason",
    "consensus_arg_orfs",
    "arg_genes",
    "virulence_hits",
    "mge_hits",
    "stress_biocide_metal_hits",
    "mobility_marker_hits",
    "mobility_class",
    "evidence_priority_tier",
    "priority_rationale",
    "selection_role",
)
with output_table.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in annotated:
        record = {field: row.get(field, "") for field in fields}
        record["selection_role"] = context[row["sequence_id"]]["selection_role"]
        writer.writerow(record)

summary = {
    "status": "PASS",
    "records": len(annotated),
    "prediction_counts": dict(sorted(Counter(row["prediction"] for row in annotated).items())),
    "evidence_tier_counts": dict(
        sorted(Counter(row["evidence_priority_tier"] for row in annotated).items())
    ),
    "arg_positive_records": sum(int(row["consensus_arg_orfs"]) > 0 for row in annotated),
    "virulence_positive_records": sum(int(row["virulence_hits"]) > 0 for row in annotated),
    "mge_positive_records": sum(int(row["mge_hits"]) > 0 for row in annotated),
    "mobility_positive_records": sum(int(row["mobility_marker_hits"]) > 0 for row in annotated),
    "selection_design": "two outcome- and evidence-selected W1 records per frozen MobiOrigin output class",
    "interpretation_boundary": "reproducibility and annotation showcase; not independent validation or accuracy evidence",
}
canonical = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
summary["summary_sha256"] = hashlib.sha256(canonical).hexdigest()
output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
summary_rc=$?
if [[ $summary_rc -ne 0 ]]; then
  echo "STOP: showcase summary generation failed (status $summary_rc)."
  exit $summary_rc
fi

echo
echo "===== FOUR-CLASS PREDICTION AND ANNOTATION TABLE ====="
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$OUTPUT_DIR/showcase_summary.tsv"
else
  sed -n '1,20p' "$OUTPUT_DIR/showcase_summary.tsv"
fi

echo
echo "PASS: the eight-sequence four-class prediction and annotation showcase is complete."
echo "Combined table: $OUTPUT_DIR/showcase_summary.tsv"
echo "Annotation report: $ANNOTATION_DIR/mobiorigin_report.html"
echo "Integrated dashboard: $INTEGRATED_VISUALIZATION_DIR/mobiorigin_dashboard.html"
echo "Editable SVG: $INTEGRATED_VISUALIZATION_DIR/mobiorigin_summary.svg"
echo "Result metadata: $OUTPUT_DIR/showcase_result.json"
echo "Interpretation: outcome-selected demonstration only; not classification accuracy evidence."

if [[ "$(uname -s)" == "Darwin" ]]; then
  open "$ANNOTATION_DIR/mobiorigin_report.html" 2>/dev/null || true
fi

exit $overall_rc
