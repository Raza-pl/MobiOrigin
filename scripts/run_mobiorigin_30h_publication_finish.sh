#!/bin/zsh

# Finish the real-assembly MobiOrigin annotation and operational benchmark
# within a 30-hour workstation window. This script intentionally does not use
# `set -e`: every stage records its own status and later independent stages may
# continue after a failure.

set +e
unsetopt ERR_EXIT 2>/dev/null || true

ROOT="${MOBIORIGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="/Users/shahbazraza/miniconda3/envs/plasflow2/bin/python"
PLASCLASS_ENV="/Users/shahbazraza/miniconda3/envs/plasclass_nar"
RUNNER_ROOT="/private/tmp/mobiorigin-secondary-comparator-runners"
RESULT_ROOT="$ROOT/results/mobiorigin_30h_publication_finish"
INPUT_ROOT="$RESULT_ROOT/inputs"
OUTPUT_ROOT="$RESULT_ROOT/outputs"
LOG_ROOT="$RESULT_ROOT/logs"
TIMING="$RESULT_ROOT/stage_timing.tsv"

MOB_DATABASE="$ROOT/evaluation/release_audit/mobiorigin_external_validation_20260817/026_mobiorigin_external_prediction_preflight/runtime_database"
ANNOTATION_DATABASE="$ROOT/data/databases"
AMRFINDER_DATABASE="/Users/shahbazraza/miniconda3/share/amrfinderplus/data/2026-08-07.1"
GENOMAD_DATABASE="$ROOT/data/databases/genomad_db"
GENOMAD_CHECKSUMS="$ROOT/evaluation/release_audit/nar_benchmark_20260727/19_genomad_preflight/database_sha256.tsv"
GENOMAD_CONTRACT="$ROOT/evaluation/release_audit/nar_benchmark_20260727/29_genomad_resource_contract_correction/genomad_runner_contract_resource_corrected.json"
PLASME_DATABASE="$ROOT/evaluation/release_audit/nar_benchmark_20260727/120_plasme_runtime_database_copy/runtime_database/DB"
PLATON_DATABASE="$ROOT/evaluation/release_audit/nar_benchmark_20260727/163_platon_database_extraction/database/db"
BENCHMARK_PROTOCOL="$ROOT/evaluation/release_audit/nar_benchmark_20260727/03_protocol_freeze/08d_protocol_freeze_report.txt"
DOCKER_CONTEXT="colima-plasflow1-nar"

PLASCLASS_CONTRACT="$RUNNER_ROOT/scripts/benchmark/contracts/plasclass_runner_contract_v1_1.json"
PLASFLOW_V1_CONTRACT="$RUNNER_ROOT/scripts/benchmark/contracts/plasflow_v1_runner_contract_v1.json"
PLASME_CONTRACT="$RUNNER_ROOT/scripts/benchmark/contracts/plasme_runner_contract_v1.json"
PLATON_CONTRACT="$RUNNER_ROOT/scripts/benchmark/contracts/platon_runner_contract_v1.json"
PLASFLOW_V1_CORRECTION="$ROOT/scripts/benchmark/finalize_plasflow_v1_native_vote.py"

mkdir -p "$INPUT_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"

if [[ ! -f "$TIMING" ]]; then
  printf 'dataset\tstage\tstatus\telapsed_seconds\treused\n' > "$TIMING"
fi

typeset -i OVERALL_STATUS=0

record_stage() {
  local dataset="$1"
  local stage="$2"
  local stage_rc="$3"
  local elapsed="$4"
  local reused="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$stage" "$stage_rc" "$elapsed" "$reused" >> "$TIMING"
}

run_stage() {
  local dataset="$1"
  local stage="$2"
  local marker="$3"
  local output_directory="$4"
  shift 4

  if [[ -s "$marker" ]]; then
    echo "Reusing completed $dataset/$stage"
    record_stage "$dataset" "$stage" 0 0 true
    return 0
  fi

  if [[ -d "$output_directory" ]] && [[ -n "$(find "$output_directory" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "STOP: incomplete non-empty output retained for $dataset/$stage: $output_directory"
    echo "No files were deleted. Inspect this directory before retrying."
    record_stage "$dataset" "$stage" 90 0 false
    OVERALL_STATUS=1
    return 90
  fi

  mkdir -p "$(dirname "$output_directory")"
  local started=$SECONDS
  echo "Starting $dataset/$stage"
  "$@" >"$LOG_ROOT/${dataset}_${stage}.stdout.log" 2>"$LOG_ROOT/${dataset}_${stage}.stderr.log"
  local stage_rc=$?
  local elapsed=$((SECONDS - started))
  record_stage "$dataset" "$stage" "$stage_rc" "$elapsed" false

  if [[ $stage_rc -ne 0 ]]; then
    echo "FAILED: $dataset/$stage (status $stage_rc; ${elapsed}s)"
    tail -30 "$LOG_ROOT/${dataset}_${stage}.stderr.log" 2>/dev/null || true
    OVERALL_STATUS=1
    return $stage_rc
  fi

  if [[ ! -s "$marker" ]]; then
    echo "FAILED: $dataset/$stage returned zero but its completion artifact is missing: $marker"
    OVERALL_STATUS=1
    return 91
  fi

  echo "Completed $dataset/$stage in ${elapsed}s"
  return 0
}

build_subset() {
  local source_fasta="$1"
  local subset_fasta="$2"
  local manifest="$3"
  local dataset="$4"

  if [[ -s "$subset_fasta" ]] && [[ -s "$manifest" ]]; then
    echo "Reusing deterministic subset for $dataset"
    return 0
  fi

  if [[ -e "$subset_fasta" ]] || [[ -e "$manifest" ]]; then
    echo "STOP: partial subset artifacts exist for $dataset; no overwrite was attempted."
    OVERALL_STATUS=1
    return 92
  fi

  "$PY" - "$source_fasta" "$subset_fasta" "$manifest" "$dataset" <<'PY'
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
manifest = Path(sys.argv[3])
dataset = sys.argv[4]
target_bases_per_bin = 4_000_000
maximum_records_per_bin = 800


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def records(path: Path):
    with open_text(path) as handle:
        header = None
        pieces = []
        index = -1
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith(">"):
                if header is not None:
                    yield index, header, "".join(pieces).upper()
                index += 1
                header = line[1:]
                pieces = []
            elif header is not None:
                pieces.append("".join(line.split()))
        if header is not None:
            yield index, header, "".join(pieces).upper()


def length_bin(length: int) -> str | None:
    if length < 1_000:
        return None
    if length < 2_000:
        return "01_1k_to_lt2k"
    if length < 5_000:
        return "02_2k_to_lt5k"
    if length < 10_000:
        return "03_5k_to_lt10k"
    if length < 50_000:
        return "04_10k_to_lt50k"
    return "05_ge50k"


candidates: dict[str, list[tuple[str, int, int, str]]] = {}
seen_ids: set[str] = set()
for index, header, sequence in records(source):
    canonical_id = header.split()[0]
    if not canonical_id:
        raise RuntimeError(f"Empty FASTA identifier at record {index + 1}")
    if canonical_id in seen_ids:
        raise RuntimeError(f"Duplicate FASTA identifier: {canonical_id}")
    seen_ids.add(canonical_id)
    group = length_bin(len(sequence))
    if group is None:
        continue
    rank = hashlib.sha256(
        f"mobiorigin-real-v1\0{dataset}\0{index}\0{header}".encode()
    ).hexdigest()
    candidates.setdefault(group, []).append((rank, index, len(sequence), header))

selected: dict[int, tuple[str, str, int, str]] = {}
for group in sorted(candidates):
    accumulated = 0
    count = 0
    for rank, index, length, header in sorted(candidates[group]):
        if count >= maximum_records_per_bin or accumulated >= target_bases_per_bin:
            break
        selected[index] = (group, rank, length, header)
        accumulated += length
        count += 1

if not selected:
    raise RuntimeError("No sequences at least 1,000 bp were available")

destination.parent.mkdir(parents=True, exist_ok=True)
rows = []
selected_order = 0
with destination.open("x") as fasta_handle:
    for index, header, sequence in records(source):
        if index not in selected:
            continue
        group, rank, expected_length, expected_header = selected[index]
        if len(sequence) != expected_length or header != expected_header:
            raise RuntimeError("FASTA changed between deterministic selection passes")
        selected_order += 1
        opaque_id = "mobreal_" + dataset + "_" + hashlib.sha256(
            f"{dataset}\0{index}\0{header}".encode()
        ).hexdigest()[:24]
        fasta_handle.write(f">{opaque_id}\n")
        for offset in range(0, len(sequence), 80):
            fasta_handle.write(sequence[offset : offset + 80] + "\n")
        rows.append(
            {
                "selected_order": selected_order,
                "opaque_id": opaque_id,
                "source_record_index": index,
                "source_header": header,
                "length_bp": len(sequence),
                "length_bin": group,
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "selection_rank_sha256": rank,
            }
        )

with manifest.open("x") as handle:
    fields = list(rows[0])
    handle.write("\t".join(fields) + "\n")
    for row in rows:
        handle.write("\t".join(str(row[field]) for field in fields) + "\n")

fasta_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
summary = {
    "schema_version": "mobiorigin-real-assembly-operational-subset-v1",
    "dataset": dataset,
    "source": str(source.resolve()),
    "selection": "SHA-256-ranked records within five fixed length bins",
    "target_bases_per_bin": target_bases_per_bin,
    "maximum_records_per_bin": maximum_records_per_bin,
    "records": len(rows),
    "bases": sum(row["length_bp"] for row in rows),
    "fasta_sha256": fasta_sha256,
}
destination.with_suffix(".subset.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, sort_keys=True))
PY
  local subset_rc=$?
  if [[ $subset_rc -ne 0 ]]; then
    OVERALL_STATUS=1
  fi
  return $subset_rc
}

run_dataset() {
  local dataset="$1"
  local source_fasta="$2"
  local subset="$INPUT_ROOT/${dataset}.fasta"
  local manifest="$INPUT_ROOT/${dataset}_manifest.tsv"
  local base="$OUTPUT_ROOT/$dataset"

  build_subset "$source_fasta" "$subset" "$manifest" "$dataset"
  if [[ $? -ne 0 ]]; then
    return 1
  fi

  run_stage "$dataset" mobiorigin_predict \
    "$base/mobiorigin/predictions.tsv" "$base/mobiorigin" \
    "$PY" -m mobiorigin predict \
      --input-fasta "$subset" \
      --output-dir "$base/mobiorigin" \
      --database-dir "$MOB_DATABASE" \
      --threads 8

  if [[ -s "$base/mobiorigin/predictions.tsv" ]]; then
    run_stage "$dataset" comprehensive_annotation \
      "$base/annotation/publication_summary.json" "$base/annotation" \
      "$PY" -m mobiorigin annotate \
        --input-fasta "$subset" \
        --output-dir "$base/annotation" \
        --database-dir "$ANNOTATION_DATABASE" \
        --threads 8 \
        --profile comprehensive \
        --amrfinder-mode official \
        --amrfinder-bin "/Users/shahbazraza/miniconda3/envs/plasflow2/bin/amrfinder" \
        --amrfinder-database "$AMRFINDER_DATABASE" \
        --predictions-tsv "$base/mobiorigin/predictions.tsv"
  else
    echo "Skipping annotation for $dataset because MobiOrigin predictions are incomplete."
    OVERALL_STATUS=1
  fi

  run_stage "$dataset" genomad \
    "$base/genomad/standardized_predictions.tsv" "$base/genomad" \
    "$PY" "$RUNNER_ROOT/scripts/benchmark/runners/genomad.py" \
      --input-fasta "$subset" \
      --output-dir "$base/genomad" \
      --environment-prefix "/Users/shahbazraza/miniconda3/envs/plasflow2" \
      --database-directory "$GENOMAD_DATABASE" \
      --database-checksums "$GENOMAD_CHECKSUMS" \
      --scope-contract "$GENOMAD_CONTRACT" \
      --threads 8 \
      --timeout-seconds 43200

  run_stage "$dataset" plasclass \
    "$base/plasclass/standardized_predictions.tsv" "$base/plasclass" \
    "$PY" "$RUNNER_ROOT/scripts/benchmark/runners/plasclass.py" \
      --input-fasta "$subset" \
      --output-dir "$base/plasclass" \
      --environment-prefix "$PLASCLASS_ENV" \
      --scope-contract "$PLASCLASS_CONTRACT" \
      --cohort-role development

  if [[ -s "$base/plasflow_v1/raw_predictions.tsv" ]] && \
     [[ -s "$base/plasflow_v1/runner_provenance.json" ]] && \
     [[ ! -s "$base/plasflow_v1/standardized_predictions.tsv" ]]; then
    run_stage "$dataset" plasflow_v1 \
      "$base/plasflow_v1/standardized_predictions.tsv" \
      "$base/plasflow_v1/native_vote_correction" \
      "$PY" "$PLASFLOW_V1_CORRECTION" \
        --input-fasta "$subset" \
        --output-dir "$base/plasflow_v1" \
        --frozen-adapter "$RUNNER_ROOT/scripts/benchmark/adapters/plasflow_v1.py"
  else
    run_stage "$dataset" plasflow_v1 \
      "$base/plasflow_v1/standardized_predictions.tsv" "$base/plasflow_v1" \
      "$PY" "$RUNNER_ROOT/scripts/benchmark/runners/plasflow_v1.py" \
        --input-fasta "$subset" \
        --output-dir "$base/plasflow_v1" \
        --docker-context "$DOCKER_CONTEXT" \
        --scope-contract "$PLASFLOW_V1_CONTRACT" \
        --cohort-role development
  fi

  run_stage "$dataset" plasme \
    "$base/plasme/standardized_predictions.tsv" "$base/plasme" \
    "$PY" "$RUNNER_ROOT/scripts/benchmark/runners/plasme.py" \
      --input-fasta "$subset" \
      --output-dir "$base/plasme" \
      --database-directory "$PLASME_DATABASE" \
      --docker-context "$DOCKER_CONTEXT" \
      --scope-contract "$BENCHMARK_PROTOCOL" \
      --cohort-role development

  run_stage "$dataset" platon \
    "$base/platon/standardized_predictions.tsv" "$base/platon" \
    "$PY" "$RUNNER_ROOT/scripts/benchmark/runners/platon.py" \
      --input-fasta "$subset" \
      --output-dir "$base/platon" \
      --database-directory "$PLATON_DATABASE" \
      --docker-context "$DOCKER_CONTEXT" \
      --scope-contract "$BENCHMARK_PROTOCOL" \
      --cohort-role development
}

echo "MobiOrigin 30-hour publication finish"
echo "Output: $RESULT_ROOT"
echo "Policy: two deterministic real-assembly subsets; no ground-truth accuracy claims"

required=(
  "$PY"
  "$ROOT/data/test/GCA_054405655.1_ASM5440565v1_genomic.fna"
  "$ROOT/data/test/assembly_b.contigs.fa.gz"
  "$MOB_DATABASE"
  "$ANNOTATION_DATABASE"
  "$AMRFINDER_DATABASE"
  "$GENOMAD_DATABASE"
  "$GENOMAD_CHECKSUMS"
  "$GENOMAD_CONTRACT"
  "$PLASCLASS_CONTRACT"
  "$PLASFLOW_V1_CONTRACT"
  "$PLASME_CONTRACT"
  "$PLATON_CONTRACT"
  "$BENCHMARK_PROTOCOL"
  "$PLASFLOW_V1_CORRECTION"
  "$PLASME_DATABASE"
  "$PLATON_DATABASE"
)

for required_path in "${required[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "STOP: required path is missing: $required_path"
    exit 2
  fi
done

if command -v colima >/dev/null 2>&1; then
  colima status --profile plasflow1-nar >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    echo "Starting the retained comparator container environment..."
    colima start --profile plasflow1-nar
    if [[ $? -ne 0 ]]; then
      echo "STOP: the comparator container environment could not start."
      exit 3
    fi
  fi
else
  echo "STOP: colima is unavailable."
  exit 3
fi

run_dataset "gca054405655" "$ROOT/data/test/GCA_054405655.1_ASM5440565v1_genomic.fna"
run_dataset "assembly_b" "$ROOT/data/test/assembly_b.contigs.fa.gz"

echo "Building publication-oriented operational summaries..."
"$PY" - "$RESULT_ROOT" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
datasets = ["gca054405655", "assembly_b"]
tools = ["mobiorigin", "genomad", "plasclass", "plasflow_v1", "plasme", "platon"]


def read_predictions(path: Path, tool: str):
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    identifier_field = "sequence_id" if tool == "mobiorigin" else "contig_id"
    label_field = "prediction" if tool == "mobiorigin" else "predicted_label"
    output = []
    for row in rows:
        label = row[label_field]
        if label == "plasmid":
            binary = "plasmid"
        elif label == "unclassified":
            binary = "unclassified"
        else:
            binary = "non-plasmid"
        output.append((row[identifier_field], binary))
    return output


timing = {}
timing_path = root / "stage_timing.tsv"
if timing_path.is_file():
    with timing_path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["status"] == "0" and row["reused"] == "false":
                timing[(row["dataset"], row["stage"])] = int(row["elapsed_seconds"])

summary_rows = []
agreement_rows = []
for dataset in datasets:
    input_summary = json.loads((root / "inputs" / f"{dataset}.subset.json").read_text())
    predictions = {}
    for tool in tools:
        if tool == "mobiorigin":
            path = root / "outputs" / dataset / tool / "predictions.tsv"
            timing_stage = "mobiorigin_predict"
        else:
            path = root / "outputs" / dataset / tool / "standardized_predictions.tsv"
            timing_stage = tool
        if not path.is_file():
            continue
        values = read_predictions(path, tool)
        predictions[tool] = values
        counts = Counter(label for _, label in values)
        summary_rows.append(
            {
                "dataset": dataset,
                "tool": tool,
                "records": len(values),
                "input_bases": input_summary["bases"],
                "plasmid_calls": counts["plasmid"],
                "non_plasmid_calls": counts["non-plasmid"],
                "unclassified_calls": counts["unclassified"],
                "prediction_coverage": (len(values) - counts["unclassified"]) / len(values),
                "plasmid_call_fraction": counts["plasmid"] / len(values),
                "wallclock_seconds_this_script": timing.get((dataset, timing_stage), ""),
                "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    reference = predictions.get("mobiorigin")
    if reference is None:
        continue
    reference_ids = [identifier for identifier, _ in reference]
    reference_labels = [label for _, label in reference]
    for tool, values in predictions.items():
        if tool == "mobiorigin":
            continue
        ids = [identifier for identifier, _ in values]
        if ids != reference_ids:
            raise RuntimeError(f"Prediction order mismatch for {dataset}/{tool}")
        labels = [label for _, label in values]
        exact = sum(left == right for left, right in zip(reference_labels, labels))
        callable_pairs = [
            (left, right)
            for left, right in zip(reference_labels, labels)
            if left != "unclassified" and right != "unclassified"
        ]
        called_agreement = (
            sum(left == right for left, right in callable_pairs) / len(callable_pairs)
            if callable_pairs
            else 0.0
        )
        agreement_rows.append(
            {
                "dataset": dataset,
                "comparison": f"mobiorigin_vs_{tool}",
                "records": len(labels),
                "exact_binary_agreement_including_unclassified": exact / len(labels),
                "jointly_called_records": len(callable_pairs),
                "agreement_among_jointly_called": called_agreement,
                "analysis_role": "label-free operational agreement; not accuracy",
            }
        )


def write_table(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


write_table(root / "operational_benchmark_summary.tsv", summary_rows)
write_table(root / "pairwise_operational_agreement.tsv", agreement_rows)

risk_rows = []
for dataset in datasets:
    result = root / "outputs" / dataset / "annotation" / "mobiorigin_annotated_results.tsv"
    if not result.is_file():
        continue
    with result.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    tier_counts = Counter(row["evidence_priority_tier"] for row in rows)
    for tier in ["A", "B", "C", "D", "E"]:
        risk_rows.append(
            {
                "dataset": dataset,
                "evidence_priority_tier": tier,
                "records": tier_counts[tier],
                "fraction": tier_counts[tier] / len(rows),
                "interpretation": "evidence-priority tier; not a clinical risk score",
            }
        )
write_table(root / "biological_evidence_tier_summary.tsv", risk_rows)

artifacts = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "SHA256SUMS.txt":
        artifacts.append((hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(root)))
with (root / "SHA256SUMS.txt").open("w") as handle:
    for digest, relative_path in artifacts:
        handle.write(f"{digest}  {relative_path}\n")

complete_tools = len(summary_rows)
print("MobiOrigin 30-hour publication finish summary")
print(f"Completed dataset/tool prediction sets: {complete_tools}/12")
print(f"Operational comparisons: {len(agreement_rows)}/10")
print(f"Checksum-tracked artifacts: {len(artifacts)}")
print(f"Output: {root}")
PY
summary_status=$?
if [[ $summary_status -ne 0 ]]; then
  OVERALL_STATUS=1
fi

available_kib=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
available_gib=$((available_kib / 1024 / 1024))

echo
echo "MobiOrigin 30-hour workflow exit status: $OVERALL_STATUS"
echo "Available storage GiB: $available_gib"
echo "Results: $RESULT_ROOT"
echo "Primary reports:"
echo "  $OUTPUT_ROOT/gca054405655/annotation/mobiorigin_report.html"
echo "  $OUTPUT_ROOT/assembly_b/annotation/mobiorigin_report.html"
echo "  $RESULT_ROOT/operational_benchmark_summary.tsv"
echo "  $RESULT_ROOT/pairwise_operational_agreement.tsv"
echo "  $RESULT_ROOT/biological_evidence_tier_summary.tsv"
echo
echo "Interpretation boundary: these real assemblies have no frozen ground truth."
echo "The benchmark supports runtime, call-rate, coverage, and agreement claims only."
echo "Evidence tiers A-E prioritize biological follow-up; they are not clinical risk scores."

exit $OVERALL_STATUS
