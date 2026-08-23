#!/usr/bin/env python3
"""Finalize a successful PlasFlow v1 run rejected by an invalid adapter check.

Official PlasFlow v1 obtains its native class ``id`` by majority vote across
three classifiers, while the probability columns are arithmetic means.  The
native vote therefore need not equal the mean-probability argmax.  This helper
reuses the untouched official output, reproduces the official final-label
policy, and invokes the otherwise unchanged frozen benchmark adapter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

EXPECTED_ADAPTER_SHA256 = (
    "431b92f4d7150220cf42a14832ed5529ad930ad16fe4f095ef6a973906fb4236"
)
EXPECTED_ADAPTER_CONTRACT_SHA256 = (
    "7e24ac29aedb6f0b106e421302cb1ee147f076c96c24d299f9b9d55d1b42c3bf"
)
EXPECTED_CONTAINER_DIGEST = (
    "sha256:e69acee3233010dbf5a5245620252bf5b9bde930ad5546473ec496992995a7da"
)
FAILURE_PREFIX = (
    "Output or adapter failure: ValueError: Native class id is inconsistent "
    "with the probability argmax"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_adapter(path: Path) -> Any:
    if sha256_file(path) != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("Frozen PlasFlow-v1 adapter identity changed")
    specification = importlib.util.spec_from_file_location(
        "mobiorigin_frozen_plasflow_v1_adapter", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Unable to load the frozen PlasFlow-v1 adapter")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    if module.CONTRACT_SHA256 != EXPECTED_ADAPTER_CONTRACT_SHA256:
        raise RuntimeError("Frozen PlasFlow-v1 adapter contract changed")
    return module


def official_final_label(
    raw_class_id: int, probabilities: dict[str, float], adapter: Any
) -> str:
    """Reproduce the official PlasFlow-v1.1 final-label policy exactly."""

    initial_label = str(adapter.PROBABILITY_FIELDS[raw_class_id])
    if probabilities[initial_label] >= adapter.DECISION_THRESHOLD:
        return initial_label
    plasmid_sum = sum(probabilities[field] for field in adapter.PLASMID_FIELDS)
    chromosome_sum = sum(probabilities[field] for field in adapter.CHROMOSOME_FIELDS)
    taxon = initial_label.split(".", 1)[1]
    taxon_sum = sum(
        probability
        for field, probability in probabilities.items()
        if field.split(".", 1)[1] == taxon
    )
    if plasmid_sum > adapter.DECISION_THRESHOLD:
        return "plasmid.unclassified"
    if chromosome_sum > adapter.DECISION_THRESHOLD:
        return "chromosome.unclassified"
    if taxon_sum > adapter.DECISION_THRESHOLD:
        return f"unclassified.{taxon}"
    return "unclassified.unclassified"


def corrected_loader(adapter: Any, mismatches: list[dict[str, str]]) -> Any:
    def load(path: Path) -> dict[str, dict[str, Any]]:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = list(reader.fieldnames or [])
            expected = adapter.RAW_METADATA_FIELDS + adapter.PROBABILITY_FIELDS
            if len(fields) != len(set(fields)):
                raise ValueError("PlasFlow-v1 output contains duplicate columns")
            if set(fields) != set(expected):
                missing = sorted(set(expected) - set(fields))
                unexpected = sorted(set(fields) - set(expected))
                raise ValueError(
                    f"Unexpected native schema; missing={missing}, unexpected={unexpected}"
                )

            rows: dict[str, dict[str, Any]] = {}
            valid_labels = adapter.allowed_native_labels()
            for line_number, row in enumerate(reader, start=2):
                raw_identifier = (row.get("contig_name") or "").strip()
                if not raw_identifier:
                    raise ValueError(f"Empty contig_name at line {line_number}")
                contig_id = adapter.canonical_id(raw_identifier)
                if contig_id in rows:
                    raise ValueError(f"Duplicate native identifier: {contig_id}")
                pandas_index = adapter.parse_integer(
                    row.get(""), "pandas index", raw_identifier
                )
                raw_index = adapter.parse_integer(
                    row.get("contig_id"), "contig_id", raw_identifier
                )
                if pandas_index != raw_index:
                    raise ValueError(f"Native row indices differ for {raw_identifier}")
                length = adapter.parse_integer(
                    row.get("contig_length"), "contig_length", raw_identifier
                )
                if length <= 0:
                    raise ValueError(f"Invalid native length for {raw_identifier}")
                raw_class_id = adapter.parse_integer(
                    row.get("id"), "class id", raw_identifier
                )
                if not 0 <= raw_class_id < len(adapter.PROBABILITY_FIELDS):
                    raise ValueError(f"Native class id outside range for {raw_identifier}")
                raw_label = (row.get("label") or "").strip()
                if raw_label not in valid_labels:
                    raise ValueError(f"Invalid native label for {raw_identifier}")
                probabilities = {
                    field: adapter.parse_probability(row.get(field), field, raw_identifier)
                    for field in adapter.PROBABILITY_FIELDS
                }
                if not math.isclose(
                    sum(probabilities.values()),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=adapter.PROBABILITY_SUM_TOLERANCE,
                ):
                    raise ValueError(f"Probabilities do not sum to one for {raw_identifier}")
                reconstructed = official_final_label(raw_class_id, probabilities, adapter)
                if reconstructed != raw_label:
                    raise ValueError(
                        "Native label does not reproduce the official vote/threshold policy "
                        f"for {raw_identifier}: {raw_label} versus {reconstructed}"
                    )
                argmax_id = max(
                    range(len(adapter.PROBABILITY_FIELDS)),
                    key=lambda index: probabilities[adapter.PROBABILITY_FIELDS[index]],
                )
                if raw_class_id != argmax_id:
                    mismatches.append(
                        {
                            "contig_id": contig_id,
                            "majority_vote_class_id": str(raw_class_id),
                            "majority_vote_label": adapter.PROBABILITY_FIELDS[raw_class_id],
                            "mean_probability_argmax_id": str(argmax_id),
                            "mean_probability_argmax_label": adapter.PROBABILITY_FIELDS[argmax_id],
                            "final_native_label": raw_label,
                        }
                    )
                predicted_label, prediction_status = adapter.map_native_label(raw_label)
                rows[contig_id] = {
                    "raw_tool_contig_id": raw_identifier,
                    "raw_contig_index": raw_index,
                    "raw_class_id": raw_class_id,
                    "raw_label": raw_label,
                    "contig_length": length,
                    "predicted_label": predicted_label,
                    "prediction_status": prediction_status,
                    "plasmid_probability": sum(
                        probabilities[field] for field in adapter.PLASMID_FIELDS
                    ),
                    "chromosome_probability": sum(
                        probabilities[field] for field in adapter.CHROMOSOME_FIELDS
                    ),
                    "max_class_probability": max(probabilities.values()),
                }
        return rows

    return load


def validate_provenance(
    provenance_path: Path, input_fasta: Path, raw_predictions: Path
) -> dict[str, Any]:
    provenance = json.loads(provenance_path.read_text())
    execution = provenance.get("execution", {})
    input_data = provenance.get("input", {})
    errors = provenance.get("errors", [])
    if provenance.get("status") != "FAIL" or provenance.get("overall_status") != 1:
        raise RuntimeError("Retained runner is not in the expected fail-closed state")
    if execution.get("returncode") != 0 or execution.get("timed_out") is not False:
        raise RuntimeError("Official PlasFlow-v1 execution did not complete successfully")
    if len(errors) != 1 or not str(errors[0]).startswith(FAILURE_PREFIX):
        raise RuntimeError("Retained failure is not the frozen native-vote assertion")
    if input_data.get("sha256") != sha256_file(input_fasta):
        raise RuntimeError("Runner input identity does not match the supplied FASTA")
    if provenance.get("adapter_sha256") != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("Runner provenance references a different adapter")
    if provenance.get("adapter_contract_sha256") != EXPECTED_ADAPTER_CONTRACT_SHA256:
        raise RuntimeError("Runner provenance references a different adapter contract")
    if provenance.get("image_reference") != (
        "quay.io/biocontainers/plasflow@" + EXPECTED_CONTAINER_DIGEST
    ):
        raise RuntimeError("Runner provenance references a different container")
    rows = sum(1 for _ in raw_predictions.open("rb")) - 1
    if rows != input_data.get("sequence_count"):
        raise RuntimeError("Raw prediction row count does not match the frozen input")
    return provenance


def write_mismatches(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "contig_id",
        "majority_vote_class_id",
        "majority_vote_label",
        "mean_probability_argmax_id",
        "mean_probability_argmax_label",
        "final_native_label",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-adapter", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    input_fasta = args.input_fasta.resolve()
    raw = output / "raw_predictions.tsv"
    provenance_path = output / "runner_provenance.json"
    final_predictions = output / "standardized_predictions.tsv"
    final_metadata = output / "adapter_metadata.json"
    audit_dir = output / "native_vote_correction"
    validation_path = audit_dir / "native_vote_correction_validation.json"

    if final_predictions.is_file() and final_metadata.is_file() and validation_path.is_file():
        validation = json.loads(validation_path.read_text())
        if (
            validation.get("status") == "PASS"
            and validation.get("standardized_predictions_sha256")
            == sha256_file(final_predictions)
        ):
            print("Reusing validated PlasFlow-v1 native-vote correction.")
            return 0
        raise RuntimeError("Existing corrected output failed identity verification")
    if final_predictions.exists() or final_metadata.exists() or audit_dir.exists():
        raise RuntimeError("Partial correction output exists; refusing to overwrite it")
    for path in (input_fasta, raw, provenance_path, args.frozen_adapter):
        if not path.is_file():
            raise FileNotFoundError(path)

    provenance = validate_provenance(provenance_path, input_fasta, raw)
    adapter = load_adapter(args.frozen_adapter)
    mismatches: list[dict[str, str]] = []
    adapter.load_raw_predictions = corrected_loader(adapter, mismatches)
    audit_dir.mkdir(parents=True)
    temporary_predictions = audit_dir / "standardized_predictions.tsv.tmp"
    temporary_metadata = audit_dir / "adapter_metadata.json.tmp"
    metadata = adapter.adapt_plasflow_v1(
        input_fasta=input_fasta,
        raw_predictions=raw,
        output_path=temporary_predictions,
        metadata_output=temporary_metadata,
    )
    expected_rows = provenance["input"]["sequence_count"]
    if metadata.get("standardized_rows") != expected_rows:
        raise RuntimeError("Corrected standardized row count is incomplete")
    if metadata.get("complete_output") is not True:
        raise RuntimeError("Corrected adapter reported missing output")
    mismatch_path = audit_dir / "native_vote_probability_argmax_mismatches.tsv"
    write_mismatches(mismatch_path, mismatches)

    corrected_metadata = json.loads(temporary_metadata.read_text())
    corrected_metadata["standardized_output"] = str(final_predictions)
    corrected_metadata["standardized_output_sha256"] = sha256_file(
        temporary_predictions
    )
    corrected_metadata["native_vote_validation_correction"] = {
        "established_correction_source": (
            "evaluation/release_audit/nar_benchmark_20260727/"
            "249_plasflow_v1_confirmatory_native_vote_validation_correction"
        ),
        "reason": (
            "native class id is a three-classifier majority vote while "
            "probabilities are classifier means"
        ),
        "vote_probability_argmax_mismatches": len(mismatches),
        "final_native_policy_rows_verified": expected_rows,
        "official_tool_rerun": False,
        "raw_predictions_changed": False,
        "frozen_adapter_source_changed": False,
        "mapping_or_threshold_changed": False,
        "ground_truth_accessed": False,
        "performance_metrics_calculated": False,
    }
    temporary_metadata.write_text(json.dumps(corrected_metadata, indent=2) + "\n")
    os.replace(temporary_predictions, final_predictions)
    os.replace(temporary_metadata, final_metadata)
    validation = {
        "schema_version": "mobiorigin-operational-plasflow-v1-native-vote-correction-v1",
        "status": "PASS",
        "overall_status": 0,
        "input_fasta_sha256": sha256_file(input_fasta),
        "raw_predictions_sha256": sha256_file(raw),
        "runner_provenance_sha256": sha256_file(provenance_path),
        "frozen_adapter_sha256": sha256_file(args.frozen_adapter),
        "records": expected_rows,
        "vote_probability_argmax_mismatches": len(mismatches),
        "official_final_policy_mismatches": 0,
        "standardized_predictions_sha256": sha256_file(final_predictions),
        "adapter_metadata_sha256": sha256_file(final_metadata),
        "mismatch_audit_sha256": sha256_file(mismatch_path),
        "official_tool_rerun": False,
        "raw_predictions_changed": False,
        "ground_truth_accessed": False,
        "performance_metrics_calculated": False,
    }
    atomic_json(validation_path, validation)
    print("===== PLASFLOW V1 NATIVE-VOTE OUTPUT FINALIZATION =====")
    print("Overall status: 0")
    print(f"Records: {expected_rows}")
    print(f"Majority-vote/mean-argmax differences: {len(mismatches)}")
    print("Official raw predictions reused: true")
    print("Ground-truth labels accessed: false")
    print("Performance metrics calculated: false")
    print(f"Standardized output: {final_predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
