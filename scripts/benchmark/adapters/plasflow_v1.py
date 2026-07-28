#!/usr/bin/env python3
"""Normalize official PlasFlow v1.1 output for the frozen NAR benchmark.

This adapter is restricted to manuscript-only comparative benchmarking.
It is not part of the PlasFlow2 prediction workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

TOOL_NAME = "PlasFlow"
TOOL_VERSION = "1.1"
SCHEMA_VERSION = "nar-comparator-adapter-v1"
CONTRACT_SHA256 = "7e24ac29aedb6f0b106e421302cb1ee147f076c96c24d299f9b9d55d1b42c3bf"
CONTAINER_DIGEST = "sha256:e69acee3233010dbf5a5245620252bf5b9bde930ad5546473ec496992995a7da"
DECISION_THRESHOLD = 0.7
PROBABILITY_SUM_TOLERANCE = 1e-5

CHROMOSOME_FIELDS = [
    "chromosome.Acidobacteria",
    "chromosome.Actinobacteria",
    "chromosome.Bacteroidetes",
    "chromosome.Chlamydiae",
    "chromosome.Chlorobi",
    "chromosome.Chloroflexi",
    "chromosome.Cyanobacteria",
    "chromosome.DeinococcusThermus",
    "chromosome.Firmicutes",
    "chromosome.Fusobacteria",
    "chromosome.Nitrospirae",
    "chromosome.other",
    "chromosome.Planctomycetes",
    "chromosome.Proteobacteria",
    "chromosome.Spirochaetes",
    "chromosome.Tenericutes",
    "chromosome.Thermotogae",
    "chromosome.Verrucomicrobia",
]

PLASMID_FIELDS = [
    "plasmid.Actinobacteria",
    "plasmid.Bacteroidetes",
    "plasmid.Chlamydiae",
    "plasmid.Cyanobacteria",
    "plasmid.DeinococcusThermus",
    "plasmid.Firmicutes",
    "plasmid.Fusobacteria",
    "plasmid.other",
    "plasmid.Proteobacteria",
    "plasmid.Spirochaetes",
]

PROBABILITY_FIELDS = CHROMOSOME_FIELDS + PLASMID_FIELDS

RAW_METADATA_FIELDS = [
    "",
    "contig_id",
    "contig_name",
    "contig_length",
    "id",
    "label",
]

OUTPUT_FIELDS = [
    "contig_id",
    "input_header",
    "length",
    "raw_tool_contig_id",
    "raw_contig_index",
    "raw_class_id",
    "raw_label",
    "predicted_label",
    "prediction_status",
    "plasmid_probability",
    "chromosome_probability",
    "max_class_probability",
    "decision_threshold",
    "source_tool",
    "source_version",
    "container_digest",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(header: str) -> str:
    """Return the first whitespace-delimited FASTA identifier."""

    value = header.strip()
    if not value:
        raise ValueError("Empty sequence identifier")
    return value.split()[0]


def load_fasta_records(path: Path) -> list[dict[str, str | int]]:
    """Read ordered FASTA headers and lengths without altering sequences."""

    if not path.is_file():
        raise FileNotFoundError(f"Input FASTA does not exist: {path}")

    records: list[dict[str, str | int]] = []
    observed: dict[str, str] = {}
    full_header: str | None = None
    sequence_length = 0

    def finalize_record() -> None:
        nonlocal full_header
        nonlocal sequence_length

        if full_header is None:
            return

        contig_id = canonical_id(full_header)

        if contig_id in observed:
            raise ValueError(
                "Canonical FASTA identifier collision: "
                f"{contig_id!r} maps to both "
                f"{observed[contig_id]!r} and {full_header!r}"
            )

        if sequence_length <= 0:
            raise ValueError(f"FASTA record has no sequence: {contig_id}")

        observed[contig_id] = full_header
        records.append(
            {
                "contig_id": contig_id,
                "input_header": full_header,
                "length": sequence_length,
            }
        )

    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                finalize_record()
                full_header = line[1:].strip()
                sequence_length = 0

                if not full_header:
                    raise ValueError(f"Empty FASTA header at line {line_number}")
                continue

            if full_header is None:
                raise ValueError(
                    "Sequence data appears before the first FASTA header " f"at line {line_number}"
                )

            sequence_length += len(line)

    finalize_record()

    if not records:
        raise ValueError(f"No FASTA records found in {path}")

    return records


def parse_integer(
    value: str | None,
    field: str,
    raw_identifier: str,
) -> int:
    """Parse a required integer field."""

    text = (value or "").strip()

    try:
        parsed = int(text)
    except ValueError as error:
        raise ValueError(f"Invalid {field} for {raw_identifier!r}: {text!r}") from error

    return parsed


def parse_probability(
    value: str | None,
    field: str,
    raw_identifier: str,
) -> float:
    """Parse and validate one native probability."""

    text = (value or "").strip()

    try:
        probability = float(text)
    except ValueError as error:
        raise ValueError(
            f"Invalid probability {field!r} for " f"{raw_identifier!r}: {text!r}"
        ) from error

    if not math.isfinite(probability):
        raise ValueError(f"Non-finite probability {field!r} for " f"{raw_identifier!r}")

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Probability {field!r} outside [0,1] for " f"{raw_identifier!r}: {probability}"
        )

    return probability


def allowed_native_labels() -> set[str]:
    """Return all labels allowed by the official threshold policy."""

    taxa = {field.split(".", 1)[1] for field in PROBABILITY_FIELDS}

    labels = set(PROBABILITY_FIELDS)
    labels.update(
        {
            "plasmid.unclassified",
            "chromosome.unclassified",
            "unclassified.unclassified",
        }
    )
    labels.update(f"unclassified.{taxon}" for taxon in taxa)
    return labels


def map_native_label(raw_label: str) -> tuple[str, str]:
    """Map the final official label to the frozen binary task."""

    if raw_label.startswith("plasmid."):
        return "plasmid", "called_plasmid"

    if raw_label.startswith("chromosome."):
        return "non-plasmid", "called_non_plasmid"

    if raw_label.startswith("unclassified."):
        return "unclassified", "native_abstention"

    raise ValueError(f"Unsupported PlasFlow v1 label: {raw_label!r}")


def load_raw_predictions(
    path: Path,
) -> dict[str, dict[str, Any]]:
    """Read and validate the official PlasFlow v1 prediction table."""

    if not path.is_file():
        raise FileNotFoundError(f"PlasFlow v1 prediction table does not exist: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])

        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("PlasFlow v1 output contains duplicate column names")

        expected_fields = RAW_METADATA_FIELDS + PROBABILITY_FIELDS
        missing_fields = set(expected_fields) - set(fieldnames)
        unexpected_fields = set(fieldnames) - set(expected_fields)

        if missing_fields:
            raise ValueError(
                "PlasFlow v1 output is missing columns: " + ", ".join(sorted(missing_fields))
            )

        if unexpected_fields:
            raise ValueError(
                "PlasFlow v1 output has unexpected columns: " + ", ".join(sorted(unexpected_fields))
            )

        raw_rows: dict[str, dict[str, Any]] = {}
        valid_labels = allowed_native_labels()

        for line_number, row in enumerate(reader, start=2):
            raw_identifier = (row.get("contig_name") or "").strip()

            if not raw_identifier:
                raise ValueError(f"{path}:{line_number}: empty contig_name")

            contig_id = canonical_id(raw_identifier)

            if contig_id in raw_rows:
                raise ValueError("Duplicate canonical PlasFlow v1 identifier: " f"{contig_id!r}")

            pandas_index = parse_integer(
                row.get(""),
                "pandas index",
                raw_identifier,
            )
            raw_contig_index = parse_integer(
                row.get("contig_id"),
                "contig_id",
                raw_identifier,
            )

            if pandas_index != raw_contig_index:
                raise ValueError(
                    "PlasFlow v1 pandas index and contig_id differ for "
                    f"{raw_identifier!r}: "
                    f"{pandas_index} versus {raw_contig_index}"
                )

            contig_length = parse_integer(
                row.get("contig_length"),
                "contig_length",
                raw_identifier,
            )

            if contig_length <= 0:
                raise ValueError(f"Invalid non-positive length for {raw_identifier!r}")

            raw_class_id = parse_integer(
                row.get("id"),
                "class id",
                raw_identifier,
            )

            if not 0 <= raw_class_id < len(PROBABILITY_FIELDS):
                raise ValueError(
                    f"Class id outside [0,27] for {raw_identifier!r}: " f"{raw_class_id}"
                )

            raw_label = (row.get("label") or "").strip()

            if raw_label not in valid_labels:
                raise ValueError(
                    f"Invalid final native label for " f"{raw_identifier!r}: {raw_label!r}"
                )

            probabilities = {
                field: parse_probability(
                    row.get(field),
                    field,
                    raw_identifier,
                )
                for field in PROBABILITY_FIELDS
            }

            probability_sum = sum(probabilities.values())

            if not math.isclose(
                probability_sum,
                1.0,
                rel_tol=0.0,
                abs_tol=PROBABILITY_SUM_TOLERANCE,
            ):
                raise ValueError(
                    "Native probabilities do not sum to one for "
                    f"{raw_identifier!r}: {probability_sum}"
                )

            argmax_index = max(
                range(len(PROBABILITY_FIELDS)),
                key=lambda index: probabilities[PROBABILITY_FIELDS[index]],
            )

            if raw_class_id != argmax_index:
                raise ValueError(
                    "Native class id is inconsistent with the probability "
                    f"argmax for {raw_identifier!r}: "
                    f"{raw_class_id} versus {argmax_index}"
                )

            plasmid_probability = sum(probabilities[field] for field in PLASMID_FIELDS)
            chromosome_probability = sum(probabilities[field] for field in CHROMOSOME_FIELDS)
            max_class_probability = max(probabilities.values())

            predicted_label, prediction_status = map_native_label(raw_label)

            raw_rows[contig_id] = {
                "raw_tool_contig_id": raw_identifier,
                "raw_contig_index": raw_contig_index,
                "raw_class_id": raw_class_id,
                "raw_label": raw_label,
                "contig_length": contig_length,
                "predicted_label": predicted_label,
                "prediction_status": prediction_status,
                "plasmid_probability": plasmid_probability,
                "chromosome_probability": chromosome_probability,
                "max_class_probability": max_class_probability,
            }

    return raw_rows


def formatted_probability(value: float) -> str:
    """Format probabilities deterministically without material rounding."""

    return f"{value:.17g}"


def adapt_plasflow_v1(
    input_fasta: Path,
    raw_predictions: Path,
    output_path: Path,
    metadata_output: Path | None = None,
) -> dict[str, Any]:
    """Create a complete standardized PlasFlow v1 prediction table."""

    fasta_records = load_fasta_records(input_fasta)
    raw_rows = load_raw_predictions(raw_predictions)

    input_ids = {str(record["contig_id"]) for record in fasta_records}
    extra_ids = sorted(set(raw_rows) - input_ids)

    if extra_ids:
        raise ValueError(
            "PlasFlow v1 output contains identifiers absent from the "
            "input FASTA: " + ", ".join(extra_ids[:20])
        )

    standardized: list[dict[str, str]] = []
    missing_ids: list[str] = []

    for record in fasta_records:
        contig_id = str(record["contig_id"])
        input_header = str(record["input_header"])
        length = int(record["length"])
        raw = raw_rows.get(contig_id)

        if raw is None:
            missing_ids.append(contig_id)
            standardized.append(
                {
                    "contig_id": contig_id,
                    "input_header": input_header,
                    "length": str(length),
                    "raw_tool_contig_id": "",
                    "raw_contig_index": "",
                    "raw_class_id": "",
                    "raw_label": "",
                    "predicted_label": "unclassified",
                    "prediction_status": "missing_output",
                    "plasmid_probability": "",
                    "chromosome_probability": "",
                    "max_class_probability": "",
                    "decision_threshold": str(DECISION_THRESHOLD),
                    "source_tool": TOOL_NAME,
                    "source_version": TOOL_VERSION,
                    "container_digest": CONTAINER_DIGEST,
                }
            )
            continue

        raw_length = int(raw["contig_length"])

        if raw_length != length:
            raise ValueError(
                f"Length mismatch for {contig_id!r}: " f"input={length}, PlasFlow={raw_length}"
            )

        standardized.append(
            {
                "contig_id": contig_id,
                "input_header": input_header,
                "length": str(length),
                "raw_tool_contig_id": str(raw["raw_tool_contig_id"]),
                "raw_contig_index": str(raw["raw_contig_index"]),
                "raw_class_id": str(raw["raw_class_id"]),
                "raw_label": str(raw["raw_label"]),
                "predicted_label": str(raw["predicted_label"]),
                "prediction_status": str(raw["prediction_status"]),
                "plasmid_probability": formatted_probability(float(raw["plasmid_probability"])),
                "chromosome_probability": formatted_probability(
                    float(raw["chromosome_probability"])
                ),
                "max_class_probability": formatted_probability(float(raw["max_class_probability"])),
                "decision_threshold": str(DECISION_THRESHOLD),
                "source_tool": TOOL_NAME,
                "source_version": TOOL_VERSION,
                "container_digest": CONTAINER_DIGEST,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(standardized)

    label_counts = Counter(row["predicted_label"] for row in standardized)
    status_counts = Counter(row["prediction_status"] for row in standardized)
    native_label_counts = Counter(row["raw_label"] for row in standardized if row["raw_label"])

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "source_tool": TOOL_NAME,
        "source_version": TOOL_VERSION,
        "container_digest": CONTAINER_DIGEST,
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_tuning_allowed": False,
        "adapter_reclassified_native_calls": False,
        "primary_task": "binary plasmid detection",
        "native_phage_class_available": False,
        "three_class_ranking_allowed": False,
        "input_fasta": str(input_fasta),
        "input_fasta_sha256": sha256_file(input_fasta),
        "raw_predictions": str(raw_predictions),
        "raw_predictions_sha256": sha256_file(raw_predictions),
        "standardized_output": str(output_path),
        "standardized_output_sha256": sha256_file(output_path),
        "input_sequences": len(fasta_records),
        "raw_prediction_rows": len(raw_rows),
        "standardized_rows": len(standardized),
        "complete_output": not missing_ids,
        "runner_success_allowed": not missing_ids,
        "label_counts": dict(sorted(label_counts.items())),
        "prediction_status_counts": dict(sorted(status_counts.items())),
        "native_label_counts": dict(sorted(native_label_counts.items())),
        "missing_output_ids": missing_ids,
        "probability_contract": {
            "native_probability_fields": 28,
            "chromosome_probability_fields": 18,
            "plasmid_probability_fields": 10,
            "row_sum_absolute_tolerance": (PROBABILITY_SUM_TOLERANCE),
            "continuous_plasmid_probability_available": True,
        },
        "identifier_contract": {
            "canonical_identifier": ("First whitespace-delimited FASTA header token."),
            "raw_identifier_retained": True,
            "collision_policy": "reject",
            "output_order": "input FASTA order",
        },
        "missing_output_policy": {
            "standardized_label": "unclassified",
            "prediction_status": "missing_output",
            "metric_semantics": "abstention",
            "runner_success_allowed": False,
        },
        "manuscript_disclosure": (
            "PlasFlow v1.1 does not provide a native phage class "
            "and is included only in binary plasmid-detection "
            "comparisons."
        ),
        "models_deserialized_by_adapter": False,
        "predictions_generated_by_adapter": False,
        "production_workflow_component": False,
    }

    if metadata_output is not None:
        metadata_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Normalize official PlasFlow v1.1 output into the frozen "
            "manuscript-only binary comparator schema. This command is "
            "not part of the PlasFlow2 prediction workflow."
        )
    )
    parser.add_argument(
        "--input-fasta",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--raw-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    """Run the adapter."""

    args = parse_args()
    metadata = adapt_plasflow_v1(
        input_fasta=args.input_fasta,
        raw_predictions=args.raw_predictions,
        output_path=args.output,
        metadata_output=args.metadata_output,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
