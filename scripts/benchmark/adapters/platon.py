#!/usr/bin/env python3
"""Normalize official Platon 1.7 output for the frozen NAR benchmark.

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

TOOL_NAME = "Platon"
TOOL_VERSION = "1.7"
SOURCE_COMMIT = "3cce2dd295348e25be4b5bd64f3622c2603d6ba0"
CONTAINER_IMAGE_ID = "sha256:74d96300053a9ce3d4f10bbb935b20631e1d8547c1df632d5f05b178eb2cbbf6"
SCHEMA_VERSION = "nar-comparator-adapter-v1"
CONTRACT_SHA256 = "b8add8c173bdd049f750133cdfd6bef2d8be42b0315c2bcf66e4182079aa6e96"

MODE = "accuracy"
METAGENOME_MODE = True
MINIMUM_LENGTH = 1_000
MAXIMUM_LENGTH = 500_000

RAW_TSV_FIELDS = [
    "ID",
    "Length",
    "Coverage",
    "# ORFs",
    "RDS",
    "Circular",
    "Inc Type(s)",
    "# Replication",
    "# Mobilization",
    "# OriT",
    "# Conjugation",
    "# AMRs",
    "# rRNAs",
    "# Plasmid Hits",
]

REQUIRED_JSON_FIELDS = {
    "id",
    "length",
    "sequence",
    "orfs",
    "is_circular",
    "inc_types",
    "amr_hits",
    "mobilization_hits",
    "orit_hits",
    "replication_hits",
    "conjugation_hits",
    "rrnas",
    "plasmid_hits",
    "coverage",
    "protein_score",
}

OUTPUT_FIELDS = [
    "contig_id",
    "input_header",
    "length",
    "raw_tool_contig_id",
    "raw_native_label",
    "predicted_label",
    "prediction_status",
    "plasmid_score",
    "decision_threshold",
    "rds",
    "is_circular",
    "inc_types",
    "replication_hit_count",
    "mobilization_hit_count",
    "orit_hit_count",
    "conjugation_hit_count",
    "amr_hit_count",
    "rrna_hit_count",
    "reference_plasmid_hit_count",
    "source_tool",
    "source_version",
    "mode",
    "metagenome_mode",
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


def load_fasta_records(
    path: Path,
    *,
    allow_empty: bool,
    source_name: str,
) -> list[dict[str, str | int]]:
    """Load ordered FASTA records without altering identifiers."""

    if not path.is_file():
        raise FileNotFoundError(f"{source_name} FASTA does not exist: {path}")

    records: list[dict[str, str | int]] = []
    observed: dict[str, str] = {}
    full_header: str | None = None
    sequence_parts: list[str] = []

    def finalize_record() -> None:
        nonlocal full_header
        nonlocal sequence_parts

        if full_header is None:
            return

        contig_id = canonical_id(full_header)
        sequence = "".join(sequence_parts)

        if contig_id in observed:
            raise ValueError(
                f"Duplicate canonical {source_name} FASTA identifier: "
                f"{contig_id!r} from {observed[contig_id]!r} and "
                f"{full_header!r}"
            )

        if not sequence:
            raise ValueError(f"{source_name} FASTA record has no sequence: {contig_id!r}")

        observed[contig_id] = full_header
        records.append(
            {
                "contig_id": contig_id,
                "input_header": full_header,
                "length": len(sequence),
                "sequence": sequence.upper(),
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
                sequence_parts = []

                if not full_header:
                    raise ValueError(f"Empty {source_name} FASTA header at line " f"{line_number}")

                continue

            if full_header is None:
                raise ValueError(
                    f"{source_name} FASTA sequence data appears before "
                    f"the first header at line {line_number}"
                )

            sequence_parts.append("".join(line.split()))

    finalize_record()

    if not records and not allow_empty:
        raise ValueError(f"No FASTA records found in {path}")

    return records


def reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Reject duplicate keys while loading native JSON."""

    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in Platon JSON: {key!r}")
        result[key] = value

    return result


def parse_finite_number(
    value: Any,
    *,
    field: str,
    contig_id: str,
) -> float:
    """Parse a finite numeric value."""

    if isinstance(value, bool):
        raise ValueError(f"Invalid Platon {field} for {contig_id!r}: {value!r}")

    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Platon {field} for {contig_id!r}: {value!r}") from error

    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite Platon {field} for {contig_id!r}: {value!r}")

    return parsed


def parse_nonnegative_integer(
    value: Any,
    *,
    field: str,
    contig_id: str,
) -> int:
    """Parse a non-negative integral value."""

    if isinstance(value, bool):
        raise ValueError(f"Invalid Platon {field} for {contig_id!r}: {value!r}")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Platon {field} for {contig_id!r}: {value!r}") from error

    if str(parsed) != str(value).strip() or parsed < 0:
        raise ValueError(f"Invalid Platon {field} for {contig_id!r}: {value!r}")

    return parsed


def load_raw_json(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate native plasmid-only Platon JSON."""

    if not path.is_file():
        raise FileNotFoundError(f"Platon JSON does not exist: {path}")

    try:
        raw = json.loads(
            path.read_text(),
            object_pairs_hook=reject_duplicate_pairs,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed Platon JSON: {path}") from error

    if not isinstance(raw, dict):
        raise ValueError("Platon JSON top level must be an object")

    records: dict[str, dict[str, Any]] = {}

    for raw_identifier, value in raw.items():
        contig_id = canonical_id(raw_identifier)

        if contig_id in records:
            raise ValueError(f"Duplicate canonical Platon JSON identifier: {contig_id!r}")

        if not isinstance(value, dict):
            raise ValueError(f"Platon JSON record must be an object: {contig_id!r}")

        missing = REQUIRED_JSON_FIELDS - set(value)

        if missing:
            raise ValueError(
                f"Platon JSON record {contig_id!r} is missing fields: " + ", ".join(sorted(missing))
            )

        if str(value["id"]) != raw_identifier:
            raise ValueError(f"Platon JSON identifier mismatch for {contig_id!r}")

        length = parse_nonnegative_integer(
            value["length"],
            field="length",
            contig_id=contig_id,
        )

        sequence = value["sequence"]

        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"Invalid Platon JSON sequence for {contig_id!r}")

        if len(sequence) != length:
            raise ValueError(f"Platon JSON sequence length mismatch for {contig_id!r}")

        if not isinstance(value["is_circular"], bool):
            raise ValueError(f"Invalid Platon circularity for {contig_id!r}")

        if not isinstance(value["orfs"], dict):
            raise ValueError(f"Invalid Platon ORF object for {contig_id!r}")

        for field in (
            "inc_types",
            "amr_hits",
            "mobilization_hits",
            "orit_hits",
            "replication_hits",
            "conjugation_hits",
            "rrnas",
            "plasmid_hits",
        ):
            if not isinstance(value[field], list):
                raise ValueError(f"Invalid Platon {field} list for {contig_id!r}")

        parse_finite_number(
            value["protein_score"],
            field="RDS",
            contig_id=contig_id,
        )

        records[contig_id] = value

    return records


def load_raw_tsv(path: Path) -> dict[str, dict[str, str]]:
    """Load and validate native plasmid-only Platon TSV."""

    if not path.is_file():
        raise FileNotFoundError(f"Platon TSV does not exist: {path}")

    rows: dict[str, dict[str, str]] = {}

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        if reader.fieldnames != RAW_TSV_FIELDS:
            raise ValueError(
                f"Unexpected Platon TSV fields: {reader.fieldnames!r}; "
                f"expected {RAW_TSV_FIELDS!r}"
            )

        for line_number, row in enumerate(reader, start=2):
            raw_identifier = (row.get("ID") or "").strip()
            contig_id = canonical_id(raw_identifier)

            if contig_id in rows:
                raise ValueError(f"Duplicate canonical Platon TSV identifier: " f"{contig_id!r}")

            parse_nonnegative_integer(
                row["Length"],
                field="TSV length",
                contig_id=contig_id,
            )

            for field in (
                "# ORFs",
                "# Replication",
                "# Mobilization",
                "# OriT",
                "# Conjugation",
                "# AMRs",
                "# rRNAs",
                "# Plasmid Hits",
            ):
                parse_nonnegative_integer(
                    row[field],
                    field=field,
                    contig_id=contig_id,
                )

            parse_finite_number(
                row["RDS"],
                field="TSV RDS",
                contig_id=contig_id,
            )

            circular = row["Circular"].strip().lower()
            if circular not in {"yes", "no"}:
                raise ValueError(
                    f"Invalid Platon TSV circularity at line " f"{line_number}: {row['Circular']!r}"
                )

            coverage = row["Coverage"].strip()
            if coverage != "NA":
                parse_finite_number(
                    coverage,
                    field="coverage",
                    contig_id=contig_id,
                )

            rows[contig_id] = dict(row)

    return rows


def format_float(value: float) -> str:
    """Format a finite float without unnecessary rounding."""

    return f"{value:.17g}"


def json_compact(value: Any) -> str:
    """Serialize structured evidence deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def validate_plasmid_evidence(
    *,
    contig_id: str,
    input_record: dict[str, str | int],
    json_record: dict[str, Any],
    tsv_record: dict[str, str],
) -> None:
    """Cross-check native JSON and TSV evidence for one plasmid call."""

    input_length = int(input_record["length"])
    json_length = int(json_record["length"])
    tsv_length = int(tsv_record["Length"])

    if input_length != json_length or input_length != tsv_length:
        raise ValueError(
            f"Platon length mismatch for {contig_id!r}: "
            f"input={input_length}, JSON={json_length}, TSV={tsv_length}"
        )

    if str(json_record["sequence"]).upper() != str(input_record["sequence"]).upper():
        raise ValueError(f"Platon JSON sequence does not match input for {contig_id!r}")

    json_rds = parse_finite_number(
        json_record["protein_score"],
        field="RDS",
        contig_id=contig_id,
    )
    tsv_rds = parse_finite_number(
        tsv_record["RDS"],
        field="TSV RDS",
        contig_id=contig_id,
    )

    if abs(tsv_rds - round(json_rds, 1)) > 1e-9:
        raise ValueError(
            f"Platon JSON/TSV RDS mismatch for {contig_id!r}: " f"JSON={json_rds}, TSV={tsv_rds}"
        )

    expected_counts = {
        "# ORFs": len(json_record["orfs"]),
        "# Replication": len(json_record["replication_hits"]),
        "# Mobilization": len(json_record["mobilization_hits"]),
        "# OriT": len(json_record["orit_hits"]),
        "# Conjugation": len(json_record["conjugation_hits"]),
        "# AMRs": len(json_record["amr_hits"]),
        "# rRNAs": len(json_record["rrnas"]),
        "# Plasmid Hits": len(json_record["plasmid_hits"]),
    }

    for field, expected in expected_counts.items():
        observed = int(tsv_record[field])
        if observed != expected:
            raise ValueError(
                f"Platon JSON/TSV {field} mismatch for {contig_id!r}: "
                f"JSON={expected}, TSV={observed}"
            )

    expected_circular = "yes" if json_record["is_circular"] else "no"
    if tsv_record["Circular"].strip().lower() != expected_circular:
        raise ValueError(f"Platon JSON/TSV circularity mismatch for {contig_id!r}")


def adapt_platon(
    input_fasta: Path,
    plasmid_fasta: Path,
    chromosome_fasta: Path,
    raw_json: Path,
    raw_tsv: Path,
    output_path: Path,
    metadata_output: Path | None = None,
) -> dict[str, Any]:
    """Create a complete standardized Platon table."""

    input_records = load_fasta_records(
        input_fasta,
        allow_empty=False,
        source_name="input",
    )
    plasmid_records = load_fasta_records(
        plasmid_fasta,
        allow_empty=True,
        source_name="plasmid-output",
    )
    chromosome_records = load_fasta_records(
        chromosome_fasta,
        allow_empty=True,
        source_name="chromosome-output",
    )
    json_records = load_raw_json(raw_json)
    tsv_records = load_raw_tsv(raw_tsv)

    input_by_id = {str(record["contig_id"]): record for record in input_records}
    plasmid_by_id = {str(record["contig_id"]): record for record in plasmid_records}
    chromosome_by_id = {str(record["contig_id"]): record for record in chromosome_records}

    input_ids = set(input_by_id)
    plasmid_ids = set(plasmid_by_id)
    chromosome_ids = set(chromosome_by_id)
    json_ids = set(json_records)
    tsv_ids = set(tsv_records)

    overlap = sorted(plasmid_ids & chromosome_ids)
    if overlap:
        raise ValueError(
            "Platon identifiers appear in both native FASTAs: " + ", ".join(overlap[:20])
        )

    extra_plasmid = sorted(plasmid_ids - input_ids)
    extra_chromosome = sorted(chromosome_ids - input_ids)

    if extra_plasmid:
        raise ValueError(
            "Platon plasmid FASTA contains identifiers absent from input: "
            + ", ".join(extra_plasmid[:20])
        )

    if extra_chromosome:
        raise ValueError(
            "Platon chromosome FASTA contains identifiers absent from input: "
            + ", ".join(extra_chromosome[:20])
        )

    if json_ids != plasmid_ids:
        raise ValueError("Platon JSON identifiers differ from native plasmid FASTA")

    if tsv_ids != plasmid_ids:
        raise ValueError("Platon TSV identifiers differ from native plasmid FASTA")

    for native_name, native_records in (
        ("plasmid", plasmid_by_id),
        ("chromosome", chromosome_by_id),
    ):
        for contig_id, native_record in native_records.items():
            input_record = input_by_id[contig_id]

            if str(native_record["sequence"]).upper() != str(input_record["sequence"]).upper():
                raise ValueError(
                    f"Platon {native_name} FASTA sequence does not "
                    f"match input for {contig_id!r}"
                )

    for contig_id in plasmid_ids:
        validate_plasmid_evidence(
            contig_id=contig_id,
            input_record=input_by_id[contig_id],
            json_record=json_records[contig_id],
            tsv_record=tsv_records[contig_id],
        )

    supported_ids = {
        contig_id
        for contig_id, record in input_by_id.items()
        if MINIMUM_LENGTH <= int(record["length"]) <= MAXIMUM_LENGTH
    }
    unsupported_ids = input_ids - supported_ids
    native_ids = plasmid_ids | chromosome_ids

    unexpected_unsupported = sorted(native_ids & unsupported_ids)
    if unexpected_unsupported:
        raise ValueError(
            "Platon returned native calls for unsupported-length inputs: "
            + ", ".join(unexpected_unsupported[:20])
        )

    missing_supported_ids = sorted(supported_ids - native_ids)

    standardized: list[dict[str, str]] = []

    for input_record in input_records:
        contig_id = str(input_record["contig_id"])
        length = int(input_record["length"])

        row = {
            "contig_id": contig_id,
            "input_header": str(input_record["input_header"]),
            "length": str(length),
            "raw_tool_contig_id": "",
            "raw_native_label": "",
            "predicted_label": "",
            "prediction_status": "",
            "plasmid_score": "",
            "decision_threshold": "",
            "rds": "",
            "is_circular": "",
            "inc_types": "",
            "replication_hit_count": "",
            "mobilization_hit_count": "",
            "orit_hit_count": "",
            "conjugation_hit_count": "",
            "amr_hit_count": "",
            "rrna_hit_count": "",
            "reference_plasmid_hit_count": "",
            "source_tool": TOOL_NAME,
            "source_version": TOOL_VERSION,
            "mode": MODE,
            "metagenome_mode": "true",
        }

        if contig_id in unsupported_ids:
            row["predicted_label"] = "unclassified"
            row["prediction_status"] = "unsupported_length"

        elif contig_id in plasmid_ids:
            evidence = json_records[contig_id]

            row.update(
                {
                    "raw_tool_contig_id": contig_id,
                    "raw_native_label": "plasmid",
                    "predicted_label": "plasmid",
                    "prediction_status": "called_plasmid",
                    "rds": format_float(float(evidence["protein_score"])),
                    "is_circular": ("true" if evidence["is_circular"] else "false"),
                    "inc_types": json_compact(evidence["inc_types"]),
                    "replication_hit_count": str(len(evidence["replication_hits"])),
                    "mobilization_hit_count": str(len(evidence["mobilization_hits"])),
                    "orit_hit_count": str(len(evidence["orit_hits"])),
                    "conjugation_hit_count": str(len(evidence["conjugation_hits"])),
                    "amr_hit_count": str(len(evidence["amr_hits"])),
                    "rrna_hit_count": str(len(evidence["rrnas"])),
                    "reference_plasmid_hit_count": str(len(evidence["plasmid_hits"])),
                }
            )

        elif contig_id in chromosome_ids:
            row.update(
                {
                    "raw_tool_contig_id": contig_id,
                    "raw_native_label": "chromosome",
                    "predicted_label": "non-plasmid",
                    "prediction_status": "called_non_plasmid",
                }
            )

        else:
            row["predicted_label"] = "unclassified"
            row["prediction_status"] = "missing_output"

        standardized.append(row)

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
    native_label_counts = Counter(
        row["raw_native_label"] for row in standardized if row["raw_native_label"]
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "source_commit": SOURCE_COMMIT,
        "container_image_id": CONTAINER_IMAGE_ID,
        "mode": MODE,
        "metagenome_mode": METAGENOME_MODE,
        "characterize_mode": False,
        "minimum_supported_length": MINIMUM_LENGTH,
        "maximum_supported_length": MAXIMUM_LENGTH,
        "threshold_tuning_allowed": False,
        "adapter_reclassified_native_calls": False,
        "calibrated_plasmid_probability_available": False,
        "rds_treated_as_probability": False,
        "native_negative_label": "chromosome",
        "standardized_negative_label": "non-plasmid",
        "confirmed_chromosome_claim": False,
        "native_phage_class_available": False,
        "three_class_claim": False,
        "input_records": len(input_records),
        "plasmid_fasta_records": len(plasmid_records),
        "chromosome_fasta_records": len(chromosome_records),
        "raw_json_records": len(json_records),
        "raw_tsv_rows": len(tsv_records),
        "standardized_rows": len(standardized),
        "complete_input_coverage": (len(standardized) == len(input_records)),
        "complete_supported_partition": not missing_supported_ids,
        "runner_success_allowed": not missing_supported_ids,
        "unsupported_length_ids": sorted(unsupported_ids),
        "missing_supported_output_ids": missing_supported_ids,
        "label_counts": dict(sorted(label_counts.items())),
        "prediction_status_counts": dict(sorted(status_counts.items())),
        "native_label_counts": dict(sorted(native_label_counts.items())),
        "input_fasta_sha256": sha256_file(input_fasta),
        "plasmid_fasta_sha256": sha256_file(plasmid_fasta),
        "chromosome_fasta_sha256": sha256_file(chromosome_fasta),
        "raw_json_sha256": sha256_file(raw_json),
        "raw_tsv_sha256": sha256_file(raw_tsv),
        "standardized_output_sha256": sha256_file(output_path),
        "models_deserialized_by_adapter": False,
        "predictions_generated_by_adapter": False,
        "manuscript_only": True,
        "production_workflow_component": False,
    }

    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Create the manuscript-only adapter CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Normalize official Platon 1.7 output for the frozen "
            "manuscript benchmark. This command is not part of the "
            "PlasFlow2 prediction workflow."
        )
    )
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--plasmid-fasta", type=Path, required=True)
    parser.add_argument("--chromosome-fasta", type=Path, required=True)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--raw-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)

    return parser


def main() -> None:
    """Run the Platon adapter CLI."""

    args = build_parser().parse_args()

    metadata = adapt_platon(
        input_fasta=args.input_fasta,
        plasmid_fasta=args.plasmid_fasta,
        chromosome_fasta=args.chromosome_fasta,
        raw_json=args.raw_json,
        raw_tsv=args.raw_tsv,
        output_path=args.output,
        metadata_output=args.metadata_output,
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
