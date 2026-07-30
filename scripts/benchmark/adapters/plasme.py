#!/usr/bin/env python3
"""Normalize official PLASMe 1.1 output for the frozen NAR benchmark.

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

TOOL_NAME = "PLASMe"
TOOL_VERSION = "1.1"
SOURCE_COMMIT = "ef0409bad9c8c9ee5d66d90812bf56b345d8dd1d"
CONTAINER_IMAGE_ID = "sha256:fbc29e53cf4b331f328241da0e7a835c" "84a50e8aa51a6baf94931aa43559f9a7"
SCHEMA_VERSION = "nar-comparator-adapter-v1"
CONTRACT_SHA256 = "735407cb3b7d91200ec9ca9643336c981" "060c735fae89d0db08fb1fa2bcc98fc"

IDENTITY_THRESHOLD = 0.9
COVERAGE_THRESHOLD = 0.9
PROBABILITY_THRESHOLD = 0.5
TRANSFORMER_UNAVAILABLE = -1.0

CANDIDATE_FIELDS = [
    "order",
    "query",
    "identity",
    "coverage",
    "PLASMe",
    "overlap",
]

OUTPUT_FIELDS = [
    "contig_id",
    "input_header",
    "length",
    "raw_candidate_present",
    "raw_order",
    "raw_identity",
    "raw_coverage",
    "raw_plasme_score",
    "raw_overlap",
    "raw_positive_fasta_present",
    "predicted_label",
    "prediction_status",
    "identity_threshold",
    "coverage_threshold",
    "probability_threshold",
    "source_tool",
    "source_version",
    "source_commit",
    "container_image_id",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
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
    """Load ordered FASTA records without changing identifiers."""

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
            raise ValueError(f"{source_name} FASTA record has no sequence: " f"{contig_id}")

        observed[contig_id] = full_header
        records.append(
            {
                "contig_id": contig_id,
                "input_header": full_header,
                "length": len(sequence),
                "sequence": sequence,
            }
        )

    with path.open() as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                finalize_record()

                full_header = line[1:].strip()
                sequence_parts = []

                if not full_header:
                    raise ValueError(f"Empty {source_name} FASTA header " f"at line {line_number}")

                continue

            if full_header is None:
                raise ValueError(
                    f"{source_name} FASTA sequence data appears "
                    f"before the first header at line {line_number}"
                )

            sequence_parts.append("".join(line.split()))

    finalize_record()

    if not records and not allow_empty:
        raise ValueError(f"No FASTA records found in {path}")

    return records


def parse_unit_interval(
    raw_value: str,
    *,
    field: str,
    raw_identifier: str,
) -> float:
    """Parse a finite numeric value in the closed unit interval."""

    text = raw_value.strip()

    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"Invalid PLASMe {field} for " f"{raw_identifier!r}: {text!r}") from error

    if not math.isfinite(value):
        raise ValueError(f"Non-finite PLASMe {field} for " f"{raw_identifier!r}: {text!r}")

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"PLASMe {field} outside [0,1] for " f"{raw_identifier!r}: {text!r}")

    return value


def parse_plasme_score(
    raw_value: str,
    *,
    raw_identifier: str,
) -> float:
    """Parse a PLASMe score or the native -1 sentinel."""

    text = raw_value.strip()

    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(
            f"Invalid PLASMe transformer score for " f"{raw_identifier!r}: {text!r}"
        ) from error

    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite PLASMe transformer score for " f"{raw_identifier!r}: {text!r}"
        )

    if value == TRANSFORMER_UNAVAILABLE:
        return value

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"PLASMe transformer score outside [0,1] " f"for {raw_identifier!r}: {text!r}"
        )

    return value


def load_candidate_rows(
    path: Path,
) -> dict[str, dict[str, str | float]]:
    """Read and validate the official PLASMe candidate CSV."""

    if not path.is_file():
        raise FileNotFoundError(f"PLASMe candidate CSV does not exist: {path}")

    rows: dict[str, dict[str, str | float]] = {}

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != CANDIDATE_FIELDS:
            raise ValueError(
                "Unexpected PLASMe candidate columns: "
                f"{reader.fieldnames!r}; expected "
                f"{CANDIDATE_FIELDS!r}"
            )

        for line_number, raw_row in enumerate(
            reader,
            start=2,
        ):
            raw_identifier = (raw_row.get("query") or "").strip()

            if not raw_identifier:
                raise ValueError(f"{path}:{line_number}: empty PLASMe query")

            contig_id = canonical_id(raw_identifier)

            if contig_id in rows:
                previous = rows[contig_id]["raw_tool_contig_id"]

                raise ValueError(
                    "Duplicate canonical PLASMe candidate "
                    f"identifier: {contig_id!r} from "
                    f"{previous!r} and {raw_identifier!r}"
                )

            order = (raw_row.get("order") or "").strip()

            if not order:
                raise ValueError(
                    f"{path}:{line_number}: empty PLASMe order " f"for {raw_identifier!r}"
                )

            identity = parse_unit_interval(
                raw_row.get("identity") or "",
                field="identity",
                raw_identifier=raw_identifier,
            )
            coverage = parse_unit_interval(
                raw_row.get("coverage") or "",
                field="coverage",
                raw_identifier=raw_identifier,
            )
            score = parse_plasme_score(
                raw_row.get("PLASMe") or "",
                raw_identifier=raw_identifier,
            )

            rows[contig_id] = {
                "raw_tool_contig_id": raw_identifier,
                "order": order,
                "identity": identity,
                "coverage": coverage,
                "plasme_score": score,
                "overlap": (raw_row.get("overlap") or "").strip(),
            }

    return rows


def format_float(value: float) -> str:
    """Format a finite number without material rounding."""

    return f"{value:.17g}"


def recomputed_positive(
    candidate: dict[str, str | float] | None,
) -> bool:
    """Recompute the frozen official balance-preset decision."""

    if candidate is None:
        return False

    identity = float(candidate["identity"])
    coverage = float(candidate["coverage"])
    score = float(candidate["plasme_score"])

    alignment_positive = identity >= IDENTITY_THRESHOLD and coverage >= COVERAGE_THRESHOLD

    transformer_positive = score != TRANSFORMER_UNAVAILABLE and score > PROBABILITY_THRESHOLD

    return alignment_positive or transformer_positive


def adapt_plasme(
    input_fasta: Path,
    positive_fasta: Path,
    candidate_csv: Path,
    output_path: Path,
    metadata_output: Path | None = None,
) -> dict[str, Any]:
    """Create a complete standardized PLASMe table."""

    input_records = load_fasta_records(
        input_fasta,
        allow_empty=False,
        source_name="input",
    )
    positive_records = load_fasta_records(
        positive_fasta,
        allow_empty=True,
        source_name="positive-output",
    )
    candidates = load_candidate_rows(candidate_csv)

    input_by_id = {str(record["contig_id"]): record for record in input_records}
    positive_by_id = {str(record["contig_id"]): record for record in positive_records}

    input_ids = set(input_by_id)
    positive_ids = set(positive_by_id)
    candidate_ids = set(candidates)

    extra_positive = sorted(positive_ids - input_ids)
    extra_candidates = sorted(candidate_ids - input_ids)

    if extra_positive:
        raise ValueError(
            "PLASMe positive FASTA contains identifiers "
            "absent from the input FASTA: " + ", ".join(extra_positive[:20])
        )

    if extra_candidates:
        raise ValueError(
            "PLASMe candidate CSV contains identifiers "
            "absent from the input FASTA: " + ", ".join(extra_candidates[:20])
        )

    for contig_id, positive in positive_by_id.items():
        source = input_by_id[contig_id]

        if str(positive["sequence"]).upper() != str(source["sequence"]).upper():
            raise ValueError(
                "PLASMe positive FASTA sequence does not " f"match the input for {contig_id!r}"
            )

    standardized: list[dict[str, str]] = []
    decision_paths: Counter[str] = Counter()

    for record in input_records:
        contig_id = str(record["contig_id"])
        candidate = candidates.get(contig_id)

        official_positive = contig_id in positive_by_id
        expected_positive = recomputed_positive(candidate)

        if official_positive != expected_positive:
            raise ValueError(
                "PLASMe official positive FASTA disagrees "
                "with the frozen balance-preset decision "
                f"for {contig_id!r}: official="
                f"{official_positive}, recomputed="
                f"{expected_positive}"
            )

        if candidate is None:
            raw_candidate_present = "false"
            raw_order = ""
            raw_identity = ""
            raw_coverage = ""
            raw_score = ""
            raw_overlap = ""
            decision_paths["no_candidate"] += 1
        else:
            raw_candidate_present = "true"
            raw_order = str(candidate["order"])
            raw_identity = format_float(float(candidate["identity"]))
            raw_coverage = format_float(float(candidate["coverage"]))
            raw_score = format_float(float(candidate["plasme_score"]))
            raw_overlap = str(candidate["overlap"])

            alignment_positive = (
                float(candidate["identity"]) >= IDENTITY_THRESHOLD
                and float(candidate["coverage"]) >= COVERAGE_THRESHOLD
            )
            transformer_positive = (
                float(candidate["plasme_score"]) != TRANSFORMER_UNAVAILABLE
                and float(candidate["plasme_score"]) > PROBABILITY_THRESHOLD
            )

            if alignment_positive and transformer_positive:
                decision_paths["alignment_and_transformer"] += 1
            elif alignment_positive:
                decision_paths["alignment_only"] += 1
            elif transformer_positive:
                decision_paths["transformer_only"] += 1
            else:
                decision_paths["called_non_plasmid"] += 1

        if official_positive:
            predicted_label = "plasmid"
            prediction_status = "called_plasmid"
        else:
            predicted_label = "non-plasmid"
            prediction_status = "called_non_plasmid"

        standardized.append(
            {
                "contig_id": contig_id,
                "input_header": str(record["input_header"]),
                "length": str(record["length"]),
                "raw_candidate_present": (raw_candidate_present),
                "raw_order": raw_order,
                "raw_identity": raw_identity,
                "raw_coverage": raw_coverage,
                "raw_plasme_score": raw_score,
                "raw_overlap": raw_overlap,
                "raw_positive_fasta_present": ("true" if official_positive else "false"),
                "predicted_label": predicted_label,
                "prediction_status": (prediction_status),
                "identity_threshold": (format_float(IDENTITY_THRESHOLD)),
                "coverage_threshold": (format_float(COVERAGE_THRESHOLD)),
                "probability_threshold": (format_float(PROBABILITY_THRESHOLD)),
                "source_tool": TOOL_NAME,
                "source_version": TOOL_VERSION,
                "source_commit": SOURCE_COMMIT,
                "container_image_id": (CONTAINER_IMAGE_ID),
            }
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
    ) as handle:
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

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "source_commit": SOURCE_COMMIT,
        "container_image_id": (CONTAINER_IMAGE_ID),
        "official_preset": "balance",
        "input_records": len(input_records),
        "positive_fasta_records": len(positive_records),
        "candidate_rows": len(candidates),
        "standardized_rows": len(standardized),
        "complete_input_coverage": (len(standardized) == len(input_records)),
        "label_counts": dict(sorted(label_counts.items())),
        "prediction_status_counts": dict(sorted(status_counts.items())),
        "decision_path_counts": dict(sorted(decision_paths.items())),
        "identity_threshold": (IDENTITY_THRESHOLD),
        "coverage_threshold": (COVERAGE_THRESHOLD),
        "probability_threshold": (PROBABILITY_THRESHOLD),
        "native_negative_label": ("non-plasmid"),
        "three_class_claim": False,
        "input_fasta_sha256": sha256_file(input_fasta),
        "positive_fasta_sha256": (sha256_file(positive_fasta)),
        "candidate_csv_sha256": (sha256_file(candidate_csv)),
        "standardized_output_sha256": (sha256_file(output_path)),
        "manuscript_only": True,
        "production_workflow_component": False,
    }

    if metadata_output is not None:
        metadata_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        metadata_output.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Create the manuscript-only adapter CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Normalize official PLASMe 1.1 output "
            "for the frozen manuscript benchmark. "
            "This command is not part of the "
            "PlasFlow2 prediction workflow."
        )
    )
    parser.add_argument(
        "--input-fasta",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--positive-fasta",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-csv",
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

    return parser


def main() -> None:
    """Run the PLASMe adapter CLI."""

    args = build_parser().parse_args()

    metadata = adapt_plasme(
        input_fasta=args.input_fasta,
        positive_fasta=args.positive_fasta,
        candidate_csv=args.candidate_csv,
        output_path=args.output,
        metadata_output=args.metadata_output,
    )

    print(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
