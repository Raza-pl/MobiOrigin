#!/usr/bin/env python3
"""Normalize PlasClass 0.1 output for the frozen NAR benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

TOOL_NAME = "PlasClass"
TOOL_VERSION = "0.1"
SCHEMA_VERSION = "nar-comparator-adapter-v1"
CONTRACT_SHA256 = "70e984280cffacea651945f1dddedee1dc948b4f2f4787191caaafd074a79adf"
DECISION_THRESHOLD = 0.5

OUTPUT_FIELDS = [
    "contig_id",
    "input_header",
    "length",
    "model_scale",
    "raw_tool_contig_id",
    "predicted_label",
    "prediction_status",
    "plasmid_score",
    "decision_threshold",
    "source_tool",
    "source_version",
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
    """Return the first whitespace-delimited identifier token."""

    value = header.strip()
    if not value:
        raise ValueError("Empty sequence identifier")
    return value.split()[0]


def model_scale(length: int) -> int:
    """Return the official PlasClass model scale for a sequence length."""

    if length < 0:
        raise ValueError("Sequence length cannot be negative")
    if length <= 5_500:
        return 1_000
    if length <= 55_000:
        return 10_000
    if length <= 300_000:
        return 100_000
    return 500_000


def load_fasta_records(
    path: Path,
) -> list[dict[str, str | int]]:
    """Load ordered FASTA identifiers, full headers, and lengths."""

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
                "model_scale": model_scale(sequence_length),
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

                if not full_header:
                    raise ValueError(f"Empty FASTA header at line {line_number}")

                sequence_length = 0
                continue

            if full_header is None:
                raise ValueError(
                    "Sequence data appears before the first FASTA " f"header at line {line_number}"
                )

            sequence_length += len(line)

    finalize_record()

    if not records:
        raise ValueError(f"No FASTA records found in {path}")

    return records


def parse_score(raw_value: str, raw_identifier: str) -> float:
    """Validate and return one PlasClass probability."""

    text = raw_value.strip()

    if not text:
        raise ValueError(f"Missing PlasClass score for {raw_identifier}")

    try:
        score = float(text)
    except ValueError as error:
        raise ValueError(f"Invalid PlasClass score for {raw_identifier}: {text}") from error

    if not math.isfinite(score):
        raise ValueError(f"Non-finite PlasClass score for {raw_identifier}: {text}")

    if not 0.0 <= score <= 1.0:
        raise ValueError(f"PlasClass score outside [0,1] for " f"{raw_identifier}: {text}")

    return score


def load_raw_scores(
    path: Path,
) -> dict[str, dict[str, str | float]]:
    """Read official two-column, headerless PlasClass output."""

    if not path.is_file():
        raise FileNotFoundError(f"PlasClass score output does not exist: {path}")

    rows: dict[str, dict[str, str | float]] = {}

    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")

            if not line.strip():
                continue

            columns = line.split("\t")

            if len(columns) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected exactly two "
                    f"tab-separated columns, found {len(columns)}"
                )

            raw_identifier = columns[0].strip()

            if not raw_identifier:
                raise ValueError(f"{path}:{line_number}: empty PlasClass identifier")

            contig_id = canonical_id(raw_identifier)
            score = parse_score(columns[1], raw_identifier)

            if contig_id in rows:
                previous = rows[contig_id]["raw_tool_contig_id"]
                raise ValueError(
                    "Duplicate canonical PlasClass identifier: "
                    f"{contig_id!r} from {previous!r} and "
                    f"{raw_identifier!r}"
                )

            rows[contig_id] = {
                "raw_tool_contig_id": raw_identifier,
                "score": score,
            }

    return rows


def formatted_score(score: float) -> str:
    """Format a probability deterministically without material rounding."""

    return f"{score:.17g}"


def adapt_plasclass(
    input_fasta: Path,
    raw_scores: Path,
    output_path: Path,
    metadata_output: Path | None = None,
) -> dict[str, Any]:
    """Create a complete standardized PlasClass prediction table."""

    fasta_records = load_fasta_records(input_fasta)
    score_rows = load_raw_scores(raw_scores)

    input_ids = {str(record["contig_id"]) for record in fasta_records}

    extra_ids = sorted(set(score_rows) - input_ids)

    if extra_ids:
        raise ValueError(
            "PlasClass output contains identifiers absent from the "
            "input FASTA: " + ", ".join(extra_ids[:20])
        )

    standardized: list[dict[str, str]] = []
    missing_ids: list[str] = []

    for record in fasta_records:
        contig_id = str(record["contig_id"])
        source = score_rows.get(contig_id)

        if source is None:
            missing_ids.append(contig_id)
            standardized.append(
                {
                    "contig_id": contig_id,
                    "input_header": str(record["input_header"]),
                    "length": str(record["length"]),
                    "model_scale": str(record["model_scale"]),
                    "raw_tool_contig_id": "",
                    "predicted_label": "unclassified",
                    "prediction_status": "missing_output",
                    "plasmid_score": "",
                    "decision_threshold": str(DECISION_THRESHOLD),
                    "source_tool": TOOL_NAME,
                    "source_version": TOOL_VERSION,
                }
            )
            continue

        score = float(source["score"])
        predicted_label = "plasmid" if score >= DECISION_THRESHOLD else "non-plasmid"
        prediction_status = (
            "called_plasmid" if predicted_label == "plasmid" else "called_non_plasmid"
        )

        standardized.append(
            {
                "contig_id": contig_id,
                "input_header": str(record["input_header"]),
                "length": str(record["length"]),
                "model_scale": str(record["model_scale"]),
                "raw_tool_contig_id": str(source["raw_tool_contig_id"]),
                "predicted_label": predicted_label,
                "prediction_status": prediction_status,
                "plasmid_score": formatted_score(score),
                "decision_threshold": str(DECISION_THRESHOLD),
                "source_tool": TOOL_NAME,
                "source_version": TOOL_VERSION,
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
    scale_counts = Counter(row["model_scale"] for row in standardized)

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "source_tool": TOOL_NAME,
        "source_version": TOOL_VERSION,
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_tuning_allowed": False,
        "negative_label_semantics": "binary non-plasmid",
        "input_fasta": str(input_fasta),
        "input_fasta_sha256": sha256_file(input_fasta),
        "raw_scores": str(raw_scores),
        "raw_scores_sha256": sha256_file(raw_scores),
        "standardized_output": str(output_path),
        "standardized_output_sha256": sha256_file(output_path),
        "input_sequences": len(fasta_records),
        "raw_score_rows": len(score_rows),
        "standardized_rows": len(standardized),
        "complete_output": not missing_ids,
        "runner_success_allowed": not missing_ids,
        "label_counts": dict(sorted(label_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "model_scale_counts": dict(sorted(scale_counts.items())),
        "missing_output_ids": missing_ids,
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
        },
        "continuous_score_available": True,
        "models_deserialized_by_adapter": False,
        "production_workflow_component": False,
    }

    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize official PlasClass 0.1 output into the "
            "frozen manuscript-only NAR comparator schema."
        )
    )
    parser.add_argument(
        "--input-fasta",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--raw-scores",
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
    args = parse_args()
    metadata = adapt_plasclass(
        input_fasta=args.input_fasta,
        raw_scores=args.raw_scores,
        output_path=args.output,
        metadata_output=args.metadata_output,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
