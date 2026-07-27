#!/usr/bin/env python3
"""Normalize geNomad 1.12.0 output for the frozen NAR benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOOL_NAME = "geNomad"
TOOL_VERSION = "1.12.0"
DATABASE_VERSION = "1.9"
DATABASE_FINGERPRINT = "1a23156892a2ee1aa149641b39f65bfb5c7a9fe8ed6c9647dc0b9fb26633677d"
CONTRACT_SHA256 = "cd5f4cbab615d931c35470df24640950c489e4d8424c523ed54eedecf35bfdee"
SCHEMA_VERSION = "nar-comparator-adapter-v1"

OUTPUT_FIELDS = [
    "contig_id",
    "input_header",
    "raw_plasmid_id",
    "raw_virus_ids",
    "predicted_label",
    "prediction_status",
    "chromosome_score",
    "plasmid_score",
    "virus_score",
    "plasmid_fdr",
    "virus_fdr",
    "plasmid_n_hallmarks",
    "virus_n_hallmarks",
    "plasmid_marker_enrichment",
    "virus_marker_enrichment",
    "source_tool",
    "source_version",
]


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(header: str) -> str:
    value = header.strip()
    if not value:
        raise ValueError("Empty sequence identifier")
    return value.split()[0]


def parent_id(raw_id: str) -> str:
    token = canonical_id(raw_id)
    return token.split("|provirus_", 1)[0]


def load_fasta_headers(file_path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    observed: dict[str, str] = {}

    with file_path.open() as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            full_header = line[1:].strip()
            contig_id = canonical_id(full_header)
            if contig_id in observed:
                raise ValueError("Canonical FASTA identifier collision: " f"{contig_id!r}")
            observed[contig_id] = full_header
            records.append((contig_id, full_header))

    if not records:
        raise ValueError(f"No FASTA records found in {file_path}")
    return records


def load_tsv(
    file_path: Path,
    required_columns: set[str],
) -> list[dict[str, str]]:
    if not file_path.is_file():
        raise FileNotFoundError(f"Required geNomad output missing: {file_path}")

    rows: list[dict[str, str]] = []
    observed_ids: set[str] = set()

    with file_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{file_path} is missing columns: " + ", ".join(sorted(missing)))

        for row in reader:
            raw_id = (row.get("seq_name") or "").strip()
            if not raw_id:
                raise ValueError(f"{file_path} contains an empty seq_name")
            if raw_id in observed_ids:
                raise ValueError(f"Duplicate geNomad identifier in {file_path}: {raw_id}")
            observed_ids.add(raw_id)
            rows.append(row)

    return rows


def validated_number(
    value: str | None,
    field: str,
    raw_id: str,
    *,
    required: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    text = (value or "").strip()
    if not text or text.upper() == "NA":
        if required:
            raise ValueError(f"Missing {field} for {raw_id}")
        return None

    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"Invalid {field} for {raw_id}: {text}") from error

    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field} for {raw_id}: {text}")
    if minimum is not None and number < minimum:
        raise ValueError(f"Out-of-range {field} for {raw_id}: {text}")
    if maximum is not None and number > maximum:
        raise ValueError(f"Out-of-range {field} for {raw_id}: {text}")
    return number


def formatted(number: float | None) -> str:
    return "" if number is None else f"{number:.10g}"


def grouped_rows(
    rows: list[dict[str, str]],
    input_ids: set[str],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        raw_id = (row.get("seq_name") or "").strip()
        mapped_id = parent_id(raw_id)
        if mapped_id not in input_ids:
            raise ValueError("geNomad output identifier is absent from input FASTA: " f"{raw_id}")
        grouped[mapped_id].append(row)

    return dict(grouped)


def numeric_values(
    rows: list[dict[str, str]],
    field: str,
    *,
    required: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw_id = (row.get("seq_name") or "").strip()
        value = validated_number(
            row.get(field),
            field,
            raw_id,
            required=required,
            minimum=minimum,
            maximum=maximum,
        )
        if value is not None:
            values.append(value)
    return values


def maximum_value(
    rows: list[dict[str, str]],
    field: str,
    **kwargs: Any,
) -> str:
    values = numeric_values(rows, field, **kwargs)
    return formatted(max(values) if values else None)


def minimum_value(
    rows: list[dict[str, str]],
    field: str,
    **kwargs: Any,
) -> str:
    values = numeric_values(rows, field, **kwargs)
    return formatted(min(values) if values else None)


def validate_rows(
    plasmid_rows: list[dict[str, str]],
    virus_rows: list[dict[str, str]],
    score_rows: list[dict[str, str]],
) -> None:
    for row in plasmid_rows:
        raw_id = row["seq_name"]
        validated_number(
            row.get("plasmid_score"),
            "plasmid_score",
            raw_id,
            required=True,
            minimum=0.0,
            maximum=1.0,
        )
        validated_number(row.get("fdr"), "fdr", raw_id, minimum=0.0, maximum=1.0)
        validated_number(row.get("n_hallmarks"), "n_hallmarks", raw_id, minimum=0.0)
        validated_number(
            row.get("marker_enrichment"),
            "marker_enrichment",
            raw_id,
            minimum=0.0,
        )

    for row in virus_rows:
        raw_id = row["seq_name"]
        validated_number(
            row.get("virus_score"),
            "virus_score",
            raw_id,
            required=True,
            minimum=0.0,
            maximum=1.0,
        )
        validated_number(row.get("fdr"), "fdr", raw_id, minimum=0.0, maximum=1.0)
        validated_number(row.get("n_hallmarks"), "n_hallmarks", raw_id, minimum=0.0)
        validated_number(
            row.get("marker_enrichment"),
            "marker_enrichment",
            raw_id,
            minimum=0.0,
        )

    for row in score_rows:
        raw_id = row["seq_name"]
        for field in (
            "chromosome_score",
            "plasmid_score",
            "virus_score",
        ):
            validated_number(
                row.get(field),
                field,
                raw_id,
                required=True,
                minimum=0.0,
                maximum=1.0,
            )


def adapt_genomad(
    input_fasta: Path,
    plasmid_summary: Path,
    virus_summary: Path,
    calibrated_scores: Path,
    output_path: Path,
    metadata_output: Path | None = None,
) -> dict[str, Any]:
    fasta_records = load_fasta_headers(input_fasta)
    input_ids = {contig_id for contig_id, _ in fasta_records}

    plasmid_rows = load_tsv(
        plasmid_summary,
        {
            "seq_name",
            "plasmid_score",
            "fdr",
            "n_hallmarks",
            "marker_enrichment",
        },
    )
    virus_rows = load_tsv(
        virus_summary,
        {
            "seq_name",
            "virus_score",
            "fdr",
            "n_hallmarks",
            "marker_enrichment",
        },
    )
    score_rows = load_tsv(
        calibrated_scores,
        {
            "seq_name",
            "chromosome_score",
            "plasmid_score",
            "virus_score",
        },
    )

    validate_rows(plasmid_rows, virus_rows, score_rows)

    plasmids = grouped_rows(plasmid_rows, input_ids)
    viruses = grouped_rows(virus_rows, input_ids)
    scores = grouped_rows(score_rows, input_ids)

    standardized: list[dict[str, str]] = []

    for contig_id, input_header in fasta_records:
        p_rows = plasmids.get(contig_id, [])
        v_rows = viruses.get(contig_id, [])
        s_rows = scores.get(contig_id, [])

        if p_rows and v_rows:
            label = "unclassified"
            prediction_status = "ambiguous_dual_call"
        elif p_rows:
            label = "plasmid"
            prediction_status = "called_plasmid"
        elif v_rows:
            label = "phage"
            prediction_status = "called_phage"
        else:
            label = "chromosome"
            prediction_status = "not_detected"

        standardized.append(
            {
                "contig_id": contig_id,
                "input_header": input_header,
                "raw_plasmid_id": ";".join(sorted(row["seq_name"] for row in p_rows)),
                "raw_virus_ids": ";".join(sorted(row["seq_name"] for row in v_rows)),
                "predicted_label": label,
                "prediction_status": prediction_status,
                "chromosome_score": maximum_value(
                    s_rows,
                    "chromosome_score",
                    required=True,
                    minimum=0.0,
                    maximum=1.0,
                ),
                "plasmid_score": maximum_value(
                    s_rows,
                    "plasmid_score",
                    required=True,
                    minimum=0.0,
                    maximum=1.0,
                ),
                "virus_score": maximum_value(
                    s_rows,
                    "virus_score",
                    required=True,
                    minimum=0.0,
                    maximum=1.0,
                ),
                "plasmid_fdr": minimum_value(
                    p_rows,
                    "fdr",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "virus_fdr": minimum_value(
                    v_rows,
                    "fdr",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "plasmid_n_hallmarks": maximum_value(p_rows, "n_hallmarks", minimum=0.0),
                "virus_n_hallmarks": maximum_value(v_rows, "n_hallmarks", minimum=0.0),
                "plasmid_marker_enrichment": maximum_value(
                    p_rows, "marker_enrichment", minimum=0.0
                ),
                "virus_marker_enrichment": maximum_value(v_rows, "marker_enrichment", minimum=0.0),
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

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "source_tool": TOOL_NAME,
        "source_version": TOOL_VERSION,
        "database_version": DATABASE_VERSION,
        "database_fingerprint_sha256": DATABASE_FINGERPRINT,
        "input_fasta": str(input_fasta),
        "input_fasta_sha256": sha256_file(input_fasta),
        "plasmid_summary_sha256": sha256_file(plasmid_summary),
        "virus_summary_sha256": sha256_file(virus_summary),
        "calibrated_scores_sha256": sha256_file(calibrated_scores),
        "standardized_output": str(output_path),
        "standardized_output_sha256": sha256_file(output_path),
        "input_sequences": len(fasta_records),
        "plasmid_summary_rows": len(plasmid_rows),
        "virus_summary_rows": len(virus_rows),
        "calibrated_score_rows": len(score_rows),
        "standardized_rows": len(standardized),
        "label_counts": dict(sorted(label_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_summary_policy": "Fail closed.",
        "not_detected_policy": "Map to chromosome.",
        "dual_call_policy": "Map to unclassified.",
        "provirus_policy": "Map child identifier to input parent.",
        "confirmatory_tuning": False,
    }

    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Normalize geNomad 1.12.0 output into the frozen " "NAR comparator schema.")
    )
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--plasmid-summary", type=Path, required=True)
    parser.add_argument("--virus-summary", type=Path, required=True)
    parser.add_argument("--calibrated-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = adapt_genomad(
        input_fasta=args.input_fasta,
        plasmid_summary=args.plasmid_summary,
        virus_summary=args.virus_summary,
        calibrated_scores=args.calibrated_scores,
        output_path=args.output,
        metadata_output=args.metadata_output,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
