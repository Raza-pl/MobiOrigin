#!/usr/bin/env python3
"""Normalize MOB-recon output for the frozen NAR comparator benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

TOOL_NAME = "MOB-recon"
TOOL_VERSION = "3.1.9"
SCHEMA_VERSION = "nar-comparator-adapter-v1"

OUTPUT_FIELDS = [
    "contig_id",
    "input_header",
    "raw_tool_contig_id",
    "predicted_label",
    "prediction_status",
    "raw_molecule_type",
    "primary_cluster_id",
    "secondary_cluster_id",
    "rep_type",
    "relaxase_type",
    "mash_nearest_neighbor",
    "mash_neighbor_distance",
    "source_tool",
    "source_version",
]


def canonical_id(header: str) -> str:
    """Return the first whitespace-delimited FASTA identifier token."""
    value = header.strip()
    if not value:
        raise ValueError("Empty contig identifier is not allowed")
    return value.split()[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_fasta_headers(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seen: dict[str, str] = {}

    with path.open() as handle:
        for line in handle:
            if not line.startswith(">"):
                continue

            full_header = line[1:].strip()
            contig_id = canonical_id(full_header)

            if contig_id in seen:
                raise ValueError(
                    "Canonical FASTA identifier collision: "
                    f"{contig_id!r} maps to both "
                    f"{seen[contig_id]!r} and {full_header!r}"
                )

            seen[contig_id] = full_header
            records.append((contig_id, full_header))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")

    return records


def load_mob_report(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])

        required = {"contig_id", "molecule_type"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                "MOB-recon report is missing required columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            raw_id = (row.get("contig_id") or "").strip()
            contig_id = canonical_id(raw_id)

            if contig_id in rows:
                previous = rows[contig_id].get("contig_id", "")
                raise ValueError(
                    "Duplicate canonical MOB-recon identifier: "
                    f"{contig_id!r} from {previous!r} and {raw_id!r}"
                )

            rows[contig_id] = row

    return rows


def normalize_molecule_type(value: str) -> tuple[str, str]:
    molecule = value.strip().lower()

    if molecule == "plasmid":
        return "plasmid", "called"
    if molecule in {"chromosome", "chromosomal"}:
        return "chromosome", "called"

    return "unclassified", "unsupported_molecule_type"


def adapt_mob_recon(
    input_fasta: Path,
    contig_report: Path,
    output_path: Path,
    metadata_output: Path | None = None,
) -> dict[str, Any]:
    """Create a complete, deterministic standardized prediction table."""
    fasta_records = load_fasta_headers(input_fasta)
    input_ids = {contig_id for contig_id, _ in fasta_records}
    tool_rows = load_mob_report(contig_report)

    extra_ids = sorted(set(tool_rows) - input_ids)
    if extra_ids:
        raise ValueError(
            "MOB-recon output contains identifiers absent from the input: "
            + ", ".join(extra_ids[:20])
        )

    standardized: list[dict[str, str]] = []
    missing_ids: list[str] = []
    unsupported_ids: list[str] = []

    for contig_id, full_header in fasta_records:
        source = tool_rows.get(contig_id)

        if source is None:
            missing_ids.append(contig_id)
            standardized.append(
                {
                    "contig_id": contig_id,
                    "input_header": full_header,
                    "raw_tool_contig_id": "",
                    "predicted_label": "unclassified",
                    "prediction_status": "missing_output",
                    "raw_molecule_type": "",
                    "primary_cluster_id": "",
                    "secondary_cluster_id": "",
                    "rep_type": "",
                    "relaxase_type": "",
                    "mash_nearest_neighbor": "",
                    "mash_neighbor_distance": "",
                    "source_tool": TOOL_NAME,
                    "source_version": TOOL_VERSION,
                }
            )
            continue

        raw_molecule = (source.get("molecule_type") or "").strip()
        predicted_label, prediction_status = normalize_molecule_type(raw_molecule)

        if prediction_status == "unsupported_molecule_type":
            unsupported_ids.append(contig_id)

        standardized.append(
            {
                "contig_id": contig_id,
                "input_header": full_header,
                "raw_tool_contig_id": (source.get("contig_id") or "").strip(),
                "predicted_label": predicted_label,
                "prediction_status": prediction_status,
                "raw_molecule_type": raw_molecule,
                "primary_cluster_id": (source.get("primary_cluster_id") or "").strip(),
                "secondary_cluster_id": (source.get("secondary_cluster_id") or "").strip(),
                "rep_type": (source.get("rep_type(s)") or "").strip(),
                "relaxase_type": (source.get("relaxase_type(s)") or "").strip(),
                "mash_nearest_neighbor": (source.get("mash_nearest_neighbor") or "").strip(),
                "mash_neighbor_distance": (source.get("mash_neighbor_distance") or "").strip(),
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
        "source_tool": TOOL_NAME,
        "source_version": TOOL_VERSION,
        "input_fasta": str(input_fasta),
        "input_fasta_sha256": sha256_file(input_fasta),
        "source_report": str(contig_report),
        "source_report_sha256": sha256_file(contig_report),
        "standardized_output": str(output_path),
        "standardized_output_sha256": sha256_file(output_path),
        "input_sequences": len(fasta_records),
        "source_report_rows": len(tool_rows),
        "standardized_rows": len(standardized),
        "label_counts": dict(sorted(label_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_output_ids": missing_ids,
        "unsupported_molecule_type_ids": unsupported_ids,
        "identifier_contract": {
            "raw_identifier": ("Complete MOB-recon contig_id retained in " "raw_tool_contig_id."),
            "canonical_identifier": ("First whitespace-delimited FASTA token."),
            "collision_policy": "Reject canonical identifier collisions.",
        },
        "missing_output_policy": "Map to unclassified.",
        "unsupported_molecule_type_policy": "Map to unclassified.",
        "continuous_score_available": False,
    }

    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize MOB-recon contig_report.txt into the frozen " "NAR comparator schema."
        )
    )
    parser.add_argument(
        "--input-fasta",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--contig-report",
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
    metadata = adapt_mob_recon(
        input_fasta=args.input_fasta,
        contig_report=args.contig_report,
        output_path=args.output,
        metadata_output=args.metadata_output,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
