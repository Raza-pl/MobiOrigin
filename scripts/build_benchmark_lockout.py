#!/usr/bin/env python3
"""Build a reproducible source-level lockout for final classifier evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from plasflow2.classify.splits import source_group_id


def _hash_fraction(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def build_lockout(
    ids_path: Path,
    labels_path: Path,
    benchmark_label_paths: list[Path],
    output_groups_path: Path,
    phage_manifest_path: Path,
    summary_path: Path,
    *,
    phage_dev_manifest_path: Path | None = None,
    phage_final_manifest_path: Path | None = None,
    phage_label: int = 2,
    phage_fraction: float = 0.10,
    phage_dev_fraction: float = 0.50,
    seed: int = 42,
) -> dict[str, object]:
    """Reserve all external benchmark sources plus a deterministic phage subset."""

    if not 0.0 <= phage_fraction <= 1.0:
        raise ValueError("phage_fraction must be between 0 and 1")
    if not 0.0 <= phage_dev_fraction <= 1.0:
        raise ValueError("phage_dev_fraction must be between 0 and 1")

    sequence_ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    labels = np.load(labels_path, mmap_mode="r")
    if len(sequence_ids) != len(labels):
        raise ValueError(
            f"Row mismatch: {len(sequence_ids)} sequence IDs versus {len(labels)} labels"
        )

    external_groups: set[str] = set()
    for path in benchmark_label_paths:
        with path.open() as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if "source_accession" not in (reader.fieldnames or []):
                raise ValueError(f"{path} does not contain a source_accession column")
            external_groups.update(
                row["source_accession"].strip() for row in reader if row["source_accession"].strip()
            )

    groups = [source_group_id(sequence_id) for sequence_id in sequence_ids]
    phage_group_counts = Counter(
        group for group, label in zip(groups, labels) if int(label) == phage_label
    )
    locked_phage_groups = {
        group for group in phage_group_counts if _hash_fraction(group, seed) < phage_fraction
    }
    phage_dev_groups = {
        group
        for group in locked_phage_groups
        if _hash_fraction(group, seed + 1_000_003) < phage_dev_fraction
    }
    phage_final_groups = locked_phage_groups - phage_dev_groups
    locked_groups = external_groups | locked_phage_groups
    locked_training_rows = [i for i, group in enumerate(groups) if group in locked_groups]
    locked_phage_rows = [
        i
        for i, (group, label) in enumerate(zip(groups, labels))
        if group in locked_phage_groups and int(label) == phage_label
    ]

    output_groups_path.parent.mkdir(parents=True, exist_ok=True)
    output_groups_path.write_text("\n".join(sorted(locked_groups)) + "\n")

    def write_phage_manifest(path: Path, selected_groups: set[str]) -> int:
        selected_rows = [
            i
            for i in locked_phage_rows
            if groups[i] in selected_groups
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(["row_index", "sequence_id", "source_group", "label"])
            for i in selected_rows:
                writer.writerow([i, sequence_ids[i], groups[i], int(labels[i])])
        return len(selected_rows)

    write_phage_manifest(phage_manifest_path, locked_phage_groups)
    phage_dev_rows = (
        write_phage_manifest(phage_dev_manifest_path, phage_dev_groups)
        if phage_dev_manifest_path is not None
        else 0
    )
    phage_final_rows = (
        write_phage_manifest(phage_final_manifest_path, phage_final_groups)
        if phage_final_manifest_path is not None
        else 0
    )

    remaining_class_counts = Counter(
        int(label) for group, label in zip(groups, labels) if group not in locked_groups
    )
    summary: dict[str, object] = {
        "seed": seed,
        "phage_fraction": phage_fraction,
        "external_groups": len(external_groups),
        "locked_phage_groups": len(locked_phage_groups),
        "locked_phage_rows": len(locked_phage_rows),
        "phage_dev_fraction": phage_dev_fraction,
        "phage_dev_groups": len(phage_dev_groups),
        "phage_dev_rows": phage_dev_rows,
        "phage_final_groups": len(phage_final_groups),
        "phage_final_rows": phage_final_rows,
        "locked_training_rows_total": len(locked_training_rows),
        "remaining_training_rows": len(labels) - len(locked_training_rows),
        "remaining_class_counts": dict(sorted(remaining_class_counts.items())),
        "benchmark_label_files": [str(path) for path in benchmark_label_paths],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--benchmark-labels",
        type=Path,
        nargs="+",
        required=True,
        help="Benchmark label TSV files containing source_accession.",
    )
    parser.add_argument("--out-groups", type=Path, required=True)
    parser.add_argument("--phage-manifest", type=Path, required=True)
    parser.add_argument("--phage-dev-manifest", type=Path)
    parser.add_argument("--phage-final-manifest", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--phage-label", type=int, default=2)
    parser.add_argument("--phage-fraction", type=float, default=0.10)
    parser.add_argument("--phage-dev-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_lockout(
        args.ids,
        args.labels,
        args.benchmark_labels,
        args.out_groups,
        args.phage_manifest,
        args.summary,
        phage_dev_manifest_path=args.phage_dev_manifest,
        phage_final_manifest_path=args.phage_final_manifest,
        phage_label=args.phage_label,
        phage_fraction=args.phage_fraction,
        phage_dev_fraction=args.phage_dev_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
