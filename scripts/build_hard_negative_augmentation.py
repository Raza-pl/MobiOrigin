#!/usr/bin/env python3
"""Build a source-balanced chromosome hard-negative feature augmentation."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
from plasflow2.classify.features import extract_features_to_npy
from plasflow2.classify.splits import source_group_id
from plasflow2.utils.fasta import iter_fasta


def build_hard_negative_augmentation(
    input_dir: Path,
    output_dir: Path,
    *,
    exclude_groups_path: Path | None = None,
    reference_ids_path: Path | None = None,
    reference_labels_path: Path | None = None,
    window_sizes: tuple[int, ...] = (1_000, 2_000, 5_000, 10_000, 20_000),
    rows_per_size: int = 8_000,
    max_windows_per_source_size: int = 250,
    max_file_bytes: int = 50_000_000,
    seed: int = 42,
    feature_chunk_size: int = 1_000,
) -> dict[str, object]:
    """Sample hard-negative windows without letting one genome dominate."""

    if rows_per_size <= 0 or max_windows_per_source_size <= 0:
        raise ValueError("sampling limits must be positive")
    if not window_sizes or any(window_size <= 0 for window_size in window_sizes):
        raise ValueError("window_sizes must contain positive values")

    excluded_groups = (
        {
            line.strip()
            for line in exclude_groups_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if exclude_groups_path is not None
        else set()
    )
    if (reference_ids_path is None) != (reference_labels_path is None):
        raise ValueError("reference_ids_path and reference_labels_path must be provided together")
    reference_ids: set[str] = set()
    reference_group_labels: dict[str, set[int]] = {}
    if reference_ids_path is not None and reference_labels_path is not None:
        reference_id_rows = [
            line.strip()
            for line in reference_ids_path.read_text().splitlines()
            if line.strip()
        ]
        reference_label_rows = np.load(reference_labels_path, mmap_mode="r")
        if len(reference_id_rows) != len(reference_label_rows):
            raise ValueError("Reference sequence IDs and labels have different row counts")
        reference_ids = set(reference_id_rows)
        for sequence_id, label in zip(reference_id_rows, reference_label_rows):
            reference_group_labels.setdefault(source_group_id(sequence_id), set()).add(int(label))
    rng = random.Random(seed)
    reservoirs: dict[int, list[tuple[str, str]]] = {
        window_size: [] for window_size in window_sizes
    }
    candidates_seen = Counter()
    processed_groups: set[str] = set()
    skipped_locked_groups: set[str] = set()
    skipped_conflicting_groups: set[str] = set()
    skipped_plasmid_header_groups: set[str] = set()
    duplicate_groups: set[str] = set()
    exact_id_overlaps = 0
    skipped_large_files: list[str] = []
    processed_files = 0

    def consider(window_size: int, item: tuple[str, str]) -> None:
        candidates_seen[window_size] += 1
        reservoir = reservoirs[window_size]
        seen = candidates_seen[window_size]
        if len(reservoir) < rows_per_size:
            reservoir.append(item)
            return
        replacement = rng.randrange(seen)
        if replacement < rows_per_size:
            reservoir[replacement] = item

    for fasta_path in sorted(input_dir.glob("*.fna")):
        if fasta_path.stat().st_size > max_file_bytes:
            skipped_large_files.append(str(fasta_path))
            continue
        processed_files += 1
        for record in iter_fasta(fasta_path):
            group = source_group_id(record.id)
            if "plasmid" in record.description.lower():
                skipped_plasmid_header_groups.add(group)
                continue
            if group in excluded_groups:
                skipped_locked_groups.add(group)
                continue
            if reference_group_labels.get(group, {1}) != {1}:
                skipped_conflicting_groups.add(group)
                continue
            if group in processed_groups:
                duplicate_groups.add(group)
                continue
            processed_groups.add(group)
            sequence = str(record.seq).upper()
            for window_size in window_sizes:
                if len(sequence) < window_size:
                    continue
                step = max(1, window_size // 2)
                positions = range(0, len(sequence) - window_size + 1, step)
                n_positions = len(positions)
                if n_positions > max_windows_per_source_size:
                    selected_positions = rng.sample(
                        positions,
                        max_windows_per_source_size,
                    )
                else:
                    selected_positions = positions
                for start in selected_positions:
                    fragment = sequence[start : start + window_size]
                    if set(fragment) <= {"A", "C", "G", "T", "N"}:
                        sequence_id = f"{group}_w{window_size}_s{start}"
                        if sequence_id in reference_ids:
                            exact_id_overlaps += 1
                            continue
                        consider(
                            window_size,
                            (sequence_id, fragment),
                        )

    sequences: list[str] = []
    sequence_ids: list[str] = []
    rows_by_size: dict[int, int] = {}
    for window_size in window_sizes:
        sampled = reservoirs[window_size]
        rng.shuffle(sampled)
        rows_by_size[window_size] = len(sampled)
        for sequence_id, sequence in sampled:
            sequence_ids.append(sequence_id)
            sequences.append(sequence)
    if not sequences:
        raise ValueError("No usable hard-negative windows were found")
    if len(sequence_ids) != len(set(sequence_ids)):
        raise RuntimeError("Hard-negative sampling produced duplicate sequence IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "seq_ids.txt"
    labels_path = output_dir / "labels.npy"
    ids_incomplete = output_dir / "seq_ids.txt.incomplete"
    labels_incomplete = output_dir / "labels.npy.incomplete"
    ids_incomplete.write_text("\n".join(sequence_ids) + "\n")
    with labels_incomplete.open("wb") as labels_fh:
        np.save(labels_fh, np.ones(len(sequence_ids), dtype=np.int64))
    os.replace(ids_incomplete, ids_path)
    os.replace(labels_incomplete, labels_path)
    feature_shape = extract_features_to_npy(
        sequences,
        output_dir / "features.npy",
        chunk_size=feature_chunk_size,
    )

    summary: dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "window_sizes": list(window_sizes),
        "rows_per_size": rows_per_size,
        "rows_by_size": {str(key): value for key, value in rows_by_size.items()},
        "total_rows": len(sequence_ids),
        "feature_shape": list(feature_shape),
        "processed_files": processed_files,
        "processed_source_groups": len(processed_groups),
        "skipped_locked_groups": len(skipped_locked_groups),
        "skipped_conflicting_groups": len(skipped_conflicting_groups),
        "skipped_plasmid_header_groups": len(skipped_plasmid_header_groups),
        "duplicate_source_groups": len(duplicate_groups),
        "exact_reference_id_overlaps": exact_id_overlaps,
        "skipped_large_files": skipped_large_files,
        "max_file_bytes": max_file_bytes,
        "max_windows_per_source_size": max_windows_per_source_size,
        "exclude_groups_path": str(exclude_groups_path) if exclude_groups_path else None,
        "reference_ids_path": str(reference_ids_path) if reference_ids_path else None,
        "reference_labels_path": str(reference_labels_path) if reference_labels_path else None,
    }
    (output_dir / "augmentation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-groups", type=Path)
    parser.add_argument("--reference-ids", type=Path)
    parser.add_argument("--reference-labels", type=Path)
    parser.add_argument(
        "--window-sizes",
        default="1000,2000,5000,10000,20000",
    )
    parser.add_argument("--rows-per-size", type=int, default=8_000)
    parser.add_argument("--max-windows-per-source-size", type=int, default=250)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-chunk-size", type=int, default=1_000)
    args = parser.parse_args()
    window_sizes = tuple(int(value.strip()) for value in args.window_sizes.split(","))
    summary = build_hard_negative_augmentation(
        args.input_dir,
        args.out,
        exclude_groups_path=args.exclude_groups,
        reference_ids_path=args.reference_ids,
        reference_labels_path=args.reference_labels,
        window_sizes=window_sizes,
        rows_per_size=args.rows_per_size,
        max_windows_per_source_size=args.max_windows_per_source_size,
        max_file_bytes=int(args.max_file_mb * 1_000_000),
        seed=args.seed,
        feature_chunk_size=args.feature_chunk_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
