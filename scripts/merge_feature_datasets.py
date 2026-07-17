#!/usr/bin/env python3
"""Atomically concatenate aligned feature datasets without loading them into RAM."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def merge_feature_datasets(
    dataset_dirs: list[Path],
    output_dir: Path,
    *,
    chunk_rows: int = 2_000,
) -> dict[str, object]:
    """Merge features, labels, and sequence IDs while enforcing alignment."""

    if len(dataset_dirs) < 2:
        raise ValueError("At least two dataset directories are required")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    sources: list[tuple[Path, np.ndarray, np.ndarray, list[str]]] = []
    feature_dim: int | None = None
    all_ids: set[str] = set()
    total_rows = 0
    for dataset_dir in dataset_dirs:
        features = np.load(dataset_dir / "features.npy", mmap_mode="r")
        labels = np.load(dataset_dir / "labels.npy", mmap_mode="r")
        sequence_ids = [
            line.strip()
            for line in (dataset_dir / "seq_ids.txt").read_text().splitlines()
            if line.strip()
        ]
        if features.ndim != 2:
            raise ValueError(f"{dataset_dir}: features must be two-dimensional")
        if len(features) != len(labels) or len(labels) != len(sequence_ids):
            raise ValueError(f"{dataset_dir}: features, labels, and IDs are misaligned")
        if feature_dim is None:
            feature_dim = int(features.shape[1])
        elif features.shape[1] != feature_dim:
            raise ValueError(f"{dataset_dir}: feature dimension does not match")
        duplicates = all_ids.intersection(sequence_ids)
        if duplicates:
            examples = ", ".join(sorted(duplicates)[:3])
            raise ValueError(f"{dataset_dir}: duplicate sequence IDs across datasets: {examples}")
        all_ids.update(sequence_ids)
        total_rows += len(labels)
        sources.append((dataset_dir, features, labels, sequence_ids))

    assert feature_dim is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    features_incomplete = output_dir / "features.npy.incomplete"
    labels_incomplete = output_dir / "labels.npy.incomplete"
    ids_incomplete = output_dir / "seq_ids.txt.incomplete"
    merged_features = np.lib.format.open_memmap(
        features_incomplete,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, feature_dim),
    )
    merged_labels = np.lib.format.open_memmap(
        labels_incomplete,
        mode="w+",
        dtype=np.int64,
        shape=(total_rows,),
    )
    offset = 0
    merged_ids: list[str] = []
    class_counts: Counter[int] = Counter()
    source_rows: dict[str, int] = {}
    try:
        for dataset_dir, features, labels, sequence_ids in sources:
            source_rows[str(dataset_dir)] = len(labels)
            for start in range(0, len(labels), chunk_rows):
                end = min(start + chunk_rows, len(labels))
                merged_features[offset + start : offset + end] = features[start:end]
                merged_labels[offset + start : offset + end] = labels[start:end]
            class_counts.update(int(value) for value in np.asarray(labels))
            merged_ids.extend(sequence_ids)
            offset += len(labels)
        merged_features.flush()
        merged_labels.flush()
        ids_incomplete.write_text("\n".join(merged_ids) + "\n")
    finally:
        del merged_features
        del merged_labels

    os.replace(features_incomplete, output_dir / "features.npy")
    os.replace(labels_incomplete, output_dir / "labels.npy")
    os.replace(ids_incomplete, output_dir / "seq_ids.txt")
    summary: dict[str, object] = {
        "source_datasets": [str(path) for path in dataset_dirs],
        "source_rows": source_rows,
        "total_rows": total_rows,
        "feature_shape": [total_rows, feature_dim],
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "unique_sequence_ids": len(all_ids),
        "chunk_rows": chunk_rows,
    }
    (output_dir / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=2_000)
    args = parser.parse_args()
    summary = merge_feature_datasets(
        args.datasets,
        args.out,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
