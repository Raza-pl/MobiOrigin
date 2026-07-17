#!/usr/bin/env python3
"""Audit a PlasFlow training dataset before model fitting.

The audit detects row-count mismatches, duplicate sequence identifiers,
accessions assigned to multiple classes, and stale feature dimensions.
It exits non-zero when any blocking integrity problem is found.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from plasflow2.classify.features import FEATURE_DIM, FEATURE_DIM_FULL
from plasflow2.classify.splits import source_group_id

logger = logging.getLogger(__name__)

_VALID_LABELS = frozenset({0, 1, 2})
_FEATURE_SCAN_ROWS = 4096


def audit_dataset(
    ids_path: Path,
    labels_path: Path,
    features_path: Path | None = None,
) -> tuple[list[str], list[dict]]:
    """Return blocking issue messages and accession-level conflict rows."""

    sequence_ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    labels = np.load(labels_path, mmap_mode="r")
    issues: list[str] = []

    if labels.ndim != 1:
        issues.append(f"labels must be one-dimensional, found shape {labels.shape}")
    elif not np.issubdtype(labels.dtype, np.integer):
        issues.append(f"labels must use an integer dtype, found {labels.dtype}")
    else:
        observed_labels = {int(label) for label in np.unique(labels)}
        invalid_labels = sorted(observed_labels - _VALID_LABELS)
        if invalid_labels:
            issues.append(f"unsupported class labels found: {invalid_labels}")
    if len(sequence_ids) != len(labels):
        issues.append(f"row mismatch: {len(sequence_ids)} IDs versus {len(labels)} labels")

    duplicate_counts = Counter(sequence_ids)
    duplicate_ids = {sid: count for sid, count in duplicate_counts.items() if count > 1}
    if duplicate_ids:
        issues.append(
            f"{len(duplicate_ids)} duplicate IDs ({sum(v - 1 for v in duplicate_ids.values())} extra rows)"
        )

    labels_by_group: dict[str, Counter[int]] = defaultdict(Counter)
    for sid, label in zip(sequence_ids, labels):
        labels_by_group[source_group_id(sid)][int(label)] += 1
    conflicts = [
        {
            "source_group": group,
            "labels": ",".join(str(label) for label in sorted(counts)),
            "counts": ",".join(f"{label}:{counts[label]}" for label in sorted(counts)),
            "total_rows": sum(counts.values()),
        }
        for group, counts in labels_by_group.items()
        if len(counts) > 1
    ]
    if conflicts:
        issues.append(
            f"{len(conflicts)} source accessions have conflicting labels "
            f"({sum(int(row['total_rows']) for row in conflicts)} affected rows)"
        )

    if features_path is not None:
        features = np.load(features_path, mmap_mode="r")
        if features.ndim != 2:
            issues.append(f"features must be two-dimensional, found shape {features.shape}")
        else:
            if len(features) != len(labels):
                issues.append(
                    f"row mismatch: {len(features)} feature rows versus {len(labels)} labels"
                )
            if features.shape[1] not in (FEATURE_DIM, FEATURE_DIM_FULL):
                issues.append(
                    "stale feature dimension: dataset has "
                    f"{features.shape[1]}, supported dimensions are {FEATURE_DIM} and {FEATURE_DIM_FULL}"
                )
            if features.dtype != np.float32:
                issues.append(f"features must use float32 storage, found {features.dtype}")
            for start in range(0, len(features), _FEATURE_SCAN_ROWS):
                end = min(start + _FEATURE_SCAN_ROWS, len(features))
                if not np.isfinite(features[start:end]).all():
                    issues.append(f"non-finite feature values found in rows {start}:{end}")
                    break

    return issues, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, default=Path("data/seq_ids.txt"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels.npy"))
    parser.add_argument("--features", type=Path, default=Path("data/features.npy"))
    parser.add_argument("--report", type=Path, default=Path("data/dataset_audit.tsv"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    issues, conflicts = audit_dataset(args.ids, args.labels, args.features)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["source_group", "labels", "counts", "total_rows"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(conflicts)

    if issues:
        for issue in issues:
            logger.error(issue)
        logger.error("Dataset audit FAILED; conflict report: %s", args.report)
        raise SystemExit(1)
    logger.info("Dataset audit passed: %d conflicting accessions", len(conflicts))


if __name__ == "__main__":
    main()
