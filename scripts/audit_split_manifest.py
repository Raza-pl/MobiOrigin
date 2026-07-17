#!/usr/bin/env python3
"""Audit source isolation, feature-row uniqueness, and benchmark lockout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit_split_manifest(
    manifest_path: Path,
    lockout_path: Path,
    report_path: Path,
) -> tuple[list[str], dict[str, object]]:
    """Return blocking issues and write an auditable split summary."""

    with manifest_path.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    locked_groups = {
        line.strip()
        for line in lockout_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    issues: list[str] = []

    row_indices = [int(row["row_index"]) for row in rows]
    feature_rows = [int(row["feature_row_index"]) for row in rows]
    if row_indices != list(range(len(rows))):
        issues.append("row_index is not contiguous and ordered")
    if len(set(feature_rows)) != len(feature_rows):
        issues.append("feature_row_index contains duplicates")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["source_group"]].add(row["split"])
    split_conflicts = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if split_conflicts:
        issues.append(f"{len(split_conflicts)} source groups cross split boundaries")

    overlap = set(group_splits) & locked_groups
    if overlap:
        issues.append(f"{len(overlap)} locked benchmark groups occur in the manifest")

    split_class_counts = Counter((row["split"], int(row["label"])) for row in rows)
    summary: dict[str, object] = {
        "n_rows": len(rows),
        "n_feature_rows": len(set(feature_rows)),
        "n_source_groups": len(group_splits),
        "locked_groups": len(locked_groups),
        "locked_overlap": len(overlap),
        "split_conflicts": len(split_conflicts),
        "split_class_counts": {
            f"{split_name}:{label}": count
            for (split_name, label), count in sorted(split_class_counts.items())
        },
        "issues": issues,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return issues, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lockout", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    issues, summary = audit_split_manifest(
        args.manifest,
        args.lockout,
        args.report,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
