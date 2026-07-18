"""Leakage-resistant dataset splitting utilities.

Training examples are usually overlapping windows cut from a source genome.
Randomly splitting those windows leaks source-specific sequence composition into
validation and test sets.  This module keeps every window from a source
accession in exactly one split.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_WINDOW_SUFFIX = re.compile(r"^(?P<source>.+)_w\d+_s\d+$")
_DATASET_SOURCE_PREFIXES = ("COMPASS_",)


def source_group_id(sequence_id: str) -> str:
    """Return the source accession encoded in a training-window identifier.

    Dataset builders use identifiers such as ``NC_012345.1_w5000_s25000``.
    Unknown identifier formats are returned unchanged, which is conservative:
    they become independent groups rather than being merged accidentally.
    """

    clean_id = sequence_id.strip()
    match = _WINDOW_SUFFIX.match(clean_id)
    source = match.group("source") if match else clean_id
    # The same accession can be imported from more than one collection.  The
    # collection prefix is provenance, not a distinct biological source, and
    # retaining it would allow an exact accession to cross split boundaries.
    for prefix in _DATASET_SOURCE_PREFIXES:
        if source.startswith(prefix):
            return source[len(prefix) :]
    return source


def load_sequence_ids(path: Path | str, expected_count: int | None = None) -> list[str]:
    """Load one sequence identifier per line and optionally verify its length."""

    ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(
            f"Sequence-ID count mismatch: {len(ids)} IDs in {path}, "
            f"but labels contain {expected_count} rows"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate sequence IDs found in {path}")
    return ids


def grouped_split_indices(
    labels: NDArray[np.int64],
    groups: list[str] | NDArray[np.str_],
    *,
    val_size: float = 0.10,
    test_size: float = 0.10,
    random_state: int = 42,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Split row indices while keeping groups intact and approximately stratified.

    ``StratifiedGroupKFold`` is used twice: first to reserve the test fold, then
    to reserve validation from the remaining groups.  Fractions are approximate
    because a source group is indivisible.
    """

    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import]

    labels = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=str)
    if labels.ndim != 1 or len(labels) != len(group_array):
        raise ValueError("labels and groups must be one-dimensional and have equal length")
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be > 0 and sum to less than 1")
    if len(np.unique(group_array)) < 3:
        raise ValueError("At least three source groups are required for train/val/test splitting")

    all_idx = np.arange(len(labels), dtype=np.int64)

    def _held_out_fold(
        indices: NDArray[np.int64], fraction: float, seed: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        n_splits = max(2, int(round(1.0 / fraction)))
        n_groups = len(np.unique(group_array[indices]))
        n_splits = min(n_splits, n_groups)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_rel, held_rel = next(
            splitter.split(indices, labels[indices], groups=group_array[indices])
        )
        return indices[train_rel], indices[held_rel]

    trainval_idx, test_idx = _held_out_fold(all_idx, test_size, random_state)
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = _held_out_fold(trainval_idx, relative_val_size, random_state + 1)

    split_groups = [set(group_array[idx]) for idx in (train_idx, val_idx, test_idx)]
    if (
        split_groups[0] & split_groups[1]
        or split_groups[0] & split_groups[2]
        or split_groups[1] & split_groups[2]
    ):
        raise RuntimeError(
            "Grouped split invariant violated: source group appears in multiple splits"
        )
    return train_idx, val_idx, test_idx


def validate_group_labels(labels: NDArray[np.int64], groups: list[str]) -> None:
    """Reject datasets where one biological source has conflicting labels."""

    labels_by_group: dict[str, set[int]] = {}
    for group, label in zip(groups, labels):
        labels_by_group.setdefault(group, set()).add(int(label))
    conflicts = {group: values for group, values in labels_by_group.items() if len(values) > 1}
    if conflicts:
        examples = ", ".join(
            f"{group}={sorted(values)}" for group, values in list(conflicts.items())[:5]
        )
        raise ValueError(
            f"Found {len(conflicts)} source groups with conflicting class labels; "
            f"examples: {examples}. Rebuild or clean the dataset before training."
        )


def write_split_manifest(
    path: Path | str,
    sequence_ids: list[str],
    labels: NDArray[np.int64],
    groups: list[str],
    train_idx: NDArray[np.int64],
    val_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    source_row_indices: NDArray[np.int64] | None = None,
) -> None:
    """Write an auditable row-level manifest for a grouped split."""

    if source_row_indices is not None and len(source_row_indices) != len(sequence_ids):
        raise ValueError("source_row_indices must align with sequence_ids")
    split_by_index = {
        **{int(i): "train" for i in train_idx},
        **{int(i): "validation" for i in val_idx},
        **{int(i): "test" for i in test_idx},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "row_index",
                "feature_row_index",
                "sequence_id",
                "source_group",
                "label",
                "split",
            ]
        )
        for i, (sequence_id, group, label) in enumerate(zip(sequence_ids, groups, labels)):
            feature_row_index = int(source_row_indices[i]) if source_row_indices is not None else i
            writer.writerow(
                [
                    i,
                    feature_row_index,
                    sequence_id,
                    group,
                    int(label),
                    split_by_index[i],
                ]
            )
