"""Tests for leakage-resistant classifier dataset splits."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from plasflow2.classify.splits import (
    grouped_split_indices,
    load_sequence_ids,
    source_group_id,
    validate_group_labels,
    write_split_manifest,
)


def test_source_group_id_removes_only_window_suffix() -> None:
    assert source_group_id("NZ_CP012345.1_w5000_s25000") == "NZ_CP012345.1"
    assert source_group_id("COMPASS_NC_016626.1_w1000_s458000") == "NC_016626.1"
    assert source_group_id("already_complete_contig") == "already_complete_contig"


def test_grouped_split_has_no_source_overlap_and_is_reproducible() -> None:
    labels = np.repeat(np.arange(3, dtype=np.int64), 40)
    groups = [f"class_{label}_source_{i // 2}" for label in range(3) for i in range(40)]

    split_a = grouped_split_indices(labels, groups, random_state=17)
    split_b = grouped_split_indices(labels, groups, random_state=17)
    for a, b in zip(split_a, split_b):
        np.testing.assert_array_equal(a, b)

    train_idx, val_idx, test_idx = split_a
    group_array = np.asarray(groups)
    train_groups = set(group_array[train_idx])
    val_groups = set(group_array[val_idx])
    test_groups = set(group_array[test_idx])
    assert not train_groups & val_groups
    assert not train_groups & test_groups
    assert not val_groups & test_groups
    for idx in split_a:
        assert set(labels[idx]) == {0, 1, 2}


def test_load_sequence_ids_rejects_count_mismatch_and_duplicates(tmp_path) -> None:
    ids = tmp_path / "seq_ids.txt"
    ids.write_text("a\nb\n")
    with pytest.raises(ValueError, match="count mismatch"):
        load_sequence_ids(ids, expected_count=3)
    ids.write_text("a\na\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_sequence_ids(ids)


def test_validate_group_labels_rejects_conflicting_sources() -> None:
    with pytest.raises(ValueError, match="conflicting class labels"):
        validate_group_labels(np.array([0, 1], dtype=np.int64), ["same", "same"])
    validate_group_labels(np.array([0, 0, 1], dtype=np.int64), ["same", "same", "other"])


def test_split_manifest_records_every_row(tmp_path) -> None:
    ids = [f"src{i}_w1000_s0" for i in range(6)]
    groups = [f"src{i}" for i in range(6)]
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    path = tmp_path / "manifest.tsv"
    write_split_manifest(
        path,
        ids,
        labels,
        groups,
        np.array([0, 2, 4]),
        np.array([1]),
        np.array([3, 5]),
        source_row_indices=np.array([10, 11, 12, 13, 14, 15]),
    )
    with path.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 6
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert [int(row["feature_row_index"]) for row in rows] == [10, 11, 12, 13, 14, 15]
