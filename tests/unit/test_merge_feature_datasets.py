"""Tests for atomic feature-dataset concatenation."""

from __future__ import annotations

import json

import numpy as np

from scripts.merge_feature_datasets import merge_feature_datasets


def _write_dataset(path, features, labels, sequence_ids) -> None:
    path.mkdir()
    np.save(path / "features.npy", np.asarray(features, dtype=np.float32))
    np.save(path / "labels.npy", np.asarray(labels, dtype=np.int64))
    (path / "seq_ids.txt").write_text("\n".join(sequence_ids) + "\n")


def test_merge_feature_datasets_preserves_order_and_alignment(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    _write_dataset(first, [[1, 2], [3, 4]], [0, 1], ["a", "b"])
    _write_dataset(second, [[5, 6]], [2], ["c"])

    summary = merge_feature_datasets([first, second], output, chunk_rows=1)

    np.testing.assert_array_equal(
        np.load(output / "features.npy"),
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
    )
    np.testing.assert_array_equal(np.load(output / "labels.npy"), [0, 1, 2])
    assert (output / "seq_ids.txt").read_text().splitlines() == ["a", "b", "c"]
    assert summary["class_counts"] == {"0": 1, "1": 1, "2": 1}
    assert json.loads((output / "merge_summary.json").read_text())["total_rows"] == 3
