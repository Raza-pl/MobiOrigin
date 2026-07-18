"""Tests for the pre-training dataset integrity audit."""

from __future__ import annotations

import numpy as np

from scripts.audit_training_dataset import audit_dataset


def test_audit_detects_duplicate_conflicting_ids_and_stale_features(tmp_path) -> None:
    ids = tmp_path / "ids.txt"
    labels = tmp_path / "labels.npy"
    features = tmp_path / "features.npy"
    ids.write_text("ACC_w1000_s0\nACC_w1000_s0\n")
    np.save(labels, np.array([0, 1], dtype=np.int64))
    np.save(features, np.zeros((2, 10), dtype=np.float32))

    issues, conflicts = audit_dataset(ids, labels, features)

    assert any("duplicate IDs" in issue for issue in issues)
    assert any("conflicting labels" in issue for issue in issues)
    assert any("stale feature dimension" in issue for issue in issues)
    assert conflicts[0]["source_group"] == "ACC"


def test_audit_accepts_clean_current_shape_dataset(tmp_path) -> None:
    from plasflow2.classify.features import FEATURE_DIM

    ids = tmp_path / "ids.txt"
    labels = tmp_path / "labels.npy"
    features = tmp_path / "features.npy"
    ids.write_text("A_w1000_s0\nB_w1000_s0\n")
    np.save(labels, np.array([0, 1], dtype=np.int64))
    np.save(features, np.zeros((2, FEATURE_DIM), dtype=np.float32))

    issues, conflicts = audit_dataset(ids, labels, features)

    assert issues == []
    assert conflicts == []


def test_audit_detects_invalid_labels_and_non_finite_features(tmp_path) -> None:
    from plasflow2.classify.features import FEATURE_DIM

    ids = tmp_path / "ids.txt"
    labels = tmp_path / "labels.npy"
    features = tmp_path / "features.npy"
    ids.write_text("A_w1000_s0\nB_w1000_s0\n")
    np.save(labels, np.array([0, 7], dtype=np.int64))
    matrix = np.zeros((2, FEATURE_DIM), dtype=np.float32)
    matrix[1, 0] = np.nan
    np.save(features, matrix)

    issues, conflicts = audit_dataset(ids, labels, features)

    assert any("unsupported class labels" in issue for issue in issues)
    assert any("non-finite feature values" in issue for issue in issues)
    assert conflicts == []
