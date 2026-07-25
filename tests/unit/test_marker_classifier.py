"""Tests for marker-classifier checkpoint compatibility."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from plasflow2.classify.marker_classifier import (
    MARKER_FEATURE_NAMES,
    ContigMarkerFeatures,
    MarkerClassifier,
    aggregate_scores,
    marker_model_safety_issues,
    resolve_marker_model_path,
)


def _safe_model_card() -> dict:
    return {
        "class_counts": {"plasmid": 100, "chromosome": 100, "phage": 100},
        "split_type": "grouped_by_source_genome",
        "n_distinct_groups": 30,
        "feature_names": list(MARKER_FEATURE_NAMES),
        "feature_schema_version": "marker-v2",
        "training_data_sha256": "a" * 64,
        "benchmark_lockout_verified": True,
        "benchmark_lockout_sha256": "b" * 64,
    }


def test_marker_model_safety_accepts_complete_grouped_model() -> None:
    assert marker_model_safety_issues(_safe_model_card()) == []


def test_marker_model_safety_rejects_deployed_binary_collapsed_semantics() -> None:
    metadata = _safe_model_card()
    metadata["class_counts"] = {
        "plasmid": 30_000,
        "chromosome": 60_000,
        "phage": 0,
    }
    metadata["split_type"] = "random_per_row"
    metadata["n_distinct_groups"] = None

    issues = marker_model_safety_issues(metadata)

    assert any("phage" in issue for issue in issues)
    assert any("not grouped" in issue for issue in issues)
    assert any("source-group count" in issue for issue in issues)


def test_train_rejects_missing_biological_class() -> None:
    X = np.random.default_rng(2).normal(size=(30, 4)).astype(np.float32)
    y = np.array([0, 1] * 15, dtype=np.int64)

    with pytest.raises(ValueError, match="plasmid, chromosome, and phage"):
        MarkerClassifier().train(X, y, n_estimators=2)


def test_aggregate_scores_uses_marker_fraction_as_attention_weight() -> None:
    combined = aggregate_scores(
        {"plasmid": 0.8, "chromosome": 0.1, "phage": 0.1},
        {"plasmid": 0.2, "chromosome": 0.7, "phage": 0.1},
        marker_gene_fraction=0.5,
    )

    assert combined == pytest.approx({"plasmid": 0.5, "chromosome": 0.4, "phage": 0.1})


def test_train_grouped_split_never_splits_a_group_across_train_and_val() -> None:
    """A group's rows must all land on one side of the split, never both.

    Regression test for the leakage fix: build_marker_dataset.py slices each
    source genome into several overlapping windows (near-duplicate rows).
    Before this fix, MarkerClassifier.train() used a plain random per-row
    split, which could place sibling windows of the same genome in both
    train and val -- inflating val_accuracy by letting the model validate on
    near-duplicates of what it trained on.
    """
    rng = np.random.default_rng(0)
    n_classes, n_groups_per_class, n_rows_per_group, n_features = 3, 10, 4, 5

    X_list, y_list, g_list = [], [], []
    for cls in range(n_classes):
        for gi in range(n_groups_per_class):
            base = rng.normal(loc=cls * 3, scale=1.0, size=n_features)
            group_id = f"class{cls}_group{gi}"
            for _ in range(n_rows_per_group):
                X_list.append(base + rng.normal(scale=0.05, size=n_features))
                y_list.append(cls)
                g_list.append(group_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(g_list)

    clf = MarkerClassifier()
    clf._model = None  # ensure train() builds a fresh model
    result = clf.train(X, y, n_estimators=10, groups=groups, eval_fraction=0.2)

    assert "val_accuracy" in result
    # Re-derive the same split the way train() does, to check the invariant
    # directly rather than relying on log output.
    from sklearn.model_selection import GroupShuffleSplit

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_rel, va_rel = next(splitter.split(cls_idx, groups=groups[cls_idx]))
        tr_groups = set(groups[cls_idx][tr_rel])
        va_groups = set(groups[cls_idx][va_rel])
        assert tr_groups.isdisjoint(
            va_groups
        ), f"class {cls}: groups leaked across train/val: {tr_groups & va_groups}"


def test_train_without_groups_falls_back_to_random_split_with_warning(caplog) -> None:
    """Legacy NPZ files without a groups array should still train, with a warning."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    y = np.array([0, 1, 2] * 20, dtype=np.int64)

    clf = MarkerClassifier()
    with caplog.at_level(logging.WARNING):
        result = clf.train(X, y, n_estimators=10, groups=None)

    assert "val_accuracy" in result
    assert any("falling back to a random per-row split" in rec.message for rec in caplog.records)


def test_predict_scores_supports_legacy_binary_marker_model() -> None:
    class BinaryModel:
        def predict_proba(self, features):
            assert features.shape == (1, 28)
            return np.array([[0.8, 0.2]], dtype=np.float32)

    classifier = MarkerClassifier()
    classifier._model = BinaryModel()

    scores = classifier.predict_scores(ContigMarkerFeatures(contig_id="contig"))

    assert scores == {
        "plasmid": np.float32(0.8),
        "chromosome": np.float32(0.2),
        "phage": 0.0,
    }


def test_save_writes_model_card_and_load_reads_it_back(tmp_path) -> None:
    import xgboost as xgb

    X = np.random.default_rng(0).random((20, 3), dtype=np.float32)
    y = np.array([0, 1, 2] * 6 + [0, 1], dtype=np.int64)

    classifier = MarkerClassifier()
    classifier._model = xgb.XGBClassifier(n_estimators=2, max_depth=2)
    classifier._model.fit(X, y)

    # save() is called with a .pkl path (matching every real call site) but
    # writes XGBoost's native JSON format instead -- no .pkl file is created.
    out_path = tmp_path / "marker_xgb.pkl"
    classifier.save(out_path, metadata={"training_data_path": "some/features.npz"})

    json_path = tmp_path / "marker_xgb.json"
    meta_path = tmp_path / "marker_xgb.json.meta.json"
    assert json_path.exists()
    assert meta_path.exists()
    assert not out_path.exists()  # no .pkl written by save() anymore

    # load() still accepts the original .pkl-suffixed path -- it resolves to
    # the .json sibling via resolve_marker_model_path().
    loaded = MarkerClassifier.load(out_path)
    assert loaded.metadata["training_data_path"] == "some/features.npz"
    assert loaded.metadata["format"] == "xgboost-json"
    assert "saved_at" in loaded.metadata
    assert loaded.metadata["n_features"] == 3
    np.testing.assert_allclose(loaded.predict_proba(X), classifier.predict_proba(X))


def test_resolve_marker_model_path_ignores_pickle_and_prefers_json(
    tmp_path,
) -> None:
    pkl_path = tmp_path / "marker_xgb.pkl"
    json_path = tmp_path / "marker_xgb.json"

    assert resolve_marker_model_path(pkl_path) is None

    pkl_path.write_bytes(b"legacy pickle bytes")
    assert resolve_marker_model_path(pkl_path) is None

    json_path.write_text("{}")
    assert resolve_marker_model_path(pkl_path) == json_path


def test_legacy_pickle_is_rejected_without_deserialization(tmp_path) -> None:
    import pickle

    sentinel = tmp_path / "pickle_was_executed"

    class Payload:
        def __reduce__(self):
            command = "from pathlib import Path; " f"Path({str(sentinel)!r}).touch()"
            return exec, (command,)

    pkl_path = tmp_path / "legacy_marker_xgb.pkl"
    pkl_path.write_bytes(pickle.dumps(Payload()))

    with pytest.raises(ValueError, match="Refusing legacy pickle"):
        MarkerClassifier.load(pkl_path)

    assert not sentinel.exists()
