"""Tests for marker-classifier checkpoint compatibility."""

from __future__ import annotations

import logging

import numpy as np
from plasflow2.classify.marker_classifier import ContigMarkerFeatures, MarkerClassifier


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

    out_path = tmp_path / "marker_xgb.pkl"
    classifier.save(out_path, metadata={"training_data_path": "some/features.npz"})

    meta_path = tmp_path / "marker_xgb.pkl.meta.json"
    assert meta_path.exists()

    loaded = MarkerClassifier.load(out_path)
    assert loaded.metadata["training_data_path"] == "some/features.npz"
    assert "saved_at" in loaded.metadata
    assert loaded.metadata["n_features"] == 3


def test_load_warns_when_model_card_missing(tmp_path, caplog) -> None:
    import pickle

    import xgboost as xgb

    X = np.random.default_rng(0).random((10, 2), dtype=np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)
    model = xgb.XGBClassifier(n_estimators=2, max_depth=2)
    model.fit(X, y)

    out_path = tmp_path / "legacy_marker_xgb.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(model, fh)

    with caplog.at_level(logging.WARNING):
        loaded = MarkerClassifier.load(out_path)

    assert loaded.metadata == {}
    assert any("No model card" in rec.message for rec in caplog.records)
