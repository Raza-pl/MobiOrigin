"""Tests for marker-classifier checkpoint compatibility."""

from __future__ import annotations

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
