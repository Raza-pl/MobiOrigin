"""Train the XGBoost marker classifier from pre-built marker features.

Usage
-----
    # 1. Build features (one-time, ~20 min)
    python scripts/build_marker_dataset.py \\
        --plasmid-dir  data/databases/plasmids/ \\
        --chrom-dir    data/gtdb_genomes/bacteria/ \\
        --model        data/models/mlp_v2.pt \\
        --mob-db       data/databases/mob_suite/mob_proteins.dmnd \\
        --max-per-class 30000 \\
        --out          data/marker_features.npz

    # 2. Train XGBoost (~2 min)
    python scripts/train_marker_model.py \\
        --features data/marker_features.npz \\
        --out      data/models/

Output
------
    data/models/marker_xgb.pkl   — trained MarkerClassifier (pickled XGBoost)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plasflow2.classify.marker_classifier import (  # noqa: E402
    MARKER_FEATURE_NAMES,
    MarkerClassifier,
    marker_classifier_available,
)
from plasflow2.utils.device import IDX_TO_CLASS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost marker classifier")
    parser.add_argument("--features", type=Path, required=True,
                        help="Marker features .npz (from build_marker_dataset.py)")
    parser.add_argument("--out", type=Path, default=Path("data/models"),
                        help="Output directory for marker_xgb.pkl")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth",    type=int, default=6)
    parser.add_argument("--lr",           type=float, default=0.1)
    args = parser.parse_args()

    if not marker_classifier_available():
        logger.error("xgboost is not installed. Run: pip install xgboost")
        sys.exit(1)

    # Load features
    data = np.load(args.features, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feat_names = [str(f) for f in data["feature_names"]] if "feature_names" in data else None
    logger.info("Loaded features: X=%s  y=%s", X.shape, y.shape)
    if feat_names:
        logger.info("Feature names (%d): %s", len(feat_names), feat_names)

    # Class distribution
    for idx, name in IDX_TO_CLASS.items():
        count = int((y == idx).sum())
        logger.info("  %-12s  %6d  (%.1f%%)", name, count, 100 * count / len(y))

    # Train
    clf = MarkerClassifier()
    result = clf.train(
        X, y,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
    )
    logger.info("Validation accuracy: %.4f", result["val_accuracy"])

    # Feature importance report — use NPZ feature names when available
    importances = result["feature_importances"]
    if feat_names and len(feat_names) == len(importances):
        importances = dict(zip(feat_names, clf._model.feature_importances_))
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("Feature importances:")
    for feat, imp in ranked:
        logger.info("  %-35s  %.4f", feat, imp)

    # Save
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "marker_xgb.pkl"
    clf.save(out_path)
    logger.info("Done — saved to %s", out_path)


if __name__ == "__main__":
    main()
