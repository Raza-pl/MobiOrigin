"""Train the XGBoost marker classifier from pre-built marker features.

Usage
-----
    # 1. Build features (one-time, ~20 min)
    python scripts/build_marker_dataset.py \\
        --plasmid-dir  data/databases/plasmids/ \\
        --chrom-dir    data/gtdb_genomes/bacteria/ \\
        --model        data/models/mlp_v2.pt \\
        --exclude-groups data/benchmark/locked_all_training_groups.txt \\
        --mob-db       data/databases/mob_suite/mob_proteins.dmnd \\
        --max-per-class 30000 \\
        --out          data/marker_features.npz

    # 2. Train XGBoost (~2 min)
    python scripts/train_marker_model.py \\
        --features data/marker_features.npz \\
        --exclude-groups data/benchmark/locked_all_training_groups.txt \\
        --out      data/models/

Output
------
    data/models/marker_xgb.json        — trained MarkerClassifier (XGBoost
                                          native JSON format, not pickle)
    data/models/marker_xgb.json.meta.json — model card (provenance)

    (MarkerClassifier.save() writes the .json file even though this script
    still names its output path with a .pkl extension below — see
    marker_classifier.py's save()/load() docstrings. Existing tooling that
    looks for marker_xgb.pkl will still find it via
    resolve_marker_model_path(), which checks for a .json/.ubj sibling
    first.)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plasflow2.classify.marker_classifier import (  # noqa: E402
    MARKER_FEATURE_NAMES,
    MARKER_PROFILE_FULL,
    MarkerClassifier,
    marker_classifier_available,
)
from plasflow2.classify.splits import validate_group_labels  # noqa: E402
from plasflow2.utils.device import IDX_TO_CLASS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost marker classifier")
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Marker features .npz (from build_marker_dataset.py)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/models"),
        help="Output directory for marker_xgb.pkl",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--allow-ungrouped",
        action="store_true",
        help=(
            "Allow an unsafe random per-row split for development only. "
            "Models trained this way are rejected by the production pipeline."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-features",
        action="store_true",
        help=(
            "Development only: allow a dataset without the verified "
            "full-annotation feature profile. Such a model remains blocked "
            "from production fusion."
        ),
    )
    parser.add_argument(
        "--exclude-groups",
        type=Path,
        default=None,
        help=(
            "Locked benchmark source groups used by the dataset builder. "
            "Required for production training."
        ),
    )
    args = parser.parse_args()

    if not marker_classifier_available():
        logger.error("xgboost is not installed. Run: pip install xgboost")
        sys.exit(1)

    # Load features
    data = np.load(args.features, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    groups = data["groups"] if "groups" in data else None
    sequence_ids = (
        [str(value) for value in data["sequence_ids"]] if "sequence_ids" in data else None
    )
    feat_names = [str(f) for f in data["feature_names"]] if "feature_names" in data else None
    feature_schema_version = (
        str(data["feature_schema_version"].item()) if "feature_schema_version" in data else None
    )
    dataset_feature_profile = (
        str(data["feature_profile"].item()) if "feature_profile" in data else None
    )
    dataset_parity_verified = (
        bool(data["training_prediction_parity_verified"].item())
        if "training_prediction_parity_verified" in data
        else False
    )
    dataset_lockout_sha256 = (
        str(data["lockout_sha256"].item()) if "lockout_sha256" in data else None
    )
    if not args.allow_incomplete_features:
        if dataset_feature_profile != MARKER_PROFILE_FULL:
            parser.error(
                "production marker training requires feature_profile="
                f"{MARKER_PROFILE_FULL!r}; dataset declares "
                f"{dataset_feature_profile!r}"
            )
        if dataset_parity_verified is not True:
            parser.error(
                "production marker training requires verified " "training/prediction feature parity"
            )

    logger.info("Loaded features: X=%s  y=%s", X.shape, y.shape)
    if groups is not None:
        logger.info(
            "Loaded groups: %d rows, %d distinct source genomes — using grouped split.",
            len(groups),
            len(set(groups.tolist())),
        )
    else:
        if not args.allow_ungrouped:
            parser.error(
                "the feature dataset has no 'groups' array; rebuild it with "
                "scripts/build_marker_dataset.py, or use --allow-ungrouped "
                "for a non-production experiment"
            )
        logger.warning(
            "No 'groups' array in %s (built before grouped-split support was added) — "
            "falling back to a random per-row split. val_accuracy may be optimistic if "
            "this dataset has multiple overlapping windows per source genome. "
            "Rebuild with the current build_marker_dataset.py to get a grouped split.",
            args.features,
        )
    if not args.allow_ungrouped:
        if args.exclude_groups is None or not args.exclude_groups.is_file():
            parser.error("--exclude-groups is required for production marker training")
        if sequence_ids is None:
            parser.error("production marker features must contain sequence_ids")
        if len(sequence_ids) != len(y) or len(set(sequence_ids)) != len(sequence_ids):
            parser.error("sequence_ids must be unique and aligned with feature rows")
        if feat_names != MARKER_FEATURE_NAMES:
            parser.error("feature_names do not exactly match the runtime schema")
        if feature_schema_version != "marker-v2":
            parser.error("feature_schema_version must be 'marker-v2'")
        if not np.isfinite(X).all():
            parser.error("feature matrix contains non-finite values")
        validate_group_labels(y, [str(value) for value in groups])

        excluded_groups = {
            line.strip() for line in args.exclude_groups.read_text().splitlines() if line.strip()
        }
        overlap = {str(value) for value in groups} & excluded_groups
        if overlap:
            parser.error(f"benchmark lockout failed: {len(overlap)} excluded groups remain")
        actual_lockout_sha256 = _sha256(args.exclude_groups)
        if dataset_lockout_sha256 != actual_lockout_sha256:
            parser.error("dataset lockout SHA-256 does not match --exclude-groups")
    else:
        actual_lockout_sha256 = None
    if feat_names:
        logger.info("Feature names (%d): %s", len(feat_names), feat_names)

    # Class distribution
    for idx, name in IDX_TO_CLASS.items():
        count = int((y == idx).sum())
        logger.info("  %-12s  %6d  (%.1f%%)", name, count, 100 * count / len(y))

    # Train
    clf = MarkerClassifier()
    result = clf.train(
        X,
        y,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        groups=groups,
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

    # Save, with a model card recording exactly what produced this checkpoint.
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "marker_xgb.pkl"
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "training_data_path": str(args.features.resolve()),
        "training_data_sha256": _sha256(args.features),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts": {IDX_TO_CLASS[idx]: int((y == idx).sum()) for idx in sorted(IDX_TO_CLASS)},
        "feature_names": feat_names,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.lr,
        },
        "val_accuracy": result["val_accuracy"],
        "split_type": ("grouped_by_source_genome" if groups is not None else "random_per_row"),
        "n_distinct_groups": len(set(groups.tolist())) if groups is not None else None,
        "feature_schema_version": feature_schema_version,
        "feature_profile": dataset_feature_profile,
        "training_prediction_parity_verified": dataset_parity_verified,
        "benchmark_lockout_verified": not args.allow_ungrouped,
        "benchmark_lockout_sha256": actual_lockout_sha256,
        "zero_variance_features": (
            [
                feat_names[index]
                for index in range(X.shape[1])
                if float(np.min(X[:, index])) == float(np.max(X[:, index]))
            ]
            if feat_names
            else []
        ),
    }
    clf.save(out_path, metadata=metadata)
    logger.info("Done — saved to %s", out_path)


if __name__ == "__main__":
    main()
