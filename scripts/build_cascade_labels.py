"""Build Stage 1 and Stage 2 label arrays for cascade MLP training.

Reads the 3-class labels.npy from data/k7_3class_experiment/ and produces two
binary label arrays.  Feature files are NOT copied — train_model.py --class-filter
handles index-level filtering directly from the original features.npy on disk.

Stage 1 — plasmid detector (plasmid vs. rest):
    Original label 0 (plasmid)    → 1
    Original label 1 (chromosome) → 0
    Original label 2 (phage)      → 0
    All 300k samples used; MLP trained with class weighting to favour recall.

Stage 2 — chromosome / phage discriminator:
    Only samples with original label 1 (chr) or 2 (phage) are used.
    train_model.py --class-filter 1,2 handles filtering in-place (no file copy).
    Output here is just the *full* original labels.npy path — Stage 2 uses the
    same label file but passes --class-filter 1,2 to remap and filter at runtime.

USAGE (from project root):
    python scripts/build_cascade_labels.py \\
        --labels data/k7_3class_experiment/labels.npy \\
        --out    data/k7_cascade_experiment/

OUTPUTS:
    data/k7_cascade_experiment/stage1_labels.npy   (300k, binary 0/1)
    data/k7_cascade_experiment/stage2_labels.npy   (symlink to original labels —
                                                     used with --class-filter 1,2)
"""

from __future__ import annotations

import argparse
import collections
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path,
                        default=Path("data/k7_3class_experiment/labels.npy"),
                        help="Original 3-class labels.npy (values 0=plasmid,1=chr,2=phage)")
    parser.add_argument("--out", type=Path,
                        default=Path("data/k7_cascade_experiment"),
                        help="Output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading 3-class labels from %s …", args.labels)
    labels = np.load(args.labels).astype(np.int64)
    dist = dict(collections.Counter(labels.tolist()).most_common())
    logger.info("Original distribution: %s  (0=plasmid, 1=chr, 2=phage)", dist)

    # ── Stage 1: plasmid (1) vs. non-plasmid (0) ──────────────────────────
    stage1 = (labels == 0).astype(np.int64)   # plasmid=1, rest=0
    s1_dist = dict(collections.Counter(stage1.tolist()).most_common())
    logger.info("Stage 1 labels (0=non-plasmid, 1=plasmid): %s", s1_dist)
    s1_ratio = s1_dist.get(0, 0) / max(s1_dist.get(1, 1), 1)
    logger.info("  Class imbalance ratio: %.1f:1 (non-plasmid:plasmid) — class weighting will compensate", s1_ratio)

    stage1_path = args.out / "stage1_labels.npy"
    np.save(stage1_path, stage1)
    logger.info("Saved Stage 1 labels → %s", stage1_path)

    # ── Stage 2: chromosome (0) vs. phage (1) ─────────────────────────────
    # train_model.py --class-filter 1,2 handles filtering at runtime from the
    # ORIGINAL labels file (chr=1→0, phage=2→1 automatically by sorted remap).
    # We just create a symlink / copy so the cascade dir is self-contained.
    stage2_link = args.out / "stage2_labels.npy"
    if stage2_link.exists() or stage2_link.is_symlink():
        stage2_link.unlink()
    stage2_link.symlink_to(args.labels.resolve())
    logger.info("Stage 2 labels → symlink to %s", args.labels)
    logger.info("  Use: train_model.py --labels %s --class-filter 1,2", stage2_link)
    logger.info("  This keeps chr+phage samples only, remaps chr→0  phage→1")

    # ── Also record the features path ─────────────────────────────────────
    features_path = args.labels.parent / "features.npy"
    if features_path.exists():
        features_link = args.out / "features.npy"
        if features_link.exists() or features_link.is_symlink():
            features_link.unlink()
        features_link.symlink_to(features_path.resolve())
        logger.info("Features → symlink to %s", features_path)
    else:
        logger.warning("features.npy not found at %s — create symlink manually", features_path)

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("═══ Cascade training commands ═══")
    logger.info("")
    logger.info("Stage 1 (plasmid vs. rest):")
    logger.info("  python scripts/train_model.py \\")
    logger.info("      --data   %s \\", args.out / "features.npy")
    logger.info("      --labels %s \\", stage1_path)
    logger.info("      --out    %s/stage1_models \\", args.out)
    logger.info("      --mlp --epochs 50")
    logger.info("")
    logger.info("Stage 2 (chromosome vs. phage, chr+phage samples only):")
    logger.info("  python scripts/train_model.py \\")
    logger.info("      --data         %s \\", args.out / "features.npy")
    logger.info("      --labels       %s \\", stage2_link)
    logger.info("      --class-filter 1,2 \\")
    logger.info("      --out          %s/stage2_models \\", args.out)
    logger.info("      --mlp --epochs 50")
    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
