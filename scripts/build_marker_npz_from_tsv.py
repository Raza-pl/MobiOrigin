"""Build marker_features.npz from benchmark TSV files.

Joins three benchmark TSV files on contig_id:
  - plasflow2_predictions.tsv  → MLP scores (plasmid/chromosome/phage)
  - annotations_with_genomad.tsv → MOB-suite + geNomad SPM features (26 cols)
  - ground_truth.tsv            → true labels

Produces a .npz compatible with train_marker_model.py.

Usage
-----
  python scripts/build_marker_npz_from_tsv.py \\
      --predictions  data/benchmark/results/plasflow2_predictions.tsv \\
      --annotations  data/benchmark/annotations_with_genomad.tsv \\
      --labels       data/benchmark/ground_truth.tsv \\
      --out          data/marker_features_genomad.npz

  # Without geNomad (just MOB-suite, for comparison):
  python scripts/build_marker_npz_from_tsv.py \\
      --predictions  data/benchmark/results/plasflow2_predictions.tsv \\
      --annotations  data/benchmark/annotations.tsv \\
      --labels       data/benchmark/ground_truth.tsv \\
      --out          data/marker_features_mobonly.npz
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Label encoding — must match IDX_TO_CLASS in plasflow2
LABEL_TO_IDX = {"plasmid": 0, "chromosome": 1, "phage": 2, "virus": 2}

# Feature columns from annotations TSV (in this order)
MOB_SUITE_COLS = [
    "is_conjugative", "is_mobilizable", "has_replicon", "has_ice", "has_rep_protein",
    "n_arg_per_kb", "n_mge_per_kb", "n_ice_per_kb", "n_rep_per_kb",
    "coding_density", "n_orfs_per_kb", "gc_content", "length_bp",
]

GENOMAD_COLS = [
    "p_marker_freq", "c_marker_freq", "v_marker_freq", "pp_marker_freq",
    "median_p_spm", "median_c_spm", "median_v_spm", "p_vs_c_logistic",
    "strand_switch_rate", "no_rbs_freq", "canonical_sd_freq", "n_plasmid_markers",
]


def load_tsv(path: Path) -> dict[str, dict]:
    """Load a TSV into {contig_id: row_dict}."""
    rows = {}
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cid = (row.get("contig_id") or row.get("sequence_id") or "").strip()
            if cid:
                rows[cid] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build marker_features.npz from benchmark TSV files"
    )
    parser.add_argument("--predictions", type=Path,
                        default=Path("data/benchmark/results/plasflow2_predictions.tsv"),
                        help="PlasFlow v2 MLP predictions TSV")
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/benchmark/annotations_with_genomad.tsv"),
                        help="MOB-suite + geNomad annotation TSV")
    parser.add_argument("--labels", type=Path,
                        default=Path("data/benchmark/ground_truth.tsv"),
                        help="Ground truth TSV with contig_id and true_label columns")
    parser.add_argument("--out", type=Path,
                        default=Path("data/marker_features_genomad.npz"),
                        help="Output .npz path")
    parser.add_argument("--label-col", default="true_label",
                        help="Column name for labels in ground_truth TSV (default: true_label)")
    args = parser.parse_args()

    log.info("Loading predictions from %s …", args.predictions)
    preds = load_tsv(args.predictions)
    log.info("  %d rows", len(preds))

    log.info("Loading annotations from %s …", args.annotations)
    annots = load_tsv(args.annotations)
    log.info("  %d rows", len(annots))

    # Detect which annotation columns are present
    sample_annot = next(iter(annots.values())) if annots else {}
    ann_cols = [c for c in MOB_SUITE_COLS if c in sample_annot]
    gn_cols  = [c for c in GENOMAD_COLS  if c in sample_annot]
    has_genomad = bool(gn_cols)
    log.info("  MOB-suite cols found: %d / %d", len(ann_cols), len(MOB_SUITE_COLS))
    log.info("  geNomad cols found:   %d / %d", len(gn_cols), len(GENOMAD_COLS))

    log.info("Loading labels from %s …", args.labels)
    labels = load_tsv(args.labels)
    log.info("  %d rows", len(labels))

    # Build feature names list (matches existing marker_features.npz ordering)
    feature_names = (
        ["mlp_plasmid_score", "mlp_chromosome_score", "mlp_phage_score"]
        + ann_cols
        + ["log10_length"]
        + gn_cols
    )
    # Remove length_bp from ann_cols since we use log10_length instead
    if "length_bp" in feature_names:
        feature_names.remove("length_bp")
    # Deduplicate log10_length
    seen = set()
    feature_names = [f for f in feature_names if not (f in seen or seen.add(f))]

    log.info("Feature vector: %d features", len(feature_names))

    # Join all three on contig_id
    shared = set(preds) & set(annots) & set(labels)
    log.info("Shared contigs (all three files): %d", len(shared))

    if not shared:
        log.error("No shared contig IDs — check that all TSVs came from the same FASTA")
        raise SystemExit(1)

    rows_X, rows_y, skipped = [], [], 0
    for cid in sorted(shared):
        label_str = labels[cid].get(args.label_col, "").strip().lower()
        if label_str not in LABEL_TO_IDX:
            skipped += 1
            continue
        y_val = LABEL_TO_IDX[label_str]

        p = preds[cid]
        a = annots[cid]

        def _f(d: dict, col: str, default: float = 0.0) -> float:
            v = d.get(col, "")
            try:
                return float(v) if v not in ("", "NA", "None") else default
            except (ValueError, TypeError):
                return default

        length_bp = _f(a, "length_bp", 1000.0)
        log10_len = math.log10(max(length_bp, 1.0))

        row = [
            _f(p, "plasmid_score"),
            _f(p, "chromosome_score"),
            _f(p, "phage_score"),
        ]

        for col in ann_cols:
            if col == "length_bp":
                continue  # replaced by log10_length
            row.append(_f(a, col))

        row.append(log10_len)

        for col in gn_cols:
            row.append(_f(a, col))

        rows_X.append(row)
        rows_y.append(y_val)

    log.info("Skipped (unknown label): %d", skipped)

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int64)
    log.info("Final dataset: X=%s  y=%s", X.shape, y.shape)

    # Class distribution
    class_names = {0: "plasmid", 1: "chromosome", 2: "phage"}
    for idx, name in class_names.items():
        count = int((y == idx).sum())
        log.info("  %-12s  %6d  (%.1f%%)", name, count, 100 * count / len(y))

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        feature_names=np.array(feature_names, dtype=str),
    )
    log.info("Saved to %s", args.out)
    log.info("")
    log.info("Next: train XGBoost on this dataset:")
    log.info("  python scripts/train_marker_model.py \\")
    log.info("      --features %s \\", args.out)
    log.info("      --out      data/models/")


if __name__ == "__main__":
    main()
