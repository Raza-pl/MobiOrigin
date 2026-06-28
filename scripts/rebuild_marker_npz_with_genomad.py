"""Rebuild marker_features_balanced_28.npz with real geNomad SPM features.

After running:
  1. build_training_fasta.py  → data/marker_work/{plasmid,chromosome,phage}_training.fna
  2. genomad annotate (per class) → data/marker_work/{label}_genomad_ann/

This script:
  - Loads the existing balanced marker_features.npz (16-feature NPZ)
  - Loads geNomad genes TSVs for each class
  - Computes 12 geNomad SPM features per contig
  - Expands the NPZ to 28 features: [3 MLP + 13 MOB-suite + log10_length + 12 geNomad]
  - Saves data/marker_features_balanced_28_genomad.npz

Usage
-----
  python scripts/rebuild_marker_npz_with_genomad.py \\
      --base-npz   data/marker_features_balanced_28.npz \\
      --ann-dir    data/marker_work \\
      --out        data/marker_features_balanced_28_genomad.npz

  # Then retrain:
  python scripts/train_marker_model.py \\
      --features data/marker_features_balanced_28_genomad.npz \\
      --out      data/models/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# geNomad feature columns (must match extract_genomad_features.GENOMAD_COLS)
GENOMAD_COLS = [
    "p_marker_freq", "c_marker_freq", "v_marker_freq", "pp_marker_freq",
    "median_p_spm", "median_c_spm", "median_v_spm", "p_vs_c_logistic",
    "strand_switch_rate", "no_rbs_freq", "canonical_sd_freq", "n_plasmid_markers",
]
N_GENOMAD = len(GENOMAD_COLS)

# Default neutral values for each geNomad feature when missing
_ZERO_GN = {
    "p_marker_freq": 0.0, "c_marker_freq": 0.0, "v_marker_freq": 0.0,
    "pp_marker_freq": 0.0, "median_p_spm": 0.0, "median_c_spm": 0.0,
    "median_v_spm": 0.0, "p_vs_c_logistic": 0.5,  # 0.5 = neutral (no signal)
    "strand_switch_rate": 0.0, "no_rbs_freq": 0.0, "canonical_sd_freq": 0.0,
    "n_plasmid_markers": 0.0,
}

# Label index → class name (must match IDX_TO_CLASS in plasflow2)
IDX_TO_CLASS = {0: "plasmid", 1: "chromosome", 2: "phage"}


def load_genomad_features_for_class(ann_dir: Path, label: str) -> dict[str, list[float]]:
    """Load geNomad SPM features for one class from its genomad annotate output.

    Looks for: {ann_dir}/{label}_genomad_ann/{label}_training_genes.tsv

    Returns {contig_id: [12 float features]} or {} if not found.
    """
    try:
        from extract_genomad_features import extract_all as gn_extract_all  # type: ignore
    except ImportError:
        log.error("Cannot import extract_genomad_features — ensure scripts/ is on sys.path")
        raise

    # geNomad annotate puts output in a subdirectory named {input_stem}_annotate/
    # e.g. plasmid_genomad_ann/plasmid_training_annotate/plasmid_training_genes.tsv
    ann_subdir = ann_dir / f"{label}_genomad_ann"
    genes_tsv = ann_subdir / f"{label}_training_annotate" / f"{label}_training_genes.tsv"

    # Fallback: geNomad v1.x may put it directly in ann_subdir
    if not genes_tsv.exists():
        flat = ann_subdir / f"{label}_training_genes.tsv"
        if flat.exists():
            genes_tsv = flat
        else:
            log.warning("[%s] genes TSV not found at:\n  %s\n  %s", label, genes_tsv, flat)
            return {}

    log.info("[%s] Loading geNomad features from %s", label, genes_tsv)
    features = gn_extract_all(genes_tsv)  # {contig_id: {col: float}}
    log.info("[%s] %d contigs with geNomad features", label, len(features))

    # Convert to ordered list of 12 values per contig
    result: dict[str, list[float]] = {}
    zero = [_ZERO_GN[c] for c in GENOMAD_COLS]
    for cid, feat_dict in features.items():
        result[cid] = [feat_dict.get(c, _ZERO_GN[c]) for c in GENOMAD_COLS]

    return result


def _extract_contig_ids_from_proteins(proteins_dir: Path, y: np.ndarray) -> list[str]:
    """Recover per-row contig IDs by reading protein FASTAs in class order.

    build_marker_dataset.py writes proteins in the same sequence order as the
    feature matrix rows (class order: plasmid → chromosome → phage).  Each ORF
    header is '{contig_id}_{orf_index}'; stripping the suffix recovers the ID.

    Args:
        proteins_dir: directory containing *_proteins.faa files
        y: label array (shape N,) to determine expected class sizes

    Returns:
        list of contig IDs aligned to rows in X/y
    """
    import re
    classes = [(0, "plasmid"), (1, "chromosome"), (2, "phage")]
    ids_by_class: dict[int, list[str]] = {}
    for cls_idx, cls_name in classes:
        faa = proteins_dir / f"{cls_name}_proteins.faa"
        if not faa.exists():
            log.warning("  Missing %s — using empty contig IDs for class %d", faa, cls_idx)
            ids_by_class[cls_idx] = [""] * int((y == cls_idx).sum())
            continue
        # Extract unique contig IDs in appearance order
        seen: dict[str, None] = {}
        with open(faa) as fh:
            for line in fh:
                if line.startswith(">"):
                    orf_id = line[1:].split()[0].strip()
                    cid = re.sub(r"_\d+$", "", orf_id)
                    if cid not in seen:
                        seen[cid] = None
        ids = list(seen.keys())
        expected = int((y == cls_idx).sum())
        log.info("  %s protein FAA → %d unique contig IDs (expected %d rows)",
                 cls_name, len(ids), expected)
        if len(ids) != expected:
            log.warning("  Count mismatch for %s: %d IDs vs %d rows in NPZ",
                        cls_name, len(ids), expected)
        ids_by_class[cls_idx] = ids

    # Reconstruct full list in row order (classes appear contiguously in y)
    all_ids: list[str] = []
    prev_idx = None
    for row_idx in range(len(y)):
        cls_idx = int(y[row_idx])
        if cls_idx != prev_idx:
            # Start of a new class block — reset per-class counter
            _counters[cls_idx] = 0
            prev_idx = cls_idx
        pos = _counters[cls_idx]
        cls_ids = ids_by_class.get(cls_idx, [])
        all_ids.append(cls_ids[pos] if pos < len(cls_ids) else "")
        _counters[cls_idx] = pos + 1
    return all_ids

_counters: dict[int, int] = {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild marker NPZ with real geNomad SPM features"
    )
    parser.add_argument("--base-npz", type=Path,
                        default=Path("data/marker_features_balanced_28.npz"),
                        help="Existing balanced NPZ (16- or 28-feature)")
    parser.add_argument("--ann-dir", type=Path,
                        default=Path("data/marker_work"),
                        help="Directory containing {label}_genomad_ann/ subdirs")
    parser.add_argument("--out", type=Path,
                        default=Path("data/marker_features_balanced_28_genomad.npz"),
                        help="Output NPZ path")
    parser.add_argument("--proteins-dir", type=Path,
                        default=Path("data/marker_work"),
                        help="Directory with {plasmid,chromosome,phage}_proteins.faa "
                             "(used to recover contig ID order)")
    parser.add_argument("--id-map", type=Path, default=None,
                        help="Optional TSV mapping row_index → contig_id (if NPZ lacks IDs)")
    args = parser.parse_args()

    # Load base NPZ
    log.info("Loading base NPZ: %s", args.base_npz)
    data = np.load(args.base_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feat_names_raw = data["feature_names"] if "feature_names" in data else None
    contig_ids_raw = data["contig_ids"] if "contig_ids" in data else None
    log.info("Base NPZ: X=%s  y=%s", X.shape, y.shape)

    n_base_features = X.shape[1]
    log.info("Base features: %d", n_base_features)

    # Determine if we need to add geNomad block or replace existing zero-padded one
    if n_base_features == 16:
        log.info("16-feature NPZ detected — will append 12 geNomad features → 28")
        gn_start = None  # append
    elif n_base_features == 28:
        log.info("28-feature NPZ detected — will replace last 12 (geNomad) columns")
        gn_start = 16  # replace columns 16-27
    else:
        log.error("Unexpected feature count: %d (expected 16 or 28)", n_base_features)
        sys.exit(1)

    # Load contig IDs if available
    if contig_ids_raw is not None:
        contig_ids = [str(c) for c in contig_ids_raw]
        log.info("Contig IDs found in NPZ: %d", len(contig_ids))
    elif args.id_map is not None:
        import csv
        with open(args.id_map) as fh:
            contig_ids = [row["contig_id"] for row in csv.DictReader(fh, delimiter="\t")]
        log.info("Loaded %d contig IDs from --id-map", len(contig_ids))
    elif args.proteins_dir is not None:
        # Recover contig ID order from protein FASTAs (written in same seq order as NPZ rows)
        contig_ids = _extract_contig_ids_from_proteins(args.proteins_dir, y)
        log.info("Recovered %d contig IDs from protein FASTAs", len(contig_ids))
    else:
        contig_ids = None
        log.warning("No contig IDs available — geNomad features will be assigned by class index order")

    # Load geNomad features per class
    class_gn: dict[int, dict[str, list[float]]] = {}
    for idx, name in IDX_TO_CLASS.items():
        class_gn[idx] = load_genomad_features_for_class(args.ann_dir, name)

    # Build new geNomad feature block
    zero_vals = [_ZERO_GN[c] for c in GENOMAD_COLS]
    gn_block = np.zeros((len(y), N_GENOMAD), dtype=np.float32)

    matched = 0
    for row_i in range(len(y)):
        label_idx = int(y[row_i])
        gn_dict = class_gn.get(label_idx, {})

        if contig_ids is not None:
            cid = contig_ids[row_i]
            if cid in gn_dict:
                gn_block[row_i] = gn_dict[cid]
                matched += 1
            else:
                # Default: neutral values
                gn_block[row_i] = zero_vals
        else:
            # No ID map — can't do per-contig lookup
            gn_block[row_i] = zero_vals

    log.info("Matched %d / %d rows with geNomad features (%.1f%%)",
             matched, len(y), 100.0 * matched / max(len(y), 1))

    # Warn if match rate is low
    if matched < len(y) * 0.5:
        log.warning("Low match rate — geNomad annotate may not have covered all training contigs")
        log.warning("Consider re-running genomad annotate on the training FASTAs")

    # Build final X matrix
    # Set p_vs_c_logistic (index 7 of geNomad block) to 0.5 for unmatched rows
    p_vs_c_idx = GENOMAD_COLS.index("p_vs_c_logistic")
    for row_i in range(len(y)):
        if gn_block[row_i, p_vs_c_idx] == 0.0:
            gn_block[row_i, p_vs_c_idx] = 0.5

    if gn_start is None:
        # Append to 16-feature base
        X_new = np.hstack([X, gn_block])
    else:
        # Replace last 12 columns (positions gn_start:gn_start+12)
        X_new = X.copy()
        X_new[:, gn_start:gn_start + N_GENOMAD] = gn_block

    # Feature names — must match MARKER_FEATURE_NAMES in marker_classifier.py
    # (log10_length=12, gc_content=13, coding_density=14, n_orfs_per_kb=15)
    base_16 = [
        "mlp_plasmid_score", "mlp_chromosome_score", "mlp_phage_score",
        "is_conjugative", "is_mobilizable", "has_replicon", "has_ice", "has_rep_protein",
        "n_arg_per_kb", "n_mge_per_kb", "n_ice_per_kb", "n_rep_per_kb",
        "log10_length", "gc_content", "coding_density", "n_orfs_per_kb",
    ]
    # If base NPZ already has feature_names, use those for the non-geNomad columns
    if feat_names_raw is not None and len(feat_names_raw) >= 16:
        base_16 = list(feat_names_raw[:16])
    feature_names = base_16 + GENOMAD_COLS
    assert len(feature_names) == 28

    log.info("Final shape: X=%s  y=%s", X_new.shape, y.shape)

    # Class distribution
    for idx, name in IDX_TO_CLASS.items():
        count = int((y == idx).sum())
        log.info("  %-12s  %6d  (%.1f%%)", name, count, 100.0 * count / len(y))

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict = dict(X=X_new, y=y, feature_names=np.array(feature_names, dtype=str))
    if contig_ids is not None:
        save_kwargs["contig_ids"] = np.array(contig_ids, dtype=str)
    np.savez_compressed(args.out, **save_kwargs)
    log.info("Saved → %s", args.out)
    log.info("")
    log.info("Next: retrain XGBoost with real geNomad features:")
    log.info("  python scripts/train_marker_model.py \\")
    log.info("      --features %s \\", args.out)
    log.info("      --out      data/models/")


if __name__ == "__main__":
    main()
