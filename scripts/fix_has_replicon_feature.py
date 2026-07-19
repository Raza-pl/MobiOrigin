#!/usr/bin/env python3
"""Patch the `has_replicon` feature in the marker-model training set.

Background
----------
`has_replicon` (MOB-suite replicon typing via nucleotide alignment against
`rep.dna.fas`) has been 0 for every one of the 90,000 rows in
`data/marker_features_balanced_28_genomad.npz` — the XGBoost marker
classifier that ships in production never saw a single positive example of
this feature during training. `has_rep_protein` (the protein-level replicon
signal) and the other biological-evidence features are populated normally;
only this one nucleotide-level feature was silently always zero.

At inference time, `predict.py` compensates with a hard-coded post hoc rule
("replicon boost": transfer 65% of non-plasmid probability mass to plasmid
when has_replicon>=1 and mlp>=0.15) — a train/serve mismatch. The model has
no learned coefficient for has_replicon; the boost is a guess standing in
for what training should have taught it directly.

This script recomputes has_replicon correctly for the 90k training rows by
aligning each class's training FASTA against `rep.dna.fas` and patching the
NPZ column in place, using the same match criteria (query coverage >= 60%,
identity >= 80%) as the original (unfinished, uncommitted) attempt at this
fix in scripts/dev/rebuild_npz_with_replicons.py. It intentionally does NOT
perform the "hard FN/FP augmentation" step from that script, which mined
extra training rows from a benchmark-prediction file — mixing benchmark and
training data risks leakage and is out of scope for this fix.

Uses `mappy` (the minimap2 Python binding) instead of shelling out to the
`minimap2` CLI, since the CLI binary is not guaranteed to be on PATH.

Usage
-----
    python scripts/fix_has_replicon_feature.py \\
        --base-npz data/marker_features_balanced_28_genomad.npz \\
        --marker-work data/marker_work \\
        --rep-db data/databases/mob_suite/rep.dna.fas \\
        --out data/marker_features_balanced_28_genomad.npz

Then retrain and redistribute the deployed model:
    python scripts/train_marker_model.py \\
        --features data/marker_features_balanced_28_genomad.npz \\
        --out data/models/
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import mappy
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# NPZ row order (matches build_marker_dataset.py / marker_features_balanced_28_genomad.npz)
CLASSES = [
    ("plasmid", "plasmid_training.fna", "plasmid_proteins.faa", 0, 30_000),
    ("chromosome", "chromosome_training.fna", "chromosome_proteins.faa", 30_000, 60_000),
    ("phage", "phage_training.fna", "phage_proteins.faa", 60_000, 90_000),
]

MIN_QCOV = 0.60
MIN_IDENTITY = 0.80


def get_contig_ids_from_proteins(faa_path: Path) -> list[str]:
    """Recover NPZ row -> contig_id order from ORF headers ('{contig_id}_{orf_index}')."""
    seen: dict[str, None] = {}
    with open(faa_path) as fh:
        for line in fh:
            if line.startswith(">"):
                orf_id = line[1:].split()[0].strip()
                cid = re.sub(r"_\d+$", "", orf_id)
                if cid not in seen:
                    seen[cid] = None
    return list(seen.keys())


def find_replicon_hits(fasta: Path, rep_db: Path, threads: int = 4) -> set[str]:
    """Return contig IDs in *fasta* with a qualifying hit against *rep_db*.

    Equivalent to `minimap2 -c -x asm20 --secondary=no rep_db fasta`, filtered
    to *replicon*-side (target) coverage >= 60% and identity >= 80% -- i.e.
    "at least 60% of a known replicon marker sequence was found in this
    window, at >=80% identity." rep_db (the reference/target index) is built
    from the ~300-1500bp replicon marker sequences; fasta windows (the query)
    are 2-10kb, so it's the replicon-side coverage that's meaningful here,
    not window-side coverage (a window is almost always much longer than the
    marker it contains). mappy's r_st/r_en/ctg_len are target-side
    coordinates when the Aligner is built from rep_db, matching PAF's
    tstart/tend/tlen for `minimap2 rep_db fasta`.
    """
    aligner = mappy.Aligner(str(rep_db), preset="asm20", n_threads=threads)
    if not aligner:
        raise RuntimeError(f"Failed to build mappy index from {rep_db}")

    hits: set[str] = set()
    n_seqs = 0
    for name, seq, _qual in mappy.fastx_read(str(fasta)):
        n_seqs += 1
        if len(seq) == 0:
            continue
        for hit in aligner.map(seq):
            tcov = (hit.r_en - hit.r_st) / hit.ctg_len if hit.ctg_len else 0.0
            identity = hit.mlen / hit.blen if hit.blen else 0.0
            if tcov >= MIN_QCOV and identity >= MIN_IDENTITY:
                hits.add(name)
                break
    log.info("  scanned %d sequences, %d with qualifying replicon hit", n_seqs, len(hits))
    return hits


def patch_has_replicon(
    X: np.ndarray, feat_names: list[str], marker_work: Path, rep_db: Path
) -> np.ndarray:
    rep_idx = feat_names.index("has_replicon")
    log.info("has_replicon column index: %d", rep_idx)
    log.info("Current has_replicon=1 count: %d / %d", int((X[:, rep_idx] > 0).sum()), len(X))

    X = X.copy()
    for label, fasta_name, faa_name, row_start, row_end in CLASSES:
        fasta = marker_work / fasta_name
        faa = marker_work / faa_name
        n_rows = row_end - row_start
        log.info("[%s] patching rows %d-%d (%d rows) ...", label, row_start, row_end - 1, n_rows)

        if not faa.exists() or not fasta.exists():
            log.warning("  missing %s or %s -- skipping class", fasta, faa)
            continue

        ids = get_contig_ids_from_proteins(faa)
        if len(ids) != n_rows:
            log.warning(
                "  ID count mismatch: %d ids vs %d NPZ rows -- truncating/padding", len(ids), n_rows
            )
            ids = ids[:n_rows] if len(ids) > n_rows else ids + [""] * (n_rows - len(ids))

        hit_ids = find_replicon_hits(fasta, rep_db)

        patched = 0
        for offset, seq_id in enumerate(ids):
            if seq_id in hit_ids:
                X[row_start + offset, rep_idx] = 1.0
                patched += 1
        log.info("  [%s] patched %d / %d rows to has_replicon=1", label, patched, n_rows)

    log.info("Total has_replicon=1 after patching: %d / %d", int((X[:, rep_idx] > 0).sum()), len(X))
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-npz", type=Path, required=True)
    parser.add_argument("--marker-work", type=Path, required=True)
    parser.add_argument("--rep-db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    data = np.load(args.base_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feat_names = [str(f) for f in data["feature_names"]]

    X_patched = patch_has_replicon(X, feat_names, args.marker_work, args.rep_db)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X_patched, y=y, feature_names=np.array(feat_names))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
