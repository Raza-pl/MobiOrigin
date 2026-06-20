"""Update MLP score columns in existing marker_features.npz.

The existing marker_features.npz has real DIAMOND biological features
(relaxase / MPF / rep protein hits) but the MLP score columns (cols 0-2)
were computed with the OLD 1365-dim model.  This script updates only those
columns using the NEW 1493-dim MLP, preserving the biological features.

Strategy
--------
Re-sample the same number of sequences per class (same seed) from the
original sources, run the new MLP, and overwrite cols 0-2.  The biological
feature columns 3-11 are left intact.

Runtime (Apple M-series): ~30 min for 90k sequences (MLP + k=6 features)

Usage
-----
    python scripts/update_marker_mlp_scores.py
    python scripts/update_marker_mlp_scores.py --dry-run   # check shapes only
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import random
import shutil
import sys
from pathlib import Path

# ── macOS ARM segfault fix: cap BLAS threads before numpy/torch import ───────
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from plasflow2.classify.features import extract_features  # noqa: E402
from plasflow2.classify.model import load_model  # noqa: E402
from plasflow2.utils.device import get_device  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

WINDOW_SIZES = (2000, 5000, 10_000)
BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Sequence loading (same sources as build_marker_dataset.py)
# ---------------------------------------------------------------------------

def _iter_fasta(path: Path, gzipped: bool = False):
    opener = gzip.open(path, "rt") if gzipped else open(path)
    with opener as fh:
        cur_id, parts = None, []
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, "".join(parts)
                cur_id = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if cur_id is not None:
            yield cur_id, "".join(parts)


def _tile(seq: str) -> list[str]:
    frags = []
    for w in WINDOW_SIZES:
        step = max(1, w // 2)
        for s in range(0, len(seq) - w + 1, step):
            frags.append(seq[s : s + w])
    return frags


def load_plasmid_seqs(n: int, seed: int) -> list[str]:
    seqs: list[str] = []
    for fname in ("plsdb.fasta", "COMPASS.fna"):
        fpath = ROOT / "data/databases/plasmids" / fname
        if fpath.exists():
            for _, seq in _iter_fasta(fpath):
                if len(seq) >= 1000:
                    seqs.append(seq.upper())
    logger.info("  Plasmid pool: %d sequences", len(seqs))
    rng = random.Random(seed)
    rng.shuffle(seqs)
    return seqs[:n]


def load_chrom_seqs(n: int, seed: int) -> list[str]:
    frags: list[str] = []
    chrom_fna = ROOT / "data/databases/chromosomes.fna"
    chrom_dir = ROOT / "data/chromosomes/bacteria"

    if chrom_fna.exists():
        for _, seq in _iter_fasta(chrom_fna):
            frags.extend(_tile(seq.upper()))
        logger.info("  chromosomes.fna: %d windows", len(frags))

    if chrom_dir.is_dir():
        files = list(chrom_dir.glob("*.fna"))
        rng2 = random.Random(seed + 1)
        rng2.shuffle(files)
        before = len(frags)
        for fpath in files[:150]:
            for sid, seq in _iter_fasta(fpath):
                if "plasmid" in sid.lower():
                    continue
                frags.extend(_tile(seq.upper()))
        logger.info("  chrom_dir added %d windows", len(frags) - before)

    rng = random.Random(seed)
    rng.shuffle(frags)
    return frags[:n]


def load_phage_seqs(n: int, seed: int) -> list[str]:
    seqs: list[str] = []
    inphared = ROOT / "data/databases/inphared/inphared_phages.fa.gz"
    if inphared.exists():
        for _, seq in _iter_fasta(inphared, gzipped=True):
            if 5000 <= len(seq) <= 300_000:
                seqs.append(seq.upper())
    logger.info("  Phage pool: %d sequences", len(seqs))
    rng = random.Random(seed)
    rng.shuffle(seqs)
    return seqs[:n]


# ---------------------------------------------------------------------------
# MLP inference
# ---------------------------------------------------------------------------

def run_mlp_scores(
    sequences: list[str],
    model,
    device,
    k6_pca_path=None,
) -> np.ndarray:
    """Return (N, 3) softmax prob array: [plasmid, chromosome, phage].

    For binary models (num_classes=2), the phage column is padded with 0.0.
    """
    import torch

    X = extract_features(sequences, k6_pca_path=k6_pca_path)
    logger.info("    Feature matrix: %s", X.shape)
    all_probs: list[np.ndarray] = []
    for start in range(0, len(X), BATCH_SIZE):
        batch = torch.tensor(X[start : start + BATCH_SIZE]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    probs_all = np.vstack(all_probs).astype(np.float32)

    # Binary model (2-class) — pad phage column with 0.0 so callers always
    # receive a (N, 3) array in [plasmid, chromosome, phage] order.
    if probs_all.shape[1] == 2:
        pad = np.zeros((len(probs_all), 1), dtype=np.float32)
        probs_all = np.hstack([probs_all, pad])
        logger.info("    Binary model: padded phage column → shape %s", probs_all.shape)

    return probs_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Update MLP score columns in marker_features.npz")
    parser.add_argument("--npz",    type=Path, default=ROOT / "data/marker_features.npz")
    parser.add_argument("--model",  type=Path, default=ROOT / "data/models/mlp_v2.pt")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print shapes and exit without updating")
    parser.add_argument("--k6-pca-path", type=Path, default=None,
                        help="Custom k=6 PCA path (default: data/models/k6_pca.pkl)")
    args = parser.parse_args()

    # Load existing features
    logger.info("Loading %s …", args.npz)
    data = np.load(args.npz, allow_pickle=True)
    X = data["X"].copy().astype(np.float32)
    y = data["y"].copy().astype(np.int64)
    feat_names = data["feature_names"]
    logger.info("Existing: X=%s  y=%s", X.shape, y.shape)

    class_counts = {int(v): int(c) for v, c in zip(*np.unique(y, return_counts=True))}
    logger.info("Class counts: %s (0=plasmid, 1=chromosome, 2=phage)", class_counts)
    logger.info("Biological features (before update):")
    logger.info("  is_conjugative nonzero: %d", (X[:, 3] > 0).sum())
    logger.info("  has_rep_protein nonzero: %d", (X[:, 7] > 0).sum())

    if args.dry_run:
        logger.info("DRY RUN: skipping MLP update")
        return

    # Load model
    logger.info("Loading MLP from %s …", args.model)
    device = get_device()
    model = load_model(args.model, device=device)
    model.eval()

    # Detect binary model and remap phage → chromosome labels
    is_binary = (model.net[11].out_features == 2)
    if is_binary:
        logger.info("Binary model detected (num_classes=2).")
        logger.info("  Remapping phage labels (y=2) → chromosome (y=1) in saved NPZ.")
        n_before = int((y == 2).sum())
        y[y == 2] = 1
        logger.info("  Remapped %d phage samples to chromosome.", n_before)
        logger.info("  New class distribution: %s", dict(zip(*np.unique(y, return_counts=True))))

    # Process each class
    n_plasmid = class_counts.get(0, 0)
    n_chrom   = class_counts.get(1, 0)
    n_phage   = class_counts.get(2, 0)

    new_cols = np.zeros((len(X), 3), dtype=np.float32)

    logger.info("=== Plasmid (%d sequences) ===", n_plasmid)
    plas_seqs = load_plasmid_seqs(n_plasmid, args.seed)
    plas_probs = run_mlp_scores(plas_seqs[:n_plasmid], model, device, k6_pca_path=args.k6_pca_path)
    new_cols[:n_plasmid] = plas_probs[:n_plasmid]
    logger.info("  Plasmid score mean (plasmid col): %.4f", plas_probs[:, 0].mean())

    logger.info("=== Chromosome (%d sequences) ===", n_chrom)
    chrom_seqs = load_chrom_seqs(n_chrom, args.seed)
    chrom_probs = run_mlp_scores(chrom_seqs[:n_chrom], model, device, k6_pca_path=args.k6_pca_path)
    new_cols[n_plasmid : n_plasmid + n_chrom] = chrom_probs[:n_chrom]
    logger.info("  Chromosome score mean (chrom col): %.4f", chrom_probs[:, 1].mean())

    logger.info("=== Phage (%d sequences) ===", n_phage)
    phage_seqs = load_phage_seqs(n_phage, args.seed)
    phage_probs = run_mlp_scores(phage_seqs[:n_phage], model, device, k6_pca_path=args.k6_pca_path)
    new_cols[n_plasmid + n_chrom :] = phage_probs[:n_phage]
    logger.info("  Phage score mean (phage col): %.4f", phage_probs[:, 2].mean())

    # Overwrite cols 0-2
    X[:, 0] = new_cols[:, 0]  # mlp_plasmid_score
    X[:, 1] = new_cols[:, 1]  # mlp_chromosome_score
    X[:, 2] = new_cols[:, 2]  # mlp_phage_score

    # Verify biological features untouched
    logger.info("Verification (biological features should be unchanged):")
    logger.info("  is_conjugative nonzero: %d", (X[:, 3] > 0).sum())
    logger.info("  has_rep_protein nonzero: %d", (X[:, 7] > 0).sum())

    # Backup + save
    backup = args.npz.parent / "marker_features_old_mlp.npz"
    shutil.copy(args.npz, backup)
    logger.info("Backed up original → %s", backup)

    np.savez(args.npz, X=X, y=y, feature_names=feat_names)
    logger.info("Saved updated marker_features.npz → %s", args.npz)
    logger.info("Done. Now run: python scripts/train_marker_model.py "
                "--features %s --out data/models/", args.npz)


if __name__ == "__main__":
    main()
