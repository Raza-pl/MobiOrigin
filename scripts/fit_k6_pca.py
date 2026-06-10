"""Fit the k=6 PCA model from current training sequences.

Samples up to 100k sequences across all three classes (plasmid, chromosome,
phage) to ensure the PCA captures k=6 variance from all sequence types.

Sources used:
  Plasmids   : data/databases/plasmids/plsdb.fasta
               data/databases/plasmids/COMPASS.fna
  Chromosomes: data/databases/chromosomes.fna  (tiled into contig windows)
  Phages     : data/databases/inphared/inphared_phages.fa.gz

Output:
  data/models/k6_pca.pkl   (128-component IncrementalPCA)

Usage:
    python scripts/fit_k6_pca.py
    python scripts/fit_k6_pca.py --n-per-class 40000 --n-components 128
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import sys
from pathlib import Path

import numpy as np

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plasflow2.classify.features import fit_k6_pca, kmer_vector, K6_RAW_DIM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

WINDOW_SIZES = (1000, 2000, 5000, 10_000)


# ---------------------------------------------------------------------------
# FASTA streaming (no BioPython dependency)
# ---------------------------------------------------------------------------

def _iter_fasta(path: Path, gzipped: bool = False):
    """Yield (full_header, sequence) from a FASTA file.

    full_header includes the entire description line (after '>') so callers
    can filter on keywords like 'plasmid' that appear in the description
    but not in the bare sequence ID.
    """
    opener = gzip.open(path, "rt") if gzipped else open(path)
    with opener as fh:
        cur_id, parts = None, []
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, "".join(parts)
                cur_id = line[1:]   # full description, not just first token
                parts = []
            else:
                parts.append(line)
        if cur_id is not None:
            yield cur_id, "".join(parts)


def _tile_sequence(seq: str, window_sizes=WINDOW_SIZES, step_fraction=0.5,
                   min_len=1000) -> list[str]:
    """Tile a sequence into contig-sized windows."""
    fragments = []
    for w in window_sizes:
        step = max(1, int(w * step_fraction))
        for start in range(0, len(seq) - w + 1, step):
            frag = seq[start:start + w]
            if len(frag) >= min_len:
                fragments.append(frag)
    return fragments


# ---------------------------------------------------------------------------
# Per-class sequence loading
# ---------------------------------------------------------------------------

def load_plasmid_seqs(plasmid_dir: Path, n: int, seed: int = 42) -> list[str]:
    """Sample n plasmid sequences from PLSDB + COMPASS."""
    seqs = []
    for fname in ("plsdb.fasta", "COMPASS.fna"):
        fpath = plasmid_dir / fname
        if not fpath.exists():
            logger.warning("Not found: %s", fpath)
            continue
        for _, seq in _iter_fasta(fpath):
            if len(seq) >= 1000:
                seqs.append(seq)
    logger.info("Plasmid sequences loaded: %d", len(seqs))
    rng = random.Random(seed)
    rng.shuffle(seqs)
    return seqs[:n]


def load_chromosome_seqs(chrom_fna: Path, n: int, seed: int = 42,
                         chrom_dir: Path | None = None) -> list[str]:
    """Tile chromosomes into contig windows and sample n.

    If chrom_dir is provided, tiles all .fna files in that directory
    (output of download_refseq_chromosomes.py) for better diversity.
    Also tiles chrom_fna if it exists, combining both sources.
    """
    fragments = []

    # Tile the base chromosomes.fna
    if chrom_fna.exists():
        for _, seq in _iter_fasta(chrom_fna):
            fragments.extend(_tile_sequence(seq))
        logger.info("  chromosomes.fna: %d windows", len(fragments))

    # Tile per-assembly FASTAs from download_refseq_chromosomes.py.
    # Cap at max_files to avoid OOM — 150 genomes × ~5k windows = 750k fragments,
    # well within RAM; we sample down to n anyway.
    if chrom_dir and chrom_dir.is_dir():
        fna_files = list(chrom_dir.glob("*.fna"))
        rng2 = random.Random(seed + 1)
        rng2.shuffle(fna_files)
        max_files = 150
        selected = fna_files[:max_files]
        logger.info("  %s: tiling %d / %d .fna files (capped for RAM) …",
                    chrom_dir, len(selected), len(fna_files))
        before = len(fragments)
        for fpath in selected:
            for seq_id, seq in _iter_fasta(fpath):
                # Skip plasmid sequences in RefSeq genomic FNA files
                if "plasmid" in seq_id.lower():
                    continue
                fragments.extend(_tile_sequence(seq))
        logger.info("  chrom_dir added %d windows", len(fragments) - before)

    logger.info("Chromosome windows total: %d", len(fragments))
    rng = random.Random(seed)
    rng.shuffle(fragments)
    return fragments[:n]


def load_phage_seqs(inphared_gz: Path, n: int, seed: int = 42) -> list[str]:
    """Sample n phage sequences from INPHARED (gzipped FASTA)."""
    seqs = []
    for _, seq in _iter_fasta(inphared_gz, gzipped=True):
        if 5_000 <= len(seq) <= 300_000:
            seqs.append(seq)
    logger.info("Phage sequences loaded: %d", len(seqs))
    rng = random.Random(seed)
    rng.shuffle(seqs)
    return seqs[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fit k=6 PCA from training sequences")
    parser.add_argument("--plasmid-dir", type=Path,
                        default=Path("data/databases/plasmids"),
                        help="Directory with plsdb.fasta and COMPASS.fna")
    parser.add_argument("--chrom-fna", type=Path,
                        default=Path("data/databases/chromosomes.fna"),
                        help="Chromosome FASTA (tiled into windows)")
    parser.add_argument("--chrom-dir", type=Path,
                        default=None,
                        help="Directory of per-assembly .fna files (output of download_refseq_chromosomes.py)")
    parser.add_argument("--inphared-gz", type=Path,
                        default=Path("data/databases/inphared/inphared_phages.fa.gz"),
                        help="INPHARED phage FASTA (gzipped)")
    parser.add_argument("--out", type=Path,
                        default=Path("data/models/k6_pca.pkl"),
                        help="Output path for PCA pickle")
    parser.add_argument("--n-per-class", type=int, default=33_000,
                        help="Sequences per class for PCA fitting (total ~3x this, max 100k)")
    parser.add_argument("--n-components", type=int, default=128,
                        help="PCA output dimensionality (default: 128)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve relative paths from the project root
    root = Path(__file__).parent.parent
    for attr in ("plasmid_dir", "chrom_fna", "inphared_gz", "out"):
        p = getattr(args, attr)
        if not p.is_absolute():
            setattr(args, attr, root / p)
    if args.chrom_dir and not args.chrom_dir.is_absolute():
        args.chrom_dir = root / args.chrom_dir

    logger.info("=== Fitting k=6 PCA (%d components) ===", args.n_components)
    logger.info("Sampling %d sequences per class", args.n_per_class)

    # Load sequences from each class
    plasmid_seqs = load_plasmid_seqs(args.plasmid_dir, args.n_per_class, args.seed)
    chrom_seqs   = load_chromosome_seqs(args.chrom_fna, args.n_per_class, args.seed,
                                        chrom_dir=args.chrom_dir)
    phage_seqs   = load_phage_seqs(args.inphared_gz, args.n_per_class, args.seed)

    all_seqs = plasmid_seqs + chrom_seqs + phage_seqs
    logger.info("Total sequences for PCA fitting: %d", len(all_seqs))

    # Shuffle so PCA batches aren't class-stratified
    rng = random.Random(args.seed)
    rng.shuffle(all_seqs)

    # Fit and save
    pca = fit_k6_pca(all_seqs, n_components=args.n_components, out_path=args.out)
    logger.info("Explained variance ratio sum: %.4f",
                float(pca.explained_variance_ratio_.sum()))
    logger.info("k=6 PCA saved → %s", args.out)
    logger.info("Done. Now run build_dataset.py — it will auto-detect the PCA.")


if __name__ == "__main__":
    main()
