"""Build the ground-truth benchmark FASTA from downloaded genomes.

Takes the downloaded chromosome + plasmid sequences and produces:
  1. A mixed FASTA (chromosome fragments + plasmid sequences)
  2. A ground truth TSV (contig_id → true_label → genome_source)
  3. Per-length-bin statistics

Fragmentation strategy
----------------------
Chromosomes: tiled into overlapping windows at 5 representative sizes
  (1 kb, 2 kb, 5 kb, 10 kb, 20 kb) with 50% step — simulates Illumina
  metagenomic assembly contigs.  Minimum fragment 1000 bp.

Plasmids:
  ≤ 50 kb:  used whole (a typical plasmid assembles as one contig)
  > 50 kb:  fragmented at 10 kb windows with 50% step (large plasmids often
             fragment in metagenomes)

Cap per genome: 2,000 chromosome fragments + all plasmid sequences.
Total expected: ~45,000 chromosome + ~1,200 plasmid fragments.

Usage
-----
    python scripts/build_benchmark.py \\
        --benchmark-dir data/benchmark \\
        --min-length 1000 \\
        --seed 42

Outputs
-------
    data/benchmark/benchmark.fna          — mixed FASTA for classification
    data/benchmark/ground_truth.tsv       — contig_id, true_label, genome, length
    data/benchmark/benchmark_stats.txt    — summary statistics
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from Bio import SeqIO  # type: ignore
from Bio.SeqRecord import SeqRecord  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHROM_WINDOW_SIZES = (1000, 2000, 5000, 10_000, 20_000)
PLASMID_LARGE_THRESHOLD = 50_000  # bp — plasmids above this are fragmented
PLASMID_FRAGMENT_SIZE   = 10_000  # bp — window size for large plasmid fragmentation
MAX_CHROM_FRAGS_PER_GENOME = 2_000


def fragment_sequence(
    seq: str,
    seq_id: str,
    window: int,
    step_fraction: float = 0.5,
    min_length: int = 1000,
) -> list[tuple[str, str]]:
    """Tile a sequence into (fragment_id, fragment_seq) pairs."""
    step = max(1, int(window * step_fraction))
    frags = []
    for start in range(0, len(seq) - window + 1, step):
        frag = seq[start: start + window]
        if len(frag) >= min_length and set(frag.upper()) <= set("ACGTN"):
            frags.append((f"{seq_id}_w{window}_s{start}", frag))
    return frags


def process_genome(
    accession: str,
    genome_dir: Path,
    min_length: int,
    seed: int,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Process one genome → (chrom_fragments, plasmid_fragments).

    Each item is (contig_id, sequence, true_label).
    true_label is 'chromosome' or 'plasmid'.
    """
    chrom_path = genome_dir / "chromosome.fna"
    plas_path  = genome_dir / "plasmids.fna"
    meta_path  = genome_dir / "metadata.json"

    if not chrom_path.exists() or chrom_path.stat().st_size == 0:
        logger.warning("  %s: chromosome.fna missing — skipping", accession)
        return [], []

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    organism = meta.get("organism", accession)

    # ── Chromosome: tile into windows ──────────────────────────────────────
    chrom_frags: list[tuple[str, str, str]] = []
    for rec in SeqIO.parse(str(chrom_path), "fasta"):
        seq = str(rec.seq).upper()
        if len(seq) < min_length:
            continue
        for w in CHROM_WINDOW_SIZES:
            frags = fragment_sequence(seq, f"{accession}_chr_{rec.id}", w, min_length=min_length)
            chrom_frags.extend((fid, fseq, "chromosome") for fid, fseq in frags)

    # Cap chromosomal fragments per genome to avoid imbalance
    rng = random.Random(seed)
    if len(chrom_frags) > MAX_CHROM_FRAGS_PER_GENOME:
        chrom_frags = rng.sample(chrom_frags, MAX_CHROM_FRAGS_PER_GENOME)

    # ── Plasmids: whole or fragmented ──────────────────────────────────────
    plas_frags: list[tuple[str, str, str]] = []
    if plas_path.exists() and plas_path.stat().st_size > 0:
        for rec in SeqIO.parse(str(plas_path), "fasta"):
            seq = str(rec.seq).upper()
            if len(seq) < min_length:
                continue
            plas_id = f"{accession}_plas_{rec.id}"
            if len(seq) <= PLASMID_LARGE_THRESHOLD:
                # Use whole plasmid sequence
                plas_frags.append((plas_id, seq, "plasmid"))
            else:
                # Fragment large plasmids
                frags = fragment_sequence(seq, plas_id, PLASMID_FRAGMENT_SIZE,
                                          step_fraction=0.5, min_length=min_length)
                plas_frags.extend((fid, fseq, "plasmid") for fid, fseq in frags)

    logger.info("  %s (%s): %d chr frags, %d plasmid frags",
                accession, organism[:50], len(chrom_frags), len(plas_frags))
    return chrom_frags, plas_frags


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark FASTA with ground truth")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"),
                        help="Directory with genomes/ subdir (default: data/benchmark)")
    parser.add_argument("--min-length", type=int, default=1000,
                        help="Minimum fragment length bp (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    args = parser.parse_args()

    genomes_dir = args.benchmark_dir / "genomes"
    if not genomes_dir.exists():
        logger.error("genomes/ dir not found at %s — run download_benchmark_genomes.py first",
                     genomes_dir)
        raise SystemExit(1)

    all_chrom: list[tuple[str, str, str]] = []
    all_plas:  list[tuple[str, str, str]] = []
    accessions = sorted(d.name for d in genomes_dir.iterdir() if d.is_dir())

    logger.info("Processing %d genome directories …", len(accessions))
    for acc in accessions:
        c, p = process_genome(acc, genomes_dir / acc, args.min_length, args.seed)
        all_chrom.extend(c)
        all_plas.extend(p)

    if not all_chrom and not all_plas:
        logger.error("No sequences found — check genomes/ directory")
        raise SystemExit(1)

    # Shuffle so chromosomes and plasmids are interleaved
    all_contigs = all_chrom + all_plas
    random.Random(args.seed).shuffle(all_contigs)

    logger.info("\nBenchmark composition:")
    label_counts = Counter(c[2] for c in all_contigs)
    total = len(all_contigs)
    for lbl, cnt in sorted(label_counts.items()):
        logger.info("  %-12s  %6d  (%5.1f%%)", lbl, cnt, 100 * cnt / total)
    logger.info("  %-12s  %6d", "TOTAL", total)

    # Length distribution
    length_bins = defaultdict(Counter)
    for _, seq, lbl in all_contigs:
        ln = len(seq)
        if ln < 2000:       bin_name = "1-2kb"
        elif ln < 5000:     bin_name = "2-5kb"
        elif ln < 10000:    bin_name = "5-10kb"
        elif ln < 20000:    bin_name = "10-20kb"
        else:               bin_name = ">20kb"
        length_bins[bin_name][lbl] += 1

    logger.info("\nLength distribution:")
    for bin_name in ["1-2kb", "2-5kb", "5-10kb", "10-20kb", ">20kb"]:
        bc = length_bins[bin_name]
        tot = sum(bc.values())
        if tot:
            logger.info("  %8s  total=%5d  chromosome=%5d  plasmid=%5d",
                        bin_name, tot, bc["chromosome"], bc["plasmid"])

    # Write benchmark FASTA
    fna_path = args.benchmark_dir / "benchmark.fna"
    gt_path  = args.benchmark_dir / "ground_truth.tsv"

    with open(fna_path, "w") as fna_fh, open(gt_path, "w") as gt_fh:
        gt_fh.write("contig_id\ttrue_label\tlength\tgenome_accession\n")
        for contig_id, seq, label in all_contigs:
            fna_fh.write(f">{contig_id}\n{seq}\n")
            # Extract accession from contig_id (first two tokens)
            parts = contig_id.split("_")
            acc = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
            gt_fh.write(f"{contig_id}\t{label}\t{len(seq)}\t{acc}\n")

    # Stats file
    stats_path = args.benchmark_dir / "benchmark_stats.txt"
    with open(stats_path, "w") as fh:
        fh.write("PlasFlow v2 Benchmark Dataset\n")
        fh.write("=" * 40 + "\n\n")
        fh.write(f"Total contigs : {total:,}\n")
        for lbl, cnt in sorted(label_counts.items()):
            fh.write(f"  {lbl:<12}  {cnt:6,}  ({100*cnt/total:.1f}%)\n")
        fh.write(f"\nGenomes processed: {len(accessions)}\n")
        fh.write(f"\nLength distribution:\n")
        for bin_name in ["1-2kb", "2-5kb", "5-10kb", "10-20kb", ">20kb"]:
            bc = length_bins[bin_name]
            tot = sum(bc.values())
            if tot:
                fh.write(f"  {bin_name:8}  chromosome={bc['chromosome']:5}  "
                         f"plasmid={bc['plasmid']:5}  total={tot:6}\n")

    logger.info("\nOutputs:")
    logger.info("  Benchmark FASTA  : %s  (%d contigs)", fna_path, total)
    logger.info("  Ground truth TSV : %s", gt_path)
    logger.info("  Stats            : %s", stats_path)
    logger.info("\nNext steps:")
    logger.info("  python scripts/run_benchmark_evaluation.py \\")
    logger.info("    --benchmark-dir %s \\", args.benchmark_dir)
    logger.info("    --model         data/models/mlp_v2.pt \\")
    logger.info("    --out           data/benchmark/results/")


if __name__ == "__main__":
    main()
