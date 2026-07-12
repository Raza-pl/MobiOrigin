#!/usr/bin/env python3
"""Fragment downloaded genomes into simulated metagenomic contigs for benchmarking.

Reads the metadata.tsv produced by download_genomes.py and generates two
benchmark datasets:

  Tier 1 — isolate-genome benchmark (clean, well-labelled)
  Tier 2 — synthetic metagenome (mixed species, class-imbalanced, harder)

For each tier the output is:
  input.fasta     — multi-FASTA to feed to classifiers (no labels)
  labels.tsv      — ground-truth (contig_id, true_label, length, source_accession,
                                  molecule_type, taxon, length_tier)

Fragmentation model
-------------------
Chromosomes are always fragmented: random cut points drawn from a log-normal
distribution (μ=log(3000), σ=0.8), giving a realistic SPAdes-like length
distribution with modal contig ~2-3 kb and a heavy right tail.

Plasmids are fragmented proportionally to their size:
  <15 kb  → kept as a single contig (small plasmids often assemble intact)
  15-50 kb → 1-3 fragments
  >50 kb  → fragmented the same way as chromosomes

Usage
-----
    python scripts/benchmark/make_benchmark.py \\
        --genomes  data/benchmark/genomes \\
        --out      data/benchmark \\
        --min-len  1000 \\
        --seed     42

    # Tier 2: mix species at realistic metagenome ratios
    python scripts/benchmark/make_benchmark.py \\
        --genomes  data/benchmark/genomes \\
        --out      data/benchmark \\
        --tier2-mix-fraction 0.03 \\   # ~3% plasmid contigs
        --tier2-n-genomes    50
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Length tier labels (for stratified evaluation) ─────────────────────────────

LENGTH_TIERS = [
    (2_000, "<2 kb"),
    (5_000, "2-5 kb"),
    (10_000, "5-10 kb"),
    (50_000, "10-50 kb"),
    (float("inf"), ">50 kb"),
]


def _length_tier(bp: int) -> str:
    for cutoff, label in LENGTH_TIERS:
        if bp <= cutoff:
            return label
    return ">50 kb"


# ── Fragmentation ──────────────────────────────────────────────────────────────


def _fragment(
    seq: str,
    accession: str,
    mol_type: str,
    min_len: int,
    rng: random.Random,
) -> list[dict]:
    """Fragment *seq* into realistic contig-like pieces.

    Returns list of dicts with keys: contig_id, sequence, true_label, length,
    source_accession, molecule_type.
    """
    n = len(seq)

    # Decide how to fragment based on molecule type and size
    if mol_type == "plasmid":
        if n < 15_000:
            # Small plasmids: keep intact
            cut_points = [0, n]
        elif n < 50_000:
            # Medium: 1-3 fragments
            n_cuts = rng.randint(1, 3)
            cuts = sorted(rng.randint(min_len, n - min_len) for _ in range(n_cuts))
            cut_points = [0] + cuts + [n]
        else:
            # Large: same fragmentation as chromosomes
            cut_points = _lognormal_cuts(seq, min_len, rng)
    else:
        # Chromosome: always fragment
        cut_points = _lognormal_cuts(seq, min_len, rng)

    contigs = []
    for i, (start, end) in enumerate(zip(cut_points[:-1], cut_points[1:])):
        fragment = seq[start:end]
        if len(fragment) < min_len:
            continue
        contig_id = f"{accession}_c{i + 1}"
        contigs.append(
            {
                "contig_id": contig_id,
                "sequence": fragment,
                "true_label": "plasmid" if mol_type == "plasmid" else "chromosome",
                "length": len(fragment),
                "source_accession": accession,
                "molecule_type": mol_type,
                "length_tier": _length_tier(len(fragment)),
            }
        )

    return contigs


def _lognormal_cuts(seq: str, min_len: int, rng: random.Random) -> list[int]:
    """Generate cut points from a log-normal length distribution."""
    n = len(seq)
    pos = 0
    cuts = [0]
    while pos < n:
        # Log-normal: modal contig ~2.7 kb, heavy right tail
        length = int(np.random.lognormal(mean=math.log(3000), sigma=0.8))
        length = max(min_len, min(length, n - pos))
        pos += length
        cuts.append(pos)
    return cuts


# ── Dataset builder ────────────────────────────────────────────────────────────


def _load_metadata(genomes_dir: Path) -> list[dict]:
    meta_path = genomes_dir / "metadata.tsv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.tsv not found in {genomes_dir}. " "Run download_genomes.py first."
        )
    with open(meta_path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _read_fasta(path: Path) -> dict[str, str]:
    """Read a FASTA file, return {accession: sequence}."""
    seqs: dict[str, str] = {}
    current_id = None
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id:
                    seqs[current_id] = "".join(buf)
                current_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.upper())
    if current_id:
        seqs[current_id] = "".join(buf)
    return seqs


def _write_dataset(contigs: list[dict], out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "input.fasta"
    labels_path = out_dir / "labels.tsv"

    with open(fasta_path, "w") as ff, open(labels_path, "w", newline="") as lf:
        writer = csv.DictWriter(
            lf,
            fieldnames=[
                "contig_id",
                "true_label",
                "length",
                "length_tier",
                "source_accession",
                "molecule_type",
                "taxon",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for c in contigs:
            ff.write(f">{c['contig_id']}\n{c['sequence']}\n")
            writer.writerow(
                {
                    "contig_id": c["contig_id"],
                    "true_label": c["true_label"],
                    "length": c["length"],
                    "length_tier": c["length_tier"],
                    "source_accession": c["source_accession"],
                    "molecule_type": c["molecule_type"],
                    "taxon": c.get("taxon", ""),
                }
            )

    n_plas = sum(1 for c in contigs if c["true_label"] == "plasmid")
    n_chr = sum(1 for c in contigs if c["true_label"] == "chromosome")
    logger.info(
        "%s → %d contigs (%d plasmid / %d chromosome) — %s",
        name,
        len(contigs),
        n_plas,
        n_chr,
        out_dir,
    )


def build_tier1(
    genomes_dir: Path,
    out_dir: Path,
    min_len: int,
    seed: int,
) -> None:
    """Tier 1: one contig set per taxon + a combined 'all species' set.

    Each assembly → fragmented contigs with ground-truth labels.
    All assemblies pooled into one input.fasta for combined evaluation.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    meta = _load_metadata(genomes_dir)

    all_contigs: list[dict] = []

    # Group by taxon
    taxa = sorted({r["taxon"] for r in meta})
    for taxon in taxa:
        taxon_rows = [r for r in meta if r["taxon"] == taxon]
        taxon_contigs: list[dict] = []

        for row in taxon_rows:
            fa_path = genomes_dir / row["assembly_fasta"]
            if not fa_path.exists():
                logger.warning("  FASTA not found: %s — skipping", fa_path)
                continue

            seqs = _read_fasta(fa_path)
            seq = seqs.get(row["accession"])
            if seq is None:
                # Try partial match on accession (version numbers vary)
                for k, v in seqs.items():
                    if k.startswith(row["accession"].split(".")[0]):
                        seq = v
                        break
            if seq is None:
                logger.warning(
                    "  Accession %s not found in %s — skipping", row["accession"], fa_path.name
                )
                continue

            frags = _fragment(
                seq,
                accession=row["accession"],
                mol_type=row["molecule_type"],
                min_len=min_len,
                rng=rng,
            )
            for f in frags:
                f["taxon"] = taxon
            taxon_contigs.extend(frags)

        if taxon_contigs:
            safe_name = taxon.replace(" ", "_").lower()
            _write_dataset(taxon_contigs, out_dir / "tier1" / safe_name, taxon)
            all_contigs.extend(taxon_contigs)

    # Combined
    rng.shuffle(all_contigs)
    _write_dataset(all_contigs, out_dir / "tier1" / "all_species", "Tier1 (all species)")


def build_tier2(
    genomes_dir: Path,
    out_dir: Path,
    min_len: int,
    seed: int,
    n_genomes: int = 50,
    plasmid_fraction: float = 0.03,
) -> None:
    """Tier 2: synthetic metagenome — class-imbalanced, mixed species.

    Samples *n_genomes* assemblies at random, fragments all of them, then
    downsamples chromosome contigs so that plasmids make up *plasmid_fraction*
    of the total — matching real wastewater metagenome class priors.
    """
    rng = random.Random(seed + 1)
    np.random.seed(seed + 1)
    meta = _load_metadata(genomes_dir)

    # Sample assemblies
    uids = list({r["uid"] for r in meta})
    rng.shuffle(uids)
    selected_uids = set(uids[:n_genomes])
    selected_rows = [r for r in meta if r["uid"] in selected_uids]

    plasmid_contigs: list[dict] = []
    chrom_contigs: list[dict] = []

    for row in selected_rows:
        fa_path = genomes_dir / row["assembly_fasta"]
        if not fa_path.exists():
            continue
        seqs = _read_fasta(fa_path)
        seq = seqs.get(row["accession"])
        if seq is None:
            for k, v in seqs.items():
                if k.startswith(row["accession"].split(".")[0]):
                    seq = v
                    break
        if seq is None:
            continue

        frags = _fragment(
            seq,
            accession=row["accession"],
            mol_type=row["molecule_type"],
            min_len=min_len,
            rng=rng,
        )
        for f in frags:
            f["taxon"] = row["taxon"]

        if row["molecule_type"] == "plasmid":
            plasmid_contigs.extend(frags)
        else:
            chrom_contigs.extend(frags)

    # Downsample chromosomes so plasmids = plasmid_fraction of total
    n_plas = len(plasmid_contigs)
    n_chr_target = int(n_plas * (1 - plasmid_fraction) / plasmid_fraction)
    n_chr_target = min(n_chr_target, len(chrom_contigs))

    rng.shuffle(chrom_contigs)
    selected_chrom = chrom_contigs[:n_chr_target]

    all_contigs = plasmid_contigs + selected_chrom
    rng.shuffle(all_contigs)

    logger.info(
        "Tier 2: sampled %d genomes, target plasmid fraction %.1f%% " "(actual: %d / %d = %.2f%%)",
        n_genomes,
        plasmid_fraction * 100,
        n_plas,
        len(all_contigs),
        100 * n_plas / max(len(all_contigs), 1),
    )
    _write_dataset(all_contigs, out_dir / "tier2_metagenome", "Tier2 (synthetic metagenome)")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--genomes", required=True, type=Path, help="Directory produced by download_genomes.py."
    )
    p.add_argument(
        "--out", required=True, type=Path, help="Root output directory for benchmark datasets."
    )
    p.add_argument(
        "--min-len", type=int, default=1000, help="Minimum contig length to keep (default 1000 bp)."
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    p.add_argument(
        "--tier",
        choices=["1", "2", "all"],
        default="all",
        help="Which tiers to build (default: all).",
    )
    p.add_argument(
        "--tier2-n-genomes",
        type=int,
        default=50,
        help="Number of genomes to mix for Tier 2 (default 50).",
    )
    p.add_argument(
        "--tier2-plasmid-fraction",
        type=float,
        default=0.03,
        help="Target plasmid fraction in Tier 2 (default 0.03 = 3%%).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.tier in ("1", "all"):
        logger.info("Building Tier 1 (per-taxon isolate benchmark)…")
        build_tier1(
            genomes_dir=args.genomes,
            out_dir=args.out,
            min_len=args.min_len,
            seed=args.seed,
        )

    if args.tier in ("2", "all"):
        logger.info("Building Tier 2 (synthetic metagenome)…")
        build_tier2(
            genomes_dir=args.genomes,
            out_dir=args.out,
            min_len=args.min_len,
            seed=args.seed,
            n_genomes=args.tier2_n_genomes,
            plasmid_fraction=args.tier2_plasmid_fraction,
        )


if __name__ == "__main__":
    main()
