"""Download hard-negative genomes for PlasFlow v2 training.

These are organisms whose secondary chromosomes or large genomic islands
score as plasmid in the benchmark, causing false positives in the 10-20kb bin.

Primary FP sources (from benchmark analysis):
  GCF_000017645.1  Burkholderia vietnamiensis G4   (68 FPs)
  GCF_001593285.1  Enterobacter cloacae ATCC 13047 (66 FPs)

Extended set — organisms with confirmed secondary chromosomes or genomic
islands that are compositionally plasmid-like:
  Burkholderia genus: many species have chromid (chromosome 2)
  Ralstonia solanacearum: 2 chromosomes
  Agrobacterium tumefaciens: second chromosome + Ti plasmid
  Sinorhizobium meliloti: 2 chromosomes + 2 mega-plasmids
  Vibrio cholerae: 2 chromosomes
  Rhizobium leguminosarum: multiple replicons

Usage
-----
    python scripts/download_hard_negatives.py --out data/hard_negatives/
    python scripts/download_hard_negatives.py --out data/hard_negatives/ --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Accession → (species, reason)
HARD_NEGATIVE_GENOMES: dict[str, tuple[str, str]] = {
    # ── Primary FP organisms (confirmed in benchmark) ─────────────────────────
    "GCF_000017645.1": (
        "Burkholderia vietnamiensis G4",
        "68 FPs: secondary chromosome scores 0.85-0.99 as plasmid",
    ),
    "GCF_001593285.1": (
        "Enterobacter cloacae ATCC 13047",
        "66 FPs: genomic islands score 0.85-0.99 as plasmid",
    ),
    # ── Burkholderia species with known secondary chromosomes ─────────────────
    "GCF_000011705.1": (
        "Burkholderia pseudomallei K96243",
        "Two-chromosome organism; chr2 is plasmid-compositionally similar",
    ),
    "GCF_000009085.1": (
        "Burkholderia mallei ATCC 23344",
        "Reduced two-chromosome genome from B. pseudomallei ancestor",
    ),
    "GCF_000012545.1": (
        "Burkholderia cenocepacia J2315",
        "Three chromosomes; chr2+3 have plasmid-like composition",
    ),
    "GCF_000195715.1": (
        "Burkholderia glumae BGR1",
        "Two chromosomes; causes rice grain rot",
    ),
    # ── Ralstonia / Cupriavidus ───────────────────────────────────────────────
    "GCF_000009125.1": (
        "Ralstonia solanacearum GMI1000",
        "Two-replicon genome; megaplasmid resembles secondary chromosome",
    ),
    # ── Rhizobiales with complex multi-replicon genomes ───────────────────────
    "GCF_000006945.2": (
        "Sinorhizobium meliloti 1021",
        "Two chromosomes + two mega-plasmids; all replicons are plasmid-like",
    ),
    "GCF_000017145.1": (
        "Agrobacterium tumefaciens C58",
        "Circular + linear chromosomes + Ti plasmid",
    ),
    # ── Vibrio (two-chromosome genus) ─────────────────────────────────────────
    "GCF_000006745.1": (
        "Vibrio cholerae O1 biovar El Tor N16961",
        "Two chromosomes; chr2 historically misclassified as megaplasmid",
    ),
    "GCF_000196235.1": (
        "Vibrio parahaemolyticus RIMD 2210633",
        "Two chromosomes; chr2 is ~1.9 Mb",
    ),
    # ── Enterobacteriaceae with large pathogenicity islands ───────────────────
    "GCF_000005845.2": (
        "Escherichia coli K-12 MG1655",
        "Well-annotated reference; pathogenicity island-free (negative control context)",
    ),
    "GCF_000026345.1": (
        "Enterobacter cloacae EcWSU1",
        "Additional Enterobacter with genomic islands",
    ),
}

NCBI_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


def _accession_to_ftp_path(accession: str) -> str:
    """Convert GCF_XXXXXXXXX.V to NCBI FTP path prefix."""
    # GCF_000017645.1 → GCF/000/017/645
    prefix = accession.split(".")[0]   # GCF_000017645
    kind   = prefix[:3]                # GCF
    nums   = prefix[4:]                # 000017645
    p1, p2, p3 = nums[0:3], nums[3:6], nums[6:9]
    return f"{NCBI_FTP_BASE}/{kind}/{p1}/{p2}/{p3}"


def download_genome(accession: str, species: str, out_dir: Path, dry_run: bool = False) -> Path | None:
    """Download the genomic FASTA for *accession* into *out_dir*.

    Fetches the assembly listing page to find the exact filename, then
    downloads *_genomic.fna.gz and decompresses it.

    Returns the path to the downloaded .fna file, or None on failure.
    """
    out_fna = out_dir / f"{accession}_genomic.fna"
    if out_fna.exists():
        logger.info("  [SKIP] %s already exists (%s)", out_fna.name, accession)
        return out_fna

    ftp_prefix = _accession_to_ftp_path(accession)
    listing_url = f"{ftp_prefix}/"

    if dry_run:
        logger.info("  [DRY RUN] Would fetch listing: %s", listing_url)
        return None

    logger.info("  Fetching listing: %s", listing_url)
    try:
        with urllib.request.urlopen(listing_url, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.error("  Failed to fetch listing for %s: %s", accession, exc)
        return None

    # Find the assembly directory name (e.g. GCF_000017645.1_GVietnG4_v1)
    import re
    dirs = re.findall(rf'href="({accession}_[^/"]+)/"', html)
    if not dirs:
        logger.error("  No assembly directory found for %s in listing", accession)
        return None
    asm_dir = dirs[0]

    gz_url = f"{ftp_prefix}/{asm_dir}/{asm_dir}_genomic.fna.gz"
    gz_path = out_dir / f"{accession}_genomic.fna.gz"

    logger.info("  Downloading: %s", gz_url)
    try:
        with urllib.request.urlopen(gz_url, timeout=120) as resp, \
             open(gz_path, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except Exception as exc:
        logger.error("  Download failed for %s: %s", accession, exc)
        gz_path.unlink(missing_ok=True)
        return None

    # Decompress
    logger.info("  Decompressing → %s", out_fna.name)
    try:
        with gzip.open(gz_path, "rb") as gz_in, open(out_fna, "wb") as fna_out:
            shutil.copyfileobj(gz_in, fna_out)
        gz_path.unlink()
    except Exception as exc:
        logger.error("  Decompression failed for %s: %s", accession, exc)
        gz_path.unlink(missing_ok=True)
        out_fna.unlink(missing_ok=True)
        return None

    size_mb = out_fna.stat().st_size / 1e6
    logger.info("  Saved: %s  (%.1f MB)", out_fna.name, size_mb)
    return out_fna


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download hard-negative chromosome genomes for PlasFlow v2"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/hard_negatives"),
        help="Output directory (default: data/hard_negatives/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without downloading",
    )
    parser.add_argument(
        "--accessions", default=None,
        help="Comma-separated subset of accessions to download (default: all)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    target = HARD_NEGATIVE_GENOMES
    if args.accessions:
        subset = {a.strip() for a in args.accessions.split(",")}
        target = {k: v for k, v in target.items() if k in subset}
        logger.info("Filtering to %d requested accessions", len(target))

    logger.info("=" * 60)
    logger.info("Downloading %d hard-negative genomes → %s", len(target), args.out)
    logger.info("=" * 60)

    ok, failed = [], []
    for accession, (species, reason) in target.items():
        logger.info("")
        logger.info("%s  —  %s", accession, species)
        logger.info("  Reason: %s", reason)
        result = download_genome(accession, species, args.out, dry_run=args.dry_run)
        if result:
            ok.append(accession)
        else:
            failed.append(accession)
        time.sleep(0.5)   # be polite to NCBI

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done: %d downloaded, %d failed", len(ok), len(failed))
    if failed:
        logger.warning("Failed: %s", failed)
    logger.info("")
    logger.info("Next step:")
    logger.info("  bash scripts/retrain_hard_neg.sh")


if __name__ == "__main__":
    main()
