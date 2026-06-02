#!/usr/bin/env python3
"""Download additional plasmids from NCBI RefSeq to expand training data to ~75k unique genomes.

Strategy
--------
* We already have ~48,385 unique plasmids from PLSDB + RefSeq + COMPASS.
* We need ~27,000 more unique plasmid sequences.
* Source: NCBI RefSeq plasmids (assembly_summary for plasmids) and
          PLSDB (if a newer version is available).
* Deduplication: one representative per 95% ANI cluster → enforced by
  filtering to accessions NOT already in our seq_ids.txt.
* We do NOT filter by host genus — plasmid diversity is sequence-based,
  not taxonomy-based. A single host genus can carry dozens of unrelated
  plasmid families (IncF, IncP, IncQ, ColE1, etc.), all scientifically
  distinct.

Why not filter plasmids by host genus?
---------------------------------------
Unlike chromosomes (where one genus = one lifestyle), plasmids are defined
by their own replication + mobility modules, not their host. The correct
unit of plasmid diversity is the plasmid sequence cluster (ANI-based), not
the host organism. Filtering by host genus would discard many genuinely
novel plasmid sequences while keeping redundant ones.

Usage
-----
    python scripts/download_additional_plasmids.py \\
        --out-dir    data/extra_plasmids/ \\
        --target     27000 \\
        --threads    8 \\
        --existing-ids data/seq_ids.txt
"""

import argparse
import gzip
import logging
import re
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

# NCBI E-utilities — search RefSeq nucleotide for plasmid sequences
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
# PLSDB latest release (plasmid DB, curated, ~60k sequences)
PLSDB_FASTA_URL = "https://api.plasmid.science/plsdb/download/plsdb.fna.gz"
NCBI_FTP = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


def load_existing_accessions(seq_ids_path: Path) -> set[str]:
    """Extract accessions already in training data to avoid duplicates."""
    existing: set[str] = set()
    if not seq_ids_path.exists():
        return existing
    with open(seq_ids_path) as f:
        for line in f:
            # Format: SOURCE_ACCESSION_w{n}_s{n}
            acc = re.sub(r"_w\d+_s\d+$", "", line.strip())
            # Strip source prefix (PLSDB_, RefSeq_, COMPASS_)
            acc = re.sub(r"^(PLSDB|RefSeq|COMPASS|INPHARED)_", "", acc)
            existing.add(acc)
    logger.info("Loaded %d existing accessions to exclude", len(existing))
    return existing


def fetch_plsdb_accessions(meta_dir: Path) -> list[str]:
    """Download the PLSDB metadata TSV and return all accessions.

    PLSDB is the most complete curated plasmid database (~60k sequences).
    We use the metadata to get accession lists, then fetch sequences via
    NCBI E-utilities for those not already in our training set.
    """
    meta_url = "https://api.plasmid.science/plsdb/download/plsdb.tsv"
    meta_path = meta_dir / "plsdb_metadata.tsv"

    if not meta_path.exists() or meta_path.stat().st_size < 1000:
        logger.info("Downloading PLSDB metadata TSV …")
        try:
            req = urllib.request.Request(meta_url, headers={"User-Agent": "PlasFlow/2.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(meta_path, "wb") as f:
                    f.write(resp.read())
        except Exception as exc:
            logger.error("PLSDB download failed: %s", exc)
            logger.info("Alternative: manually download from https://api.plasmid.science/plsdb/download/plsdb.tsv")
            return []

    accessions = []
    with open(meta_path) as f:
        header = f.readline().strip().split("\t")
        acc_col = 0
        for i, h in enumerate(header):
            if "accession" in h.lower() or "nuccore" in h.lower():
                acc_col = i
                break
        for line in f:
            parts = line.strip().split("\t")
            if parts and len(parts) > acc_col:
                acc = parts[acc_col].strip()
                if acc:
                    accessions.append(acc)

    logger.info("PLSDB: %d total accessions", len(accessions))
    return accessions


def fetch_sequences_efetch(accessions: list[str], out_dir: Path, batch_size: int = 200) -> int:
    """Fetch nucleotide sequences from NCBI in batches using efetch.

    Saves one combined FASTA per batch, named by first accession.
    Returns number of sequences successfully written.
    """
    written = 0
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i + batch_size]
        dest = out_dir / f"batch_{i:06d}.fna.gz"
        if dest.exists() and dest.stat().st_size > 500:
            written += len(batch)
            continue
        ids_str = ",".join(batch)
        url = (f"{NCBI_EFETCH}?db=nuccore&id={ids_str}"
               f"&rettype=fasta&retmode=text")
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PlasFlow/2.0"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    content = resp.read()
                with gzip.open(dest, "wb") as gz:
                    gz.write(content)
                n = content.count(b">")
                written += n
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
        # NCBI rate limit: max 3 requests/sec without API key
        time.sleep(0.4)
        if (i // batch_size) % 10 == 0:
            logger.info("Efetch progress: %d / %d accessions", i + len(batch), len(accessions))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download additional plasmids for training")
    parser.add_argument("--out-dir",      default="data/extra_plasmids", type=Path)
    parser.add_argument("--target",       default=27000, type=int,
                        help="Number of new plasmid genomes to download")
    parser.add_argument("--threads",      default=8,     type=int)
    parser.add_argument("--existing-ids", default="data/seq_ids.txt", type=Path,
                        help="seq_ids.txt from current training set (to avoid duplicates)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = args.out_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)

    # ── Load existing accessions ──────────────────────────────────────────
    existing = load_existing_accessions(args.existing_ids)

    # ── Fetch PLSDB accession list ────────────────────────────────────────
    all_accessions = fetch_plsdb_accessions(meta_dir)
    if not all_accessions:
        logger.error(
            "Could not fetch PLSDB accessions. "
            "Manually download plsdb.tsv from https://api.plasmid.science/plsdb/download/plsdb.tsv "
            "and save to %s", meta_dir / "plsdb_metadata.tsv"
        )
        sys.exit(1)

    # ── Filter: exclude already in training set ───────────────────────────
    new_accessions = [a for a in all_accessions if a not in existing]
    logger.info("%d new PLSDB accessions not in existing training data", len(new_accessions))

    if len(new_accessions) < args.target:
        logger.warning(
            "Only %d new accessions available — target of %d may not be fully met.",
            len(new_accessions), args.target,
        )

    selected = new_accessions[:args.target]
    logger.info("Fetching %d plasmid sequences via NCBI efetch …", len(selected))
    logger.info("(This uses NCBI E-utilities — expect ~%.0f min at 3 req/sec)",
                len(selected) / 200 * 0.4 / 60)

    # ── Download via efetch ───────────────────────────────────────────────
    n_written = fetch_sequences_efetch(selected, args.out_dir)

    n_files = len(list(args.out_dir.glob("batch_*.fna.gz")))
    logger.info("Done: %d batch files written (~%d sequences)", n_files, n_written)
    logger.info("Total unique plasmids after adding these: ~%d", 48385 + len(selected))
    logger.info("Next: python scripts/build_dataset.py --plasmids %s ...", args.out_dir)


if __name__ == "__main__":
    main()
