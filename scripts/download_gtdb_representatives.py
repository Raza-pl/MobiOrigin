#!/usr/bin/env python3
"""Download GTDB representative genomes for PlasFlow v2 model retraining.

Strategy
--------
* Bacteria  : one genome per genus → target 5,000–10,000 genomes
* Archaea   : one genome per genus → target ~3,000 genomes
* Filter    : GTDB-representative genomes only (highest-quality per species cluster)
* Source    : NCBI RefSeq / GenBank via GTDB metadata TSV

Usage
-----
    python scripts/download_gtdb_representatives.py \\
        --out-dir  data/gtdb_genomes/ \\
        --max-bacteria  8000 \\
        --max-archaea   3000 \\
        --threads       8

Output
------
    data/gtdb_genomes/bacteria/   — one .fna.gz per genome
    data/gtdb_genomes/archaea/    — one .fna.gz per genome
    data/gtdb_genomes/download_summary.tsv
"""

import argparse
import gzip
import logging
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GTDB_BASE = "https://data.gtdb.ecogenomic.org/releases/latest"
BAC_META  = f"{GTDB_BASE}/bac120_metadata.tsv.gz"
ARC_META  = f"{GTDB_BASE}/ar53_metadata.tsv.gz"
NCBI_FTP  = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


# ---------------------------------------------------------------------------
# GTDB metadata parsing
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with retry."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        logger.info("[cache] %s", dest.name)
        return True
    logger.info("Downloading %s → %s", desc or url.split("/")[-1], dest)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as exc:
            logger.warning("Attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    return False


def parse_gtdb_metadata(meta_gz: Path, domain: str) -> list[dict]:
    """Parse GTDB metadata TSV and return one representative per genus.

    Key columns used:
        accession               — NCBI accession (GCF_/GCA_)
        gtdb_representative     — 't' if this is the GTDB genus representative
        gtdb_taxonomy           — full taxonomy string
        checkm_completeness     — genome completeness %
        checkm_contamination    — genome contamination %
    """
    reps: dict[str, dict] = {}  # genus → best genome

    logger.info("Parsing %s metadata …", domain)
    with gzip.open(meta_gz, "rt") as fh:
        header = fh.readline().lstrip("#").strip().split("\t")
        col = {h: i for i, h in enumerate(header)}

        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(col):
                continue

            is_rep  = parts[col.get("gtdb_representative", 0)].strip().lower() == "t"
            if not is_rep:
                continue

            accession = parts[col.get("accession", 0)].strip()
            taxonomy  = parts[col.get("gtdb_taxonomy", col.get("taxonomy", 1))].strip()
            comp      = float(parts[col.get("checkm_completeness", 2)] or 0)
            cont      = float(parts[col.get("checkm_contamination", 3)] or 0)

            # Quality filter: ≥90% complete, ≤5% contaminated
            if comp < 90.0 or cont > 5.0:
                continue

            # Extract genus from taxonomy string
            # Format: d__Bacteria;p__Firmicutes;c__...;g__Lactobacillus;s__...
            genus_m = re.search(r"g__([^;]+)", taxonomy)
            genus = genus_m.group(1).strip() if genus_m else "unknown"
            if genus in ("", "unknown"):
                continue

            # Keep only one genome per genus (first representative encountered)
            if genus not in reps:
                reps[genus] = {
                    "accession": accession,
                    "genus": genus,
                    "taxonomy": taxonomy,
                    "completeness": comp,
                    "contamination": cont,
                    "domain": domain,
                }

    logger.info("%s: %d unique genera with representative genomes", domain, len(reps))
    return list(reps.values())


# ---------------------------------------------------------------------------
# NCBI genome download
# ---------------------------------------------------------------------------

def accession_to_ftp_path(accession: str) -> str | None:
    """Convert GCF_/GCA_ accession to NCBI FTP directory URL.

    GCF_000001405.40 → /all/GCF/000/001/405/GCF_000001405.40_<assembly>/
    We need to fetch the directory listing to get the exact folder name.
    """
    # GTDB prefixes: RS_GCF_... or GB_GCA_... — use re.sub, NOT lstrip
    # (lstrip strips individual chars, so lstrip("GB_") turns GCA_ into CA_)
    acc_clean = re.sub(r"^(RS|GB)_", "", accession)
    m = re.match(r"(GC[FA])_(\d{3})(\d{3})(\d{3})", acc_clean)
    if not m:
        return None
    prefix, p1, p2, p3 = m.groups()
    return f"{NCBI_FTP}/{prefix}/{p1}/{p2}/{p3}/"


def download_genome(entry: dict, out_dir: Path) -> tuple[str, bool]:
    """Download a single genome FASTA. Returns (accession, success)."""
    accession = entry["accession"]
    acc_clean = re.sub(r"^(RS|GB)_", "", accession)
    dest = out_dir / f"{acc_clean}.fna.gz"

    if dest.exists() and dest.stat().st_size > 10_000:
        return accession, True

    ftp_base = accession_to_ftp_path(acc_clean)
    if not ftp_base:
        logger.warning("Cannot parse accession: %s", accession)
        return accession, False

    # Fetch HTTPS directory listing to find assembly folder name
    try:
        req = urllib.request.Request(ftp_base, headers={"User-Agent": "PlasFlow/2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            listing = resp.read().decode(errors="replace")
    except Exception as exc:
        logger.debug("Directory listing failed for %s: %s", acc_clean, exc)
        return accession, False

    # NCBI HTTPS listing: href="GCF_000001405.40_GRCh38.p14/"
    # Strip version from accession for matching (GCF_000001405 matches GCF_000001405.40_...)
    acc_no_ver = acc_clean.split(".")[0]
    folder_m = re.search(rf'href="({re.escape(acc_no_ver)}[^"]+)/"', listing)
    if not folder_m:
        logger.debug("No assembly folder found for %s", acc_clean)
        return accession, False

    folder = folder_m.group(1).rstrip("/")
    fna_url = f"{ftp_base}{folder}/{folder}_genomic.fna.gz"

    for attempt in range(3):
        try:
            req2 = urllib.request.Request(fna_url, headers={"User-Agent": "PlasFlow/2.0"})
            with urllib.request.urlopen(req2, timeout=60) as resp:
                with open(dest, "wb") as f:
                    f.write(resp.read())
            return accession, True
        except Exception:
            if dest.exists():
                dest.unlink()
            if attempt < 2:
                time.sleep(2 ** attempt)
    logger.debug("Failed to download %s", accession)
    return accession, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download GTDB representative genomes")
    parser.add_argument("--out-dir",       default="data/gtdb_genomes",  type=Path)
    parser.add_argument("--max-bacteria",  default=8000,  type=int,
                        help="Max bacterial genomes (one per genus, default 8000)")
    parser.add_argument("--max-archaea",   default=3000,  type=int,
                        help="Max archaeal genomes (one per genus, default 3000)")
    parser.add_argument("--threads",       default=8,     type=int,
                        help="Parallel download threads")
    parser.add_argument("--min-completeness", default=90.0, type=float)
    parser.add_argument("--max-contamination", default=5.0, type=float)
    args = parser.parse_args()

    out = args.out_dir
    bac_dir = out / "bacteria"
    arc_dir = out / "archaea"
    meta_dir = out / "metadata"
    bac_dir.mkdir(parents=True, exist_ok=True)
    arc_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ── Download GTDB metadata ────────────────────────────────────────────
    bac_meta_gz = meta_dir / "bac120_metadata.tsv.gz"
    arc_meta_gz = meta_dir / "ar53_metadata.tsv.gz"
    download_file(BAC_META, bac_meta_gz, "bacterial metadata")
    download_file(ARC_META, arc_meta_gz, "archaeal metadata")

    # ── Parse: one rep per genus ──────────────────────────────────────────
    bac_reps = parse_gtdb_metadata(bac_meta_gz, "Bacteria")
    arc_reps = parse_gtdb_metadata(arc_meta_gz, "Archaea")

    # Sort by completeness descending, cap at requested max
    bac_reps.sort(key=lambda x: -x["completeness"])
    arc_reps.sort(key=lambda x: -x["completeness"])
    bac_reps = bac_reps[:args.max_bacteria]
    arc_reps = arc_reps[:args.max_archaea]

    logger.info("Selected %d bacterial genera, %d archaeal genera",
                len(bac_reps), len(arc_reps))

    # ── Write summary before download ─────────────────────────────────────
    summary_path = out / "download_summary.tsv"
    with open(summary_path, "w") as f:
        f.write("accession\tgenus\tdomain\tcompleteness\tcontamination\ttaxonomy\n")
        for entry in bac_reps + arc_reps:
            f.write(f"{entry['accession']}\t{entry['genus']}\t{entry['domain']}\t"
                    f"{entry['completeness']}\t{entry['contamination']}\t{entry['taxonomy']}\n")
    logger.info("Summary written to %s", summary_path)

    # ── Download genomes in parallel ──────────────────────────────────────
    def download_batch(entries: list[dict], out_dir: Path, label: str) -> None:
        done = already = failed = 0
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {pool.submit(download_genome, e, out_dir): e for e in entries}
            for fut in as_completed(futures):
                acc, ok = fut.result()
                if ok:
                    done += 1
                else:
                    failed += 1
                total = done + failed
                if total % 100 == 0:
                    logger.info("%s: %d / %d downloaded, %d failed",
                                label, done, len(entries), failed)
        logger.info("%s complete: %d downloaded, %d failed", label, done, failed)

    logger.info("Downloading %d bacterial genomes …", len(bac_reps))
    download_batch(bac_reps, bac_dir, "Bacteria")

    logger.info("Downloading %d archaeal genomes …", len(arc_reps))
    download_batch(arc_reps, arc_dir, "Archaea")

    # ── Final count ───────────────────────────────────────────────────────
    n_bac = len(list(bac_dir.glob("*.fna.gz")))
    n_arc = len(list(arc_dir.glob("*.fna.gz")))
    logger.info("Final: %d bacterial + %d archaeal genomes downloaded", n_bac, n_arc)
    logger.info("Next step: python scripts/build_dataset.py --chroms %s --archaea %s ...",
                bac_dir, arc_dir)


if __name__ == "__main__":
    main()
