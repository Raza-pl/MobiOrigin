"""AMRFinderPlus — NCBI's curated antimicrobial resistance gene finder.

AMRFinderPlus catches genes missed by CARD and SARG, especially newer
acquisitions, point mutations, and curated gene families.  It is the
reference-quality database used by NCBI for clinical genome annotation.

Pipeline:
    proteins.faa  →  amrfinder --protein  →  amrfinder_hits.tsv
                  →  parse_amrfinder_hits()  →  [ARGHit(source="AMR")]

Setup (one-time):
    conda install -c bioconda ncbi-amrfinderplus
    amrfinder_update --force_update    # downloads latest DB (~400 MB)
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
from pathlib import Path

from plasflow2.annotate.args import ARGHit

logger = logging.getLogger(__name__)

AMR_MIN_IDENTITY = 80.0
AMR_MIN_COVERAGE = 80.0


def amrfinder_available() -> bool:
    try:
        result = subprocess.run(["amrfinder", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_amrfinder(
    protein_fasta: Path | str,
    out_tsv: Path | str,
    threads: int = 8,
    organism: str | None = None,
) -> Path:
    protein_fasta = Path(protein_fasta)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    # AMRFinderPlus caps threads at 10 on macOS
    safe_threads = min(threads, 10)

    cmd = [
        "amrfinder",
        "--protein",  str(protein_fasta),
        "--output",   str(out_tsv),
        "--threads",  str(safe_threads),
        "--plus",
    ]
    if organism:
        cmd += ["--organism", organism]

    logger.info("Running AMRFinderPlus: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("AMRFinderPlus failed (exit %d): %s", result.returncode, result.stderr[:500])
        raise RuntimeError(f"AMRFinderPlus failed with exit code {result.returncode}")
    return out_tsv


def parse_amrfinder_hits(
    tsv_path: Path | str,
    min_identity: float = AMR_MIN_IDENTITY,
    min_coverage: float = AMR_MIN_COVERAGE,
) -> list[ARGHit]:
    tsv_path = Path(tsv_path)
    hits: list[ARGHit] = []
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return hits

    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("Element type", "").strip() != "AMR":
                continue
            try:
                identity = float(row.get("% Identity to reference sequence", 0) or 0)
                coverage = float(row.get("% Coverage of reference sequence", 0) or 0)
            except ValueError:
                continue
            if identity < min_identity or coverage < min_coverage:
                continue

            orf_id    = row.get("Protein identifier", "").strip()
            gene_name = row.get("Gene symbol", "").strip() or row.get("Sequence name", "")
            drug_cls  = row.get("Class", "unknown").strip() or "unknown"
            subclass  = row.get("Subclass", "").strip()
            accession = row.get("Accession of closest sequence", "").strip()
            mechanism = row.get("Method", "unknown").strip()
            contig_id = re.sub(r"_\d+$", "", orf_id) if orf_id else ""
            try:
                evalue = float(row.get("E-value", 0) or 0)
            except (ValueError, KeyError):
                evalue = 0.0

            hits.append(ARGHit(
                contig_id=contig_id,
                gene_name=gene_name,
                aro_accession=accession,
                amr_family=subclass or drug_cls,
                drug_class=drug_cls,
                resistance_mechanism=mechanism,
                identity=identity,
                coverage=coverage,
                evalue=evalue,
                source="AMR",
                _orf_id=orf_id,
            ))

    logger.info("Parsed %d AMRFinderPlus hits from %s", len(hits), tsv_path)
    return hits


def annotate_amrfinder(
    proteins_faa: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    organism: str | None = None,
    min_identity: float = AMR_MIN_IDENTITY,
    min_coverage: float = AMR_MIN_COVERAGE,
) -> list[ARGHit]:
    """Run AMRFinderPlus on pre-predicted proteins and return ARGHit list."""
    if not amrfinder_available():
        logger.warning(
            "AMRFinderPlus not found — skipping. "
            "Install: conda install -c bioconda ncbi-amrfinderplus && amrfinder_update"
        )
        return []

    proteins_faa = Path(proteins_faa)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = work_dir / "amrfinder_hits.tsv"

    if out_tsv.exists() and out_tsv.stat().st_size > 0:
        logger.info("Reusing cached AMRFinderPlus hits from %s", out_tsv)
    else:
        run_amrfinder(proteins_faa, out_tsv, threads=threads, organism=organism)

    return parse_amrfinder_hits(out_tsv, min_identity=min_identity, min_coverage=min_coverage)
