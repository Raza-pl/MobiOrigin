"""BacMet2 — Biocide and Metal Resistance Gene annotation via DIAMOND.

Pipeline:
    proteins.faa → run_diamond(bacmet.dmnd) → bacmet_hits.tsv
                 → parse_bacmet_hits()       → [BacMetHit]

Database setup (one-time):
    bash scripts/setup_bacmet_ice_diamond.sh

Header format in BacMet2_EXP.fasta:
    >BAC0001|abeM|tr|Q5FAM9|Q5FAM9_ACIBA Multidrug efflux pump AbeM OS=...
    Fields: BacMet_ID | Gene_name | db | UniProt_acc | entry_name  description
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

BACMET_MIN_IDENTITY = 80.0
BACMET_MIN_COVERAGE = 80.0

# sseqid format: BAC0001|abeM|tr|Q5FAM9|Q5FAM9_ACIBA
_BACMET_ID_RE = re.compile(r"^(BAC\d+)\|([^|]+)")


@dataclass
class BacMetHit:
    """Single DIAMOND hit against BacMet2."""

    contig_id: str
    gene_name: str  # e.g. "abeM"
    bacmet_id: str  # e.g. "BAC0001"
    resistance_class: str  # "Bio" | "Met" | "Bio/Met"
    compound: str  # comma-separated compound names (shortened)
    identity: float
    coverage: float
    evalue: float
    _orf_id: str = field(default="", repr=False, compare=False)


def load_bacmet_metadata(tsv_path: Path | str) -> dict[str, dict]:  # type: ignore
    """Load Bacmet_list.tsv → dict keyed by BacMet_ID.

    Columns: BacMet_ID, Gene_name, Class, Accession, Organism, Length, Location, Compound
    Returns: {BAC0001: {gene_name, resistance_class, compound_short}}
    """
    meta: dict[str, dict] = {}  # type: ignore
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        logger.warning("BacMet metadata not found at %s", tsv_path)
        return meta

    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            bacmet_id = row.get("BacMet_ID", "").strip()
            if not bacmet_id:
                continue
            compound_raw = row.get("Compound", "").strip()
            compounds = []
            for part in compound_raw.split("],"):
                name = part.split("[class:")[0].strip().rstrip(",").strip()
                if name:
                    compounds.append(name)
            compound_short = "; ".join(compounds[:5])
            if len(compounds) > 5:
                compound_short += f" (+{len(compounds)-5} more)"
            meta[bacmet_id] = {
                "gene_name": row.get("Gene_name", "").strip(),
                "resistance_class": row.get("Class", "").strip(),
                "compound": compound_short,
            }

    logger.info("Loaded %d BacMet entries from %s", len(meta), tsv_path)
    return meta


def parse_bacmet_hits(
    tsv_path: Path | str,
    bacmet_meta: dict | None = None,  # type: ignore
    min_identity: float = BACMET_MIN_IDENTITY,
    min_coverage: float = BACMET_MIN_COVERAGE,
) -> list[BacMetHit]:
    """Parse DIAMOND tabular output against BacMet2 DB."""
    tsv_path = Path(tsv_path)
    meta = bacmet_meta or {}
    hits: list[BacMetHit] = []

    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return hits

    with open(tsv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            qseqid, sseqid, pident, qcovhsp, evalue, _stitle = (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                parts[4],
                parts[5],
            )
            try:
                ident = float(pident)
                cov = float(qcovhsp)
            except ValueError:
                continue
            if ident < min_identity or cov < min_coverage:
                continue

            contig_id = re.sub(r"_\d+$", "", qseqid)
            m = _BACMET_ID_RE.match(sseqid)
            if m:
                bacmet_id = m.group(1)
                gene_name = m.group(2)
            else:
                bacmet_id = sseqid
                gene_name = sseqid

            entry = meta.get(bacmet_id, {})
            hits.append(
                BacMetHit(
                    contig_id=contig_id,
                    gene_name=entry.get("gene_name", gene_name),
                    bacmet_id=bacmet_id,
                    resistance_class=entry.get("resistance_class", "unknown"),
                    compound=entry.get("compound", ""),
                    identity=ident,
                    coverage=cov,
                    evalue=float(evalue),
                    _orf_id=qseqid,
                )
            )

    logger.info("Parsed %d BacMet hits from %s", len(hits), tsv_path)
    return hits


def annotate_bacmet(
    proteins_faa: Path | str,
    bacmet_db: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    min_identity: float = BACMET_MIN_IDENTITY,
    min_coverage: float = BACMET_MIN_COVERAGE,
) -> list[BacMetHit]:
    """Run DIAMOND against BacMet2 and return BacMetHit list."""
    proteins_faa = Path(proteins_faa)
    bacmet_db = Path(bacmet_db)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata from xlsx in same directory as the DB
    meta_candidates = list(bacmet_db.parent.glob("Bacmet_list.tsv")) + list(
        bacmet_db.parent.glob("*.tsv")
    )
    bacmet_meta = load_bacmet_metadata(meta_candidates[0]) if meta_candidates else {}

    out_tsv = work_dir / "bacmet_hits.tsv"
    if out_tsv.exists() and out_tsv.stat().st_size > 0:
        logger.info("Reusing cached BacMet hits from %s", out_tsv)
    else:
        db_stem = str(bacmet_db).removesuffix(".dmnd")
        cmd = [
            "diamond",
            "blastp",
            "--query",
            str(proteins_faa),
            "--db",
            db_stem,
            "--out",
            str(out_tsv),
            "--outfmt",
            "6",
            "qseqid",
            "sseqid",
            "pident",
            "qcovhsp",
            "evalue",
            "stitle",
            "--id",
            str(min_identity),
            "--query-cover",
            str(min_coverage),
            "--threads",
            str(threads),
            "--sensitive",
            "--max-target-seqs",
            "1",
            "--quiet",
        ]
        logger.info("Running DIAMOND BacMet: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("BacMet DIAMOND failed: %s", result.stderr[:300])
            return []

    return parse_bacmet_hits(out_tsv, bacmet_meta, min_identity, min_coverage)
