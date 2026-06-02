"""ICEberg3 — Integrative and Conjugative Element annotation via DIAMOND.

Pipeline:
    proteins.faa → run_diamond(ice.dmnd) → ice_hits.tsv
                 → parse_ice_hits()      → [ICEHit]

Database setup (one-time):
    bash scripts/setup_bacmet_ice_diamond.sh

Header format in ICEberg3_experimental.fasta:
    >ICEberg|1010 gi|1224956895|ref|ATB17827.1| Integrase [Pseudomonas aeruginosa]
    ICE ID = 1010, protein accession = ATB17827.1, function = Integrase

Annotation file (ice_experimental_list.xlsx):
    ID (FASTA header) | Col2 | Col3 | Class (gene function)
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ICE_MIN_IDENTITY = 70.0   # ICE proteins are more diverse — lower threshold
ICE_MIN_COVERAGE = 70.0

# sseqid from DIAMOND: first token of FASTA header
# Header: >ICEberg|1010 gi|...|ref|ATB17827.1| Integrase [Organism]
_ICE_ID_RE = re.compile(r"ICEberg\|(\d+)")


@dataclass
class ICEHit:
    """Single DIAMOND hit against ICEberg3."""
    contig_id: str
    ice_id: str         # ICEberg element ID, e.g. "1010"
    gene_function: str  # e.g. "Integrase", "relaxase", "conjugal transfer"
    identity: float
    coverage: float
    evalue: float
    _orf_id: str = field(default="", repr=False, compare=False)


def load_ice_metadata(tsv_path: Path | str) -> dict[str, str]:
    """Load ice_experimental_list.tsv → dict mapping protein accession → gene function.

    Columns: ID (FASTA header), col2, col3, Class (gene function)
    Returns: {protein_accession: gene_function}
    """
    meta: dict[str, str] = {}
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        logger.warning("ICE metadata not found at %s", tsv_path)
        return meta

    with open(tsv_path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            header = row[0].strip()
            func = row[3].strip()
            if not header or not func:
                continue
            acc_match = re.search(r"ref\|([^|]+)\|", header)
            if acc_match:
                meta[acc_match.group(1)] = func

    logger.info("Loaded %d ICE gene annotations from %s", len(meta), tsv_path)
    return meta


def parse_ice_hits(
    tsv_path: Path | str,
    ice_meta: dict[str, str] | None = None,
    min_identity: float = ICE_MIN_IDENTITY,
    min_coverage: float = ICE_MIN_COVERAGE,
) -> list[ICEHit]:
    """Parse DIAMOND tabular output against ICEberg3 DB."""
    tsv_path = Path(tsv_path)
    meta = ice_meta or {}
    hits: list[ICEHit] = []

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
            qseqid, sseqid, pident, qcovhsp, evalue, stitle = (
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
            )
            try:
                ident = float(pident)
                cov = float(qcovhsp)
            except ValueError:
                continue
            if ident < min_identity or cov < min_coverage:
                continue

            contig_id = re.sub(r"_\d+$", "", qseqid)

            # ICE element ID from sseqid (DIAMOND uses first word of header as sseqid)
            # sseqid = "ICEberg|1010"
            ice_m = _ICE_ID_RE.search(sseqid)
            ice_id = ice_m.group(1) if ice_m else sseqid

            # Gene function: from metadata (accession lookup) or stitle
            # stitle contains the full header text from the DB
            acc_m = re.search(r"ref\|([^|]+)\|", stitle)
            acc = acc_m.group(1) if acc_m else ""
            gene_function = meta.get(acc, "")
            if not gene_function:
                # Fall back to first bracketed term or last token before organism
                fn_m = re.search(r"\|\s+([^[]+?)\s*\[", stitle)
                gene_function = fn_m.group(1).strip() if fn_m else "unknown"

            hits.append(ICEHit(
                contig_id=contig_id,
                ice_id=ice_id,
                gene_function=gene_function,
                identity=ident,
                coverage=cov,
                evalue=float(evalue),
                _orf_id=qseqid,
            ))

    logger.info("Parsed %d ICE hits from %s", len(hits), tsv_path)
    return hits


def annotate_ice(
    proteins_faa: Path | str,
    ice_db: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    min_identity: float = ICE_MIN_IDENTITY,
    min_coverage: float = ICE_MIN_COVERAGE,
) -> list[ICEHit]:
    """Run DIAMOND against ICEberg3 and return ICEHit list."""
    proteins_faa = Path(proteins_faa)
    ice_db = Path(ice_db)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata from xlsx in same directory as DB
    meta_candidates = list(ice_db.parent.glob("ice_experimental_list.tsv")) + \
                      list(ice_db.parent.glob("*.tsv"))
    ice_meta = load_ice_metadata(meta_candidates[0]) if meta_candidates else {}

    out_tsv = work_dir / "ice_hits.tsv"
    if out_tsv.exists() and out_tsv.stat().st_size > 0:
        logger.info("Reusing cached ICE hits from %s", out_tsv)
    else:
        db_stem = str(ice_db).removesuffix(".dmnd")
        cmd = [
            "diamond", "blastp",
            "--query", str(proteins_faa),
            "--db", db_stem,
            "--out", str(out_tsv),
            "--outfmt", "6", "qseqid", "sseqid", "pident", "qcovhsp", "evalue", "stitle",
            "--id", str(min_identity),
            "--query-cover", str(min_coverage),
            "--threads", str(threads),
            "--sensitive", "--max-target-seqs", "1", "--quiet",
        ]
        logger.info("Running DIAMOND ICEberg3: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ICE DIAMOND failed: %s", result.stderr[:300])
            return []

    return parse_ice_hits(out_tsv, ice_meta, min_identity, min_coverage)
