"""Plasmid database matching — closest known plasmid per contig.

For each classified plasmid contig, searches the combined PLSDB/RefSeq/COMPASS
nucleotide database using minimap2 (asm20 preset) or BLAST and reports:

  plasmid_db_match    — best-hit accession (e.g. PLSDB_NZ_CP073379.1)
  plasmid_db_source   — source DB (PLSDB / RefSeq / COMPASS)
  plasmid_db_ani      — approximate nucleotide identity % of the best hit
  plasmid_db_cov      — query coverage % of the best hit
  plasmid_db_organism — organism field from the DB FASTA header (if present)

Why this matters
----------------
The contig-level taxonomy (from DIAMOND + LCA) tells you the *predicted host
organism* based on protein-coding genes.  The plasmid-DB match tells you
which *known plasmid* this contig is closest to.  Comparing the two reveals:

  - Mobilisability:  a contig that matches a well-characterised conjugative
    plasmid (e.g. IncF, IncP) is likely transmissible between species.
  - Novel plasmids:  low identity (<90 %) to any known plasmid suggests the
    contig may represent a novel plasmid lineage.
  - Host range:  if the DB organism matches the contig taxonomy, the plasmid
    is likely resident; a mismatch indicates recent horizontal transfer.

Tool requirements
-----------------
minimap2 must be installed (conda install -c bioconda minimap2).  If minimap2
is unavailable the function returns empty results and logs a warning rather
than raising.

Database setup (handled by scripts/setup_databases.sh)
-------------------------------------------------------
The combined plasmid FASTA is built from PLSDB + RefSeq plasmids + COMPASS:

    cat data/databases/plasmids/PLSDB.fna \\
        data/databases/plasmids/RefSeq.fna \\
        data/databases/plasmids/COMPASS.fna \\
        > data/databases/plasmids/combined.fna

No pre-indexing needed — minimap2 builds an in-memory index at runtime.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum thresholds for reporting a plasmid-DB match
PLAS_MIN_ANI = 85.0  # % sequence identity
PLAS_MIN_COV = 50.0  # % query contig coverage
PLAS_MIN_MAPQ = 10  # minimap2 MAPQ


@dataclass
class PlasmidDBHit:
    """Best-match result for a single contig against the plasmid database."""

    contig_id: str
    match_acc: str  # matched accession in the plasmid DB
    source_db: str  # PLSDB / RefSeq / COMPASS / unknown
    ani: float  # approximate nucleotide identity %
    query_cov: float  # % of query contig covered by the alignment
    organism: str  # organism from FASTA header (empty if not embedded)


def _infer_source(acc: str) -> str:
    if acc.startswith("PLSDB_"):
        return "PLSDB"
    if acc.startswith("RefSeq_"):
        return "RefSeq"
    if acc.startswith("COMPASS_"):
        return "COMPASS"
    return "unknown"


def _parse_organism(desc: str) -> str:
    """Extract organism name from a FASTA description field if present.

    Handles formats like:
        NZ_CP073379.1 Klebsiella pneumoniae plasmid pKP1, complete sequence
        PLSDB_OY754463.1 [no organism embedded]
    """
    # Common patterns: bracketed organism or second token onwards
    m = re.search(r"\[([^\]]+)\]", desc)
    if m:
        return m.group(1)
    # For plain descriptions, return everything after the accession token
    parts = desc.strip().split(None, 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _build_combined_fasta(plasmid_db_dir: Path) -> Path | None:
    """Locate the best plasmid reference FASTA for minimap2 matching.

    Strategy: prefer PLSDB alone (5 GB) over the full combined FASTA (13 GB).
    minimap2 --split-prefix works around the 4 GB index limit but still loads
    chunks into RAM; a 13 GB reference often causes SIGKILL (OOM) on machines
    with <32 GB RAM.  PLSDB alone covers >95% of known plasmid sequences and
    fits comfortably in 16 GB RAM.

    Falls back to combined.fna if PLSDB is absent, or builds combined from
    whatever sources are available.
    """
    # Prefer PLSDB only — best RAM/coverage trade-off (accept both naming conventions)
    plsdb = next(
        (
            plasmid_db_dir / n
            for n in ("plsdb.fasta", "PLSDB.fna", "plsdb.fna")
            if (plasmid_db_dir / n).exists()
        ),
        None,
    )
    if plsdb is not None:
        size_gb = plsdb.stat().st_size / 1e9
        logger.info(
            "Using %s as plasmid reference (%.1f GB) — avoids OOM on large combined FASTA",
            plsdb.name,
            size_gb,
        )
        return plsdb

    # Fall back to pre-built combined FASTA
    combined = plasmid_db_dir / "combined.fna"
    if combined.exists():
        logger.warning(
            "PLSDB.fna not found — using combined.fna (%.1f GB). "
            "This may OOM on machines with <32 GB RAM.",
            combined.stat().st_size / 1e9,
        )
        return combined

    # Build combined from whatever's available
    sources = ["RefSeq.fna", "COMPASS.fna"]
    available = [plasmid_db_dir / s for s in sources if (plasmid_db_dir / s).exists()]
    if not available:
        logger.warning("No plasmid FASTA files found in %s", plasmid_db_dir)
        return None

    logger.info("Building combined plasmid FASTA from: %s", [str(p) for p in available])
    with open(combined, "wb") as out:
        for src in available:
            with open(src, "rb") as inp:
                out.write(inp.read())
    logger.info("Combined plasmid FASTA written to %s", combined)
    return combined


def _minimap2_available() -> bool:
    try:
        subprocess.run(["minimap2", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def run_plasmid_db_search(
    query_fasta: Path | str,
    plasmid_db_dir: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    min_ani: float = PLAS_MIN_ANI,
    min_cov: float = PLAS_MIN_COV,
) -> list[PlasmidDBHit]:
    """Search plasmid contigs against the combined plasmid nucleotide database.

    Uses minimap2 (asm20 preset) — appropriate for comparing assembled contigs
    (typically 1–200 kb) against known plasmid sequences.

    Args:
        query_fasta:    FASTA of plasmid-classified contigs.
        plasmid_db_dir: Directory containing PLSDB.fna / RefSeq.fna / COMPASS.fna.
        work_dir:       Directory for intermediate files.
        threads:        minimap2 threads.
        min_ani:        Minimum % identity to report a hit (default 85 %).
        min_cov:        Minimum query coverage % (default 50 %).

    Returns:
        List of PlasmidDBHit — one per contig with a hit above thresholds.
        Contigs with no match are not included (caller should use .get()).
    """
    query_fasta = Path(query_fasta)
    plasmid_db_dir = Path(plasmid_db_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if not _minimap2_available():
        logger.warning(
            "minimap2 not found — skipping plasmid-DB matching. "
            "Install with: conda install -c bioconda minimap2"
        )
        return []

    combined_fasta = _build_combined_fasta(plasmid_db_dir)
    if combined_fasta is None:
        return []

    paf_path = work_dir / "plasmid_db_hits.paf"
    split_prefix = work_dir / "mm2_split"

    # Cache: if PAF already exists and is non-empty, skip minimap2
    if paf_path.exists() and paf_path.stat().st_size > 0:
        logger.info("Plasmid-DB: reusing cached PAF from %s", paf_path)
        return _parse_paf(paf_path, min_ani=min_ani, min_cov=min_cov)

    # --split-prefix lets minimap2 chunk a large reference (>4 GB) into multiple
    # index files rather than loading the whole thing into RAM at once.  Without it
    # a 13 GB combined FASTA requires ~60–80 GB RAM and OOMs on most machines.
    cmd = [
        "minimap2",
        "-x",
        "asm20",  # ~85–95 % identity preset
        "--secondary=no",  # best hit only per query
        "-t",
        str(threads),
        "--split-prefix",
        str(split_prefix),  # chunked indexing for large reference
        str(combined_fasta),
        str(query_fasta),
    ]
    logger.info(
        "Running minimap2 plasmid-DB search (split-prefix for large ref): %s", " ".join(cmd)
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("minimap2 failed (exit %d): %s", result.returncode, result.stderr[:800])
            return []
        paf_path.write_text(result.stdout)
    except Exception as exc:
        logger.warning("minimap2 plasmid-DB search error: %s", exc)
        return []

    return _parse_paf(paf_path, min_ani=min_ani, min_cov=min_cov)


def _parse_paf(paf_path: Path, min_ani: float, min_cov: float) -> list[PlasmidDBHit]:
    """Parse minimap2 PAF output into PlasmidDBHit objects.

    PAF columns (0-indexed):
        0  query_name
        1  query_len
        2  query_start
        3  query_end
        4  strand
        5  target_name
        6  target_len
        7  target_start
        8  target_end
        9  residue_matches
        10 alignment_block_len
        11 mapq
        ... optional tags including de:f: (divergence) or NM:i: (mismatches)
    """
    hits: list[PlasmidDBHit] = []
    best_by_query: dict[str, PlasmidDBHit] = {}

    with open(paf_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue
            try:
                qname = parts[0]
                qlen = int(parts[1])
                qstart = int(parts[2])
                qend = int(parts[3])
                tname = parts[5]
                res_match = int(parts[9])
                aln_len = int(parts[10])
                mapq = int(parts[11])
            except (ValueError, IndexError):
                continue

            if mapq < PLAS_MIN_MAPQ or aln_len == 0:
                continue

            ani = 100.0 * res_match / aln_len
            cov = 100.0 * (qend - qstart) / max(qlen, 1)

            if ani < min_ani or cov < min_cov:
                continue

            # Extract organism from optional tg:Z: tag (if present) or from tname
            organism = ""
            for tag in parts[12:]:
                if tag.startswith("tg:Z:"):
                    organism = tag[5:]
                    break

            hit = PlasmidDBHit(
                contig_id=qname,
                match_acc=tname,
                source_db=_infer_source(tname),
                ani=round(ani, 2),
                query_cov=round(cov, 2),
                organism=organism,
            )

            # Keep best (highest ANI × coverage) hit per query
            prev = best_by_query.get(qname)
            if prev is None or (hit.ani * hit.query_cov) > (prev.ani * prev.query_cov):
                best_by_query[qname] = hit

    hits = list(best_by_query.values())
    logger.info("Plasmid-DB search: %d contigs matched above thresholds", len(hits))
    return hits


def annotate_plasmid_db(
    plasmid_fasta: Path | str,
    plasmid_db_dir: Path | str,
    work_dir: Path | str,
    threads: int = 8,
) -> dict[str, PlasmidDBHit]:
    """Run plasmid-DB search and return results keyed by contig_id.

    Args:
        plasmid_fasta:   FASTA of plasmid-classified contigs.
        plasmid_db_dir:  Directory with PLSDB.fna / RefSeq.fna / COMPASS.fna.
        work_dir:        Directory for intermediate files.
        threads:         CPU threads for minimap2.

    Returns:
        Dict mapping contig_id → PlasmidDBHit.  Contigs with no match above
        thresholds are absent from the dict.
    """
    hits = run_plasmid_db_search(
        query_fasta=plasmid_fasta,
        plasmid_db_dir=plasmid_db_dir,
        work_dir=work_dir,
        threads=threads,
    )
    return {h.contig_id: h for h in hits}
