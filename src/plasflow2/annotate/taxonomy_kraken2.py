"""Kraken2-based contig taxonomy — fast k-mer classification fallback.

Kraken2 classifies nucleotide sequences directly using an exact k-mer index,
making it ~50× faster than DIAMOND blastp for large datasets.  It is used as
a FALLBACK for contigs where DIAMOND finds no hits — typically short or
repetitive contigs with few/no detectable ORFs.

DIAMOND result always takes priority: higher sensitivity in protein space,
genus/species-level resolution.  Kraken2 contributes phylum/class-level
classification for the ~40-60% of metagenomic contigs that DIAMOND misses.

Pipeline
--------
nucleotide FASTA  →  kraken2 (k-mer index search)  →  kraken2_output.txt
                  →  parse_kraken2_output()          →  taxid per contig
                  →  taxid_to_lineage() via nodes/names.dmp
                  →  dict[contig_id → TaxResult]

Database setup (one-time, ~8 GB download, ~30 sec runtime)
-----------------------------------------------------------
    bash scripts/setup_kraken2_db.sh
    # Downloads pre-built standard-8 database to data/databases/kraken2/

Auto-detected from data/databases/kraken2/ (hash.k2d must be present).

Reference
---------
Wood DE, Lu J, Langmead B. Improved metagenomic analysis with Kraken 2.
Genome Biology 20, 257 (2019). https://doi.org/10.1186/s13059-019-1891-0
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from plasflow2.annotate.taxonomy import TaxResult

logger = logging.getLogger(__name__)

# NCBI rank names → GTDB-style rank names
_NCBI_TO_GTDB_RANK = {
    "superkingdom": "domain",
    "phylum": "phylum",
    "class": "class",
    "order": "order",
    "family": "family",
    "genus": "genus",
    "species": "species",
}

# GTDB prefix for each rank
_RANK_PREFIX = {
    "domain": "d__",
    "phylum": "p__",
    "class": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
    "species": "s__",
}

_RANK_ORDER = ["domain", "phylum", "class", "order", "family", "genus", "species"]


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def kraken2_available() -> bool:
    """Return True if kraken2 is on PATH."""
    try:
        subprocess.run(["kraken2", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def find_kraken2_db(search_dirs: list[Path] | None = None) -> Path | None:
    """Find the first Kraken2 database directory containing hash.k2d.

    Checks *search_dirs* in order, then falls back to the default location
    data/databases/kraken2/ relative to the project root.
    """
    candidates: list[Path] = list(search_dirs or [])
    # Default: project root / data/databases/kraken2
    _default = Path(__file__).parent.parent.parent.parent / "data" / "databases" / "kraken2"
    candidates.append(_default)

    for d in candidates:
        if d.is_dir() and (d / "hash.k2d").exists():
            return d
    return None


# ---------------------------------------------------------------------------
# NCBI taxonomy loader (reuses nodes/names.dmp from Kaiju setup)
# ---------------------------------------------------------------------------


def _load_ncbi_taxonomy(
    nodes_dmp: Path,
    names_dmp: Path,
) -> tuple[dict[int, int], dict[int, str], dict[int, str]]:
    """Load NCBI taxonomy into parent, rank, and name dicts.

    Returns:
        parent: taxid → parent taxid
        rank:   taxid → rank string (e.g. 'genus')
        name:   taxid → scientific name
    """
    parent: dict[int, int] = {}
    rank: dict[int, str] = {}
    name: dict[int, str] = {}

    logger.debug("Loading NCBI taxonomy from %s …", nodes_dmp)
    with open(nodes_dmp) as fh:
        for line in fh:
            parts = line.split("|")
            if len(parts) < 3:
                continue
            taxid = int(parts[0].strip())
            ptaxid = int(parts[1].strip())
            rk = parts[2].strip().lower()
            parent[taxid] = ptaxid
            rank[taxid] = rk

    with open(names_dmp) as fh:
        for line in fh:
            parts = line.split("|")
            if len(parts) < 4:
                continue
            taxid = int(parts[0].strip())
            nm = parts[1].strip()
            nm_class = parts[3].strip()
            if nm_class == "scientific name":
                name[taxid] = nm

    logger.debug("Loaded %d taxonomy nodes", len(parent))
    return parent, rank, name


def taxid_to_lineage(
    taxid: int,
    parent: dict[int, int],
    rank: dict[int, str],
    name: dict[int, str],
) -> tuple[str, str, str]:
    """Walk NCBI taxonomy from taxid to root, build GTDB-style lineage.

    Returns:
        (lineage_str, deepest_rank, deepest_taxon)
        e.g. ("d__Bacteria;p__Proteobacteria", "phylum", "p__Proteobacteria")
    """
    levels: dict[str, str] = {}
    current = taxid
    visited: set[int] = set()

    while current not in (0, 1) and current not in visited:
        visited.add(current)
        rk = _NCBI_TO_GTDB_RANK.get(rank.get(current, ""), "")
        nm = name.get(current, "")
        if rk and nm and nm.lower() not in ("root", "cellular organisms", ""):
            prefix = _RANK_PREFIX.get(rk, "")
            levels[rk] = f"{prefix}{nm}"
        current = parent.get(current, 1)

    # Build lineage in rank order
    parts = [levels[r] for r in _RANK_ORDER if r in levels]
    if not parts:
        return "", "unclassified", ""

    deepest_rank = next((r for r in reversed(_RANK_ORDER) if r in levels), "unclassified")
    deepest_taxon = levels.get(deepest_rank, "")
    return ";".join(parts), deepest_rank, deepest_taxon


# ---------------------------------------------------------------------------
# Run Kraken2
# ---------------------------------------------------------------------------


def run_kraken2(
    fasta_path: Path | str,
    db_dir: Path | str,
    out_file: Path | str,
    report_file: Path | str,
    threads: int = 8,
    confidence: float = 0.1,
) -> Path:
    """Run Kraken2 on a nucleotide FASTA.

    Args:
        fasta_path:   Input nucleotide FASTA (contigs).
        db_dir:       Kraken2 database directory (contains hash.k2d).
        out_file:     Per-sequence output file path.
        report_file:  Summary report file path.
        threads:      CPU threads.
        confidence:   Confidence threshold (0–1). Higher = fewer but more
                      accurate classifications. 0.1 is a good default for
                      metagenomic contigs ≥1 kb.

    Returns:
        Path to the per-sequence output file.
    """
    fasta_path = Path(fasta_path)
    db_dir = Path(db_dir)
    out_file = Path(out_file)
    report_file = Path(report_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kraken2",
        "--db",
        str(db_dir),
        "--threads",
        str(threads),
        "--confidence",
        str(confidence),
        "--output",
        str(out_file),
        "--report",
        str(report_file),
        "--report-minimizer-data",
        str(fasta_path),
    ]
    # Handle compressed input
    if str(fasta_path).endswith(".gz"):
        cmd.insert(1, "--gzip-compressed")
    elif str(fasta_path).endswith(".bz2"):
        cmd.insert(1, "--bzip2-compressed")

    logger.info(
        "Running Kraken2 (threads=%d, confidence=%.2f): %s",
        threads,
        confidence,
        " ".join(cmd),
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Kraken2 failed (exit %d): %s", result.returncode, result.stderr[:400])
        raise RuntimeError(f"kraken2 failed with exit code {result.returncode}")

    # Log summary stats from stderr
    for line in result.stderr.splitlines():
        if "sequences classified" in line or "sequences unclassified" in line:
            logger.info("Kraken2: %s", line.strip())

    return out_file


# ---------------------------------------------------------------------------
# Parse Kraken2 output
# ---------------------------------------------------------------------------


def parse_kraken2_output(out_file: Path | str) -> dict[str, int]:
    """Parse Kraken2 per-sequence output → contig_id → taxid.

    Kraken2 output format (tab-separated):
        C/U  sequence_id  taxid  length  kmer_hits

    Only classified sequences (C) are returned.

    Args:
        out_file: Path to kraken2 --output file.

    Returns:
        Dict mapping contig_id → NCBI taxid (classified contigs only).
    """
    result: dict[str, int] = {}
    with open(out_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            status = parts[0]
            if status != "C":
                continue
            contig_id = parts[1].strip()
            taxid = int(parts[2].strip())
            if taxid > 0:
                result[contig_id] = taxid

    logger.info("Kraken2: %d contigs classified", len(result))
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def assign_taxonomy_kraken2(
    fasta_path: Path | str,
    db_dir: Path | str,
    nodes_dmp: Path | str,
    names_dmp: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    confidence: float = 0.1,
    existing_taxonomy: dict[str, TaxResult] | None = None,
) -> dict[str, TaxResult]:
    """Full Kraken2 taxonomy pipeline: run → parse → taxid→lineage → TaxResult.

    Args:
        fasta_path:          Input nucleotide FASTA (all contigs).
        db_dir:              Kraken2 database directory.
        nodes_dmp:           NCBI taxonomy nodes.dmp.
        names_dmp:           NCBI taxonomy names.dmp.
        work_dir:            Directory for intermediate files.
        threads:             CPU threads for Kraken2.
        confidence:          Kraken2 confidence threshold (default 0.1).
        existing_taxonomy:   If provided, only classify contigs that are absent
                             from this dict (DIAMOND fallback mode — DIAMOND
                             results always take priority).

    Returns:
        Dict mapping contig_id → TaxResult.
        When existing_taxonomy is provided, only returns NEW classifications
        (the caller merges them into the existing dict).
    """
    fasta_path = Path(fasta_path)
    db_dir = Path(db_dir)
    nodes_dmp = Path(nodes_dmp)
    names_dmp = Path(names_dmp)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    out_file = work_dir / "kraken2_output.txt"
    report_file = work_dir / "kraken2_report.txt"

    # 1. Run Kraken2 (or reuse cached result)
    if out_file.exists() and out_file.stat().st_size > 0:
        logger.info("Kraken2: reusing cached output from %s", out_file)
    else:
        run_kraken2(
            fasta_path=fasta_path,
            db_dir=db_dir,
            out_file=out_file,
            report_file=report_file,
            threads=threads,
            confidence=confidence,
        )

    # 2. Parse taxid per contig
    taxid_by_contig = parse_kraken2_output(out_file)

    # Filter to only contigs missing from DIAMOND results
    if existing_taxonomy:
        taxid_by_contig = {
            cid: tid for cid, tid in taxid_by_contig.items() if cid not in existing_taxonomy
        }
        logger.info(
            "Kraken2 fallback: %d new classifications for contigs missing from DIAMOND",
            len(taxid_by_contig),
        )

    if not taxid_by_contig:
        return {}

    # 3. Load NCBI taxonomy
    parent, rank, name = _load_ncbi_taxonomy(nodes_dmp, names_dmp)

    # 4. Convert taxid → TaxResult
    results: dict[str, TaxResult] = {}
    unclassified = 0

    for contig_id, taxid in taxid_by_contig.items():
        lineage, deepest_rank, deepest_taxon = taxid_to_lineage(taxid, parent, rank, name)
        if deepest_rank == "unclassified" or not deepest_taxon:
            unclassified += 1
            continue
        results[contig_id] = TaxResult(
            contig_id=contig_id,
            lineage=lineage,
            rank=deepest_rank,
            taxon=deepest_taxon,
            num_hits=1,  # Kraken2 gives single classification
            agreement=1.0,
        )

    classified = len(results)
    logger.info(
        "Kraken2: %d contigs → TaxResult (%d resolved lineage, %d dropped/root-level)",
        len(taxid_by_contig),
        classified,
        unclassified,
    )
    return results
