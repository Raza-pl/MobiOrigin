"""Virulence factor annotation via DIAMOND + VFDB (set A, validated VFs only).

Pipeline:
    plasmid FASTA → call_orfs() → proteins.faa
                  → run_diamond(vfdb.dmnd) → vfdb_hits.tsv
                  → parse_vfdb_hits() → [VFHit]

Database setup (one-time, handled by scripts/setup_databases.sh):
    wget http://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz
    gunzip VFDB_setA_pro.fas.gz
    diamond makedb --in VFDB_setA_pro.fas -d data/databases/vfdb/vfdb_setA

VFDB set A contains only experimentally validated virulence factors (core dataset).
Set B (broader, unvalidated) is available but generates more false positives.

Header format example:
    >VFG000068(gb|AAD42099) mgtC [mgtC (VF0091)] [Salmonella enterica]
     ├── VFG000068: VFDB gene ID
     ├── gb|AAD42099: GenBank accession
     ├── mgtC: gene name
     ├── VF0091: virulence factor group ID
     └── Salmonella enterica: source organism
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from plasflow2.annotate.args import call_orfs

logger = logging.getLogger(__name__)

# DIAMOND thresholds — 60 % identity is the VFDB community standard for
# detecting divergent virulence factor homologues in environmental metagenomes.
VFDB_MIN_IDENTITY = 60.0
VFDB_MIN_COVERAGE = 80.0

# Header regex for VFDB set A protein FASTA
# >VFG000068(gb|AAD42099) mgtC [mgtC (VF0091)] [Salmonella enterica]
_VFDB_HEADER_RE = re.compile(
    r"^(?P<vfg_id>VFG\d+)"
    r"\((?:gb|ref)\|(?P<accession>[^|)]+)\)"
    r"\s+(?P<gene_name>\S+)"
    r"(?:\s+\[(?P<vf_group>[^\]]+)\])?"
    r"(?:\s+\[(?P<organism>[^\]]+)\])?"
)


def load_vfdb_metadata(index_path: Path | str) -> dict[str, str]:
    """Load vfdb_indx.txt → dict mapping VFG accession → functional category.

    File format (tab-separated, no header):
        VFG050302(gb|WP_001045627.1)  VFC0001  Adherence

    Returns: {WP_001045627.1: "Adherence"}
    """
    meta: dict[str, str] = {}
    index_path = Path(index_path)
    if not index_path.exists():
        return meta
    with open(index_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            vfg_field = parts[0]  # e.g. VFG050302(gb|WP_001045627.1)
            category = parts[2].strip()
            # Extract accession from parentheses
            acc_m = re.search(r"\((?:gb|ref)\|([^|)]+)\)", vfg_field)
            if acc_m:
                meta[acc_m.group(1)] = category
            # Also key by VFG ID alone
            vfg_m = re.match(r"(VFG\d+)", vfg_field)
            if vfg_m:
                meta[vfg_m.group(1)] = category
    logger.info("Loaded %d VFDB category entries from %s", len(meta), index_path)
    return meta


@dataclass
class VFHit:
    """Single DIAMOND hit against the VFDB virulence factor database."""

    contig_id: str
    gene_name: str  # e.g. "mgtC", "stx1A"
    vfg_id: str  # VFDB gene ID, e.g. "VFG000068"
    vf_group: str  # VF group name, e.g. "mgtC (VF0091)"
    vf_category: str  # Functional category from vfdb_indx.txt, e.g. "Adherence", "Toxin"
    organism: str  # Source organism, e.g. "Salmonella enterica"
    identity: float  # % amino-acid identity
    coverage: float  # % query coverage
    evalue: float
    # Internal: ORF id used for gene-level table, not exposed in summary reports
    _orf_id: str = field(default="", repr=False, compare=False)


def _parse_vfdb_stitle(stitle: str) -> tuple[str, str, str, str]:
    """Parse gene_name, vfg_id, vf_group, organism from a VFDB stitle field.

    The stitle field in DIAMOND outfmt 6 is the full FASTA header minus the '>'.

    Returns:
        (gene_name, vfg_id, vf_group, organism) — any field may be empty string
        if the header doesn't match the expected pattern.
    """
    m = _VFDB_HEADER_RE.match(stitle.strip())
    if not m:
        # Fallback: use whatever is in the title as gene_name
        return stitle.strip()[:40], "", "", ""
    return (
        m.group("gene_name") or "",
        m.group("vfg_id") or "",
        m.group("vf_group") or "",
        m.group("organism") or "",
    )


def run_vfdb_diamond(
    protein_fasta: Path | str,
    vfdb: Path | str,
    out_tsv: Path | str,
    threads: int = 8,
    min_identity: float = VFDB_MIN_IDENTITY,
    min_coverage: float = VFDB_MIN_COVERAGE,
) -> None:
    """Run DIAMOND BLASTp against the VFDB database.

    Args:
        protein_fasta: ORF-called protein sequences (.faa).
        vfdb: Path to DIAMOND-formatted VFDB database (.dmnd).
        out_tsv: Output path for DIAMOND tabular results.
        threads: CPU threads.
        min_identity: Minimum amino-acid identity % (default 60).
        min_coverage: Minimum query coverage % (default 80).
    """
    protein_fasta = Path(protein_fasta)
    vfdb = Path(vfdb)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "diamond",
        "blastp",
        "--query",
        str(protein_fasta),
        "--db",
        str(vfdb),
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
    ]
    logger.info("Running DIAMOND (VFDB): %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DIAMOND (VFDB) failed: %s", result.stderr[:500])
        raise RuntimeError(f"DIAMOND (VFDB) failed with exit code {result.returncode}")


def parse_vfdb_hits(
    tsv_path: Path | str,
    vfdb_meta: dict[str, str] | None = None,
) -> list[VFHit]:
    """Parse DIAMOND tabular output against VFDB into VFHit objects."""
    tsv_path = Path(tsv_path)
    meta = vfdb_meta or {}
    hits: list[VFHit] = []

    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return hits

    with open(tsv_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            qseqid, _sseqid, pident, qcovhsp, evalue, stitle = row[:6]
            contig_id = "_".join(qseqid.rsplit("_", 1)[:-1]) if "_" in qseqid else qseqid
            gene_name, vfg_id, vf_group, organism = _parse_vfdb_stitle(stitle)

            # Resolve category: try VFG ID first, then accession from stitle
            vf_category = meta.get(vfg_id, "")
            if not vf_category:
                acc_m = re.search(r"\((?:gb|ref)\|([^|)]+)\)", stitle)
                if acc_m:
                    vf_category = meta.get(acc_m.group(1), "")

            hits.append(
                VFHit(
                    contig_id=contig_id,
                    gene_name=gene_name,
                    vfg_id=vfg_id,
                    vf_group=vf_group,
                    vf_category=vf_category,
                    organism=organism,
                    identity=float(pident),
                    coverage=float(qcovhsp),
                    evalue=float(evalue),
                    _orf_id=qseqid,
                )
            )

    logger.info("Parsed %d VFDB virulence factor hits from %s", len(hits), tsv_path)
    return hits


def annotate_vf(
    fasta_path: Path | str,
    vfdb: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    min_identity: float = VFDB_MIN_IDENTITY,
    min_coverage: float = VFDB_MIN_COVERAGE,
    reuse_proteins: Path | str | None = None,
) -> list[VFHit]:
    """End-to-end virulence factor annotation: ORF prediction → DIAMOND → hits.

    Args:
        fasta_path: Nucleotide FASTA of plasmid contigs.
        vfdb: Path to DIAMOND .dmnd database built from VFDB set A proteins.
        work_dir: Directory for intermediate files.
        threads: CPU threads for DIAMOND.
        min_identity: Minimum amino-acid identity % (default 60).
        min_coverage: Minimum query coverage % (default 80).
        reuse_proteins: If provided, skip ORF prediction and use this .faa file
            directly (e.g. already predicted for ARG annotation).

    Returns:
        List of VFHit across all contigs.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    proteins_path = Path(reuse_proteins) if reuse_proteins else work_dir / "proteins.faa"
    vfdb_tsv = work_dir / "vfdb_hits.tsv"

    if reuse_proteins is None:
        call_orfs(fasta_path, proteins_path)
    else:
        logger.info("Reusing pre-predicted ORFs from %s", proteins_path)

    # Load VFDB category index (auto-detect from DB directory)
    vfdb_path = Path(vfdb)
    idx_candidates = list(vfdb_path.parent.glob("vfdb_indx.txt")) + list(
        vfdb_path.parent.glob("*.txt")
    )
    vfdb_meta = load_vfdb_metadata(idx_candidates[0]) if idx_candidates else {}

    if vfdb_tsv.exists() and vfdb_tsv.stat().st_size > 0:
        logger.info("Reusing cached VFDB hits from %s", vfdb_tsv)
    else:
        run_vfdb_diamond(
            proteins_path,
            vfdb,
            vfdb_tsv,
            threads=threads,
            min_identity=min_identity,
            min_coverage=min_coverage,
        )
    return parse_vfdb_hits(vfdb_tsv, vfdb_meta)
