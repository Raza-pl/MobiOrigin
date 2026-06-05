"""Mobile Genetic Element (MGE) detection via DIAMOND + MGE protein database.

Detects:
  - Insertion sequences (IS elements) — single-module transposons with one or
    two transposase ORFs; the most common MGEs on plasmids.
  - Composite transposons — flanked by IS elements carrying cargo genes.
  - Complex transposons — e.g. Tn3 family with resolvase + transposase.
  - Integrons — intI1/intI2 integrase genes; important AMR gene mobilisers.

Pipeline:
    plasmid FASTA → call_orfs() → proteins.faa
                  → run_diamond(isfinder.dmnd) → mge_hits.tsv
                  → parse_mge_hits() → [MGEHit]

Database setup (one-time, handled by scripts/setup_databases.sh):
    # Pärnänen et al. 2018 MGE database — IS*, integrons, transposons from NCBI
    # CDS translated to protein, then DIAMOND database built:
    diamond makedb --in data/databases/mge/mge_proteins.faa \\
                   -d data/databases/mge/isfinder

Database: Pärnänen et al. (2018) Nature Communications 9:3891
    https://github.com/KatariinaParnanen/MobileGeneticElementDatabase
    - ~2,000 unique MGE CDS sequences (99% identity clustered)
    - Covers IS*, ISCR*, intI1/intI2 (integrons), tniA/B, tnpA (transposons),
      qacEdelta (quaternary ammonium resistance cassettes), Tn916-family ORFs
    - Sourced from NCBI nucleotide database annotations

Header format (NCBI-style gene name + accession):
    >IS1_1 gb|AAA62386.1| IS1 transposase [Escherichia coli]
    >intI1_1 gb|AAB59737.1| integron integrase IntI1 [E. coli]
    General pattern: >{gene_name}_{n} {description}
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

# 70 % identity / 80 % coverage — transposases diverge faster than housekeeping
# genes but the DDE catalytic domain is well conserved. 70 % captures divergent
# IS copies on environmental plasmids while limiting spurious hits to non-MGE
# DDE-fold proteins (e.g. RNase H, integrases).
MGE_MIN_IDENTITY = 70.0
MGE_MIN_COVERAGE = 80.0

# IS family inference — covers ISfinder names AND Pärnänen NCBI gene names
_IS_FAMILY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Specific IS families (check before generic IS prefix)
    (re.compile(r"\bIS26\b", re.I), "IS26"),
    (re.compile(r"\bIS21\b", re.I), "IS21"),
    (re.compile(r"\bIS30\b", re.I), "IS30"),
    (re.compile(r"\bIS66\b", re.I), "IS66"),
    (re.compile(r"\bIS91\b", re.I), "IS91"),
    (re.compile(r"\bIS110\b", re.I), "IS110"),
    (re.compile(r"\bIS200\b|\bIS605\b", re.I), "IS200/IS605"),
    (re.compile(r"\bIS256\b", re.I), "IS256"),
    (re.compile(r"\bIS630\b", re.I), "IS630"),
    (re.compile(r"\bISCR\b", re.I), "ISCR"),
    (re.compile(r"\bIS1\b", re.I), "IS1"),
    (re.compile(r"\bIS3\b", re.I), "IS3"),
    (re.compile(r"\bIS4\b", re.I), "IS4"),
    (re.compile(r"\bIS5\b", re.I), "IS5"),
    (re.compile(r"\bIS6\b", re.I), "IS6"),
    # Transposons
    (re.compile(r"\bTn3\b|\bTn903\b|\bTn1000\b", re.I), "Tn3"),
    (re.compile(r"\bTn10\b|\bTn5\b|\bTn7\b|\bTn916\b", re.I), "Complex Tn"),
    # tnpA/tniA/tnpR + optional digit suffix (tnpA1, tnpA2, tniB3, etc.)
    (re.compile(r"\btni[AB]\d*\b|\btnp[AR]\d*\b|\btransposase\b", re.I), "Transposon"),
    # Integrons (intI1/intI2 integrase, istA/istB cassette genes)
    (re.compile(r"\bintegron\b|\bintI\b|\bintI1\b|\bintI2\b", re.I), "Integron"),
    (re.compile(r"\bistA\b|\bistB\b", re.I), "Integron"),
    # Other MGE types
    (re.compile(r"\bqacE\b|\bqacEdelta\b", re.I), "qacE/Integron"),
    (re.compile(r"\bMITE\b", re.I), "MITE"),
]


def _extract_gene_name_from_sseqid(sseqid: str) -> str:
    """Extract the gene_name component from a Pärnänen-format sseqid.

    Pärnänen database headers use: {row_id}_{gene_name}_{NCBI_accession}
    e.g. "904_tnpA_CP000353.2" → "tnpA"
         "2597_IS91_MNXT01000023.1" → "IS91"
         "1818_tnpA_7_10_LN890520.1" → "tnpA_7_10"

    Also handles plain ISfinder names: "ISAba1", "IS26", "intI1".
    """
    # Already a clean name (ISfinder style) — no leading digits
    if not re.match(r"^\d+_", sseqid):
        return sseqid
    # Strip leading row_id: "904_tnpA_CP000353.2" → "tnpA_CP000353.2"
    without_id = re.sub(r"^\d+_", "", sseqid)
    # Strip trailing NCBI accession: ends with _LETTERS+DIGITS+.DIGITS or _LETTERS+DIGITS
    without_accession = re.sub(r"_[A-Z]{1,6}\d{5,}(\.\d+)?$", "", without_id, flags=re.I)
    return without_accession if without_accession else without_id


def _infer_is_family(name: str, description: str) -> str:
    """Infer the IS/MGE family from element name and description.

    Handles both ISfinder-style names (ISAba1, IS26) and Pärnänen database
    NCBI-style gene names embedded in sseqids (904_tnpA_CP000353.2).

    Key fix: replace underscores with spaces before pattern matching, because
    `_` is a `\\w` character so `\\b` word boundaries don't fire around
    `_tnpA_` — making `\\btnpA\\b` fail on "904_tnpA_CP000353.2".
    """
    # Extract just the gene_name component from Pärnänen-format IDs
    gene = _extract_gene_name_from_sseqid(name)
    # Strip trailing _<number> suffix (e.g. "IS1_3" → "IS1", "tnpA1" stays)
    clean_name = re.sub(r"_\d+$", "", gene)
    # Replace underscores with spaces so \b word boundaries work correctly
    text = f"{clean_name.replace('_', ' ')} {description.replace('_', ' ')}"
    for pattern, family in _IS_FAMILY_PATTERNS:
        if pattern.search(text):
            return family
    # Generic IS prefix fallback (e.g. ISAba1 → "ISAba", ISSoc5 → "ISSoc")
    m = re.match(r"^IS([A-Za-z0-9]{1,4})", clean_name, re.I)
    if m:
        return f"IS{m.group(1)}"
    return "Unknown"


def load_mge_metadata(tsv_path: Path | str) -> dict[str, dict]:  # type: ignore
    """Load mge_database.tsv → dict keyed by gene_name (lower-case).

    Columns: ID, Sub_class (IS family e.g. IS26), gene_name, Class, Length
    Returns: {gene_name_lower: {sub_class, mge_class}}
    """
    meta: dict[str, dict] = {}  # type: ignore
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        return meta
    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sub  = row.get("Sub_class", "").strip()
            cls  = row.get("Class", "").strip()
            entry = {"sub_class": sub, "mge_class": cls}

            # Index by gene_name (e.g. "tnpA") — primary lookup key
            gene = row.get("gene_name", "").strip().lower()
            if gene:
                meta[gene] = entry

            # Also index by cleaned gene_name without trailing digits ("tnpA1" → "tnpA")
            gene_base = re.sub(r"\d+$", "", gene)
            if gene_base and gene_base != gene:
                meta.setdefault(gene_base, entry)

            # Also index by full ID for exact sseqid matching (fallback)
            full_id = row.get("ID", "").strip().lower()
            if full_id:
                meta.setdefault(full_id, entry)
    logger.info("Loaded %d MGE family entries from %s", len(meta), tsv_path)
    return meta


@dataclass
class MGEHit:
    """Single DIAMOND hit against the ISfinder MGE protein database."""

    contig_id: str
    is_name: str     # ISfinder element name, e.g. "ISAba1"
    is_family: str   # IS family from mge_database or inferred, e.g. "IS26", "Tn3"
    mge_class: str   # Broader class: "Insertion sequences", "Transposons", "Integron"
    description: str # Free-text description from ISfinder header
    identity: float  # % amino-acid identity to ISfinder reference
    coverage: float  # % query coverage
    evalue: float
    # Internal: ORF id used for gene-level table, not exposed in summary reports
    _orf_id: str = field(default="", repr=False, compare=False)


def _parse_isfinder_stitle(stitle: str) -> tuple[str, str]:
    """Extract IS element name and description from ISfinder DIAMOND stitle.

    ISfinder headers are free-form but typically start with the IS name:
        ISAba1 AcinetobacterBA...  → ("ISAba1", "Acinetobacter ...")
        IS26 transposase IS26      → ("IS26", "transposase IS26")
        ISSoc5 IS5 family ...      → ("ISSoc5", "IS5 family ...")

    Returns:
        (is_name, description)
    """
    stitle = stitle.strip()
    # IS name is the first whitespace-delimited token if it matches IS/Tn/MITE pattern
    parts = stitle.split(None, 1)
    if not parts:
        return stitle, ""
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if re.match(r"^(IS|Tn|MITE|ICE|IME|CRISPRas)", first, re.I):
        return first, rest
    return first, rest


def run_mge_diamond(
    protein_fasta: Path | str,
    mge_db: Path | str,
    out_tsv: Path | str,
    threads: int = 8,
    min_identity: float = MGE_MIN_IDENTITY,
    min_coverage: float = MGE_MIN_COVERAGE,
) -> None:
    """Run DIAMOND BLASTp against the ISfinder/MGE protein database."""
    protein_fasta = Path(protein_fasta)
    mge_db = Path(mge_db)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "diamond",
        "blastp",
        "--query",
        str(protein_fasta),
        "--db",
        str(mge_db),
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
    logger.info("Running DIAMOND (ISfinder/MGE): %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DIAMOND (MGE) failed: %s", result.stderr[:500])
        raise RuntimeError(f"DIAMOND (MGE) failed with exit code {result.returncode}")


def parse_mge_hits(
    tsv_path: Path | str,
    mge_meta: dict | None = None,  # type: ignore
) -> list[MGEHit]:
    """Parse DIAMOND output against ISfinder into MGEHit objects.

    Args:
        tsv_path: DIAMOND tabular output.
        mge_meta: Optional dict from load_mge_metadata() for enriched family/class.
    """
    tsv_path = Path(tsv_path)
    meta = mge_meta or {}
    hits: list[MGEHit] = []

    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return hits

    with open(tsv_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            qseqid, sseqid, pident, qcovhsp, evalue, stitle = row[:6]
            contig_id = "_".join(qseqid.rsplit("_", 1)[:-1]) if "_" in qseqid else qseqid

            # Extract the gene name from the subject sequence ID.
            # The Pärnänen database uses headers like "904_tnpA_CP000353.2";
            # the stitle column just repeats the sseqid without a real description.
            # We parse gene_name from sseqid for both metadata lookup and family
            # inference — this is more reliable than parsing the stitle.
            gene_name = _extract_gene_name_from_sseqid(sseqid)
            is_name = gene_name  # human-readable element name shown in reports
            description = stitle.strip()

            # Metadata lookup: try progressively simpler keys
            gene_key_exact = gene_name.lower()
            gene_key_no_suffix = re.sub(r"_\d+$", "", gene_name).lower()
            gene_key_no_digits = re.sub(r"\d+$", "", gene_name).lower()
            entry = (meta.get(gene_key_exact)
                     or meta.get(gene_key_no_suffix)
                     or meta.get(gene_key_no_digits)
                     or {})
            is_family = entry.get("sub_class") or _infer_is_family(gene_name, description)
            mge_class = entry.get("mge_class", "")

            hits.append(
                MGEHit(
                    contig_id=contig_id,
                    is_name=is_name,
                    is_family=is_family,
                    mge_class=mge_class,
                    description=description[:120],
                    identity=float(pident),
                    coverage=float(qcovhsp),
                    evalue=float(evalue),
                    _orf_id=qseqid,
                )
            )

    logger.info("Parsed %d MGE hits from %s", len(hits), tsv_path)
    return hits


def annotate_mge(
    fasta_path: Path | str,
    mge_db: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    min_identity: float = MGE_MIN_IDENTITY,
    min_coverage: float = MGE_MIN_COVERAGE,
    reuse_proteins: Path | str | None = None,
) -> list[MGEHit]:
    """End-to-end MGE annotation: ORF prediction → DIAMOND → parsed hits.

    Args:
        fasta_path: Nucleotide FASTA of plasmid contigs.
        mge_db: Path to DIAMOND .dmnd database built from ISfinder proteins.
        work_dir: Directory for intermediate files.
        threads: CPU threads for DIAMOND.
        min_identity: Minimum amino-acid identity % (default 70).
        min_coverage: Minimum query coverage % (default 80).
        reuse_proteins: Reuse pre-predicted ORF .faa (from ARG annotation step)
            to avoid running pyrodigal twice.

    Returns:
        List of MGEHit across all contigs.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    proteins_path = Path(reuse_proteins) if reuse_proteins else work_dir / "proteins.faa"
    mge_tsv = work_dir / "mge_hits.tsv"

    if reuse_proteins is None:
        call_orfs(fasta_path, proteins_path)
    else:
        logger.info("Reusing pre-predicted ORFs from %s", proteins_path)

    # Load MGE family metadata from mge_database.xlsx (auto-detect from DB dir)
    mge_db_path = Path(mge_db)
    meta_candidates = list(mge_db_path.parent.glob("mge_database.tsv")) + \
                      list(mge_db_path.parent.glob("*.tsv"))
    mge_meta = load_mge_metadata(meta_candidates[0]) if meta_candidates else {}

    if mge_tsv.exists() and mge_tsv.stat().st_size > 0:
        logger.info("Reusing cached MGE hits from %s", mge_tsv)
        return parse_mge_hits(mge_tsv, mge_meta)

    run_mge_diamond(
        proteins_path,
        mge_db,
        mge_tsv,
        threads=threads,
        min_identity=min_identity,
        min_coverage=min_coverage,
    )
    return parse_mge_hits(mge_tsv, mge_meta)
