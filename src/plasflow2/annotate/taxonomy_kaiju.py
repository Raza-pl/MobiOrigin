"""Kaiju-based contig taxonomy — fast protein k-mer classification.

Kaiju is a k-mer based taxonomy classifier that works in translated protein
space.  It is 20–50× faster than DIAMOND blastp at comparable sensitivity for
assembled contigs because it uses an exact BWT/FM-index rather than heuristic
seed-and-extend alignment.

Pipeline
--------
proteins.faa  →  kaiju (FM-index search)  →  raw_out.tsv
              →  kaiju-addTaxonNames       →  named_out.tsv  (adds lineage)
              →  parse_kaiju_named()       →  dict[contig_id → TaxResult]

The output TaxResult objects are identical to those from the DIAMOND path, so
the rest of the pipeline (risk scoring, HTML report, predictions.tsv) is
completely unaware of which engine was used.

Database setup (one-time, ~10 min, ~25 GB download)
----------------------------------------------------
Run:  bash scripts/setup_kaiju_db.sh
or manually:

    mkdir -p data/databases/kaiju
    cd data/databases/kaiju

    # Download NCBI taxonomy (needed to resolve taxid → lineage)
    wget https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
    tar xf taxdump.tar.gz nodes.dmp names.dmp

    # Build kaiju FM-index from RefSeq proteins (nr_euk + nr_prok ~25 GB download)
    kaiju-makedb -s refseq -t 16 -o .

    # This writes: kaiju_db_refseq.fmi  (the FM-index used at runtime)

Alternatively, build from the taxonomy_proteins.faa already in this repo:

    kaiju-mkbwt -n 16 -a ACDEFGHIKLMNPQRSTVWY \
        -o data/databases/kaiju/kaiju_custom \
        data/databases/taxonomy/taxonomy_proteins.faa
    kaiju-mkfmi data/databases/kaiju/kaiju_custom

    # Then download NCBI taxonomy separately (for taxid resolution)
    # If your FASTA headers already contain GTDB-style lineage strings,
    # use the --use-gtdb-lineage flag in assign_taxonomy_kaiju().

Key files expected
------------------
    data/databases/kaiju/
        kaiju_db.fmi        ← FM-index (any name; auto-detected)
        nodes.dmp           ← NCBI taxonomy nodes
        names.dmp           ← NCBI taxonomy names

Reference
---------
Menzel P, Ng KL, Krogh A. Fast and sensitive taxonomic classification for
metagenomics with Kaiju. Nature Communications 7, 11257 (2016).
https://doi.org/10.1038/ncomms11257
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections import Counter
from pathlib import Path

from plasflow2.annotate.taxonomy import TaxResult

logger = logging.getLogger(__name__)

# GTDB rank prefixes for lineage parsing (shared with taxonomy.py)
_GTDB_RANK_PREFIXES = ["d__", "p__", "c__", "o__", "f__", "g__", "s__"]
_GTDB_RANK_NAMES = ["domain", "phylum", "class", "order", "family", "genus", "species"]
_PREFIX_TO_RANK = dict(zip(_GTDB_RANK_PREFIXES, _GTDB_RANK_NAMES))

# NCBI rank names mapped to our GTDB-style rank names (for ncbi lineage mode)
_NCBI_RANK_MAP = {
    "superkingdom": "domain",
    "phylum": "phylum",
    "class": "class",
    "order": "order",
    "family": "family",
    "genus": "genus",
    "species": "species",
}

# Minimum fraction of ORFs in a contig that must agree on a taxon at a rank
_MIN_AGREEMENT = 0.5


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def kaiju_available() -> bool:
    """Return True if kaiju and kaiju-addTaxonNames are both on PATH."""
    for tool in ("kaiju", "kaiju-addTaxonNames"):
        try:
            subprocess.run([tool, "--help"], capture_output=True)
        except FileNotFoundError:
            return False
    return True


# ---------------------------------------------------------------------------
# Database auto-detection
# ---------------------------------------------------------------------------


def find_kaiju_db(kaiju_dir: Path) -> Path | None:
    """Find the first .fmi file in *kaiju_dir*.

    kaiju-makedb writes files like kaiju_db_refseq.fmi or kaiju_db_nr.fmi.
    We also accept a manually-named kaiju_db.fmi.
    """
    for p in sorted(kaiju_dir.glob("*.fmi")):
        return p
    return None


# ---------------------------------------------------------------------------
# Run kaiju
# ---------------------------------------------------------------------------


def run_kaiju(
    protein_fasta: Path | str,
    kaiju_db: Path | str,
    nodes_dmp: Path | str,
    out_tsv: Path | str,
    threads: int = 8,
    greedy_mode: bool = True,
    greedy_mismatches: int = 5,
) -> Path:
    """Run kaiju in protein mode (-p) on pre-predicted ORFs.

    Kaiju -p (protein mode) accepts amino-acid sequences directly —
    perfect for reusing proteins.faa from the ARG annotation step.
    No 6-frame translation is needed, which is part of why it is faster
    than DIAMOND blastx.

    Args:
        protein_fasta:      Amino-acid FASTA (from pyrodigal call_orfs).
        kaiju_db:           Kaiju FM-index (.fmi file).
        nodes_dmp:          NCBI taxonomy nodes.dmp.
        out_tsv:            Output path for raw kaiju TSV.
        threads:            CPU threads (-z flag).
        greedy_mode:        Use greedy/MEM mode (better sensitivity, still fast).
        greedy_mismatches:  Allowed mismatches in greedy mode (-e flag).

    Returns:
        Path to the raw kaiju output TSV.
    """
    protein_fasta = Path(protein_fasta)
    kaiju_db = Path(kaiju_db)
    nodes_dmp = Path(nodes_dmp)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaiju",
        "-t",
        str(nodes_dmp),
        "-f",
        str(kaiju_db),
        "-i",
        str(protein_fasta),
        "-o",
        str(out_tsv),
        "-z",
        str(threads),
        "-p",  # protein mode — input is already amino acids
        "-v",  # verbose (writes progress to stderr)
    ]
    if greedy_mode:
        cmd += ["-a", "greedy", "-e", str(greedy_mismatches)]

    logger.info("Running kaiju (protein mode, threads=%d): %s", threads, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("kaiju failed (exit %d): %s", result.returncode, result.stderr[:600])
        raise RuntimeError(f"kaiju failed with exit code {result.returncode}")

    logger.info("kaiju finished — output: %s", out_tsv)
    return out_tsv


# ---------------------------------------------------------------------------
# Add taxonomy names (taxid → full lineage string)
# ---------------------------------------------------------------------------


def add_taxon_names(
    kaiju_tsv: Path | str,
    nodes_dmp: Path | str,
    names_dmp: Path | str,
    named_tsv: Path | str,
    ranks: str = "superkingdom,phylum,class,order,family,genus,species",
) -> Path:
    """Run kaiju-addTaxonNames to append lineage columns to kaiju output.

    Produces a TSV with additional columns for each requested rank.

    Args:
        kaiju_tsv:   Raw kaiju output (from run_kaiju).
        nodes_dmp:   NCBI taxonomy nodes.dmp.
        names_dmp:   NCBI taxonomy names.dmp.
        named_tsv:   Output path for the annotated TSV.
        ranks:       Comma-separated rank list to expand.

    Returns:
        Path to the named TSV.
    """
    kaiju_tsv = Path(kaiju_tsv)
    nodes_dmp = Path(nodes_dmp)
    names_dmp = Path(names_dmp)
    named_tsv = Path(named_tsv)

    cmd = [
        "kaiju-addTaxonNames",
        "-t",
        str(nodes_dmp),
        "-n",
        str(names_dmp),
        "-i",
        str(kaiju_tsv),
        "-o",
        str(named_tsv),
        "-r",
        ranks,
        "-p",  # protein mode flag (pass-through)
    ]
    logger.info("Running kaiju-addTaxonNames: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("kaiju-addTaxonNames failed: %s", result.stderr[:400])
        raise RuntimeError("kaiju-addTaxonNames failed")
    return named_tsv


# ---------------------------------------------------------------------------
# Parse kaiju-addTaxonNames output
# ---------------------------------------------------------------------------


def _ncbi_lineage_to_gtdb_style(rank_values: dict[str, str]) -> tuple[str, str, str]:
    """Convert NCBI rank→name dict to GTDB-style lineage, rank, taxon strings.

    Args:
        rank_values: e.g. {"superkingdom": "Bacteria", "phylum": "Proteobacteria", ...}

    Returns:
        (lineage_str, deepest_rank, deepest_taxon)
        lineage_str: "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;..."
    """
    prefix_map = {
        "superkingdom": "d__",
        "phylum": "p__",
        "class": "c__",
        "order": "o__",
        "family": "f__",
        "genus": "g__",
        "species": "s__",
    }
    rank_order = ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]

    parts: list[str] = []
    deepest_rank = "unclassified"
    deepest_taxon = ""

    for rank in rank_order:
        name = rank_values.get(rank, "").strip()
        if name and name.lower() not in ("", "na", "n/a", "unclassified", "root"):
            prefix = prefix_map[rank]
            parts.append(f"{prefix}{name}")
            deepest_rank = _NCBI_RANK_MAP.get(rank, rank)
            deepest_taxon = f"{prefix}{name}"

    return ";".join(parts), deepest_rank, deepest_taxon


def parse_kaiju_named(named_tsv: Path | str) -> dict[str, list[tuple[str, str, str]]]:
    """Parse kaiju-addTaxonNames output into per-ORF taxonomy hits.

    kaiju-addTaxonNames output columns (tab-separated):
        0  status         C (classified) or U (unclassified)
        1  orf_id         e.g. "contig_42_7" (pyrodigal orf_id)
        2  taxid          NCBI taxonomy ID
        3  score          length of best match (or sum in greedy mode)
        4  accessions     matching DB accessions (semicolon-separated)
        5  fragment       matched amino-acid fragment
        6+ rank columns   one per rank requested in kaiju-addTaxonNames -r

    Returns:
        Dict mapping orf_id → list of (lineage_str, rank, taxon) tuples.
        Only classified ORFs are included.
    """
    named_tsv = Path(named_tsv)
    hits: dict[str, list[tuple[str, str, str]]] = {}

    rank_order = ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]

    with open(named_tsv) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            status = cols[0].strip()
            if status != "C":
                continue

            orf_id = cols[1].strip()

            # Rank name columns start at index 6; their order matches -r argument
            rank_values: dict[str, str] = {}
            for i, rank in enumerate(rank_order):
                col_idx = 6 + i
                if col_idx < len(cols):
                    rank_values[rank] = cols[col_idx].strip()

            lineage_str, rank, taxon = _ncbi_lineage_to_gtdb_style(rank_values)
            if taxon:
                hits.setdefault(orf_id, []).append((lineage_str, rank, taxon))

    logger.info("Parsed kaiju hits for %d classified ORFs from %s", len(hits), named_tsv)
    return hits


# ---------------------------------------------------------------------------
# LCA aggregation: ORF-level hits → contig-level TaxResult
# ---------------------------------------------------------------------------


def _lca_from_orf_hits(
    orf_hits: list[tuple[str, str, str]],
    min_agreement: float = _MIN_AGREEMENT,
) -> tuple[str, str, str]:
    """Compute LCA from a list of (lineage, rank, taxon) tuples.

    Walks from species up to domain, returning the deepest rank where
    ≥ min_agreement fraction of ORFs agree.

    Returns:
        (lineage_str, rank, taxon) at the LCA level.
    """
    rank_order = ["species", "genus", "family", "order", "class", "phylum", "domain"]
    prefix_map = {
        "domain": "d__",
        "phylum": "p__",
        "class": "c__",
        "order": "o__",
        "family": "f__",
        "genus": "g__",
        "species": "s__",
    }

    n = len(orf_hits)
    if n == 0:
        return "", "unclassified", ""

    # Extract taxon name at each rank for every hit
    def _extract_at_rank(lineage: str, targetprefix: str) -> str:
        for part in lineage.split(";"):
            if part.startswith(targetprefix):
                return part
        return ""

    for rank in rank_order:
        prefix = prefix_map[rank]
        taxa_at_rank = [_extract_at_rank(lin, prefix) for lin, _, _ in orf_hits]
        taxa_at_rank = [t for t in taxa_at_rank if t]
        if not taxa_at_rank:
            continue
        counter: Counter[str] = Counter(taxa_at_rank)
        best_taxon, best_count = counter.most_common(1)[0]
        if best_count / n >= min_agreement:
            # Rebuild lineage up to this rank
            sample_lineage = next(
                lin for lin, _, _ in orf_hits if _extract_at_rank(lin, prefix) == best_taxon
            )
            # Truncate lineage at this rank
            parts = sample_lineage.split(";")
            truncated = []
            for part in parts:
                truncated.append(part)
                if part.startswith(prefix):
                    break
            return ";".join(truncated), rank, best_taxon

    return "", "unclassified", ""


def aggregate_kaiju_by_contig(
    orf_hits: dict[str, list[tuple[str, str, str]]],
    min_agreement: float = _MIN_AGREEMENT,
) -> dict[str, TaxResult]:
    """Aggregate per-ORF kaiju hits to per-contig TaxResult using LCA.

    The orf_id format is ``<contig_id>_<n>`` (pyrodigal convention).
    We strip the trailing ``_<n>`` to recover the contig_id.

    Args:
        orf_hits:      Output of parse_kaiju_named() — orf_id → hit list.
        min_agreement: Minimum fraction of ORFs agreeing at a rank for LCA.

    Returns:
        Dict mapping contig_id → TaxResult.
    """
    # Group by contig
    contig_hits: dict[str, list[tuple[str, str, str]]] = {}
    for orf_id, hits in orf_hits.items():
        contig_id = re.sub(r"_\d+$", "", orf_id)
        contig_hits.setdefault(contig_id, []).extend(hits)

    results: dict[str, TaxResult] = {}
    for contig_id, hits in contig_hits.items():
        lineage, rank, taxon = _lca_from_orf_hits(hits, min_agreement)
        if rank == "unclassified":
            continue
        # Compute agreement fraction at the resolved rank
        n = len(hits)
        prefix = taxon[:3] if taxon else ""  # noqa: F841
        agree = sum(1 for lin, _, _ in hits if taxon in lin) / max(n, 1)
        results[contig_id] = TaxResult(
            contig_id=contig_id,
            lineage=lineage,
            rank=rank,
            taxon=taxon,
            num_hits=n,
            agreement=round(agree, 3),
        )

    logger.info(
        "Kaiju contig taxonomy: %d / %d contigs classified",
        len(results),
        len(contig_hits),
    )
    return results


# ---------------------------------------------------------------------------
# Main entry point — mirrors assign_taxonomy() interface
# ---------------------------------------------------------------------------


def assign_taxonomy_kaiju(
    protein_fasta: Path | str,
    kaiju_db: Path | str,
    nodes_dmp: Path | str,
    names_dmp: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    min_agreement: float = _MIN_AGREEMENT,
    greedy_mode: bool = True,
    greedy_mismatches: int = 5,
) -> dict[str, TaxResult]:
    """Full Kaiju taxonomy pipeline: run → add names → parse → aggregate by contig.

    This function is the Kaiju equivalent of ``assign_taxonomy()`` in taxonomy.py.
    It accepts the same pre-predicted proteins.faa from the ARG step so no extra
    ORF prediction is needed.

    Args:
        protein_fasta:      Amino-acid FASTA (pyrodigal proteins.faa).
        kaiju_db:           Kaiju FM-index (.fmi file).
        nodes_dmp:          NCBI taxonomy nodes.dmp.
        names_dmp:          NCBI taxonomy names.dmp.
        work_dir:           Directory for intermediate files.
        threads:            CPU threads for kaiju.
        min_agreement:      Min fraction of ORFs agreeing at a rank for LCA.
        greedy_mode:        Kaiju greedy mode (better sensitivity, still fast).
        greedy_mismatches:  Mismatches allowed in greedy mode.

    Returns:
        Dict mapping contig_id → TaxResult, identical structure to DIAMOND path.
    """
    protein_fasta = Path(protein_fasta)
    kaiju_db = Path(kaiju_db)
    nodes_dmp = Path(nodes_dmp)
    names_dmp = Path(names_dmp)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_tsv = work_dir / "kaiju_raw.tsv"
    named_tsv = work_dir / "kaiju_named.tsv"

    # 1. Run kaiju
    run_kaiju(
        protein_fasta=protein_fasta,
        kaiju_db=kaiju_db,
        nodes_dmp=nodes_dmp,
        out_tsv=raw_tsv,
        threads=threads,
        greedy_mode=greedy_mode,
        greedy_mismatches=greedy_mismatches,
    )

    # 2. Add taxon names (taxid → lineage columns)
    add_taxon_names(
        kaiju_tsv=raw_tsv,
        nodes_dmp=nodes_dmp,
        names_dmp=names_dmp,
        named_tsv=named_tsv,
    )

    # 3. Parse and aggregate
    orf_hits = parse_kaiju_named(named_tsv)
    return aggregate_kaiju_by_contig(orf_hits, min_agreement=min_agreement)
