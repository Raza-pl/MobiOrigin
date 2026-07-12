#!/usr/bin/env python3
"""Download complete RefSeq genomes with labelled plasmids for benchmarking.

Usage
-----
    python scripts/benchmark/download_genomes.py \\
        --out data/benchmark/genomes \\
        --per-taxon 60 \\
        --email your@email.com

Output
------
    data/benchmark/genomes/
        metadata.tsv          — one row per downloaded sequence (accession, taxon,
                                molecule_type [chromosome|plasmid], length)
        {accession}.fasta     — one FASTA per RefSeq assembly (chr + all plasmids)

The metadata.tsv is the ground-truth label file used by make_benchmark.py.

Taxa included
-------------
Tier 1 — clinical AMR priority pathogens (ESKAPE + E. coli):
    Enterococcus faecium, Staphylococcus aureus, Klebsiella pneumoniae,
    Acinetobacter baumannii, Pseudomonas aeruginosa, Enterobacter cloacae,
    Escherichia coli

Tier 2 — additional breadth:
    Salmonella enterica, Bacillus cereus, Vibrio cholerae,
    Clostridioides difficile, Streptococcus pyogenes
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Target taxa ───────────────────────────────────────────────────────────────

TIER1_TAXA = [
    "Enterococcus faecium",
    "Staphylococcus aureus",
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Pseudomonas aeruginosa",
    "Enterobacter cloacae",
    "Escherichia coli",
]

TIER2_TAXA = [
    "Salmonella enterica",
    "Bacillus cereus",
    "Vibrio cholerae",
    "Clostridioides difficile",
    "Streptococcus pyogenes",
]

# ── NCBI Entrez helpers ────────────────────────────────────────────────────────


def _require_biopython() -> None:
    try:
        import Bio  # noqa: F401
    except ImportError:
        sys.exit(
            "BioPython is required:\n"
            "    conda install -c conda-forge biopython\n"
            "or: pip install biopython"
        )


def _search_assemblies(taxon: str, n: int, email: str) -> list[str]:
    """Return up to n RefSeq assembly UIDs for complete genomes of *taxon*."""
    from Bio import Entrez

    Entrez.email = email
    query = (
        f'"{taxon}"[Organism] AND "complete genome"[Assembly Level] '
        f'AND "reference genome"[filter] OR '
        f'"{taxon}"[Organism] AND "complete genome"[Assembly Level] '
        f'AND "latest refseq"[filter]'
    )
    handle = Entrez.esearch(db="assembly", term=query, retmax=n * 3, usehistory="y")
    rec = Entrez.read(handle)
    handle.close()
    uids = rec.get("IdList", [])
    logger.info("  %s: found %d assembly UIDs", taxon, len(uids))
    return uids[:n]


def _get_assembly_accessions(uid: str, email: str) -> list[str]:
    """Return RefSeq nucleotide accessions (chr + plasmids) for an assembly UID."""
    from Bio import Entrez

    Entrez.email = email
    # Fetch assembly summary to get the RefSeq accession
    handle = Entrez.esummary(db="assembly", id=uid, report="full")
    summary = Entrez.read(handle, validate=False)
    handle.close()

    doc = summary["DocumentSummarySet"]["DocumentSummary"][0]
    refseq_cat = str(doc.get("RefSeq_category", ""))
    rs_acc = str(doc.get("AssemblyAccession", ""))

    if not rs_acc.startswith("GCF_"):
        return []
    # Avoid non-reference / anomalous assemblies
    if "suppressed" in str(doc.get("AssemblyStatus", "")).lower():
        return []

    # Use nuccore link to get all sequences in this assembly
    handle = Entrez.elink(
        dbfrom="assembly", db="nuccore", id=uid, linkname="assembly_nuccore_refseq"
    )
    links = Entrez.read(handle)
    handle.close()

    try:
        ids = [lnk["Id"] for lnk in links[0]["LinkSetDb"][0]["Link"]]
    except (IndexError, KeyError):
        ids = []

    logger.debug("    assembly %s (%s): %d sequences", rs_acc, refseq_cat, len(ids))
    return ids


def _fetch_sequences(nuccore_ids: list[str], email: str) -> list[dict]:
    """Fetch FASTA + metadata for a list of nuccore IDs."""
    from Bio import Entrez, SeqIO

    Entrez.email = email
    if not nuccore_ids:
        return []

    batch_size = 50
    results = []
    for i in range(0, len(nuccore_ids), batch_size):
        batch = nuccore_ids[i : i + batch_size]
        handle = Entrez.efetch(
            db="nuccore",
            id=",".join(batch),
            rettype="gb",
            retmode="text",
        )
        for rec in SeqIO.parse(handle, "genbank"):
            mol_type = "chromosome"
            desc_lower = rec.description.lower()
            if any(kw in desc_lower for kw in ("plasmid", "plas.", "mega-plasmid", "megaplasmid")):
                mol_type = "plasmid"
            # Also check the source feature
            for feat in rec.features:
                if feat.type == "source":
                    quals = feat.qualifiers
                    if "plasmid" in quals:
                        mol_type = "plasmid"
                    break
            results.append(
                {
                    "accession": rec.id,
                    "description": rec.description,
                    "length": len(rec.seq),
                    "molecule_type": mol_type,
                    "sequence": str(rec.seq),
                }
            )
        handle.close()
        time.sleep(0.4)  # NCBI rate limit: max 3 req/s without API key

    return results


# ── Main download logic ────────────────────────────────────────────────────────


def download(
    out_dir: Path,
    taxa: list[str],
    per_taxon: int,
    email: str,
    min_plasmid_count: int = 1,
) -> Path:
    """Download genomes for *taxa* and write metadata.tsv + per-assembly FASTAs.

    Returns path to metadata.tsv.
    """
    _require_biopython()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "metadata.tsv"
    meta_rows: list[dict] = []

    for taxon in taxa:
        logger.info("Processing taxon: %s", taxon)
        uids = _search_assemblies(taxon, per_taxon, email)
        downloaded = 0

        for uid in uids:
            if downloaded >= per_taxon:
                break
            try:
                nuc_ids = _get_assembly_accessions(uid, email)
                if not nuc_ids:
                    continue

                seqs = _fetch_sequences(nuc_ids, email)
                if not seqs:
                    continue

                # Require at least min_plasmid_count labelled plasmids
                n_plasmids = sum(1 for s in seqs if s["molecule_type"] == "plasmid")
                if n_plasmids < min_plasmid_count:
                    logger.debug("  skip UID %s — no labelled plasmids", uid)
                    continue

                # Write all sequences for this assembly into one FASTA
                assembly_fa = out_dir / f"{uid}.fasta"
                with open(assembly_fa, "w") as fh:
                    for s in seqs:
                        fh.write(f">{s['accession']} {s['description']}\n{s['sequence']}\n")

                for s in seqs:
                    meta_rows.append(
                        {
                            "uid": uid,
                            "accession": s["accession"],
                            "taxon": taxon,
                            "molecule_type": s["molecule_type"],
                            "length": s["length"],
                            "assembly_fasta": str(assembly_fa.name),
                        }
                    )

                logger.info(
                    "  ✓ UID %s: %d sequences (%d plasmids)",
                    uid,
                    len(seqs),
                    n_plasmids,
                )
                downloaded += 1
                time.sleep(0.5)

            except Exception as exc:
                logger.warning("  ✗ UID %s failed: %s", uid, exc)
                time.sleep(1.0)

    # Write metadata TSV
    with open(meta_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["uid", "accession", "taxon", "molecule_type", "length", "assembly_fasta"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    n_chr = sum(1 for r in meta_rows if r["molecule_type"] == "chromosome")
    n_plas = sum(1 for r in meta_rows if r["molecule_type"] == "plasmid")
    logger.info(
        "Done: %d assemblies, %d chromosomes, %d plasmids → %s",
        len({r["uid"] for r in meta_rows}),
        n_chr,
        n_plas,
        meta_path,
    )
    return meta_path


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True, type=Path, help="Output directory for genomes.")
    p.add_argument(
        "--per-taxon", type=int, default=60, help="Max assemblies per taxon (default 60)."
    )
    p.add_argument(
        "--email", required=True, help="Email address for NCBI Entrez (required by NCBI)."
    )
    p.add_argument(
        "--tier",
        choices=["1", "2", "all"],
        default="1",
        help="Which taxa to download (1=ESKAPE+Ecoli, 2=extra breadth, all=both; default 1).",
    )
    p.add_argument(
        "--min-plasmids",
        type=int,
        default=1,
        help="Skip assemblies with fewer than N labelled plasmids (default 1).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    taxa: list[str] = []
    if args.tier in ("1", "all"):
        taxa += TIER1_TAXA
    if args.tier in ("2", "all"):
        taxa += TIER2_TAXA

    download(
        out_dir=args.out,
        taxa=taxa,
        per_taxon=args.per_taxon,
        email=args.email,
        min_plasmid_count=args.min_plasmids,
    )


if __name__ == "__main__":
    main()
