#!/usr/bin/env python3
"""Fix empty d__ (domain) fields in the existing taxonomy_proteins.faa.

The original build_taxonomy_db.py used NCBI rank "superkingdom" to fill in
d__, but NCBI taxonomy dropped that rank for many lineages, leaving d__ empty.
This script patches the headers in the existing .faa (no re-running pyrodigal)
and rebuilds only the DIAMOND database — takes ~5 min instead of ~40 min.

Strategy:
  1. Load the assembly_summary taxid lookup (genome accession → taxid).
  2. Load NCBI taxonomy (nodes.dmp + names.dmp) and build a
     taxid → domain (Bacteria / Archaea / Eukaryota / Viruses) table
     by walking every taxid up to one of the known root domain taxids.
  3. Read taxonomy_proteins.faa line by line.  For each header, parse the
     contig accession from the protein ID (e.g. NZ_CP027667.1 from
     NZ_CP027667.1_186), look up the genome accession in assembly_summary,
     look up the domain, and inject it into the lineage string.
  4. Write taxonomy_proteins.faa (in-place, via a temp file) and rebuild
     DIAMOND database.

Usage:
    python scripts/fix_taxonomy_headers.py
    python scripts/fix_taxonomy_headers.py --out data/databases/taxonomy \
        --taxdump data/databases/taxonomy/taxdump \
        --threads 16
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# NCBI taxids for the top-level domains
_DOMAIN_TAXIDS: dict[int, str] = {
    2:     "Bacteria",
    2157:  "Archaea",
    2759:  "Eukaryota",
    10239: "Viruses",
    12884: "Viroids",
}

# Phylum-name → domain fallback (when taxid lookup fails)
# Covers the most common phyla found in wastewater / soil metagenomes
_PHYLUM_TO_DOMAIN: dict[str, str] = {
    # Bacteria
    "Pseudomonadota": "Bacteria", "Proteobacteria": "Bacteria",
    "Bacillota": "Bacteria",      "Firmicutes": "Bacteria",
    "Bacteroidota": "Bacteria",   "Bacteroidetes": "Bacteria",
    "Actinomycetota": "Bacteria", "Actinobacteria": "Bacteria",
    "Verrucomicrobiota": "Bacteria",
    "Planctomycetota": "Bacteria",
    "Chloroflexota": "Bacteria",
    "Spirochaetota": "Bacteria",
    "Campylobacterota": "Bacteria", "Epsilonproteobacteria": "Bacteria",
    "Myxococcota": "Bacteria",
    "Cyanobacteriota": "Bacteria",  "Cyanobacteria": "Bacteria",
    "Desulfobacterota": "Bacteria",
    "Patescibacteria": "Bacteria",
    "Nitrospirota": "Bacteria",
    "Acidobacteriota": "Bacteria",
    "Chlamydiota": "Bacteria",
    "Fusobacteriota": "Bacteria",
    "Deinococcota": "Bacteria",
    "Synergistota": "Bacteria",
    "Thermodesulfobacteriota": "Bacteria",
    # Archaea
    "Halobacteriota": "Archaea",   "Euryarchaeota": "Archaea",
    "Thermoproteota": "Archaea",   "Crenarchaeota": "Archaea",
    "Nanoarchaeota": "Archaea",    "Asgardarchaeota": "Archaea",
    "Thermoplasmatota": "Archaea",
    "Methanobacteriota": "Archaea",
}


def load_taxdump(taxdump_dir: Path) -> dict[int, str]:
    """Build taxid → domain string by walking NCBI taxonomy to root."""
    nodes = taxdump_dir / "nodes.dmp"
    names = taxdump_dir / "names.dmp"

    logger.info("Parsing names.dmp …")
    taxid_to_name: dict[int, str] = {}
    with open(names) as f:
        for line in f:
            p = [x.strip() for x in line.split("|")]
            if len(p) >= 4 and p[3] == "scientific name":
                taxid_to_name[int(p[0])] = p[1]

    logger.info("Parsing nodes.dmp …")
    taxid_to_parent: dict[int, int] = {}
    with open(nodes) as f:
        for line in f:
            p = [x.strip() for x in line.split("|")]
            if len(p) >= 2:
                taxid_to_parent[int(p[0])] = int(p[1])

    logger.info("Building taxid → domain table …")
    taxid_to_domain: dict[int, str] = {}
    for tid in list(taxid_to_parent.keys()) + list(_DOMAIN_TAXIDS.keys()):
        if tid in taxid_to_domain:
            continue
        # Walk to root, collect ancestry
        path: list[int] = []
        cur = tid
        visited: set[int] = set()
        while cur not in visited and cur != 1:
            visited.add(cur)
            path.append(cur)
            if cur in _DOMAIN_TAXIDS:
                domain = _DOMAIN_TAXIDS[cur]
                # Assign to all taxids on this path
                for t in path:
                    taxid_to_domain[t] = domain
                break
            parent = taxid_to_parent.get(cur, 1)
            if parent == cur:
                break
            cur = parent

    logger.info("Built domain table for %d taxids", len(taxid_to_domain))
    return taxid_to_domain


def load_assembly_taxids(summary_dir: Path) -> dict[str, int]:
    """Load genome_accession → taxid from bacteria + archaea assembly summaries."""
    acc_to_taxid: dict[str, int] = {}
    for fname in ("bacteria_assembly_summary.txt", "archaea_assembly_summary.txt"):
        p = summary_dir / fname
        if not p.exists():
            logger.warning("Summary file not found: %s", p)
            continue
        with open(p) as f:
            cols: list[str] = []
            for line in f:
                if line.startswith("#") and "assembly_accession" in line:
                    cols = line.lstrip("#").strip().split("\t")
                    continue
                if not cols or line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) <= max(0, 1):
                    continue
                try:
                    acc_idx  = cols.index("assembly_accession")
                    tid_idx  = cols.index("taxid")
                    acc  = parts[acc_idx].strip()
                    taxid = int(parts[tid_idx].strip())
                    # Store with and without version, and GCF/GCA prefix only
                    acc_to_taxid[acc] = taxid
                    acc_to_taxid[acc.split(".")[0]] = taxid
                    # Also strip assembly name suffix: GCF_000007905.1_ASM790v1 → GCF_000007905.1
                    short = "_".join(acc.split("_")[:2])
                    acc_to_taxid[short] = taxid
                    acc_to_taxid[short.split(".")[0]] = taxid
                except (ValueError, IndexError):
                    continue
    logger.info("Loaded %d accession→taxid mappings", len(acc_to_taxid))
    return acc_to_taxid


def _domain_from_phylum(lineage: str) -> str:
    """Infer domain from phylum name in existing lineage string."""
    m = re.search(r"p__([^;]+)", lineage)
    if not m:
        return ""
    phylum = m.group(1).strip()
    return _PHYLUM_TO_DOMAIN.get(phylum, "")


def fix_headers(
    faa_in: Path,
    faa_out: Path,
    taxid_to_domain: dict[int, str],
    acc_to_taxid: dict[str, int],
) -> dict[str, int]:
    """Rewrite protein FASTA headers with correct d__ domain field.

    Returns counts: {"fixed": N, "phylum_fallback": N, "unknown": N, "total": N}
    """
    # Regex to extract contig accession from protein ID like NZ_CP027667.1_186
    # Handles: NZ_XXXXX.1_N, NC_XXXXX.1_N, GCF_XXXXX.1_contigN_N
    _CONTIG_ACC_RE = re.compile(r"^([A-Z0-9]+_[A-Z0-9]+\.\d+)")

    counts = {"fixed": 0, "phylum_fallback": 0, "unknown": 0, "total": 0}

    def get_domain(prot_id: str, lineage: str) -> str:
        """Resolve domain for one protein."""
        # 1. Already has domain?
        m = re.match(r"d__([^;]+)", lineage)
        if m and m.group(1).strip():
            return m.group(1).strip()

        # 2. Try taxid lookup via contig accession
        cm = _CONTIG_ACC_RE.match(prot_id)
        if cm:
            contig_acc = cm.group(1)
            # Try with version, without version, and various prefixes
            for key in (contig_acc, contig_acc.split(".")[0]):
                tid = acc_to_taxid.get(key)
                if tid:
                    dom = taxid_to_domain.get(tid, "")
                    if dom:
                        counts["fixed"] += 1
                        return dom

        # 3. Infer from phylum in lineage
        dom = _domain_from_phylum(lineage)
        if dom:
            counts["phylum_fallback"] += 1
            return dom

        counts["unknown"] += 1
        return ""

    with open(faa_in) as fin, open(faa_out, "w") as fout:
        prot_id = ""
        lineage = ""
        for line in fin:
            if line.startswith(">"):
                counts["total"] += 1
                parts = line[1:].rstrip().split(None, 1)
                prot_id = parts[0]
                lineage = parts[1] if len(parts) > 1 else ""

                domain = get_domain(prot_id, lineage)

                # Inject/replace d__ field
                if lineage:
                    # Replace existing d__xxx (empty or not) with correct value
                    new_lineage = re.sub(
                        r"d__[^;]*",
                        f"d__{domain}",
                        lineage,
                        count=1,
                    )
                else:
                    new_lineage = f"d__{domain}"

                fout.write(f">{prot_id} {new_lineage}\n")
            else:
                fout.write(line)

        if counts["total"] % 100_000 == 0:
            logger.info("  Processed %d proteins …", counts["total"])

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix empty d__ domain in taxonomy_proteins.faa and rebuild DIAMOND DB."
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/databases/taxonomy"),
        help="Taxonomy database directory (default: data/databases/taxonomy)",
    )
    parser.add_argument(
        "--taxdump", type=Path, default=None,
        help="Directory containing nodes.dmp + names.dmp (default: <out>/taxdump)",
    )
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Threads for DIAMOND makedb (default: 8)",
    )
    args = parser.parse_args()

    out_dir     = args.out
    taxdump_dir = args.taxdump or out_dir / "taxdump"
    faa_in      = out_dir / "taxonomy_proteins.faa"
    faa_fixed   = out_dir / "taxonomy_proteins.faa"   # overwrite in-place via temp
    dmnd_out    = out_dir / "refseq_taxonomy.dmnd"

    for p, name in [(faa_in, "taxonomy_proteins.faa"),
                    (taxdump_dir / "nodes.dmp", "nodes.dmp"),
                    (taxdump_dir / "names.dmp", "names.dmp")]:
        if not p.exists():
            logger.error("Required file not found: %s", p)
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Step 1: Load NCBI taxonomy")
    logger.info("=" * 60)
    taxid_to_domain = load_taxdump(taxdump_dir)

    logger.info("=" * 60)
    logger.info("Step 2: Load assembly summaries")
    logger.info("=" * 60)
    acc_to_taxid = load_assembly_taxids(taxdump_dir)

    logger.info("=" * 60)
    logger.info("Step 3: Fix protein headers (%s)", faa_in)
    logger.info("=" * 60)

    # Write to a temp file, then replace
    with tempfile.NamedTemporaryFile(
        mode="w", dir=out_dir, suffix=".faa.tmp", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    counts = fix_headers(faa_in, tmp_path, taxid_to_domain, acc_to_taxid)
    tmp_path.replace(faa_in)   # atomic replace

    total = counts["total"]
    logger.info(
        "Headers fixed: %d taxid-lookup | %d phylum-fallback | %d unknown | %d total",
        counts["fixed"], counts["phylum_fallback"], counts["unknown"], total,
    )
    pct_good = 100 * (total - counts["unknown"]) / max(total, 1)
    logger.info("Domain assigned: %.1f%% of proteins", pct_good)

    logger.info("=" * 60)
    logger.info("Step 4: Rebuild DIAMOND database")
    logger.info("=" * 60)

    if dmnd_out.exists():
        dmnd_out.unlink()

    cmd = [
        "diamond", "makedb",
        "--in", str(faa_in),
        "--db", str(out_dir / "refseq_taxonomy"),
        "--threads", str(args.threads),
        "--quiet",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DIAMOND failed:\n%s", result.stderr)
        sys.exit(1)

    size_mb = dmnd_out.stat().st_size / 1e6
    logger.info("DIAMOND database rebuilt: %s  (%.0f MB)", dmnd_out, size_mb)
    logger.info("=" * 60)
    logger.info("Done — taxonomy DB ready with correct domain labels.")
    logger.info("Run:  plasflow2 run --input assembly.fasta --output results/ --threads %d", args.threads)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
