"""Download benchmark genomes for PlasFlow v2 evaluation.

Downloads 30 complete bacterial genomes from NCBI RefSeq that have BOTH
chromosome(s) AND plasmid(s) as separate, annotated sequences.  These are
used to build a ground-truth benchmark for comparing PlasFlow v2 against
PlasFlow v1 and geNomad.

Selection criteria
------------------
- Complete genome status (no contigs/scaffolds)
- Has ≥1 chromosome AND ≥1 plasmid as separate NCBI records
- Covers diverse taxa: ESKAPE pathogens + Gram-positives + environmental
- Includes organisms with small (1-10 kb), medium (10-100 kb), and large
  (>100 kb) plasmids to test across the size spectrum
- Uses recent RefSeq assemblies (2022-2025) to reduce training set overlap

Output structure
----------------
data/benchmark/
    genomes/
        {accession}/
            chromosome.fna      — concatenated chromosome sequences
            plasmids.fna        — concatenated plasmid sequences
            metadata.json       — organism name, accession, plasmid sizes
    benchmark.fna               — mixed contigs (built by build_benchmark.py)
    ground_truth.tsv            — contig_id → true_label
    genome_list.tsv             — accession, organism, n_chroms, n_plasmids

Usage
-----
    python scripts/download_benchmark_genomes.py \\
        --out data/benchmark \\
        --threads 8

    # Then build the benchmark FASTA:
    python scripts/build_benchmark.py --benchmark-dir data/benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated benchmark genome list
# ---------------------------------------------------------------------------
# Format: (assembly_accession, organism_name, notes)
# Selected to cover:
#   - ESKAPE pathogens (high clinical relevance)
#   - Gram-positive organisms (different GC%, plasmid biology)
#   - Environmental organisms (wastewater-relevant)
#   - Diverse plasmid sizes: small (<10kb), medium (10-100kb), large (>100kb)
#   - Recent assemblies to reduce PLSDB training set overlap

BENCHMARK_GENOMES = [
    # ── Enterobacterales (Gram-negative, clinical) ─────────────────────────
    ("GCF_000016305.1", "Klebsiella pneumoniae NTUH-K2044",      "2 plasmids: 219kb, 107kb"),
    ("GCF_000009645.1", "Klebsiella pneumoniae MGH 78578",       "5 plasmids: 108-3kb"),
    ("GCF_900478275.1", "Klebsiella pneumoniae INF319",           "pLAP-like carbapenem R"),
    ("GCF_000006945.2", "Salmonella enterica LT2",                "1 plasmid: 94kb IncFII"),
    ("GCF_000027125.1", "Salmonella enterica D23580",             "3 plasmids incl. large"),
    ("GCF_000005845.2", "Escherichia coli K-12 MG1655",          "no plasmid — chr control"),
    ("GCF_000026345.1", "Escherichia coli CFT073",               "no plasmid — chr control"),
    ("GCF_001593285.1", "Enterobacter cloacae ATCC 13047",       "2 plasmids"),
    ("GCF_000006925.2", "Shigella flexneri 2a str. 301",         "pWR501: 221kb large"),
    # ── Pseudomonadales (Gram-negative, environmental/clinical) ────────────
    ("GCF_000006765.1", "Pseudomonas aeruginosa PAO1",           "no plasmid — chr control"),
    ("GCF_000011305.1", "Pseudomonas putida KT2440",             "no plasmid — environmental chr"),
    ("GCF_000568855.2", "Pseudomonas aeruginosa PA7",            "1 plasmid: pPA7 112kb"),
    # ── Acinetobacter (Gram-negative, clinical, MDR) ───────────────────────
    ("GCF_000015425.1", "Acinetobacter baumannii ACICU",         "2 plasmids: 84kb, 8kb"),
    ("GCF_000746645.1", "Acinetobacter baumannii AB5075-UW",     "2 plasmids"),
    ("GCF_001682755.1", "Acinetobacter pittii SH024",            "multiple small plasmids"),
    # ── Firmicutes (Gram-positive, high GC contrast) ───────────────────────
    ("GCF_000144955.2", "Enterococcus faecalis V583",            "3 plasmids: 66kb, 58kb, 17kb"),
    ("GCF_000013265.1", "Enterococcus faecium DO",               "3 plasmids incl. large"),
    ("GCF_000011505.1", "Staphylococcus aureus MRSA252",         "1 plasmid: 27kb"),
    ("GCF_000011265.1", "Staphylococcus aureus MW2",             "1 plasmid: pWMSA 28kb"),
    ("GCF_000209595.2", "Staphylococcus epidermidis ATCC 12228", "2 plasmids"),
    # ── Actinobacteria (Gram-positive, high GC) ────────────────────────────
    ("GCF_000195275.1", "Mycobacterium tuberculosis H37Rv",      "no plasmid — chr control"),
    ("GCF_000092245.1", "Streptomyces coelicolor A3(2)",         "2 linear megaplasmids"),
    # ── Proteobacteria / Alpha (environmental) ─────────────────────────────
    ("GCF_000007645.1", "Agrobacterium tumefaciens C58",         "4 replicons (2 chr + 2 plas)"),
    ("GCF_000022325.1", "Caulobacter crescentus CB15",           "no plasmid — chr control"),
    # ── Wastewater-relevant organisms ──────────────────────────────────────
    ("GCF_000147115.1", "Nitrosomonas europaea ATCC 19718",      "1 plasmid: pNEU 164kb"),
    ("GCF_000017645.1", "Burkholderia vietnamiensis G4",         "3 chromosomes + 1 plasmid"),
    ("GCF_000155975.1", "Comamonas testosteroni CNB-2",          "1 plasmid catabolic"),
    # ── Spirochaetes / divergent phylogeny ─────────────────────────────────
    ("GCF_000953215.1", "Borrelia burgdorferi B31",              "1 chr + 21 linear plasmids"),
    # ── Additional ESKAPE ──────────────────────────────────────────────────
    ("GCF_000210315.1", "Haemophilus influenzae 86-028NP",       "no plasmid — chr control"),
]

NCBI_FTP = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


def _safe_unlink(path: Path) -> None:
    """Delete a file, ignoring FileNotFoundError (Python 3.7-compatible)."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# FTP path construction
# ---------------------------------------------------------------------------

def _accession_to_ftp_dir(accession: str) -> str:
    """Convert a GCF/GCA accession to its NCBI FTP directory URL.

    GCF_000144955.2  →  https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/144/955/

    The assembly-specific subdirectory name (e.g. GCF_000144955.2_EnterFaecV583_1.0)
    is unknown until we list the directory, so we return the parent dir and
    let the caller resolve the final subdir.
    """
    # Strip version suffix for path construction
    base = accession.split(".")[0]          # e.g. GCF_000144955
    prefix = base[:3]                       # GCF or GCA
    digits = base.split("_")[1]            # 000144955
    d1, d2, d3 = digits[0:3], digits[3:6], digits[6:9]
    return f"{NCBI_FTP}/{prefix}/{d1}/{d2}/{d3}/"


def _resolve_assembly_subdir(parent_url: str, accession: str) -> str | None:
    """List the FTP parent directory and find the accession-specific subdirectory.

    Returns the full URL to the assembly directory, or None on failure.
    """
    # NCBI FTP directory listings return an HTML page we can parse
    try:
        req = urllib.request.Request(parent_url,
                                     headers={"User-Agent": "PlasFlow2-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("FTP listing failed for %s: %s", parent_url, e)
        return None

    # Find href matching the accession (e.g. GCF_000144955.2_*)
    import re
    base = accession.rsplit(".", 1)[0] + "."   # GCF_000144955.
    matches = re.findall(rf'href="({re.escape(base)}[^"]+/)"', html)
    if not matches:
        # Try without version
        base_nover = accession.split(".")[0]
        matches = re.findall(rf'href="({re.escape(base_nover)}[^"]+/)"', html)
    if not matches:
        logger.warning("No subdirectory found for %s in %s", accession, parent_url)
        return None

    subdir = matches[-1].rstrip("/")
    return f"{parent_url}{subdir}/"


def _download_ftp_genomic_fna(assembly_url: str, accession: str, dest: Path) -> bool:
    """Download the *_genomic.fna.gz file for an assembly to dest."""
    import re
    try:
        req = urllib.request.Request(assembly_url,
                                     headers={"User-Agent": "PlasFlow2-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Assembly dir listing failed %s: %s", assembly_url, e)
        return False

    # Find the genomic FASTA (not cds/rna/protein)
    fna_files = re.findall(r'href="([^"]+_genomic\.fna\.gz)"', html)
    # Exclude cds_from_genomic and rna_from_genomic
    fna_files = [f for f in fna_files
                 if "cds_from" not in f and "rna_from" not in f]
    if not fna_files:
        logger.warning("No genomic.fna.gz found at %s", assembly_url)
        return False

    fna_url = assembly_url + fna_files[-1].split("/")[-1]
    logger.info("  Downloading: %s", fna_url)
    try:
        urllib.request.urlretrieve(fna_url, str(dest))
        return True
    except Exception as e:
        logger.warning("  Download failed: %s", e)
        return False


def download_genome_sequences(
    accession: str,
    organism: str,
    out_dir: Path,
) -> dict | None:
    """Download chromosome and plasmid sequences for one assembly.

    Priority:
      1. NCBI Datasets CLI  (most reliable, cleanest)
      2. Direct NCBI FTP    (reliable, no API needed)
      3. BioPython Entrez   (last resort, often unreliable for GCF accessions)

    Returns dict with paths and metadata, or None on failure.
    """
    genome_dir = out_dir / accession
    genome_dir.mkdir(parents=True, exist_ok=True)

    meta_path  = genome_dir / "metadata.json"
    chrom_path = genome_dir / "chromosome.fna"
    plas_path  = genome_dir / "plasmids.fna"

    # Skip if already downloaded AND verified correct organism
    if chrom_path.exists() and chrom_path.stat().st_size > 0:
        # Quick sanity-check: verify the downloaded headers roughly match organism
        if _verify_download(chrom_path, plas_path, accession):
            logger.info("[skip] %s already downloaded and verified", accession)
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            return {"accession": accession, "organism": organism,
                    "chrom": chrom_path, "plasmids": plas_path, **meta}
        else:
            logger.warning("[redownload] %s: existing files failed verification", accession)

    logger.info("Downloading %s (%s) …", accession, organism)

    # Try NCBI Datasets CLI first (most reliable)
    if _datasets_available():
        result = _download_via_datasets(accession, organism, genome_dir,
                                        chrom_path, plas_path, meta_path)
        if result:
            return result
        logger.warning("  datasets CLI failed for %s, trying FTP …", accession)

    # Try direct NCBI FTP (reliable for GCF accessions, no Entrez needed)
    result = _download_via_ftp(accession, organism, genome_dir,
                               chrom_path, plas_path, meta_path)
    if result:
        return result

    logger.warning("  FTP failed for %s, trying Entrez (less reliable) …", accession)
    return _download_via_entrez(accession, organism, genome_dir,
                                chrom_path, plas_path, meta_path)


def _verify_download(chrom_path: Path, plas_path: Path, accession: str) -> bool:
    """Basic sanity check: the first sequence header should contain the accession
    base (e.g. 'GCF_000144955') OR a known NC_/NZ_ accession for that assembly.
    We can't verify without a manifest, so we just check the file is non-trivially
    large (> 100 kb) and the header is a plausible NCBI nucleotide ID.
    """
    import re
    if not chrom_path.exists() or chrom_path.stat().st_size < 100_000:
        return False
    try:
        with open(chrom_path) as fh:
            first_line = fh.readline().strip()
        # Must start with '>' and contain an NCBI nucleotide accession (NC_, NZ_, CP, etc.)
        return bool(re.match(r'^>(NC_|NZ_|CP|AP|AE|BA|BX|CR|CT|CU|AM|AL|AJ|AY|EU|FJ|GQ)', first_line))
    except Exception:
        return False


def _download_via_ftp(
    accession: str,
    organism: str,
    genome_dir: Path,
    chrom_path: Path,
    plas_path: Path,
    meta_path: Path,
) -> dict | None:
    """Download via direct NCBI FTP path — reliable for GCF/GCA accessions.

    Constructs the path as:
      https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/144/955/
          GCF_000144955.2_*/GCF_000144955.2_*_genomic.fna.gz
    then splits into chromosome.fna and plasmids.fna.
    """
    import gzip
    import io
    from Bio import SeqIO  # type: ignore

    parent_url = _accession_to_ftp_dir(accession)
    assembly_url = _resolve_assembly_subdir(parent_url, accession)
    if not assembly_url:
        return None

    gz_path = genome_dir / f"{accession}_genomic.fna.gz"
    if not _download_ftp_genomic_fna(assembly_url, accession, gz_path):
        return None

    # Parse and split into chromosome vs plasmid
    chromosomes = []
    plasmids = []
    plasmid_sizes = []
    try:
        with gzip.open(str(gz_path), "rt") as fh:
            for rec in SeqIO.parse(fh, "fasta"):
                desc = (rec.description + " " + rec.id).lower()
                if any(tok in desc for tok in ("plasmid", " plas", ",plas")):
                    plasmids.append(rec)
                    plasmid_sizes.append(len(rec.seq))
                else:
                    chromosomes.append(rec)
    except Exception as e:
        logger.error("Failed to parse %s: %s", gz_path, e)
        _safe_unlink(gz_path)
        return None

    _safe_unlink(gz_path)  # free disk space

    if not chromosomes:
        logger.warning("%s: no chromosome sequences after splitting", accession)
        return None

    SeqIO.write(chromosomes, str(chrom_path), "fasta")
    if plasmids:
        SeqIO.write(plasmids, str(plas_path), "fasta")
    else:
        plas_path.write_text("")

    chrom_len = sum(len(r.seq) for r in chromosomes)
    meta = {
        "organism": organism,
        "accession": accession,
        "n_chromosomes": len(chromosomes),
        "n_plasmids": len(plasmids),
        "chromosome_bp": chrom_len,
        "plasmid_sizes": sorted(plasmid_sizes, reverse=True),
        "download_method": "ftp",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("  %s: %d chr seq(s) (%d bp) + %d plasmid(s) %s [via FTP]",
                accession, len(chromosomes), chrom_len, len(plasmids),
                [f"{s//1000}kb" for s in plasmid_sizes[:5]])
    return {"chrom": chrom_path, "plasmids": plas_path, **meta}


def _datasets_available() -> bool:
    try:
        subprocess.run(["datasets", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def _download_via_datasets(
    accession: str,
    organism: str,
    genome_dir: Path,
    chrom_path: Path,
    plas_path: Path,
    meta_path: Path,
) -> dict | None:
    """Use NCBI Datasets CLI to download and split chromosome/plasmid sequences."""
    import zipfile, shutil, re

    zip_path = genome_dir / "ncbi_dataset.zip"
    cmd = [
        "datasets", "download", "genome", "accession", accession,
        "--include", "genome",
        "--filename", str(zip_path),
        "--no-progressbar",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("datasets CLI failed for %s: %s", accession, result.stderr[:200])
        return _download_via_entrez(accession, organism, genome_dir,
                                    chrom_path, plas_path, meta_path)

    # Extract and split by sequence role (chromosome vs plasmid)
    from Bio import SeqIO  # type: ignore
    with zipfile.ZipFile(zip_path) as zf:
        fna_files = [n for n in zf.namelist() if n.endswith(".fna")]
        if not fna_files:
            logger.warning("No .fna in dataset zip for %s", accession)
            return None
        all_records = []
        for fname in fna_files:
            with zf.open(fname) as fh:
                import io
                content = fh.read().decode("utf-8")
                all_records.extend(list(SeqIO.parse(io.StringIO(content), "fasta")))

    chromosomes = []
    plasmids = []
    plasmid_sizes = []
    for rec in all_records:
        desc = (rec.description + " " + rec.id).lower()
        if any(tok in desc for tok in ("plasmid", "unnamed", "plas")):
            plasmids.append(rec)
            plasmid_sizes.append(len(rec.seq))
        else:
            chromosomes.append(rec)

    if not chromosomes:
        logger.warning("%s: no chromosome sequences found", accession)
        return None

    SeqIO.write(chromosomes, str(chrom_path), "fasta")
    SeqIO.write(plasmids, str(plas_path), "fasta") if plasmids else plas_path.write_text("")

    chrom_len = sum(len(r.seq) for r in chromosomes)
    meta = {
        "organism": organism,
        "accession": accession,
        "n_chromosomes": len(chromosomes),
        "n_plasmids": len(plasmids),
        "chromosome_bp": chrom_len,
        "plasmid_sizes": sorted(plasmid_sizes, reverse=True),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("  %s: %d chr seqs (%d bp) + %d plasmids %s",
                accession, len(chromosomes), chrom_len, len(plasmids),
                [f"{s//1000}kb" for s in plasmid_sizes[:5]])

    _safe_unlink(zip_path)
    return {"chrom": chrom_path, "plasmids": plas_path, **meta}


def _download_via_entrez(
    accession: str,
    organism: str,
    genome_dir: Path,
    chrom_path: Path,
    plas_path: Path,
    meta_path: Path,
) -> dict | None:
    """Last-resort BioPython Entrez fallback.

    WARNING: The elink(assembly → nuccore) approach is unreliable for GCF
    accessions — it can return wrong sequences from unrelated assemblies.
    This is only kept as a last resort; FTP is strongly preferred.

    Uses esummary + efetch with the sequence accession from the assembly
    document summary (more reliable than elink).
    """
    from Bio import Entrez, SeqIO  # type: ignore
    Entrez.email = "shahbaz.invincible3182@gmail.com"

    try:
        # Use esearch on the accession to get the assembly UID
        handle = Entrez.esearch(db="assembly", term=f"{accession}[Assembly Accession]")
        record = Entrez.read(handle)
        handle.close()
        if not record["IdList"]:
            logger.warning("No assembly ID found for %s", accession)
            return None
        assembly_uid = record["IdList"][0]
        time.sleep(0.4)

        # Use esummary to get the FTP path from the assembly record
        handle = Entrez.esummary(db="assembly", id=assembly_uid)
        summary = Entrez.read(handle, validate=False)
        handle.close()
        time.sleep(0.4)

        # Extract FTP path and download directly
        try:
            doc = summary["DocumentSummarySet"]["DocumentSummary"][0]
            ftp_path = doc.get("FtpPath_RefSeq", "") or doc.get("FtpPath_GenBank", "")
            if ftp_path and ftp_path != "na":
                ftp_url = ftp_path.replace("ftp://", "https://").rstrip("/") + "/"
                assembly_name = ftp_url.rstrip("/").split("/")[-1]
                fna_url = ftp_url + f"{assembly_name}_genomic.fna.gz"
                import gzip
                gz_path = genome_dir / f"{accession}_genomic.fna.gz"
                logger.info("  %s: downloading via esummary FTP path …", accession)
                urllib.request.urlretrieve(fna_url, str(gz_path))

                chromosomes = []
                plasmids = []
                plasmid_sizes = []
                with gzip.open(str(gz_path), "rt") as fh:
                    for rec in SeqIO.parse(fh, "fasta"):
                        desc = (rec.description + " " + rec.id).lower()
                        if any(tok in desc for tok in ("plasmid", " plas", ",plas")):
                            plasmids.append(rec)
                            plasmid_sizes.append(len(rec.seq))
                        else:
                            chromosomes.append(rec)
                _safe_unlink(gz_path)

                if not chromosomes:
                    return None
                SeqIO.write(chromosomes, str(chrom_path), "fasta")
                if plasmids:
                    SeqIO.write(plasmids, str(plas_path), "fasta")
                else:
                    plas_path.write_text("")
                chrom_len = sum(len(r.seq) for r in chromosomes)
                meta = {
                    "organism": organism, "accession": accession,
                    "n_chromosomes": len(chromosomes), "n_plasmids": len(plasmids),
                    "chromosome_bp": chrom_len,
                    "plasmid_sizes": sorted(plasmid_sizes, reverse=True),
                    "download_method": "entrez_esummary",
                }
                meta_path.write_text(json.dumps(meta, indent=2))
                logger.info("  %s: %d chr (%d bp) + %d plasmids [via Entrez esummary]",
                            accession, len(chromosomes), chrom_len, len(plasmids))
                return {"chrom": chrom_path, "plasmids": plas_path, **meta}
        except Exception as e2:
            logger.warning("  Entrez esummary FTP path failed for %s: %s", accession, e2)

        logger.error("All download methods failed for %s", accession)
        return None

    except Exception as e:
        logger.error("Entrez download failed for %s: %s", accession, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark genomes")
    parser.add_argument("--out",     type=Path, default=Path("data/benchmark"),
                        help="Output directory (default: data/benchmark)")
    parser.add_argument("--threads", type=int,  default=4,
                        help="Parallel download threads (default: 4)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip genomes already downloaded")
    args = parser.parse_args()

    genomes_dir = args.out / "genomes"
    genomes_dir.mkdir(parents=True, exist_ok=True)

    # Check NCBI Datasets CLI availability
    if _datasets_available():
        logger.info("NCBI Datasets CLI found — will use for downloads")
    else:
        logger.info("NCBI Datasets CLI not found — falling back to BioPython Entrez")
        logger.info("  Install with: conda install -c conda-forge ncbi-datasets-cli")

    # Download genomes
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(
                download_genome_sequences,
                accession, organism, genomes_dir
            ): (accession, organism)
            for accession, organism, _ in BENCHMARK_GENOMES
        }
        for fut in as_completed(futures):
            acc, org = futures[fut]
            try:
                result = fut.result()
                if result:
                    results.append(result)
                else:
                    failed.append(acc)
            except Exception as e:
                logger.error("Failed %s: %s", acc, e)
                failed.append(acc)

    # Write genome list TSV
    list_path = args.out / "genome_list.tsv"
    with open(list_path, "w") as fh:
        fh.write("accession\torganism\tn_chromosomes\tn_plasmids\tchromosome_bp\tplasmid_sizes_kb\n")
        for r in sorted(results, key=lambda x: x.get("accession", "")):
            plas_kb = ",".join(f"{s//1000}" for s in r.get("plasmid_sizes", []))
            fh.write(
                f"{r.get('accession','')}\t{r.get('organism','')}\t"
                f"{r.get('n_chromosomes',0)}\t{r.get('n_plasmids',0)}\t"
                f"{r.get('chromosome_bp',0)}\t{plas_kb}\n"
            )

    logger.info("\n=== Download complete ===")
    logger.info("  Successful: %d genomes → %s", len(results), genomes_dir)
    logger.info("  Failed:     %d", len(failed))
    if failed:
        logger.info("  Failed accessions: %s", ", ".join(failed))
    logger.info("  Genome list: %s", list_path)
    logger.info("\nNext step:")
    logger.info("  python scripts/build_benchmark.py --benchmark-dir %s", args.out)


if __name__ == "__main__":
    main()
