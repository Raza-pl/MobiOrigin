#!/usr/bin/env python3
"""Download PlasFlow v2 example sequences from NCBI for installation testing.

Downloads a small representative set covering all three contig classes:
    - 2 plasmids  (pUC19, a conjugative IncP plasmid)
    - 1 phage     (Lambda)
    - 1 chromosome fragment (E. coli K-12, 50 kb)

Saves everything to example/test_assembly.fasta.

Usage:
    python example/download_test_data.py
    python example/download_test_data.py --out example/my_test.fasta --email you@example.com
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Sequences to download
# ---------------------------------------------------------------------------

SEQUENCES = [
    # (accession, expected_class, description)
    ("L09137",   "plasmid",     "pUC19 - small cloning vector (2.7 kb)"),
    ("X51505",   "plasmid",     "pR388 - IncW conjugative plasmid (33.8 kb)"),
    ("J02459",   "phage",       "Lambda phage - classic model phage (48.5 kb)"),
]

# E. coli K-12 MG1655 chromosome fragment (50 kb from position 1-50000)
CHROM_ACCESSION = "NC_000913"
CHROM_START = 1
CHROM_END   = 50_000
CHROM_LABEL = "chromosome"
CHROM_DESC  = "E. coli K-12 MG1655 chromosome fragment (50 kb)"


def fetch_sequence(
    accession: str,
    email: str,
    start: int | None = None,
    stop: int | None = None,
    retries: int = 3,
) -> str:
    """Fetch a GenBank record in FASTA format via Entrez."""
    try:
        from Bio import Entrez  # type: ignore[import]
    except ImportError:
        print("ERROR: Biopython is not installed.", file=sys.stderr)
        print("  Install with: pip install biopython", file=sys.stderr)
        sys.exit(1)

    Entrez.email = email
    kwargs: dict = {
        "db": "nucleotide",
        "id": accession,
        "rettype": "fasta",
        "retmode": "text",
    }
    if start is not None:
        kwargs["seq_start"] = start
        kwargs["seq_stop"] = stop

    for attempt in range(1, retries + 1):
        try:
            handle = Entrez.efetch(**kwargs)
            data = handle.read()
            handle.close()
            if data and data.startswith(">"):
                return data
            raise ValueError(f"Unexpected NCBI response for {accession}: {data[:80]!r}")
        except Exception as exc:
            if attempt < retries:
                wait = 5 * attempt
                print(f"  Retry {attempt}/{retries} for {accession} (error: {exc}) — waiting {wait}s …")
                time.sleep(wait)
            else:
                raise


def rename_fasta_header(fasta_text: str, new_name: str, description: str) -> str:
    """Replace the first FASTA header with a clean, descriptive name."""
    lines = fasta_text.strip().splitlines()
    if not lines:
        return fasta_text
    new_header = f">{new_name} {description}"
    return "\n".join([new_header] + lines[1:]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PlasFlow v2 example test sequences from NCBI"
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "test_assembly.fasta"),
        help="Output FASTA path (default: example/test_assembly.fasta)",
    )
    parser.add_argument(
        "--email",
        default="plasflow2.test@example.com",
        help="Email for NCBI Entrez (required by NCBI policy; any valid address works)",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("PlasFlow v2 — Downloading example test sequences from NCBI")
    print(f"Output: {out_path}")
    print()

    fasta_records: list[str] = []
    errors: list[str] = []

    for accession, label, desc in SEQUENCES:
        print(f"  Fetching {accession} ({label}) — {desc} …")
        try:
            raw = fetch_sequence(accession, args.email)
            renamed = rename_fasta_header(raw, f"{accession}_{label}", desc)
            fasta_records.append(renamed)
            print(f"    OK — {len(renamed)} characters")
            time.sleep(0.4)  # NCBI rate limit: max 3 requests/sec without API key
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            errors.append(f"{accession}: {exc}")

    # Chromosome fragment (with coordinate slice)
    print(f"  Fetching {CHROM_ACCESSION} ({CHROM_LABEL}) — {CHROM_DESC} …")
    try:
        raw = fetch_sequence(CHROM_ACCESSION, args.email, start=CHROM_START, stop=CHROM_END)
        renamed = rename_fasta_header(
            raw,
            f"{CHROM_ACCESSION}:{CHROM_START}-{CHROM_END}_{CHROM_LABEL}",
            CHROM_DESC,
        )
        fasta_records.append(renamed)
        print(f"    OK — {len(renamed)} characters")
    except Exception as exc:
        print(f"    FAILED: {exc}", file=sys.stderr)
        errors.append(f"{CHROM_ACCESSION}: {exc}")

    if not fasta_records:
        print("\nERROR: No sequences could be downloaded. Check your internet connection.", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w") as fh:
        fh.write("".join(fasta_records))

    print()
    print(f"Downloaded {len(fasta_records)} sequences → {out_path}")
    print()
    print("Sequences in test_assembly.fasta:")
    for accession, label, desc in SEQUENCES:
        status = "OK" if not any(accession in e for e in errors) else "FAILED"
        print(f"  [{status}] {accession}  {label:<12}  {desc}")
    chrom_status = "OK" if not any(CHROM_ACCESSION in e for e in errors) else "FAILED"
    print(f"  [{chrom_status}] {CHROM_ACCESSION}  {CHROM_LABEL:<12}  {CHROM_DESC}")

    if errors:
        print(f"\nWarning: {len(errors)} sequences failed to download:")
        for e in errors:
            print(f"  - {e}")
        print("The test can still run with the sequences that did download.")

    print()
    print("Next step — test classification (fast, no databases needed):")
    print(f"  plasflow2 classify --input {out_path} --output example/test_predictions.tsv")
    print()
    print("Or run the full test:")
    print("  bash scripts/test_installation.sh")


if __name__ == "__main__":
    main()
