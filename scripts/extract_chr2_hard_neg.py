#!/usr/bin/env python3
"""
extract_chr2_hard_neg.py — Extract J2315 chromid (NC_009720.1) as a
standalone hard-negative FASTA so build_dataset.py can give it a
dedicated sampling share.

ROOT CAUSE of J2315 FP at score 0.999:
  GCF_000017645.1_genomic.fna contains 4 sequences:
    NC_009715.1  Chr1    3.87 Mb  (main chromosome)
    NC_009720.1  Chr2    3.22 Mb  ← chromid; evolved from plasmid, now essential
    NC_009716.1  Chr3    0.88 Mb  (megaplasmid — rightly labelled chr in our hard-neg)
    NC_010804.1  Plasmid 0.09 Mb  (labelled chromosome in hard-neg — harmless noise)

  With --hard-negative-max 10000 shared across 13 genomes, chr2 contributes only
  ~367 windows labelled "chromosome". Against 50,000 PLSDB plasmid training
  windows (many Burkholderia-like), 367 is insufficient signal.

FIX:
  1. Extract NC_009720.1 into its own .fna file in data/hard_negatives/
  2. Keep the original GCF_000017645.1_genomic.fna (so chr1/chr3 still contribute)
  3. Increase --hard-negative-max to 30000 in retrain_k7_v2.sh

Effect:
  - Chr2 now has TWO entries in the sampling pool: once as part of the full genome
    file and once standalone. Proportional sampling doubles its contribution.
  - With 3x budget (30,000 vs 10,000) and 2x chr2 entries, chr2 gets ~6x more
    windows: ~367 → ~2,200.
  - Ratio improves from 136:1 (50k plasmid : 367 chr2) to ~23:1 (50k : 2.2k).

Usage:
  python scripts/extract_chr2_hard_neg.py
  python scripts/extract_chr2_hard_neg.py --hard-neg-dir data/hard_negatives
  python scripts/extract_chr2_hard_neg.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract J2315 chr2 as a standalone hard-neg FASTA")
    parser.add_argument(
        "--hard-neg-dir", type=Path, default=Path("data/hard_negatives"),
        help="Directory containing hard-negative genome FASTA files (default: data/hard_negatives)",
    )
    parser.add_argument(
        "--source-genome", default="GCF_000017645.1_genomic.fna",
        help="Filename of the J2315 genome file inside --hard-neg-dir",
    )
    parser.add_argument(
        "--chr2-accession", default="NC_009720.1",
        help="Accession of the J2315 chromid to extract (default: NC_009720.1)",
    )
    parser.add_argument(
        "--out-file", default="GCF_000017645.1_chr2_chromid.fna",
        help="Output filename inside --hard-neg-dir (default: GCF_000017645.1_chr2_chromid.fna)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be done without writing files",
    )
    args = parser.parse_args()

    source = args.hard_neg_dir / args.source_genome
    out = args.hard_neg_dir / args.out_file

    if not source.exists():
        sys.exit(
            f"ERROR: Source genome not found: {source}\n"
            f"  Make sure you have downloaded hard-negative genomes to {args.hard_neg_dir}/\n"
            f"  Run: python scripts/download_hard_negatives.py"
        )

    # Parse FASTA — split on headers
    records: list[tuple[str, str]] = []  # (header, seq)
    current_header = ""
    current_seq_parts: list[str] = []

    with open(source) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_header:
                    records.append((current_header, "".join(current_seq_parts)))
                current_header = line
                current_seq_parts = []
            else:
                current_seq_parts.append(line)
        if current_header:
            records.append((current_header, "".join(current_seq_parts)))

    print(f"Source: {source}")
    print(f"  {len(records)} sequences found:")
    for hdr, seq in records:
        acc = hdr.lstrip(">").split()[0]
        print(f"    {acc:20s}  {len(seq):>12,} bp  {'← TARGET' if acc == args.chr2_accession else ''}")
    print()

    # Find the target sequence
    target_records = [
        (hdr, seq) for hdr, seq in records
        if hdr.lstrip(">").split()[0] == args.chr2_accession
    ]

    if not target_records:
        # Try partial match in header
        target_records = [
            (hdr, seq) for hdr, seq in records
            if args.chr2_accession in hdr
        ]

    if not target_records:
        sys.exit(
            f"ERROR: Accession '{args.chr2_accession}' not found in {source}\n"
            f"  Available accessions: {[hdr.lstrip('>').split()[0] for hdr, _ in records]}"
        )

    hdr, seq = target_records[0]
    print(f"Extracted: {hdr.lstrip('>').split()[0]}  ({len(seq):,} bp)")

    if args.dry_run:
        print(f"\nDRY RUN — would write to: {out}")
        print("  No files written. Remove --dry-run to proceed.")
        return

    if out.exists():
        print(f"WARNING: {out} already exists — overwriting.")

    with open(out, "w") as fh:
        fh.write(f"{hdr}\n")
        # Write in 60-char lines
        for i in range(0, len(seq), 60):
            fh.write(seq[i : i + 60] + "\n")

    print(f"\nWrote: {out}")
    print(f"  Size: {out.stat().st_size:,} bytes")
    print()
    print("NEXT STEPS:")
    print("  1. Run the extraction for any other FP organisms:")
    print("     python scripts/extract_chr2_hard_neg.py --dry-run   # verify first")
    print()
    print("  2. Retrain with increased hard-neg budget:")
    print("     nohup bash scripts/retrain_k7_v2.sh > data/retrain_k7_v2.log 2>&1 &")
    print()
    print("  The new training run will sample chr2 TWICE (from the original genome file")
    print("  and from the standalone file) at 3x total budget → ~2,200 chr2 windows")
    print("  vs ~367 previously. This should substantially reduce the J2315 FP rate.")


if __name__ == "__main__":
    main()
