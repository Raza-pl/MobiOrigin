#!/usr/bin/env python3
"""
collect_composition_fp_hardnegs.py — Extract exact composition FP windows for MLP hard-neg retrain.

PROBLEM
-------
15 composition-driven FPs are chromosomal sequences with plasmid-like k-mer profiles.
Their source organisms ARE already in data/fp_hard_negatives/ (added in previous retrain),
but these specific windows at specific offsets still slip through — the MLP generalises
enough that it hasn't learned these exact high-plasmid-score positions.

FIX
---
Extract the exact 10kb/20kb windows at the exact offsets that currently FP, and add
them to data/fp_hard_negatives/ as an additional chromosome FASTA.  The next MLP
retrain (retrain_fp_hardneg.sh) will include these exact windows in the dataset,
directly teaching the MLP that these specific sequences are chromosome.

RUN FROM PROJECT ROOT
---------------------
    python scripts/collect_composition_fp_hardnegs.py

OUTPUT
------
    data/fp_hard_negatives/composition_fps_exact.fna   (15 sequences)
    scripts/retrain_fp_hardneg.sh should be run after this to include them.
"""
from __future__ import annotations

import csv
import re
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
FP_HITS   = ROOT / "results/fp_validation/fp_plsdb_hits.tsv"
FP_MARKER = ROOT / "results/fp_validation/fp_marker_summary.tsv"
HN_DIR    = ROOT / "data/fp_hard_negatives"
OUT_FASTA = HN_DIR / "composition_fps_exact.fna"

# ── Identify composition FP contig IDs ────────────────────────────────────────
plsdb_confirmed: set[str] = set()
with open(FP_HITS) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row.get("is_known_plasmid") == "YES":
            plsdb_confirmed.add(row["contig_id"])

composition_fps: list[str] = []
with open(FP_MARKER) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        cid = row["contig_id"]
        if cid in plsdb_confirmed:
            continue
        has_marker = any(row.get(c, "0") == "1"
                         for c in ["is_conjugative", "is_mobilizable", "has_rep_protein"])
        if not has_marker:
            composition_fps.append(cid)

logger.info("Composition FP windows to extract: %d", len(composition_fps))

# ── Parse window coordinates from contig ID ───────────────────────────────────
# Format: GCF_000006945.2_chr_NC_003197.2_w10000_s2335000
# GCF    = GCF_000006945.2
# chrom  = NC_003197.2
# w      = window_size = 10000
# s      = start_offset = 2335000
ID_RE = re.compile(
    r"^(GCF_\d+\.\d+)_chr_((?:NC|NZ|CP|NW)_[^_]+\.\d+)_w(\d+)_s(\d+)$"
)


def parse_contig_id(cid: str):
    m = ID_RE.match(cid)
    if not m:
        return None
    gcf, accession, window_size, start = m.groups()
    return gcf, accession, int(window_size), int(start)


# ── Find source FASTA for each composition FP ─────────────────────────────────
# We look for GCF_xxxxxxx.x_chromosome.fna in HN_DIR
def find_chromosome_fasta(gcf: str) -> Path | None:
    # Try chromosome.fna first, then genomic.fna
    for suffix in ["_chromosome.fna", "_genomic.fna"]:
        p = HN_DIR / f"{gcf}{suffix}"
        if p.exists():
            return p
    return None


# ── Extract exact window from FASTA ──────────────────────────────────────────
def extract_window(fasta_path: Path, accession: str, start: int, length: int) -> str | None:
    """
    Extract [start : start+length] from the sequence with matching accession.
    FASTA may contain multiple contigs; we search for a header containing the accession.
    """
    target_seq: list[str] = []
    in_target = False

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                header = line[1:].split()[0]
                # Match if the accession appears in the header (handles version suffix)
                in_target = (header == accession or
                             accession in header or
                             header in accession)
                if not in_target and target_seq:
                    break  # already found target, now past it
            elif in_target:
                target_seq.append(line)

    if not target_seq:
        return None

    full_seq = "".join(target_seq)
    end = start + length
    if end > len(full_seq):
        logger.warning("  Sequence %s length %d < requested end %d — clipping",
                       accession, len(full_seq), end)
        end = len(full_seq)
    if start >= len(full_seq):
        logger.warning("  Sequence %s length %d < start %d — skipping",
                       accession, len(full_seq), start)
        return None

    return full_seq[start:end]


# ── Main extraction loop ──────────────────────────────────────────────────────
written = 0
skipped = []

with open(OUT_FASTA, "w") as out:
    for cid in composition_fps:
        coords = parse_contig_id(cid)
        if coords is None:
            logger.warning("Cannot parse contig ID: %s", cid)
            skipped.append(cid)
            continue

        gcf, accession, window_size, start = coords
        fasta = find_chromosome_fasta(gcf)
        if fasta is None:
            logger.warning("No chromosome FASTA found for %s (GCF: %s)", cid, gcf)
            skipped.append(cid)
            continue

        seq = extract_window(fasta, accession, start, window_size)
        if seq is None:
            logger.warning("Could not extract window from %s", fasta)
            skipped.append(cid)
            continue

        out.write(f">{cid}\n{seq}\n")
        written += 1
        logger.info("  ✓  %s  (gcf=%s  acc=%s  start=%d  len=%d)",
                    cid, gcf, accession, start, len(seq))

logger.info("")
logger.info("Extracted %d / %d composition FP windows → %s", written, len(composition_fps), OUT_FASTA)

if skipped:
    logger.warning("Skipped %d windows:", len(skipped))
    for s in skipped:
        logger.warning("  %s", s)

print("\n=== Next step ===")
print("Add these exact hard-negative windows to the MLP retrain:")
print(f"  {OUT_FASTA} is already in data/fp_hard_negatives/")
print("  retrain_fp_hardneg.sh picks up all *.fna in that directory.")
print()
print("Run:")
print("  nohup bash scripts/retrain_fp_hardneg.sh \\")
print("      > data/retrain_composition_fp_hardneg.log 2>&1 &")
print("  tail -f data/retrain_composition_fp_hardneg.log")
print()
print("Expected: 15 composition FPs eliminated → corrected F1 ≈ 0.812")
