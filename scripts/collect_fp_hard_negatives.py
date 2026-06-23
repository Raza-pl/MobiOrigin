#!/usr/bin/env python3
"""collect_fp_hard_negatives.py

Collects chromosome FASTA files from the 29 genomes that PlasFlow v2 falsely
predicts as plasmid (false positives on the benchmark).  These become hard
negatives in the next training cycle so the model learns to distinguish
plasmid-like chromosomes from true plasmids.

Usage
-----
    python scripts/collect_fp_hard_negatives.py

Reads
-----
    data/benchmark/ground_truth.tsv
    data/benchmark/results/plasflow2_predictions.tsv
    data/benchmark/genomes/<accession>/chromosome.fna

Writes
------
    data/fp_hard_negatives/<accession>_chromosome.fna   (one per FP genome)

The output directory is used by retrain_fp_hardneg.sh as a second
--hard-negative-dir source.
"""

import csv
import shutil
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
BENCH       = ROOT / "data" / "benchmark"
GT_PATH     = BENCH / "ground_truth.tsv"
PRED_PATH   = BENCH / "results" / "plasflow2_predictions.tsv"
GENOME_DIR  = BENCH / "genomes"
OUT_DIR     = ROOT / "data" / "fp_hard_negatives"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ground truth ──────────────────────────────────────────────────────
    gt: dict[str, str] = {}
    with open(GT_PATH) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gt[row["contig_id"]] = row["true_label"]

    # ── Load v2 predictions ────────────────────────────────────────────────────
    preds: dict[str, str] = {}
    with open(PRED_PATH) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["contig_id"] in gt:
                preds[row["contig_id"]] = row["predicted"]

    # ── Find FP windows and their source genomes ───────────────────────────────
    fp_seqs = [s for s in gt if gt[s] != "plasmid" and preds.get(s) == "plasmid"]
    fp_accs: Counter = Counter()
    for s in fp_seqs:
        acc = s.split("_chr_")[0]
        fp_accs[acc] += 1

    log.info("Total FP windows: %d from %d genomes", len(fp_seqs), len(fp_accs))

    # ── Copy chromosome sequences ─────────────────────────────────────────────
    copied, skipped, missing = 0, 0, 0
    for acc, n_fp in fp_accs.most_common():
        chr_fna = GENOME_DIR / acc / "chromosome.fna"
        dst     = OUT_DIR / f"{acc}_chromosome.fna"

        if not chr_fna.exists():
            log.warning("chromosome.fna not found for %s (skipping)", acc)
            missing += 1
            continue

        if dst.exists():
            log.info("  SKIP  %s (%d FPs) — already in output dir", acc, n_fp)
            skipped += 1
            continue

        shutil.copy2(chr_fna, dst)
        log.info("  COPY  %s (%d FPs) → %s", acc, n_fp, dst.name)
        copied += 1

    log.info("")
    log.info("Done — copied=%d  skipped=%d  missing=%d", copied, skipped, missing)
    log.info("Output dir: %s  (%d files total)", OUT_DIR,
             sum(1 for _ in OUT_DIR.glob("*.fna")))
    log.info("")
    log.info("Next step:")
    log.info("  nohup bash scripts/retrain_fp_hardneg.sh "
             "> data/retrain_fp_hardneg.log 2>&1 &")


if __name__ == "__main__":
    main()
