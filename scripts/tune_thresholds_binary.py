#!/usr/bin/env python3
"""tune_thresholds_binary.py

Finds the optimal per-length plasmid decision threshold for the binary MLP by
sweeping thresholds independently in each length bin and maximising plasmid F1.

Usage
-----
    python scripts/tune_thresholds_binary.py \
        --predictions  data/benchmark/results/plasflow2_predictions.tsv \
        --ground-truth data/benchmark/ground_truth.tsv \
        --out          data/fp_hardneg_experiment/threshold_sweep.json

Output JSON
-----------
    {
      "bins": {
        "10-20kb": {"best_thr": 0.87, "tp": 270, "fp": 95, "fn": 109,
                    "precision": 0.740, "recall": 0.712, "f1": 0.726},
        ...
      },
      "overall": {"plasmid_f1": ..., "macro_f1": ...},
      "length_threshold_tiers": [...]   # paste into predict.py
    }
"""

import argparse
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BINS = [
    (1_000,  2_000,  "1-2kb"),
    (2_000,  5_000,  "2-5kb"),
    (5_000,  10_000, "5-10kb"),
    (10_000, 20_000, "10-20kb"),
    (20_000, 10**9,  ">20kb"),
]


def sweep_bin(seqs_in_bin: list[dict], thr_range) -> dict:
    """Return best-F1 threshold info for a single length bin."""
    best = {"f1": 0.0, "thr": 0.5, "tp": 0, "fp": 0, "fn": 0, "p": 0.0, "r": 0.0}
    n_plas = sum(1 for s in seqs_in_bin if s["true"] == "plasmid")
    for thr in thr_range:
        tp = sum(1 for s in seqs_in_bin if s["true"] == "plasmid" and s["score"] >= thr)
        fp = sum(1 for s in seqs_in_bin if s["true"] != "plasmid" and s["score"] >= thr)
        fn = n_plas - tp
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best["f1"]:
            best = {"f1": f1, "thr": thr, "tp": tp, "fp": fp, "fn": fn,
                    "p": p, "r": r}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions",  required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out",          required=True)
    ap.add_argument("--thr-step",     type=float, default=0.01)
    args = ap.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    gt: dict[str, str] = {}
    lengths: dict[str, int] = {}
    with open(args.ground_truth) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gt[row["contig_id"]] = row["true_label"]
            lengths[row["contig_id"]] = int(row["length"])

    scores: dict[str, float] = {}
    with open(args.predictions) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["contig_id"] in gt:
                scores[row["contig_id"]] = float(row["plasmid_score"])

    log.info("Loaded %d sequences (%d with scores)",
             len(gt), sum(1 for s in gt if s in scores))

    all_seqs = [
        {"id": s, "true": gt[s], "score": scores.get(s, 0.0), "len": lengths[s]}
        for s in gt
    ]

    thr_range = [round(t * args.thr_step, 4)
                 for t in range(int(0.50 / args.thr_step),
                                int(1.00 / args.thr_step) + 1)]

    # ── Sweep each bin independently ───────────────────────────────────────────
    bin_results: dict[str, dict] = {}
    best_thresholds: dict[str, float] = {}

    log.info("")
    log.info("%-10s  %-6s  %-4s  %-4s  %-4s  %-6s  %-6s  %-6s  %-6s",
             "Bin", "BestT", "TP", "FP", "FN", "P", "R", "F1", "N_true")
    for lo, hi, label in BINS:
        seqs = [s for s in all_seqs if lo <= s["len"] < hi]
        n_true = sum(1 for s in seqs if s["true"] == "plasmid")
        if n_true == 0:
            log.info("%-10s  %-6s  %-4s  %-4s  %-4s  (no true plasmids)", label, "—", "—", "—", "—")
            best_thresholds[label] = 0.99
            bin_results[label] = {"best_thr": 0.99, "n_true": 0, "f1": 0.0}
            continue
        res = sweep_bin(seqs, thr_range)
        best_thresholds[label] = res["thr"]
        bin_results[label] = {
            "best_thr": res["thr"], "n_true": n_true,
            "tp": res["tp"], "fp": res["fp"], "fn": res["fn"],
            "precision": round(res["p"], 4),
            "recall":    round(res["r"], 4),
            "f1":        round(res["f1"], 4),
        }
        log.info("%-10s  %-6.2f  %-4d  %-4d  %-4d  %-6.4f  %-6.4f  %-6.4f  %-6d",
                 label, res["thr"], res["tp"], res["fp"], res["fn"],
                 res["p"], res["r"], res["f1"], n_true)

    # ── Compute overall F1 at per-bin optimal thresholds ─────────────────────
    final_preds: dict[str, str] = {}
    for s in all_seqs:
        bin_label = next((lb for lo, hi, lb in BINS if lo <= s["len"] < hi), ">20kb")
        thr = best_thresholds.get(bin_label, 0.95)
        final_preds[s["id"]] = "plasmid" if s["score"] >= thr else "chromosome"

    def f1(cls: str) -> float:
        tp = sum(1 for s in gt if gt[s] == cls and final_preds.get(s) == cls)
        fp = sum(1 for s in gt if gt[s] != cls and final_preds.get(s) == cls)
        fn = sum(1 for s in gt if gt[s] == cls and final_preds.get(s) != cls)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return round(2 * p * r / (p + r) if (p + r) else 0.0, 4)

    plas_f1 = f1("plasmid")
    chr_f1  = f1("chromosome")
    macro   = round((plas_f1 + chr_f1) / 2, 4)

    log.info("")
    log.info("Overall at per-bin optimal thresholds:")
    log.info("  Plasmid F1:    %.4f", plas_f1)
    log.info("  Chromosome F1: %.4f", chr_f1)
    log.info("  Macro F1:      %.4f", macro)

    # ── Build LENGTH_THRESHOLD_TIERS snippet ──────────────────────────────────
    # Maps bin label → (max_bp, plasmid_thr)
    # phage and chr thresholds kept from current predict.py defaults
    bin_to_max = {"1-2kb": 2000, "2-5kb": 4999, "5-10kb": 9999,
                  "10-20kb": 19999, ">20kb": None}
    tier_lines = ["LENGTH_THRESHOLD_TIERS = ["]
    for _, _, lb in BINS:
        max_bp = bin_to_max[lb]
        thr    = best_thresholds[lb]
        if max_bp:
            tier_lines.append(f"    ({max_bp:6d}, {thr:.2f},  0.85, 0.70),  # {lb}")
        else:
            tier_lines.append(f"    (float('inf'), {thr:.2f}, 0.82, 0.68),  # {lb}")
    tier_lines.append("]")
    tier_str = "\n".join(tier_lines)

    log.info("")
    log.info("Suggested LENGTH_THRESHOLD_TIERS for predict.py:")
    log.info("\n%s", tier_str)

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "bins":   bin_results,
        "overall": {
            "plasmid_f1":    plas_f1,
            "chromosome_f1": chr_f1,
            "macro_f1":      macro,
        },
        "length_threshold_tiers": tier_str,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
