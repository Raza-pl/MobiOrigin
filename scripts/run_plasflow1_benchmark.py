"""Run PlasFlow v1 on the benchmark FASTA and compute precision/recall/F1.

This script lets you replace the hardcoded published numbers in
run_benchmark_evaluation.py with *actual* PlasFlow v1 predictions on
the same benchmark dataset — a fair apples-to-apples comparison.

PlasFlow v1 requires:
  - Python 3 (3.6-3.9)
  - TensorFlow ≤ 1.15 (conda-installable)
  - The PlasFlow conda package itself

INSTALLATION (one-time, ~5 min):
    conda create -n plasflow1 python=3.7 -y
    conda activate plasflow1
    conda install -c bioconda plasflow -y

Then run this script from the PlasFlow v2 project root:
    conda run -n plasflow1 python scripts/run_plasflow1_benchmark.py \\
        --benchmark-dir data/benchmark \\
        --out           data/benchmark/results/

Outputs
-------
    data/benchmark/results/
        plasflow1_predictions.tsv     — raw PlasFlow v1 output (reformatted)
        plasflow1_metrics.json        — precision/recall/F1 per class
        plasflow1_by_length.csv       — per-length-bin metrics
        plasflow1_pr_curve.csv        — P-R curve at multiple thresholds
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PlasFlow v1 wrapper
# ---------------------------------------------------------------------------

def _find_plasflow() -> str | None:
    """Return path to plasflow.py script, or None if not found."""
    # Try direct command
    if shutil.which("plasflow.py"):
        return "plasflow.py"
    # Try site-packages (common conda install location)
    try:
        import site
        for sp in site.getsitepackages():
            candidate = Path(sp) / "plasflow" / "plasflow.py"
            if candidate.exists():
                return str(candidate)
    except Exception:
        pass
    return None


def run_plasflow1(
    fasta_path: Path,
    out_tsv: Path,
    threshold: float = 0.7,
    batch_size: int = 10000,
) -> Path | None:
    """Run PlasFlow v1 on a FASTA and return path to its raw TSV output."""
    plasflow = _find_plasflow()
    if plasflow is None:
        logger.error(
            "plasflow.py not found.\n"
            "Install with:\n"
            "  conda create -n plasflow1 python=3.7 -y\n"
            "  conda activate plasflow1\n"
            "  conda install -c bioconda plasflow -y\n"
            "Then re-run with:  conda run -n plasflow1 python scripts/run_plasflow1_benchmark.py ..."
        )
        return None

    raw_tsv = out_tsv.parent / "plasflow1_raw.tsv"
    cmd = [
        sys.executable, plasflow,
        "--input",     str(fasta_path),
        "--output",    str(raw_tsv),
        "--threshold", str(threshold),
        "--batch_size", str(batch_size),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("PlasFlow v1 failed:\n%s", result.stderr[:800])
        return None
    if result.stdout:
        logger.debug(result.stdout[:400])
    logger.info("PlasFlow v1 finished → %s", raw_tsv)
    return raw_tsv


def parse_plasflow1_output(raw_tsv: Path) -> list[dict]:
    """Parse PlasFlow v1 TSV output into standard predictions format.

    PlasFlow v1 columns (tab-separated, no header in some versions):
      id  label  prob_chromosome  prob_plasmid  prob_phage  contig_length

    Returns list of dicts with keys: contig_id, predicted, confidence,
    plasmid_score, chromosome_score, phage_score
    """
    rows = []
    with open(raw_tsv) as fh:
        first = fh.readline().strip()
        # Detect if first line is a header
        has_header = first.startswith("id") or first.startswith("contig")
        if not has_header:
            # Reprocess first line as data
            fh.seek(0)

        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            contig_id  = parts[0]
            label      = parts[1].strip().lower()
            # Map v1 labels to our standard labels
            if "plasmid" in label:
                predicted = "plasmid"
            elif "chromosome" in label or "chrom" in label:
                predicted = "chromosome"
            elif "phage" in label or "virus" in label:
                predicted = "phage"
            else:
                predicted = "unclassified"

            try:
                chrom_score = float(parts[2]) if len(parts) > 2 else 0.0
                plas_score  = float(parts[3]) if len(parts) > 3 else 0.0
                phage_score = float(parts[4]) if len(parts) > 4 else 0.0
            except ValueError:
                chrom_score = plas_score = phage_score = 0.0

            confidence = max(chrom_score, plas_score, phage_score)
            rows.append({
                "contig_id":        contig_id,
                "predicted":        predicted,
                "confidence":       round(confidence, 4),
                "plasmid_score":    round(plas_score, 4),
                "chromosome_score": round(chrom_score, 4),
                "phage_score":      round(phage_score, 4),
            })

    logger.info("Parsed %d predictions from PlasFlow v1 output", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Metrics (same logic as run_benchmark_evaluation.py — self-contained here)
# ---------------------------------------------------------------------------

def load_ground_truth(gt_tsv: Path) -> dict[str, dict]:
    gt = {}
    with open(gt_tsv) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gt[row["contig_id"]] = {
                "true_label": row["true_label"],
                "length":     int(row["length"]),
                "accession":  row.get("genome_accession", ""),
            }
    return gt


def compute_metrics(
    predictions: list[dict],
    ground_truth: dict[str, dict],
    labels: list[str] = None,
) -> dict:
    from collections import defaultdict, Counter
    if labels is None:
        labels = ["plasmid", "chromosome", "phage"]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total = 0
    for pred in predictions:
        cid = pred["contig_id"]
        if cid not in ground_truth:
            continue
        true       = ground_truth[cid]["true_label"]
        pred_label = pred["predicted"]
        total += 1
        for lbl in labels:
            if true == lbl and pred_label == lbl:
                tp[lbl] += 1
            elif pred_label == lbl and true != lbl:
                fp[lbl] += 1
            elif true == lbl and pred_label != lbl:
                fn[lbl] += 1
    metrics = {}
    for lbl in labels:
        prec = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) > 0 else 0.0
        rec  = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[lbl] = {
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "tp": tp[lbl], "fp": fp[lbl], "fn": fn[lbl],
        }
    correct = sum(tp[lbl] for lbl in labels)
    metrics["overall_accuracy"] = round(correct / total, 4) if total > 0 else 0.0
    metrics["total_evaluated"]  = total
    return metrics


def compute_pr_curve(
    predictions: list[dict],
    ground_truth: dict[str, dict],
    pos_class: str = "plasmid",
) -> list[dict]:
    import numpy as np
    thresholds = [round(t, 2) for t in list(range(50, 100, 2))]
    thresholds = [t / 100 for t in thresholds]
    curve = []
    for thresh in thresholds:
        preds_at = [
            {**p, "predicted": pos_class if p.get(f"{pos_class}_score", 0) >= thresh else "other"}
            for p in predictions if p["contig_id"] in ground_truth
        ]
        m = compute_metrics(preds_at, ground_truth, [pos_class])
        curve.append({
            "threshold": thresh,
            "precision": m[pos_class]["precision"],
            "recall":    m[pos_class]["recall"],
            "f1":        m[pos_class]["f1"],
            "tp": m[pos_class]["tp"],
            "fp": m[pos_class]["fp"],
            "fn": m[pos_class]["fn"],
        })
    return curve


def compute_per_length_metrics(
    predictions: list[dict],
    ground_truth: dict[str, dict],
) -> dict:
    bins = {
        "1-2kb":   (1000,   2000),
        "2-5kb":   (2000,   5000),
        "5-10kb":  (5000,  10000),
        "10-20kb": (10000, 20000),
        ">20kb":   (20000, 10**9),
    }
    results = {}
    for bin_name, (lo, hi) in bins.items():
        bin_preds = [
            p for p in predictions
            if p["contig_id"] in ground_truth
            and lo <= ground_truth[p["contig_id"]]["length"] < hi
        ]
        if not bin_preds:
            continue
        m = compute_metrics(bin_preds, ground_truth)
        results[bin_name] = {
            "n_contigs":         len(bin_preds),
            "plasmid_precision": m["plasmid"]["precision"],
            "plasmid_recall":    m["plasmid"]["recall"],
            "plasmid_f1":        m["plasmid"]["f1"],
            "overall_accuracy":  m["overall_accuracy"],
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PlasFlow v1 on benchmark and compute metrics"
    )
    parser.add_argument("--benchmark-dir", type=Path, required=True,
                        help="Benchmark directory (must contain benchmark.fna + ground_truth.tsv)")
    parser.add_argument("--out",           type=Path, default=None,
                        help="Output directory (default: benchmark-dir/results/)")
    parser.add_argument("--threshold",     type=float, default=0.7,
                        help="PlasFlow v1 classification threshold (default: 0.7)")
    parser.add_argument("--batch-size",    type=int,   default=10000,
                        help="PlasFlow v1 batch size (default: 10000)")
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Skip running PlasFlow v1 if plasflow1_raw.tsv already exists")
    args = parser.parse_args()

    out_dir = args.out or (args.benchmark_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = args.benchmark_dir / "benchmark.fna"
    gt_path    = args.benchmark_dir / "ground_truth.tsv"

    for p in [fasta_path, gt_path]:
        if not p.exists():
            logger.error("%s not found — run build_benchmark.py first", p)
            raise SystemExit(1)

    # Load ground truth
    gt = load_ground_truth(gt_path)
    from collections import Counter
    logger.info("Ground truth: %d contigs", len(gt))
    for lbl, cnt in Counter(v["true_label"] for v in gt.values()).most_common():
        logger.info("  %-12s  %6d", lbl, cnt)

    # ── Run PlasFlow v1 ────────────────────────────────────────────────────
    raw_tsv   = out_dir / "plasflow1_raw.tsv"
    final_tsv = out_dir / "plasflow1_predictions.tsv"

    if args.reuse_predictions and raw_tsv.exists() and raw_tsv.stat().st_size > 0:
        logger.info("Reusing existing %s", raw_tsv)
    else:
        result = run_plasflow1(fasta_path, raw_tsv,
                               threshold=args.threshold,
                               batch_size=args.batch_size)
        if result is None:
            raise SystemExit(1)

    predictions = parse_plasflow1_output(raw_tsv)

    # Write reformatted predictions
    with open(final_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=predictions[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(predictions)

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = compute_metrics(predictions, gt)
    (out_dir / "plasflow1_metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("\n=== PlasFlow v1 metrics (threshold=%.2f) ===", args.threshold)
    for cls in ["plasmid", "chromosome", "phage"]:
        m = metrics[cls]
        logger.info("  %-12s  P=%.4f  R=%.4f  F1=%.4f  (TP=%d FP=%d FN=%d)",
                    cls, m["precision"], m["recall"], m["f1"],
                    m["tp"], m["fp"], m["fn"])
    logger.info("  overall accuracy: %.4f  (n=%d)",
                metrics["overall_accuracy"], metrics["total_evaluated"])

    # ── PR curve ───────────────────────────────────────────────────────────
    pr_curve = compute_pr_curve(predictions, gt)
    with open(out_dir / "plasflow1_pr_curve.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=pr_curve[0].keys())
        writer.writeheader()
        writer.writerows(pr_curve)
    best_f1 = max(pr_curve, key=lambda x: x["f1"])
    logger.info("\n  Best plasmid F1 = %.4f at threshold=%.2f  (P=%.4f  R=%.4f)",
                best_f1["f1"], best_f1["threshold"],
                best_f1["precision"], best_f1["recall"])

    # ── Per-length ─────────────────────────────────────────────────────────
    per_length = compute_per_length_metrics(predictions, gt)
    if per_length:
        with open(out_dir / "plasflow1_by_length.csv", "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["length_bin"] + list(next(iter(per_length.values())).keys())
            )
            writer.writeheader()
            for bin_name, m in per_length.items():
                writer.writerow({"length_bin": bin_name, **m})
        logger.info("\nPer-length plasmid F1:")
        for bin_name, m in per_length.items():
            logger.info("  %8s  F1=%.4f  P=%.4f  R=%.4f  (n=%d)",
                        bin_name, m["plasmid_f1"],
                        m["plasmid_precision"], m["plasmid_recall"], m["n_contigs"])

    logger.info("\nAll outputs written to %s", out_dir)
    logger.info("\nNext step — merge into comparison table:")
    logger.info("  python scripts/run_benchmark_evaluation.py \\")
    logger.info("    --benchmark-dir %s \\", args.benchmark_dir)
    logger.info("    --plasflow1-metrics %s", out_dir / "plasflow1_metrics.json")


if __name__ == "__main__":
    main()
