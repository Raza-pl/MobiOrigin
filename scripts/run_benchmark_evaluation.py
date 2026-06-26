"""Run PlasFlow v2 on the benchmark and compute precision/recall/F1.

Produces paper-ready metrics comparing PlasFlow v2 against geNomad
(if installed) and PlasFlow v1 published numbers.

Metrics computed
----------------
For each class (plasmid / chromosome / phage):
  - Precision, Recall, F1-score
  - At threshold = 0.98 (current default) and at multiple thresholds
    (0.70, 0.80, 0.90, 0.95, 0.98) to show precision-recall tradeoff

Additionally:
  - Per-length-bin accuracy (1-2kb, 2-5kb, 5-10kb, 10-20kb, >20kb)
  - Confusion matrix
  - Area under precision-recall curve (AUPRC) for plasmid class
  - Comparison table with PlasFlow v1 published numbers and geNomad

Usage
-----
    # Step 1: basic evaluation
    python scripts/run_benchmark_evaluation.py \\
        --benchmark-dir data/benchmark \\
        --model         data/models/mlp_v2.pt \\
        --out           data/benchmark/results/

    # Step 2: also compare with geNomad (requires geNomad + genomad_db)
    python scripts/run_benchmark_evaluation.py \\
        --benchmark-dir  data/benchmark \\
        --model          data/models/mlp_v2.pt \\
        --genomad-db     /path/to/genomad_db \\
        --out            data/benchmark/results/

Outputs
-------
    data/benchmark/results/
        plasflow2_predictions.tsv     — full predictions at default threshold
        plasflow2_metrics.json        — precision/recall/F1 per class
        plasflow2_confusion.csv       — confusion matrix
        plasflow2_by_length.csv       — per-length-bin metrics
        plasflow2_pr_curve.csv        — precision-recall curve data (plasmid)
        genomad_metrics.json          — geNomad metrics (if run)
        comparison_table.csv          — side-by-side comparison for paper
        figures/                      — matplotlib figures (if matplotlib installed)
"""

from __future__ import annotations

# ── macOS ARM segfault fix ──────────────────────────────────────────────────
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
# ───────────────────────────────────────────────────────────────────────────

import argparse
import csv
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Published PlasFlow v1 numbers (from Krawczyk et al. 2018, Table 1)
# Measured on simulated metagenomic reads from 40 complete genomes
PLASFLOW_V1_PUBLISHED = {
    "plasmid":    {"precision": 0.963, "recall": 0.917, "f1": 0.939},
    "chromosome": {"precision": 0.982, "recall": 0.992, "f1": 0.987},
    "overall_accuracy": 0.977,
    "note": "Krawczyk et al. NAR 2018 — simulated metagenome from 40 complete genomes, threshold=0.7",
}


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def load_ground_truth(gt_tsv: Path) -> dict[str, dict]:
    """Load ground truth TSV → {contig_id: {true_label, length, accession}}."""
    gt = {}
    with open(gt_tsv) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gt[row["contig_id"]] = {
                "true_label":   row["true_label"],
                "length":       int(row["length"]),
                "accession":    row.get("genome_accession", ""),
            }
    return gt


def run_plasflow2_classify(
    fasta_path: Path,
    model_path: Path,
    out_tsv: Path,
    threshold: float = 0.70,
    plasmid_threshold: float = 0.95,
    context: str = "unspecified",
    marker_model_path: Path | None = None,
    annotation_tsv: Path | None = None,
    marker_alpha_base: float = 0.3,
) -> list[dict]:
    """Run PlasFlow v2 MLP ± marker XGBoost classification."""
    from plasflow2.utils.fasta import load_fasta
    from plasflow2.classify.predict import predict, apply_prior_correction, CONTEXT_PRIORS

    logger.info("Loading sequences from %s …", fasta_path)
    records = load_fasta(str(fasta_path), min_length=1)
    sequences  = [str(r.seq) for r in records]
    seq_ids    = [r.id for r in records]

    if marker_model_path:
        logger.info("Running PlasFlow v2 MLP + Marker XGBoost on %d sequences …", len(sequences))
    else:
        logger.info("Running PlasFlow v2 MLP on %d sequences (threshold=%.2f, plasmid_threshold=%.2f) …",
                    len(sequences), threshold, plasmid_threshold)

    # Run WITHOUT prior correction for fair comparison (v1 and geNomad don't apply context priors)
    preds = predict(
        sequences=sequences,
        sequence_ids=seq_ids,
        model_path=model_path,
        threshold=threshold,
        plasmid_threshold=plasmid_threshold,
        argmax_fallback=False,
        source_context="unspecified",  # no prior correction for benchmark
        apply_prior=False,
        marker_model_path=marker_model_path,
        annotation_tsv=annotation_tsv,
        use_pyrodigal=True,
        marker_alpha_base=marker_alpha_base,
    )

    rows = []
    for p in preds:
        bio = p.bio_evidence or {}
        mlp = p.mlp_scores or {}
        xgb = p.xgb_scores or {}
        rows.append({
            "contig_id":        p.sequence_id,
            "predicted":        p.label,
            "confidence":       round(p.confidence, 6),
            # Final blended scores (what drove the label decision)
            "plasmid_score":    round(p.scores.get("plasmid", 0.0), 6),
            "chromosome_score": round(p.scores.get("chromosome", 0.0), 6),
            "phage_score":      round(p.scores.get("phage", 0.0), 6),
            # Raw MLP scores before XGBoost blending (empty string if no marker stage)
            "mlp_plasmid":      round(mlp["plasmid"], 6) if "plasmid" in mlp else "",
            "mlp_chromosome":   round(mlp["chromosome"], 6) if "chromosome" in mlp else "",
            "mlp_phage":        round(mlp["phage"], 6) if "phage" in mlp else "",
            # XGBoost second-stage scores
            "xgb_plasmid":      round(xgb["plasmid"], 6) if "plasmid" in xgb else "",
            "xgb_chromosome":   round(xgb["chromosome"], 6) if "chromosome" in xgb else "",
            # Biological evidence from annotation TSV (empty string if TSV not used)
            "is_conjugative":   int(bio["is_conjugative"]) if "is_conjugative" in bio else "",
            "is_mobilizable":   int(bio["is_mobilizable"]) if "is_mobilizable" in bio else "",
            "has_replicon":     int(bio["has_replicon"]) if "has_replicon" in bio else "",
            "has_ice":          int(bio["has_ice"]) if "has_ice" in bio else "",
            "has_rep_protein":  int(bio["has_rep_protein"]) if "has_rep_protein" in bio else "",
            "n_rep_per_kb":     round(bio["n_rep_per_kb"], 4) if "n_rep_per_kb" in bio else "",
            # What drove the final prediction
            "evidence_type":    p.evidence_type or "",
        })

    # Write predictions
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return rows


def compute_metrics(
    predictions: list[dict],
    ground_truth: dict[str, dict],
    labels: list[str] = None,
) -> dict:
    """Compute precision, recall, F1 for each class."""
    if labels is None:
        labels = ["plasmid", "chromosome", "phage"]

    # Build confusion matrix counts
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total = 0

    for pred in predictions:
        cid = pred["contig_id"]
        if cid not in ground_truth:
            continue
        true = ground_truth[cid]["true_label"]
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
    metrics["total_evaluated"] = total
    return metrics


def compute_pr_curve(
    predictions: list[dict],
    ground_truth: dict[str, dict],
    pos_class: str = "plasmid",
    thresholds: list[float] = None,
) -> list[dict]:
    """Compute precision-recall at multiple thresholds for one class."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.02)]

    curve = []
    for thresh in thresholds:
        preds_at_thresh = []
        for pred in predictions:
            cid = pred["contig_id"]
            if cid not in ground_truth:
                continue
            score = pred.get(f"{pos_class}_score", 0.0)
            preds_at_thresh.append({
                "contig_id": cid,
                "predicted": pos_class if score >= thresh else "other",
                "confidence": score,
            })
        m = compute_metrics(preds_at_thresh, ground_truth, [pos_class])
        curve.append({
            "threshold":  thresh,
            "precision":  m[pos_class]["precision"],
            "recall":     m[pos_class]["recall"],
            "f1":         m[pos_class]["f1"],
            "tp": m[pos_class]["tp"],
            "fp": m[pos_class]["fp"],
            "fn": m[pos_class]["fn"],
        })

    return curve


def compute_per_length_metrics(
    predictions: list[dict],
    ground_truth: dict[str, dict],
) -> dict:
    """Compute plasmid precision/recall per contig length bin."""
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
            "n_contigs":          len(bin_preds),
            "plasmid_precision":  m["plasmid"]["precision"],
            "plasmid_recall":     m["plasmid"]["recall"],
            "plasmid_f1":         m["plasmid"]["f1"],
            "overall_accuracy":   m["overall_accuracy"],
        }
    return results


def run_genomad(
    fasta_path: Path,
    genomad_db: Path,
    out_dir: Path,
    threads: int = 8,
) -> dict | None:
    """Run geNomad end-to-end and parse plasmid predictions."""
    if not genomad_db.exists():
        logger.warning("geNomad DB not found at %s — skipping", genomad_db)
        return None

    try:
        subprocess.run(["genomad", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.warning("geNomad not installed — skipping (conda install -c conda-forge -c bioconda genomad)")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "genomad", "end-to-end",
        "--cleanup",
        "--threads", str(threads),
        str(fasta_path),
        str(out_dir),
        str(genomad_db),
    ]
    logger.info("Running geNomad: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("geNomad failed: %s", result.stderr[:400])
        return None

    # Parse plasmid summary
    stem = fasta_path.stem
    summary_dir = out_dir / f"{stem}_summary"
    plas_summary = next(summary_dir.glob("*_plasmid_summary.tsv"), None)
    if not plas_summary:
        logger.warning("geNomad plasmid summary not found in %s", summary_dir)
        return None

    genomad_plasmids: set[str] = set()
    with open(plas_summary) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            genomad_plasmids.add(row["seq_name"].strip())

    logger.info("geNomad identified %d plasmids", len(genomad_plasmids))
    return {"plasmid_ids": genomad_plasmids, "summary_path": str(plas_summary)}


def genomad_to_predictions(
    genomad_plasmids: set[str],
    all_contig_ids: list[str],
) -> list[dict]:
    """Convert geNomad plasmid set to predictions list for metrics computation."""
    return [
        {
            "contig_id":  cid,
            "predicted":  "plasmid" if cid in genomad_plasmids else "chromosome",
            "confidence": 1.0,
            "plasmid_score": 1.0 if cid in genomad_plasmids else 0.0,
        }
        for cid in all_contig_ids
    ]


def build_comparison_table(
    pf2_metrics: dict,
    genomad_metrics: dict | None,
    pf1_actual_metrics: dict | None = None,
) -> list[dict]:
    """Build comparison table for paper.

    pf1_actual_metrics: if provided (from run_plasflow1_benchmark.py),
    use actual v1 predictions on the same benchmark instead of published numbers.
    """
    rows = []

    if pf1_actual_metrics:
        # Actual v1 run on the same benchmark — preferred for paper
        for cls in ["plasmid", "chromosome"]:
            m = pf1_actual_metrics.get(cls, {})
            rows.append({
                "tool":      "PlasFlow v1",
                "class":     cls,
                "precision": m.get("precision", "—"),
                "recall":    m.get("recall",    "—"),
                "f1":        m.get("f1",        "—"),
                "note":      f"same benchmark, threshold=0.7, n={pf1_actual_metrics.get('total_evaluated',0)}",
            })
    else:
        # Fall back to published numbers from Krawczyk et al. 2018
        for cls in ["plasmid", "chromosome"]:
            v1 = PLASFLOW_V1_PUBLISHED.get(cls, {})
            rows.append({
                "tool":      "PlasFlow v1",
                "class":     cls,
                "precision": v1.get("precision", "—"),
                "recall":    v1.get("recall",    "—"),
                "f1":        v1.get("f1",        "—"),
                "note":      PLASFLOW_V1_PUBLISHED["note"],
            })

    # PlasFlow v2
    for cls in ["plasmid", "chromosome", "phage"]:
        m = pf2_metrics.get(cls, {})
        rows.append({
            "tool":      "PlasFlow v2 (this work)",
            "class":     cls,
            "precision": m.get("precision", "—"),
            "recall":    m.get("recall",    "—"),
            "f1":        m.get("f1",        "—"),
            "note":      f"threshold=0.95, no prior correction, n={pf2_metrics.get('total_evaluated',0)}",
        })

    # geNomad
    if genomad_metrics:
        for cls in ["plasmid"]:
            m = genomad_metrics.get(cls, {})
            rows.append({
                "tool":      "geNomad",
                "class":     cls,
                "precision": m.get("precision", "—"),
                "recall":    m.get("recall",    "—"),
                "f1":        m.get("f1",        "—"),
                "note":      "plasmid/virus detection only",
            })

    return rows


def generate_figures(
    pr_curve: list[dict],
    per_length: dict,
    metrics: dict,
    out_dir: Path,
) -> None:
    """Generate matplotlib figures for the paper."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not available — skipping figure generation")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Precision-recall curve for plasmid class
    fig, ax = plt.subplots(figsize=(6, 5))
    recalls    = [p["recall"]    for p in pr_curve]
    precisions = [p["precision"] for p in pr_curve]
    f1s        = [p["f1"]        for p in pr_curve]
    ax.plot(recalls, precisions, "b-o", markersize=4, label="PlasFlow v2")
    # Highlight default threshold (0.98)
    best = max(pr_curve, key=lambda x: x["threshold"] <= 0.98)
    ax.plot(best["recall"], best["precision"], "r*", markersize=12,
            label=f"threshold=0.98 (F1={best['f1']:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Plasmid detection: precision-recall curve", fontsize=12)
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve_plasmid.pdf", dpi=300)
    fig.savefig(out_dir / "pr_curve_plasmid.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: pr_curve_plasmid.pdf/png")

    # 2. Per-length-bin plasmid F1
    if per_length:
        fig, ax = plt.subplots(figsize=(7, 4))
        bins  = list(per_length.keys())
        f1s   = [per_length[b]["plasmid_f1"] for b in bins]
        precs = [per_length[b]["plasmid_precision"] for b in bins]
        recs  = [per_length[b]["plasmid_recall"]    for b in bins]
        x = range(len(bins))
        w = 0.25
        ax.bar([i - w for i in x], precs, w, label="Precision", color="#2196F3")
        ax.bar([i     for i in x], recs,  w, label="Recall",    color="#4CAF50")
        ax.bar([i + w for i in x], f1s,   w, label="F1",        color="#FF9800")
        ax.set_xticks(list(x))
        ax.set_xticklabels(bins, fontsize=10)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Plasmid detection by contig length", fontsize=12)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "plasmid_by_length.pdf", dpi=300)
        fig.savefig(out_dir / "plasmid_by_length.png", dpi=150)
        plt.close(fig)
        logger.info("Saved: plasmid_by_length.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PlasFlow v2 with ground truth")
    parser.add_argument("--benchmark-dir",    type=Path, required=True)
    parser.add_argument("--model",            type=Path, default=Path("data/models/mlp_v2.pt"))
    parser.add_argument("--genomad-db",       type=Path, default=None,
                        help="geNomad database directory (enables geNomad comparison)")
    parser.add_argument("--plasflow1-metrics", type=Path, default=None,
                        help="Path to plasflow1_metrics.json from run_plasflow1_benchmark.py "
                             "(uses actual v1 predictions instead of published numbers)")
    parser.add_argument("--threads",          type=int,  default=8)
    parser.add_argument("--out",              type=Path, default=None)
    parser.add_argument("--marker-model",     type=Path, default=None,
                        help="Path to marker_xgb.pkl for second-stage XGBoost")
    parser.add_argument("--annotation-tsv",   type=Path, default=None,
                        help="Path to DIAMOND annotation TSV from annotate_sequences.py")
    parser.add_argument("--marker-alpha-base", type=float, default=0.3,
                        help="Minimum XGBoost blend weight (default 0.3). Use 0.0 to blend only when biological evidence present.")
    parser.add_argument("--stage1-model",     type=Path, default=None,
                        help="Cascade Stage 1 model (plasmid vs. rest). "
                             "When given with --stage2-model, uses cascade_predict().")
    parser.add_argument("--stage2-model",     type=Path, default=None,
                        help="Cascade Stage 2 model (chromosome vs. phage).")
    args = parser.parse_args()

    out_dir = args.out or (args.benchmark_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = args.benchmark_dir / "benchmark.fna"
    gt_path    = args.benchmark_dir / "ground_truth.tsv"

    if not fasta_path.exists():
        logger.error("benchmark.fna not found — run build_benchmark.py first")
        raise SystemExit(1)

    # Load ground truth
    logger.info("Loading ground truth from %s …", gt_path)
    gt = load_ground_truth(gt_path)
    logger.info("  %d labeled contigs", len(gt))
    from collections import Counter
    for lbl, cnt in Counter(v["true_label"] for v in gt.values()).most_common():
        logger.info("    %-12s  %6d", lbl, cnt)

    # ── PlasFlow v2 evaluation ─────────────────────────────────────────────
    use_cascade = bool(args.stage1_model and args.stage2_model)
    pf2_tsv = out_dir / ("plasflow2_cascade_predictions.tsv" if use_cascade
                         else "plasflow2_predictions.tsv")

    # Cache is valid only for pure MLP runs (not cascade, not marker, not annotation).
    use_cache = (
        not use_cascade
        and pf2_tsv.exists()
        and pf2_tsv.stat().st_size > 0
        and args.marker_model is None
        and args.annotation_tsv is None
    )
    if use_cache:
        logger.info("Reusing cached PlasFlow v2 predictions from %s", pf2_tsv)
        with open(pf2_tsv) as fh:
            pf2_preds = list(csv.DictReader(fh, delimiter="\t"))
            for p in pf2_preds:
                p["plasmid_score"]    = float(p["plasmid_score"])
                p["chromosome_score"] = float(p["chromosome_score"])
                p["phage_score"]      = float(p["phage_score"])
                p["confidence"]       = float(p["confidence"])
    elif use_cascade:
        logger.info("Cascade mode: Stage1=%s  Stage2=%s", args.stage1_model, args.stage2_model)
        from plasflow2.utils.fasta import load_fasta
        from plasflow2.classify.predict import cascade_predict
        records   = load_fasta(str(fasta_path), min_length=1)
        sequences = [str(r.seq) for r in records]
        seq_ids   = [r.id for r in records]
        preds = cascade_predict(
            sequences=sequences,
            sequence_ids=seq_ids,
            stage1_model_path=args.stage1_model,
            stage2_model_path=args.stage2_model,
        )
        pf2_preds = [
            {
                "contig_id":        p.sequence_id,
                "predicted":        p.label,
                "confidence":       p.confidence,
                "plasmid_score":    p.scores.get("plasmid", 0.0),
                "chromosome_score": p.scores.get("chromosome", 0.0),
                "phage_score":      p.scores.get("phage", 0.0),
            }
            for p in preds
        ]
        pf2_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(pf2_tsv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=pf2_preds[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(pf2_preds)
    else:
        if pf2_tsv.exists() and (args.marker_model or args.annotation_tsv):
            logger.info("Marker model/annotation provided — ignoring cached predictions")
        pf2_preds = run_plasflow2_classify(
            fasta_path=fasta_path,
            model_path=args.model,
            out_tsv=pf2_tsv,
            marker_model_path=args.marker_model,
            annotation_tsv=args.annotation_tsv,
            marker_alpha_base=args.marker_alpha_base,
        )

    # Core metrics at default threshold
    pf2_metrics = compute_metrics(pf2_preds, gt)
    (out_dir / "plasflow2_metrics.json").write_text(json.dumps(pf2_metrics, indent=2))

    logger.info("\n=== PlasFlow v2 metrics (default threshold=0.95) ===")
    for cls in ["plasmid", "chromosome", "phage"]:
        m = pf2_metrics[cls]
        logger.info("  %-12s  P=%.4f  R=%.4f  F1=%.4f  (TP=%d FP=%d FN=%d)",
                    cls, m["precision"], m["recall"], m["f1"],
                    m["tp"], m["fp"], m["fn"])
    logger.info("  overall accuracy: %.4f  (n=%d)",
                pf2_metrics["overall_accuracy"], pf2_metrics["total_evaluated"])

    # Precision-recall curve
    logger.info("\nComputing precision-recall curve …")
    pr_curve = compute_pr_curve(pf2_preds, gt)
    with open(out_dir / "plasflow2_pr_curve.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=pr_curve[0].keys())
        writer.writeheader()
        writer.writerows(pr_curve)
    best_f1 = max(pr_curve, key=lambda x: x["f1"])
    logger.info("  Best F1 for plasmid: %.4f at threshold=%.2f  (P=%.4f R=%.4f)",
                best_f1["f1"], best_f1["threshold"],
                best_f1["precision"], best_f1["recall"])

    # Per-length metrics
    per_length = compute_per_length_metrics(pf2_preds, gt)
    with open(out_dir / "plasflow2_by_length.csv", "w", newline="") as fh:
        if per_length:
            writer = csv.DictWriter(fh,
                fieldnames=["length_bin"] + list(next(iter(per_length.values())).keys()))
            writer.writeheader()
            for bin_name, m in per_length.items():
                writer.writerow({"length_bin": bin_name, **m})
    logger.info("\nPer-length plasmid F1:")
    for bin_name, m in per_length.items():
        logger.info("  %8s  F1=%.4f  P=%.4f  R=%.4f  (n=%d)",
                    bin_name, m["plasmid_f1"], m["plasmid_precision"],
                    m["plasmid_recall"], m["n_contigs"])

    # Metrics at multiple thresholds
    logger.info("\nMetrics at multiple plasmid thresholds:")
    threshold_rows = []
    for thresh in [0.70, 0.80, 0.90, 0.95, 0.98]:
        preds_at_thresh = [
            {**p, "predicted": "plasmid" if p["plasmid_score"] >= thresh
             else ("phage" if p["phage_score"] >= 0.70 else "chromosome")}
            for p in pf2_preds if p["contig_id"] in gt
        ]
        m = compute_metrics(preds_at_thresh, gt)
        pm = m["plasmid"]
        threshold_rows.append({"threshold": thresh, **pm,
                                "overall_accuracy": m["overall_accuracy"]})
        logger.info("  thresh=%.2f  plasmid P=%.4f R=%.4f F1=%.4f  overall=%.4f",
                    thresh, pm["precision"], pm["recall"], pm["f1"],
                    m["overall_accuracy"])

    with open(out_dir / "plasflow2_threshold_sweep.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=threshold_rows[0].keys())
        writer.writeheader()
        writer.writerows(threshold_rows)

    # ── Classified-only metrics (excludes unclassified sequences) ────────────
    # Important for paper: PlasFlow v1 always assigns a class (no "unclassified").
    # Our model deliberately withholds judgment below threshold, so raw metrics
    # penalise us unfairly vs v1. Report both for transparency.
    classified_preds = [p for p in pf2_preds if p["predicted"] != "unclassified"]
    if classified_preds:
        classified_gt = {cid: gt[cid] for cid in (p["contig_id"] for p in classified_preds)
                         if cid in gt}
        classified_metrics = compute_metrics(classified_preds, classified_gt)
        (out_dir / "plasflow2_classified_only_metrics.json").write_text(
            json.dumps(classified_metrics, indent=2))
        n_classified = len([p for p in classified_preds if p["contig_id"] in gt])
        n_total = len([p for p in pf2_preds if p["contig_id"] in gt])
        logger.info("\n=== PlasFlow v2 — classified sequences only (%d/%d = %.1f%%) ===",
                    n_classified, n_total, 100 * n_classified / n_total)
        for cls in ["plasmid", "chromosome", "phage"]:
            m = classified_metrics[cls]
            logger.info("  %-12s  P=%.4f  R=%.4f  F1=%.4f",
                        cls, m["precision"], m["recall"], m["f1"])
        logger.info("  overall accuracy (classified): %.4f", classified_metrics["overall_accuracy"])
        logger.info("  NOTE: unclassified sequences count as FN for their true class above")
        logger.info("        classified-only removes this penalty for recall comparison with v1")

    # ── geNomad comparison ─────────────────────────────────────────────────
    genomad_metrics = None
    if args.genomad_db:
        genomad_out = out_dir / "genomad"
        genomad_result = run_genomad(fasta_path, args.genomad_db, genomad_out, args.threads)
        if genomad_result:
            all_ids = [p["contig_id"] for p in pf2_preds]
            genomad_preds = genomad_to_predictions(
                genomad_result["plasmid_ids"], all_ids
            )
            genomad_metrics = compute_metrics(genomad_preds, gt, ["plasmid"])
            (out_dir / "genomad_metrics.json").write_text(
                json.dumps(genomad_metrics, indent=2))
            gm = genomad_metrics["plasmid"]
            logger.info("\n=== geNomad metrics ===")
            logger.info("  plasmid  P=%.4f  R=%.4f  F1=%.4f",
                        gm["precision"], gm["recall"], gm["f1"])

    # ── Load PlasFlow v1 actual metrics (optional) ─────────────────────────
    pf1_actual_metrics = None
    if args.plasflow1_metrics and args.plasflow1_metrics.exists():
        pf1_actual_metrics = json.loads(args.plasflow1_metrics.read_text())
        logger.info("\nUsing actual PlasFlow v1 metrics from %s", args.plasflow1_metrics)
    elif args.plasflow1_metrics:
        logger.warning("--plasflow1-metrics path not found: %s — falling back to published numbers",
                       args.plasflow1_metrics)

    # ── Comparison table ───────────────────────────────────────────────────
    comparison = build_comparison_table(pf2_metrics, genomad_metrics, pf1_actual_metrics)
    comp_path = out_dir / "comparison_table.csv"
    with open(comp_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    logger.info("\n=== Comparison table (saved to %s) ===", comp_path)
    logger.info("%-25s %-12s %-10s %-10s %-10s", "Tool", "Class",
                "Precision", "Recall", "F1")
    logger.info("-" * 70)
    for row in comparison:
        logger.info("%-25s %-12s %-10s %-10s %-10s",
                    row["tool"], row["class"],
                    row["precision"], row["recall"], row["f1"])

    # ── Figures ────────────────────────────────────────────────────────────
    generate_figures(pr_curve, per_length, pf2_metrics, out_dir / "figures")

    logger.info("\n=== All outputs saved to %s ===", out_dir)
    logger.info("\nKey numbers for paper:")
    pm = pf2_metrics["plasmid"]
    logger.info("  PlasFlow v2 plasmid: P=%.3f  R=%.3f  F1=%.3f",
                pm["precision"], pm["recall"], pm["f1"])
    logger.info("  PlasFlow v1 plasmid: P=%.3f  R=%.3f  F1=%.3f (published)",
                PLASFLOW_V1_PUBLISHED["plasmid"]["precision"],
                PLASFLOW_V1_PUBLISHED["plasmid"]["recall"],
                PLASFLOW_V1_PUBLISHED["plasmid"]["f1"])


if __name__ == "__main__":
    main()
