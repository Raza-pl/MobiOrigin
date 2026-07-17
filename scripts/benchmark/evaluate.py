#!/usr/bin/env python3
"""Evaluate all benchmark tool outputs against ground-truth labels.

Reads the labels.tsv produced by make_benchmark.py and each tool's output
directory (produced by run_tools.sh), then computes per-tool metrics:

    Precision, Recall, F1, MCC  (binary: plasmid vs. non-plasmid)
    ... stratified by contig length tier and by tool

Output
------
    {out}/metrics_overall.tsv   — one row per tool
    {out}/metrics_by_length.tsv — one row per (tool, length_tier)
    {out}/confusion.tsv         — raw TP/FP/TN/FN per tool
    {out}/per_contig.tsv        — every contig's true + predicted label (all tools)

Usage
-----
    python scripts/benchmark/evaluate.py \\
        --results  data/benchmark/results/tier1_all \\
        --labels   data/benchmark/tier1/all_species/labels.tsv \\
        --out      data/benchmark/eval/tier1_all
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

TOOLS = ["plasflow2", "genomad", "plasclass", "rfplasmid", "mobrecon"]

# These parsers are expected to emit one row for essentially every input
# contig.  Positive-only tools such as geNomad are intentionally excluded.
FULL_COVERAGE_TOOLS = {"plasflow2", "plasclass", "rfplasmid", "mobrecon"}

LENGTH_TIERS = ["<2 kb", "2-5 kb", "5-10 kb", "10-50 kb", ">50 kb"]


# ── Tool output parsers ────────────────────────────────────────────────────────


def _parse_plasflow2(results_dir: Path) -> dict[str, str]:
    """plasflow2: results/all_predictions.tsv — columns: sequence_id, label, ..."""
    tsv = results_dir / "plasflow2" / "all_predictions.tsv"
    if not tsv.exists():
        logger.warning("plasflow2: %s not found", tsv)
        return {}
    out = {}
    with open(tsv) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sid = row.get("sequence_id") or row.get("contig_id", "")
            label = row.get("label", "unclassified")
            out[sid] = "plasmid" if label == "plasmid" else "non-plasmid"
    logger.info("plasflow2: %d predictions", len(out))
    return out


def _parse_genomad(results_dir: Path) -> dict[str, str]:
    """geNomad: *_plasmid_summary/plasmid.fna or *_plasmid.tsv.

    geNomad's end-to-end creates:
      genomad/{prefix}_find-proviruses/{prefix}_plasmid.tsv
    and:
      genomad/{prefix}_summary/{prefix}_plasmid_summary.tsv
    We use the summary TSV if available, otherwise the plasmid TSV.
    """
    gdir = results_dir / "genomad"
    if not gdir.exists():
        return {}

    # Find summary TSV
    summaries = list(gdir.glob("**/*plasmid_summary.tsv"))
    plasmid_ids: set[str] = set()

    if summaries:
        with open(summaries[0]) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                sid = row.get("seq_name", "")
                if sid:
                    plasmid_ids.add(sid)
    else:
        # Fallback: look for plasmid.fna and extract IDs
        fastas = list(gdir.glob("**/*plasmid.fna"))
        for fa in fastas:
            with open(fa) as fh:
                for line in fh:
                    if line.startswith(">"):
                        plasmid_ids.add(line[1:].split()[0])

    # geNomad doesn't output predictions for every contig — unlisted = chromosome.
    # We return partial results; evaluate() handles missing IDs as non-plasmid.
    out = {sid: "plasmid" for sid in plasmid_ids}
    logger.info("genomad: %d plasmid predictions", len(out))
    return out


def _parse_plasclass(results_dir: Path) -> dict[str, str]:
    """PlasClass: plasclass_scores.csv — columns: name, score (0-1 plasmid prob)."""
    csv_path = results_dir / "plasclass" / "plasclass_scores.csv"
    if not csv_path.exists():
        logger.warning("plasclass: %s not found", csv_path)
        return {}
    THRESHOLD = 0.5
    out = {}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = row.get("name") or row.get("seq_name") or row.get("id", "")
            try:
                score = float(row.get("score") or row.get("plasmid_score", 0.0))
            except ValueError:
                score = 0.0
            out[sid] = "plasmid" if score >= THRESHOLD else "non-plasmid"
    logger.info("plasclass: %d predictions", len(out))
    return out


def _parse_rfplasmid(results_dir: Path) -> dict[str, str]:
    """RFPlasmid: outputRFPlasmid.txt — columns: seqname, prediction, ...

    RFPlasmid predictions are 'Plasmid', 'Chromosome', 'Uncertain'.
    """
    txt = results_dir / "rfplasmid" / "outputRFPlasmid.txt"
    if not txt.exists():
        # Some versions write to a different path
        alt = list((results_dir / "rfplasmid").glob("*.txt"))
        if alt:
            txt = alt[0]
        else:
            logger.warning("rfplasmid: output not found in %s", results_dir / "rfplasmid")
            return {}
    out = {}
    with open(txt) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sid = row.get("seqname") or row.get("Sequence", "")
            pred = (row.get("prediction") or row.get("Prediction", "")).lower()
            out[sid] = "plasmid" if "plasmid" in pred else "non-plasmid"
    logger.info("rfplasmid: %d predictions", len(out))
    return out


def _parse_mobrecon(results_dir: Path) -> dict[str, str]:
    """MOB-recon: contig_report.txt — columns: file_id, cluster_id, contig_id,
    molecule_type, ...  molecule_type in ('plasmid', 'chromosome').
    """
    txt = results_dir / "mobrecon" / "contig_report.txt"
    if not txt.exists():
        logger.warning("mobrecon: %s not found", txt)
        return {}
    out = {}
    with open(txt) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sid = row.get("contig_id") or row.get("sequence_id", "")
            mol = (row.get("molecule_type") or "").lower()
            out[sid] = "plasmid" if mol == "plasmid" else "non-plasmid"
    logger.info("mobrecon: %d predictions", len(out))
    return out


PARSERS = {
    "plasflow2": _parse_plasflow2,
    "genomad": _parse_genomad,
    "plasclass": _parse_plasclass,
    "rfplasmid": _parse_rfplasmid,
    "mobrecon": _parse_mobrecon,
}


def _load_run_status(results_dir: Path) -> dict[str, str]:
    """Load the final status recorded for each tool by ``run_tools.sh``."""

    timing_path = results_dir / "timing.tsv"
    if not timing_path.exists():
        return {}
    statuses: dict[str, str] = {}
    with open(timing_path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            tool = row.get("tool", "").strip()
            if tool:
                # Later rows win. This handles a failed first attempt followed
                # by a successful retry, or a timeout recorded after launch.
                statuses[tool] = row.get("status", "").strip().lower()
    return statuses


def _assess_tool(
    tool: str,
    predictions: dict[str, str],
    run_status: dict[str, str],
    n_labels: int,
) -> tuple[bool, str, float]:
    """Return availability, reason, and prediction coverage for a tool run."""

    coverage = len(predictions) / n_labels if n_labels else 0.0
    recorded = run_status.get(tool)
    if recorded is not None and recorded not in ("", "ok", "success", "completed"):
        return False, f"run status: {recorded}", coverage
    if tool in FULL_COVERAGE_TOOLS and coverage < 0.99:
        return False, f"incomplete output: {coverage:.1%} contig coverage", coverage
    if not predictions and recorded is None:
        return False, "no output and no successful run status", coverage
    return True, "ok", coverage


# ── Metrics ────────────────────────────────────────────────────────────────────


def _metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_total": tp + fp + tn + fn,
        "n_plasmid": tp + fn,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    results_dir: Path,
    labels_tsv: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    labels: dict[str, dict] = {}
    with open(labels_tsv) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            labels[row["contig_id"]] = row
    logger.info("Loaded %d ground-truth labels", len(labels))

    # Load all tool predictions
    predictions: dict[str, dict[str, str]] = {}
    run_status = _load_run_status(results_dir)
    tool_status_rows: list[dict] = []
    valid_tools: list[str] = []
    for tool in TOOLS:
        if tool in PARSERS:
            predictions[tool] = PARSERS[tool](results_dir)
        else:
            predictions[tool] = {}
        available, reason, coverage = _assess_tool(tool, predictions[tool], run_status, len(labels))
        tool_status_rows.append(
            {
                "tool": tool,
                "available": str(available).lower(),
                "run_status": run_status.get(tool, "not_recorded"),
                "prediction_count": len(predictions[tool]),
                "coverage": round(coverage, 4),
                "reason": reason,
            }
        )
        if available:
            valid_tools.append(tool)
        else:
            logger.warning("%s excluded from metrics: %s", tool, reason)

    _write_tsv(
        tool_status_rows,
        out_dir / "tool_status.tsv",
        ["tool", "available", "run_status", "prediction_count", "coverage", "reason"],
    )

    # Per-contig table
    per_contig_rows: list[dict] = []
    for cid, gt in labels.items():
        row: dict = {
            "contig_id": cid,
            "true_label": gt["true_label"],
            "length": gt["length"],
            "length_tier": gt.get("length_tier", ""),
            "taxon": gt.get("taxon", ""),
        }
        for tool in valid_tools:
            preds = predictions[tool]
            # Missing prediction → default to "non-plasmid" (tools only report positives)
            row[f"pred_{tool}"] = preds.get(cid, "non-plasmid")
        per_contig_rows.append(row)

    # Write per_contig.tsv
    per_contig_path = out_dir / "per_contig.tsv"
    fieldnames = ["contig_id", "true_label", "length", "length_tier", "taxon"] + [
        f"pred_{t}" for t in valid_tools
    ]
    with open(per_contig_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(per_contig_rows)
    logger.info("Per-contig table → %s", per_contig_path)

    # ── Overall metrics ───────────────────────────────────────────────────────
    overall_rows: list[dict] = []
    confusion_rows: list[dict] = []

    for tool in valid_tools:
        tp = fp = tn = fn = 0
        for row in per_contig_rows:
            true = row["true_label"]
            pred = row[f"pred_{tool}"]
            if true == "plasmid" and pred == "plasmid":
                tp += 1
            elif true != "plasmid" and pred == "plasmid":
                fp += 1
            elif true != "plasmid" and pred != "plasmid":
                tn += 1
            else:
                fn += 1
        m = _metrics(tp, fp, tn, fn)
        overall_rows.append({"tool": tool, **m})
        confusion_rows.append({"tool": tool, "tp": tp, "fp": fp, "tn": tn, "fn": fn})
        logger.info(
            "%-12s P=%.3f R=%.3f F1=%.3f MCC=%.3f",
            tool,
            m["precision"],
            m["recall"],
            m["f1"],
            m["mcc"],
        )

    _write_tsv(
        overall_rows,
        out_dir / "metrics_overall.tsv",
        [
            "tool",
            "precision",
            "recall",
            "f1",
            "mcc",
            "tp",
            "fp",
            "tn",
            "fn",
            "n_total",
            "n_plasmid",
        ],
    )

    # ── Metrics by length tier ────────────────────────────────────────────────
    by_len_rows: list[dict] = []
    for tool in valid_tools:
        for tier in LENGTH_TIERS:
            tier_rows = [r for r in per_contig_rows if r["length_tier"] == tier]
            if not tier_rows:
                continue
            tp = fp = tn = fn = 0
            for row in tier_rows:
                true = row["true_label"]
                pred = row[f"pred_{tool}"]
                if true == "plasmid" and pred == "plasmid":
                    tp += 1
                elif true != "plasmid" and pred == "plasmid":
                    fp += 1
                elif true != "plasmid" and pred != "plasmid":
                    tn += 1
                else:
                    fn += 1
            m = _metrics(tp, fp, tn, fn)
            by_len_rows.append({"tool": tool, "length_tier": tier, **m})

    _write_tsv(
        by_len_rows,
        out_dir / "metrics_by_length.tsv",
        [
            "tool",
            "length_tier",
            "precision",
            "recall",
            "f1",
            "mcc",
            "tp",
            "fp",
            "tn",
            "fn",
            "n_total",
            "n_plasmid",
        ],
    )

    # ── Metrics by taxon ──────────────────────────────────────────────────────
    taxa = sorted({r["taxon"] for r in per_contig_rows if r["taxon"]})
    by_taxon_rows: list[dict] = []
    for tool in valid_tools:
        for taxon in taxa:
            t_rows = [r for r in per_contig_rows if r["taxon"] == taxon]
            tp = fp = tn = fn = 0
            for row in t_rows:
                true = row["true_label"]
                pred = row[f"pred_{tool}"]
                if true == "plasmid" and pred == "plasmid":
                    tp += 1
                elif true != "plasmid" and pred == "plasmid":
                    fp += 1
                elif true != "plasmid" and pred != "plasmid":
                    tn += 1
                else:
                    fn += 1
            m = _metrics(tp, fp, tn, fn)
            by_taxon_rows.append({"tool": tool, "taxon": taxon, **m})

    _write_tsv(
        by_taxon_rows,
        out_dir / "metrics_by_taxon.tsv",
        [
            "tool",
            "taxon",
            "precision",
            "recall",
            "f1",
            "mcc",
            "tp",
            "fp",
            "tn",
            "fn",
            "n_total",
            "n_plasmid",
        ],
    )

    logger.info("Evaluation complete → %s", out_dir)
    _print_summary_table(overall_rows)


def _write_tsv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("  → %s", path)


def _print_summary_table(rows: list[dict]) -> None:
    print(f"\n{'Tool':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'MCC':>8}")
    print("-" * 50)
    for r in sorted(rows, key=lambda x: -x["f1"]):
        print(
            f"{r['tool']:<14} {r['precision']:>10.3f} {r['recall']:>8.3f} "
            f"{r['f1']:>8.3f} {r['mcc']:>8.3f}"
        )
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results", required=True, type=Path, help="Directory produced by run_tools.sh."
    )
    p.add_argument("--labels", required=True, type=Path, help="labels.tsv from make_benchmark.py.")
    p.add_argument("--out", required=True, type=Path, help="Output directory for evaluation TSVs.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    evaluate(args.results, args.labels, args.out)


if __name__ == "__main__":
    main()
