#!/usr/bin/env python3
"""Evaluate all benchmark tool outputs against ground-truth labels.

Reads the labels.tsv produced by make_benchmark.py and each tool's output
directory (produced by run_tools.sh), then computes per-tool metrics:

    Precision, Recall, Specificity, Balanced Accuracy, F1, MCC
    Unclassified fraction and prediction coverage
    ... stratified by contig length tier and by tool

Output
------
    {out}/metrics_overall.tsv   — one row per tool
    {out}/metrics_by_length.tsv — one row per (tool, length_tier)
    {out}/metrics_by_taxon.tsv  — one row per (tool, taxon)
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

TOOLS = ["plasflow2", "plasflow_v1", "genomad", "plasclass", "rfplasmid", "mobrecon"]

# These parsers are expected to emit one row for essentially every input
# contig.  Positive-only tools such as geNomad are intentionally excluded.
FULL_COVERAGE_TOOLS = {"plasflow2", "plasflow_v1", "plasclass", "rfplasmid", "mobrecon"}

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
            label = (row.get("label") or "unclassified").lower()
            out[sid] = (
                label
                if label in {"plasmid", "chromosome", "phage", "unclassified"}
                else "unclassified"
            )
    logger.info("plasflow2: %d predictions", len(out))
    return out


def _parse_genomad(results_dir: Path) -> dict[str, str]:
    """Parse frozen standardized geNomad output or legacy positive-only output."""
    gdir = results_dir / "genomad"
    if not gdir.exists():
        return {}

    standardized = gdir / "standardized_predictions.tsv"
    if standardized.exists():
        out: dict[str, str] = {}
        with standardized.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "contig_id",
                "predicted_label",
                "prediction_status",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    "geNomad standardized output is missing columns: " + ", ".join(sorted(missing))
                )

            for row in reader:
                contig_id = (row.get("contig_id") or "").strip()
                label = (row.get("predicted_label") or "").strip().lower()

                if not contig_id:
                    raise ValueError("geNomad standardized output has an empty ID")
                if contig_id in out:
                    raise ValueError("Duplicate geNomad standardized identifier: " f"{contig_id}")
                if label not in {
                    "plasmid",
                    "chromosome",
                    "phage",
                    "unclassified",
                }:
                    raise ValueError(f"Invalid geNomad standardized label: {label!r}")

                out[contig_id] = label

        logger.info("genomad adapter: %d predictions", len(out))
        return out

    logger.warning(
        "genomad: using legacy positive-only output fallback; "
        "confirmatory runs require standardized_predictions.tsv"
    )

    summaries = sorted(gdir.glob("**/*plasmid_summary.tsv"))
    if len(summaries) > 1:
        raise ValueError("Multiple geNomad plasmid summary files were found")

    plasmid_ids: set[str] = set()

    if summaries:
        with summaries[0].open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                raw_id = (row.get("seq_name") or "").strip()
                if raw_id:
                    plasmid_ids.add(raw_id.split("|provirus_", 1)[0])
    else:
        fastas = sorted(gdir.glob("**/*plasmid.fna"))
        if len(fastas) > 1:
            raise ValueError("Multiple geNomad plasmid FASTA files were found")
        if fastas:
            with fastas[0].open() as handle:
                for line in handle:
                    if line.startswith(">"):
                        plasmid_ids.add(line[1:].split()[0])

    out = {contig_id: "plasmid" for contig_id in sorted(plasmid_ids)}
    logger.info(
        "genomad legacy fallback: %d plasmid predictions",
        len(out),
    )
    return out


def _parse_plasclass(results_dir: Path) -> dict[str, str]:
    """Parse frozen standardized PlasClass output or legacy score CSV."""

    plasclass_dir = results_dir / "plasclass"
    if not plasclass_dir.exists():
        return {}

    standardized = plasclass_dir / "standardized_predictions.tsv"

    if standardized.exists():
        output: dict[str, str] = {}
        expected_status = {
            "plasmid": "called_plasmid",
            "non-plasmid": "called_non_plasmid",
            "unclassified": "missing_output",
        }

        with standardized.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "contig_id",
                "predicted_label",
                "prediction_status",
                "plasmid_score",
                "decision_threshold",
                "source_tool",
                "source_version",
            }
            missing = required - set(reader.fieldnames or [])

            if missing:
                raise ValueError(
                    "PlasClass standardized output is missing columns: "
                    + ", ".join(sorted(missing))
                )

            for row in reader:
                contig_id = (row.get("contig_id") or "").strip()
                label = (row.get("predicted_label") or "").strip().lower()
                prediction_status = (row.get("prediction_status") or "").strip()
                score_text = (row.get("plasmid_score") or "").strip()
                threshold_text = (row.get("decision_threshold") or "").strip()
                source_tool = (row.get("source_tool") or "").strip()
                source_version = (row.get("source_version") or "").strip()

                if not contig_id:
                    raise ValueError("PlasClass standardized output has an empty ID")

                if contig_id in output:
                    raise ValueError("Duplicate PlasClass standardized identifier: " f"{contig_id}")

                if label not in expected_status:
                    raise ValueError(f"Invalid PlasClass standardized label: {label!r}")

                if prediction_status != expected_status[label]:
                    raise ValueError(
                        "PlasClass prediction status is inconsistent with "
                        f"label for {contig_id}: "
                        f"{prediction_status!r} versus {label!r}"
                    )

                try:
                    decision_threshold = float(threshold_text)
                except ValueError as error:
                    raise ValueError(
                        "Invalid PlasClass decision threshold for "
                        f"{contig_id}: {threshold_text!r}"
                    ) from error

                if not math.isfinite(decision_threshold) or decision_threshold != 0.5:
                    raise ValueError(
                        "PlasClass decision threshold must equal 0.5; "
                        f"{contig_id} declares {threshold_text!r}"
                    )

                if source_tool != "PlasClass":
                    raise ValueError(
                        "Invalid PlasClass source_tool for " f"{contig_id}: {source_tool!r}"
                    )

                if source_version != "0.1":
                    raise ValueError(
                        "Invalid PlasClass source_version for " f"{contig_id}: {source_version!r}"
                    )

                if label == "unclassified":
                    if score_text:
                        raise ValueError(
                            "PlasClass abstention must not contain a score: " f"{contig_id}"
                        )
                else:
                    try:
                        score = float(score_text)
                    except ValueError as error:
                        raise ValueError(
                            "Invalid PlasClass score for " f"{contig_id}: {score_text!r}"
                        ) from error

                    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise ValueError(
                            "PlasClass score must be finite and within "
                            f"[0,1] for {contig_id}: {score_text!r}"
                        )

                    expected_label = "plasmid" if score >= decision_threshold else "non-plasmid"

                    if label != expected_label:
                        raise ValueError(
                            "PlasClass score and label are inconsistent for "
                            f"{contig_id}: score={score_text}, label={label}"
                        )

                output[contig_id] = label

        logger.info(
            "plasclass adapter: %d predictions",
            len(output),
        )
        return output

    legacy_scores = plasclass_dir / "plasclass_scores.csv"

    if not legacy_scores.exists():
        logger.warning(
            "plasclass: standardized_predictions.tsv not found in %s",
            plasclass_dir,
        )
        return {}

    logger.warning(
        "plasclass: using legacy score-CSV fallback; confirmatory runs "
        "require standardized_predictions.tsv"
    )

    threshold = 0.5
    output: dict[str, str] = {}

    with legacy_scores.open() as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            contig_id = (row.get("name") or row.get("seq_name") or row.get("id") or "").strip()

            if not contig_id:
                raise ValueError("PlasClass legacy score output has an empty identifier")

            if contig_id in output:
                raise ValueError("Duplicate PlasClass legacy identifier: " f"{contig_id}")

            score_text = (row.get("score") or row.get("plasmid_score") or "").strip()

            try:
                score = float(score_text)
            except ValueError as error:
                raise ValueError(
                    "Invalid PlasClass legacy score for " f"{contig_id}: {score_text!r}"
                ) from error

            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    "PlasClass legacy score must be finite and within "
                    f"[0,1] for {contig_id}: {score_text!r}"
                )

            output[contig_id] = "plasmid" if score >= threshold else "non-plasmid"

    logger.info(
        "plasclass legacy fallback: %d predictions",
        len(output),
    )
    return output


def _parse_plasflow_v1(
    results_dir: Path,
) -> dict[str, str]:
    """Parse the frozen standardized PlasFlow v1.1 output."""

    standardized = results_dir / "plasflow_v1" / "standardized_predictions.tsv"

    if not standardized.exists():
        return {}

    output: dict[str, str] = {}
    expected_statuses = {
        "plasmid": {"called_plasmid"},
        "non-plasmid": {"called_non_plasmid"},
        "unclassified": {
            "native_abstention",
            "missing_output",
        },
    }
    expected_digest = "sha256:e69acee3233010dbf5a5245620252bf5" "b9bde930ad5546473ec496992995a7da"

    with standardized.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "contig_id",
            "raw_label",
            "predicted_label",
            "prediction_status",
            "plasmid_probability",
            "chromosome_probability",
            "max_class_probability",
            "decision_threshold",
            "source_tool",
            "source_version",
            "container_digest",
        }
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "PlasFlow v1 standardized output is missing "
                "columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            contig_id = (row.get("contig_id") or "").strip()
            raw_label = (row.get("raw_label") or "").strip()
            label = (row.get("predicted_label") or "").strip().lower()
            prediction_status = (row.get("prediction_status") or "").strip()
            threshold_text = (row.get("decision_threshold") or "").strip()
            source_tool = (row.get("source_tool") or "").strip()
            source_version = (row.get("source_version") or "").strip()
            container_digest = (row.get("container_digest") or "").strip()

            if not contig_id:
                raise ValueError("PlasFlow v1 standardized output has " "an empty ID")

            if contig_id in output:
                raise ValueError("Duplicate PlasFlow v1 standardized " f"identifier: {contig_id}")

            if label not in expected_statuses:
                raise ValueError("Invalid PlasFlow v1 standardized label: " f"{label!r}")

            if prediction_status not in expected_statuses[label]:
                raise ValueError(
                    "PlasFlow v1 prediction status is "
                    "inconsistent with its label for "
                    f"{contig_id}: {prediction_status!r} "
                    f"versus {label!r}"
                )

            try:
                decision_threshold = float(threshold_text)
            except ValueError as error:
                raise ValueError(
                    "Invalid PlasFlow v1 threshold for " f"{contig_id}: {threshold_text!r}"
                ) from error

            if not math.isfinite(decision_threshold) or decision_threshold != 0.7:
                raise ValueError(
                    "PlasFlow v1 threshold must equal 0.7; "
                    f"{contig_id} declares "
                    f"{threshold_text!r}"
                )

            if source_tool != "PlasFlow":
                raise ValueError(
                    "Invalid PlasFlow v1 source_tool for " f"{contig_id}: {source_tool!r}"
                )

            if source_version != "1.1":
                raise ValueError(
                    "Invalid PlasFlow v1 source_version for " f"{contig_id}: {source_version!r}"
                )

            if container_digest != expected_digest:
                raise ValueError(
                    "Invalid PlasFlow v1 container digest "
                    f"for {contig_id}: "
                    f"{container_digest!r}"
                )

            score_fields = [
                "plasmid_probability",
                "chromosome_probability",
                "max_class_probability",
            ]
            score_texts = {field: (row.get(field) or "").strip() for field in score_fields}

            if prediction_status == "missing_output":
                if raw_label:
                    raise ValueError(
                        "PlasFlow v1 missing output must not " f"contain a raw label: {contig_id}"
                    )

                if any(score_texts.values()):
                    raise ValueError(
                        "PlasFlow v1 missing output must not " f"contain probabilities: {contig_id}"
                    )
            else:
                scores: dict[str, float] = {}

                for field, text in score_texts.items():
                    try:
                        value = float(text)
                    except ValueError as error:
                        raise ValueError(
                            "Invalid PlasFlow v1 score " f"{field!r} for {contig_id}: " f"{text!r}"
                        ) from error

                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError(
                            "PlasFlow v1 score must be "
                            "finite and within [0,1] for "
                            f"{contig_id}: {field}={text!r}"
                        )

                    scores[field] = value

                aggregate = scores["plasmid_probability"] + scores["chromosome_probability"]

                if not math.isclose(
                    aggregate,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-5,
                ):
                    raise ValueError(
                        "PlasFlow v1 aggregate "
                        "probabilities do not sum to one "
                        f"for {contig_id}: {aggregate}"
                    )

                if label == "plasmid" and not raw_label.startswith("plasmid."):
                    raise ValueError(
                        "PlasFlow v1 plasmid label is "
                        "inconsistent with its native label "
                        f"for {contig_id}: {raw_label!r}"
                    )

                if label == "non-plasmid" and not raw_label.startswith("chromosome."):
                    raise ValueError(
                        "PlasFlow v1 non-plasmid label is "
                        "inconsistent with its native label "
                        f"for {contig_id}: {raw_label!r}"
                    )

                if prediction_status == "native_abstention" and not raw_label.startswith(
                    "unclassified."
                ):
                    raise ValueError(
                        "PlasFlow v1 native abstention is "
                        "inconsistent with its native label "
                        f"for {contig_id}: {raw_label!r}"
                    )

            output[contig_id] = label

    logger.info(
        "plasflow_v1 adapter: %d predictions",
        len(output),
    )
    return output


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
            if "plasmid" in pred:
                out[sid] = "plasmid"
            elif "chromosome" in pred:
                out[sid] = "chromosome"
            else:
                out[sid] = "unclassified"
    logger.info("rfplasmid: %d predictions", len(out))
    return out


def _parse_mobrecon(results_dir: Path) -> dict[str, str]:
    """Parse the frozen MOB-recon adapter output.

    The standardized adapter is required for confirmatory evaluation. A raw
    ``contig_report.txt`` fallback is retained for older development runs and
    canonicalizes description-bearing FASTA identifiers.
    """
    standardized = results_dir / "mobrecon" / "standardized_predictions.tsv"
    if standardized.exists():
        out: dict[str, str] = {}
        with standardized.open() as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            required = {"contig_id", "predicted_label"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    "MOB-recon standardized output is missing columns: "
                    + ", ".join(sorted(missing))
                )
            for row in reader:
                sid = (row.get("contig_id") or "").strip()
                label = (row.get("predicted_label") or "unclassified").lower()
                if not sid:
                    raise ValueError("MOB-recon standardized output has an empty ID")
                if sid in out:
                    raise ValueError(f"Duplicate MOB-recon standardized identifier: {sid}")
                out[sid] = (
                    label if label in {"plasmid", "chromosome", "unclassified"} else "unclassified"
                )
        logger.info("mobrecon adapter: %d predictions", len(out))
        return out

    txt = results_dir / "mobrecon" / "contig_report.txt"
    if not txt.exists():
        logger.warning("mobrecon: %s not found", txt)
        return {}
    logger.warning(
        "mobrecon: using legacy raw-output fallback; confirmatory runs require "
        "standardized_predictions.tsv"
    )
    out = {}
    with open(txt) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            raw_sid = row.get("contig_id") or row.get("sequence_id", "")
            sid = raw_sid.strip().split()[0] if raw_sid.strip() else ""
            if not sid:
                raise ValueError("MOB-recon raw output has an empty identifier")
            if sid in out:
                raise ValueError(f"Duplicate canonical MOB-recon identifier: {sid}")
            mol = (row.get("molecule_type") or "").lower()
            if mol == "plasmid":
                out[sid] = "plasmid"
            elif mol in {"chromosome", "chromosomal"}:
                out[sid] = "chromosome"
            else:
                out[sid] = "unclassified"
    logger.info("mobrecon: %d predictions", len(out))
    return out


PARSERS = {
    "plasflow2": _parse_plasflow2,
    "plasflow_v1": _parse_plasflow_v1,
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
    *,
    requires_full_coverage: bool | None = None,
) -> tuple[bool, bool, str, float]:
    """Return metric inclusion, availability, reason, and raw coverage.

    An attempted tool is always retained in the primary analysis. Failed,
    filtered, or missing rows are imputed as ``unclassified`` downstream, as
    required by the frozen benchmark protocol. A tool with neither output nor
    a recorded attempt is excluded because it was not evaluated.
    """

    coverage = len(predictions) / n_labels if n_labels else 0.0
    if requires_full_coverage is None:
        requires_full_coverage = tool in FULL_COVERAGE_TOOLS
    recorded = run_status.get(tool)
    if recorded is not None and recorded not in ("", "ok", "success", "completed"):
        return (
            True,
            False,
            f"run status: {recorded}; missing predictions scored unclassified",
            coverage,
        )
    if not predictions and recorded is None:
        return False, False, "not attempted: no output or run status", coverage
    if requires_full_coverage and coverage < 0.99:
        return (
            True,
            False,
            f"incomplete output: {coverage:.1%}; missing predictions scored unclassified",
            coverage,
        )
    return True, True, "ok", coverage


# ── Metrics ────────────────────────────────────────────────────────────────────


def _metrics(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    n_unclassified: int = 0,
) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    n_total = tp + fp + tn + fn
    unclassified_fraction = n_unclassified / n_total if n_total else 0.0
    prediction_coverage = 1.0 - unclassified_fraction if n_total else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_unclassified": n_unclassified,
        "unclassified_fraction": round(unclassified_fraction, 4),
        "prediction_coverage": round(prediction_coverage, 4),
        "n_total": n_total,
        "n_plasmid": tp + fn,
    }


def _confusion(rows: list[dict], prediction_field: str) -> dict[str, int]:
    """Count binary outcomes while retaining abstentions explicitly."""

    tp = fp = tn = fn = n_unclassified = 0
    for row in rows:
        true = row["true_label"]
        pred = row[prediction_field]
        if pred == "unclassified":
            n_unclassified += 1
        if true == "plasmid" and pred == "plasmid":
            tp += 1
        elif true != "plasmid" and pred == "plasmid":
            fp += 1
        elif true != "plasmid" and pred != "plasmid":
            tn += 1
        else:
            fn += 1
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_unclassified": n_unclassified,
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
    tool_availability: dict[str, bool] = {}
    genomad_standardized = (results_dir / "genomad" / "standardized_predictions.tsv").exists()
    for tool in TOOLS:
        if tool in PARSERS:
            predictions[tool] = PARSERS[tool](results_dir)
        else:
            predictions[tool] = {}
        requires_full_coverage = tool in FULL_COVERAGE_TOOLS or (
            tool == "genomad" and genomad_standardized
        )
        included, available, reason, coverage = _assess_tool(
            tool,
            predictions[tool],
            run_status,
            len(labels),
            requires_full_coverage=requires_full_coverage,
        )
        tool_availability[tool] = available
        tool_status_rows.append(
            {
                "tool": tool,
                "available": str(available).lower(),
                "included_in_metrics": str(included).lower(),
                "run_status": run_status.get(tool, "not_recorded"),
                "prediction_count": len(predictions[tool]),
                "coverage": round(coverage, 4),
                "reason": reason,
            }
        )
        if included:
            valid_tools.append(tool)
            if not available:
                logger.warning("%s retained with abstentions: %s", tool, reason)
        else:
            logger.warning("%s excluded from metrics: %s", tool, reason)

    _write_tsv(
        tool_status_rows,
        out_dir / "tool_status.tsv",
        [
            "tool",
            "available",
            "included_in_metrics",
            "run_status",
            "prediction_count",
            "coverage",
            "reason",
        ],
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
            # geNomad is a positive-only detector. Full-coverage classifier
            # adapters must expose abstentions rather than silently turning
            # missing rows into confident negative calls.
            missing_label = (
                "non-plasmid"
                if (tool == "genomad" and tool_availability[tool] and not genomad_standardized)
                else "unclassified"
            )
            row[f"pred_{tool}"] = preds.get(cid, missing_label)
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
        counts = _confusion(per_contig_rows, f"pred_{tool}")
        m = _metrics(**counts)
        overall_rows.append({"tool": tool, **m})
        confusion_rows.append({"tool": tool, **counts})
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
            "specificity",
            "balanced_accuracy",
            "f1",
            "mcc",
            "tp",
            "fp",
            "tn",
            "fn",
            "n_unclassified",
            "unclassified_fraction",
            "prediction_coverage",
            "n_total",
            "n_plasmid",
        ],
    )
    _write_tsv(
        confusion_rows,
        out_dir / "confusion.tsv",
        ["tool", "tp", "fp", "tn", "fn", "n_unclassified"],
    )

    # ── Metrics by length tier ────────────────────────────────────────────────
    by_len_rows: list[dict] = []
    for tool in valid_tools:
        for tier in LENGTH_TIERS:
            tier_rows = [r for r in per_contig_rows if r["length_tier"] == tier]
            if not tier_rows:
                continue
            counts = _confusion(tier_rows, f"pred_{tool}")
            m = _metrics(**counts)
            by_len_rows.append({"tool": tool, "length_tier": tier, **m})

    _write_tsv(
        by_len_rows,
        out_dir / "metrics_by_length.tsv",
        [
            "tool",
            "length_tier",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "f1",
            "mcc",
            "tp",
            "fp",
            "tn",
            "fn",
            "n_unclassified",
            "unclassified_fraction",
            "prediction_coverage",
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
            counts = _confusion(t_rows, f"pred_{tool}")
            m = _metrics(**counts)
            by_taxon_rows.append({"tool": tool, "taxon": taxon, **m})

    _write_tsv(
        by_taxon_rows,
        out_dir / "metrics_by_taxon.tsv",
        [
            "tool",
            "taxon",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "f1",
            "mcc",
            "tp",
            "fp",
            "tn",
            "fn",
            "n_unclassified",
            "unclassified_fraction",
            "prediction_coverage",
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
