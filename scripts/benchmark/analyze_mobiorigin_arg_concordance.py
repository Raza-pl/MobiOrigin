#!/usr/bin/env python3
"""Analyze label-free concordance between MobiOrigin predictions and ARG cargo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_RECORDS = 3000
EXPECTED = {
    "predictions": "d0b225830d7e89445c527ba4dd92843a4954d88b05eab153035ba30858bb789c",
    "summary": "8140ec4d0f7e1fd7ed21d254081f34ffd4e3954d5ec15a3b418b97b244462a8c",
    "consensus": "96e00487de377c34f391658de34d0066e58fe557d20cfeff5f4679f8da290887",
}
LABELS = ("chromosome", "plasmid", "phage", "unclassified")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require_fields(fields: Sequence[str] | None, required: set[str], name: str) -> None:
    missing = required - set(fields or [])
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")


def load_predictions(path: Path, expected: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_fields(reader.fieldnames, {"sequence_id", "prediction"}, "predictions")
        for line, row in enumerate(reader, 2):
            identifier = row["sequence_id"].strip()
            prediction = row["prediction"].strip()
            if not identifier or prediction not in LABELS:
                raise ValueError(f"Invalid prediction row {line}")
            rows.append({"sequence_id": identifier, "prediction": prediction})
    identifiers = [row["sequence_id"] for row in rows]
    if len(rows) != expected or len(set(identifiers)) != expected:
        raise ValueError("Prediction row accounting failed")
    return rows


def load_summary(path: Path, expected: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {
        "sequence_id",
        "predicted_orfs",
        "card_hits",
        "amrfinderplus_hits",
        "sarg_hits",
        "consensus_arg_orfs",
    }
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_fields(reader.fieldnames, required, "annotation summary")
        for line, row in enumerate(reader, 2):
            values: dict[str, Any] = {"sequence_id": row["sequence_id"].strip()}
            for field in required - {"sequence_id"}:
                try:
                    values[field] = int(row[field])
                except ValueError as error:
                    raise ValueError(f"Invalid annotation count at row {line}") from error
                if values[field] < 0:
                    raise ValueError(f"Negative annotation count at row {line}")
            rows.append(values)
    identifiers = [row["sequence_id"] for row in rows]
    if len(rows) != expected or len(set(identifiers)) != expected:
        raise ValueError("Annotation summary row accounting failed")
    return rows


def load_consensus(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    required = {"sequence_id", "orf_id", "source", "gene_symbol", "drug_class"}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_fields(reader.fieldnames, required, "ARG consensus")
        seen: set[str] = set()
        for line, row in enumerate(reader, 2):
            identifier = row["sequence_id"].strip()
            orf = row["orf_id"].strip()
            if not identifier or not orf or orf in seen:
                raise ValueError(f"Invalid or duplicate consensus ORF at row {line}")
            seen.add(orf)
            rows.append({field: row[field].strip() for field in required})
    return rows


def evaluate_rows(
    predictions: Sequence[Mapping[str, str]],
    summary: Sequence[Mapping[str, Any]],
    consensus: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    prediction_ids = [row["sequence_id"] for row in predictions]
    summary_ids = [row["sequence_id"] for row in summary]
    if prediction_ids != summary_ids:
        raise ValueError("Prediction and annotation rows are incomplete or out of order")
    prediction_by_id = {row["sequence_id"]: row["prediction"] for row in predictions}
    consensus_counts = Counter(row["sequence_id"] for row in consensus)
    if any(identifier not in prediction_by_id for identifier in consensus_counts):
        raise ValueError("ARG consensus contains an unknown sequence identifier")
    for row in summary:
        if int(row["consensus_arg_orfs"]) != consensus_counts[row["sequence_id"]]:
            raise ValueError("Summary and consensus ARG counts disagree")

    grouped: dict[str, dict[str, int | float]] = {}
    for label in LABELS:
        selected = [
            row for row, prediction in zip(summary, predictions, strict=True)
            if prediction["prediction"] == label
        ]
        records = len(selected)
        positive = sum(int(row["consensus_arg_orfs"]) > 0 for row in selected)
        grouped[label] = {
            "records": records,
            "ARG_positive_records": positive,
            "ARG_positive_fraction": positive / records if records else 0.0,
            "consensus_ARG_ORFs": sum(int(row["consensus_arg_orfs"]) for row in selected),
            "CARD_hits": sum(int(row["card_hits"]) for row in selected),
            "SARG_hits": sum(int(row["sarg_hits"]) for row in selected),
            "AMRFinderPlus_hits": sum(int(row["amrfinderplus_hits"]) for row in selected),
        }

    genes: Counter[str] = Counter()
    drug_classes: Counter[str] = Counter()
    evidence_sources: Counter[str] = Counter()
    for row in consensus:
        if prediction_by_id[row["sequence_id"]] != "plasmid":
            continue
        if row["gene_symbol"]:
            genes[row["gene_symbol"]] += 1
        if row["source"]:
            evidence_sources[row["source"]] += 1
        for value in row["drug_class"].split(";"):
            normalized = value.strip()
            if normalized:
                drug_classes[normalized] += 1
    result: dict[str, Any] = {
        "schema_version": "mobiorigin-arg-prediction-concordance-result-v1",
        "analysis_role": "post_hoc_label_free_functional_evidence_concordance",
        "records": len(predictions),
        "by_prediction_label": grouped,
        "predicted_plasmid_profile": {
            "consensus_evidence_sources": dict(sorted(evidence_sources.items())),
            "consensus_gene_counts": dict(sorted(genes.items())),
            "consensus_drug_class_counts": dict(sorted(drug_classes.items())),
        },
        "interpretation": {
            "ARG_cargo_supports_functional_plausibility": True,
            "ARG_cargo_proves_plasmid_origin": False,
            "ARG_absence_refutes_plasmid_origin": False,
            "ground_truth_labels_accessed": False,
            "descriptive_only": True,
        },
    }
    return result


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def verify_authorization(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    recorded = value.get("authorization_sha256")
    if (
        not isinstance(value, dict)
        or not isinstance(recorded, str)
        or canonical_hash(value, "authorization_sha256") != recorded
        or value.get("status") != "AUTHORIZED"
        or value.get("analysis_role")
        != "post_hoc_label_free_functional_evidence_concordance"
        or value.get("execution_passes_authorized") != 1
        or value.get("ground_truth_label_access_authorized") is not False
    ):
        raise RuntimeError("Concordance authorization is invalid")
    return value


def run_analysis(arguments: argparse.Namespace) -> int:
    authorization = verify_authorization(arguments.authorization)
    if arguments.output_dir.exists():
        raise RuntimeError("Fresh output directory required")
    for name, path in (
        ("predictions", arguments.predictions),
        ("summary", arguments.annotation_summary),
        ("consensus", arguments.arg_consensus),
    ):
        if sha256_file(path) != EXPECTED[name]:
            raise RuntimeError(f"Frozen input identity changed: {name}")
    predictions = load_predictions(arguments.predictions, EXPECTED_RECORDS)
    summary = load_summary(arguments.annotation_summary, EXPECTED_RECORDS)
    consensus = load_consensus(arguments.arg_consensus)
    result = evaluate_rows(predictions, summary, consensus)
    result["input_sha256"] = dict(EXPECTED)
    result["authorization_sha256"] = authorization["authorization_sha256"]
    result["result_sha256"] = canonical_hash(result, "result_sha256")

    arguments.output_dir.mkdir(parents=True)
    atomic_text(
        arguments.output_dir / "arg_prediction_concordance_result.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    header = (
        "prediction_label\trecords\tARG_positive_records\tARG_positive_fraction\t"
        "consensus_ARG_ORFs\tCARD_hits\tSARG_hits\tAMRFinderPlus_hits\n"
    )
    rows = []
    for label in LABELS:
        values = result["by_prediction_label"][label]
        rows.append(
            "\t".join(
                [
                    label,
                    str(values["records"]),
                    str(values["ARG_positive_records"]),
                    f"{values['ARG_positive_fraction']:.9f}",
                    str(values["consensus_ARG_ORFs"]),
                    str(values["CARD_hits"]),
                    str(values["SARG_hits"]),
                    str(values["AMRFinderPlus_hits"]),
                ]
            )
        )
    atomic_text(
        arguments.output_dir / "arg_concordance_by_prediction_label.tsv",
        header + "\n".join(rows) + "\n",
    )
    profile = result["predicted_plasmid_profile"]
    profile_rows = ["profile_type\tvalue\tcount"]
    for key, label in (
        ("consensus_evidence_sources", "evidence_source"),
        ("consensus_gene_counts", "gene"),
        ("consensus_drug_class_counts", "drug_class"),
    ):
        for value, count in profile[key].items():
            profile_rows.append(f"{label}\t{value}\t{count}")
    atomic_text(
        arguments.output_dir / "predicted_plasmid_arg_profile.tsv",
        "\n".join(profile_rows) + "\n",
    )
    print("MobiOrigin ARG/prediction concordance analysis complete.")
    return 0


def self_test() -> int:
    predictions = [
        {"sequence_id": "a", "prediction": "plasmid"},
        {"sequence_id": "b", "prediction": "plasmid"},
        {"sequence_id": "c", "prediction": "chromosome"},
        {"sequence_id": "d", "prediction": "phage"},
    ]
    summary = [
        {"sequence_id": "a", "predicted_orfs": 3, "card_hits": 1, "sarg_hits": 1, "amrfinderplus_hits": 1, "consensus_arg_orfs": 1},
        {"sequence_id": "b", "predicted_orfs": 2, "card_hits": 0, "sarg_hits": 0, "amrfinderplus_hits": 0, "consensus_arg_orfs": 0},
        {"sequence_id": "c", "predicted_orfs": 2, "card_hits": 1, "sarg_hits": 0, "amrfinderplus_hits": 0, "consensus_arg_orfs": 1},
        {"sequence_id": "d", "predicted_orfs": 1, "card_hits": 0, "sarg_hits": 0, "amrfinderplus_hits": 0, "consensus_arg_orfs": 0},
    ]
    consensus = [
        {"sequence_id": "a", "orf_id": "a_1", "source": "CARD", "gene_symbol": "blaX", "drug_class": "beta-lactam"},
        {"sequence_id": "c", "orf_id": "c_1", "source": "CARD", "gene_symbol": "tetX", "drug_class": "tetracycline"},
    ]
    first = evaluate_rows(predictions, summary, consensus)
    second = evaluate_rows(predictions, summary, consensus)
    if first != second:
        raise RuntimeError("Synthetic concordance is not deterministic")
    plasmid = first["by_prediction_label"]["plasmid"]
    if plasmid["records"] != 2 or plasmid["ARG_positive_records"] != 1:
        raise RuntimeError("Synthetic plasmid ARG concordance was not reproduced")
    if first["predicted_plasmid_profile"]["consensus_gene_counts"] != {"blaX": 1}:
        raise RuntimeError("Synthetic predicted-plasmid profile was not reproduced")
    print("MobiOrigin ARG concordance analyzer synthetic self-test: PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--annotation-summary", type=Path)
    parser.add_argument("--arg-consensus", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    if not arguments.self_test:
        required = (
            "authorization",
            "predictions",
            "annotation_summary",
            "arg_consensus",
            "output_dir",
        )
        missing = [name for name in required if getattr(arguments, name) is None]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")
    return arguments


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(self_test() if args.self_test else run_analysis(args))
