#!/usr/bin/env python3
"""Evaluate the frozen prospective external MobiOrigin-versus-geNomad cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.evaluate_confirmatory import (  # noqa: E402
    BINARY_PREDICTIONS,
    BINARY_TRUTH,
    THREE_CLASSES,
    THREE_PREDICTIONS,
    TruthRecord,
    _encode_counts,
    binary_label,
    binary_metrics_from_counts,
    cluster_bootstrap,
    holm_adjust,
    paired_bootstrap_p_value,
    percentile_interval,
    three_class_metrics_from_counts,
)

EXPECTED_RECORDS = 3000
EXPECTED_COMPARISON_CONTRACT_SHA256 = (
    "78ea5ef0348377f76599b179b49f9240ea2fc25b083a681bb19e3eb3419b3e70"
)
EXPECTED_PREDICTION_FREEZE_SHA256 = (
    "a49798b0bf0f613f58ab2258377c4ecd73fc717ea262e024a7452b2cfd290d83"
)
EXPECTED_LABEL_SHA256 = "6099e798b581028da63e11dacb5bc03eb16e0da86226d72fcc14bf575c4c0fd3"
EXPECTED_MOBIORIGIN_SHA256 = "d0b225830d7e89445c527ba4dd92843a4954d88b05eab153035ba30858bb789c"
EXPECTED_GENOMAD_SHA256 = "e635e80dcc68e141b8a6cc3955dc651386b152278bbb1ad0de2154693902b27c"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_818
CONFIDENCE_LEVEL = 0.95
TOOLS = ("mobiorigin", "genomad")
LENGTH_BINS = (
    "1k_to_lt2k",
    "2k_to_lt5k",
    "5k_to_lt10k",
    "10k_to_lt50k",
    "50k_to_500k",
)
THREE_SCALARS = (
    "macro_f1",
    "macro_recall_balanced_accuracy",
    "accuracy",
    "multiclass_mcc",
    "prediction_coverage",
    "unclassified_fraction",
)
BINARY_SCALARS = (
    "f1",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "mcc",
    "prediction_coverage",
    "unclassified_fraction",
)


@dataclass(frozen=True)
class ExternalTruth:
    """One released external truth record."""

    contig_id: str
    prediction_order: int
    truth: str
    length_bin: str
    source_cluster: str
    length_bp: int


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


def require_fields(fieldnames: Sequence[str] | None, required: set[str], name: str) -> None:
    missing = required - set(fieldnames or [])
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")


def load_truth(path: Path, expected_records: int = EXPECTED_RECORDS) -> list[ExternalTruth]:
    required = {
        "prediction_order",
        "opaque_contig_id",
        "class",
        "length_bin",
        "source_accession",
        "fragment_length_bp",
    }
    records: list[ExternalTruth] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_fields(reader.fieldnames, required, "external label map")
        for line, row in enumerate(reader, 2):
            identifier = row["opaque_contig_id"].strip()
            source = row["source_accession"].strip()
            truth = row["class"].strip()
            length_bin = row["length_bin"].strip()
            if not identifier or identifier in seen:
                raise ValueError(f"Invalid or duplicate truth identifier at row {line}")
            seen.add(identifier)
            if not source:
                raise ValueError(f"Empty source accession at row {line}")
            if truth not in THREE_CLASSES or length_bin not in LENGTH_BINS:
                raise ValueError(f"Invalid biological field at row {line}")
            try:
                order = int(row["prediction_order"])
                length = int(row["fragment_length_bp"])
            except ValueError as error:
                raise ValueError(f"Invalid numeric truth field at row {line}") from error
            if length <= 0:
                raise ValueError(f"Non-positive fragment length at row {line}")
            records.append(ExternalTruth(identifier, order, truth, length_bin, source, length))
    if len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} truth rows, observed {len(records)}")
    if [record.prediction_order for record in records] != list(range(1, expected_records + 1)):
        raise ValueError("Truth rows are not in frozen prediction order")
    if len({record.source_cluster for record in records}) != expected_records:
        raise ValueError("External source-accession clusters are not unique")
    return records


def load_mobiorigin(path: Path, identifiers: Sequence[str]) -> list[str]:
    predictions: list[str] = []
    observed: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_fields(reader.fieldnames, {"sequence_id", "prediction"}, "MobiOrigin output")
        for line, row in enumerate(reader, 2):
            label = row["prediction"].strip()
            if label not in THREE_PREDICTIONS:
                raise ValueError(f"Invalid MobiOrigin label at row {line}: {label!r}")
            observed.append(row["sequence_id"].strip())
            predictions.append(label)
    if observed != list(identifiers):
        raise ValueError("MobiOrigin identifiers are incomplete or out of order")
    return predictions


def load_genomad(path: Path, identifiers: Sequence[str]) -> list[str]:
    predictions: list[str] = []
    observed: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_fields(
            reader.fieldnames,
            {"contig_id", "predicted_label", "prediction_status"},
            "geNomad output",
        )
        for line, row in enumerate(reader, 2):
            label = row["predicted_label"].strip()
            if label not in THREE_PREDICTIONS or not row["prediction_status"].strip():
                raise ValueError(f"Invalid geNomad prediction at row {line}")
            observed.append(row["contig_id"].strip())
            predictions.append(label)
    if observed != list(identifiers):
        raise ValueError("geNomad identifiers are incomplete or out of order")
    return predictions


def point_and_indicators(
    records: Sequence[ExternalTruth], predictions: Mapping[str, Sequence[str]]
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, np.ndarray]]]:
    truth_three = [record.truth for record in records]
    truth_binary = [binary_label(value) for value in truth_three]
    points: dict[str, dict[str, dict[str, float]]] = {
        "three_class": {},
        "plasmid_binary": {},
    }
    indicators: dict[str, dict[str, np.ndarray]] = {
        "three_class": {},
        "plasmid_binary": {},
    }
    for tool in TOOLS:
        counts, indicator = _encode_counts(
            truth_three, predictions[tool], THREE_CLASSES, THREE_PREDICTIONS
        )
        points["three_class"][tool] = {
            name: float(value) for name, value in three_class_metrics_from_counts(counts).items()
        }
        indicators["three_class"][tool] = indicator
        binary_predictions = [binary_label(value) for value in predictions[tool]]
        binary_counts, binary_indicator = _encode_counts(
            truth_binary, binary_predictions, BINARY_TRUTH, BINARY_PREDICTIONS
        )
        points["plasmid_binary"][tool] = {
            name: float(value) for name, value in binary_metrics_from_counts(binary_counts).items()
        }
        indicators["plasmid_binary"][tool] = binary_indicator
    return points, indicators


def evaluate(
    records: Sequence[ExternalTruth],
    predictions: Mapping[str, Sequence[str]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if set(predictions) != set(TOOLS):
        raise ValueError("External evaluator requires exactly MobiOrigin and geNomad")
    if any(len(values) != len(records) for values in predictions.values()):
        raise ValueError("Prediction rows do not match truth rows")
    points, indicators = point_and_indicators(records, predictions)
    bootstrap_records = [
        TruthRecord(
            record.contig_id,
            record.prediction_order,
            record.truth,
            record.length_bin,
            record.source_cluster,
            record.length_bp,
        )
        for record in records
    ]
    bootstrap = cluster_bootstrap(bootstrap_records, indicators, replicates=replicates, seed=seed)
    intervals: list[dict[str, Any]] = []
    for endpoint, endpoint_tools in bootstrap.items():
        for tool, metrics in endpoint_tools.items():
            for metric, values in metrics.items():
                lower, upper = percentile_interval(values, CONFIDENCE_LEVEL)
                intervals.append(
                    {
                        "endpoint": endpoint,
                        "tool": tool,
                        "metric": metric,
                        "lower": lower,
                        "upper": upper,
                    }
                )

    primary_specs = (
        ("three_class", "macro_f1", "three_class_macro_f1"),
        ("plasmid_binary", "f1", "plasmid_binary_f1"),
    )
    paired: list[dict[str, Any]] = []
    for endpoint, metric, preregistered_name in primary_specs:
        differences = (
            bootstrap[endpoint]["mobiorigin"][metric] - bootstrap[endpoint]["genomad"][metric]
        )
        lower, upper = percentile_interval(differences, CONFIDENCE_LEVEL)
        paired.append(
            {
                "endpoint": endpoint,
                "metric": metric,
                "preregistered_name": preregistered_name,
                "mobiorigin": points[endpoint]["mobiorigin"][metric],
                "genomad": points[endpoint]["genomad"][metric],
                "difference": points[endpoint]["mobiorigin"][metric]
                - points[endpoint]["genomad"][metric],
                "lower": lower,
                "upper": upper,
                "p_value": paired_bootstrap_p_value(differences),
            }
        )
    adjusted = holm_adjust([float(row["p_value"]) for row in paired])
    for row, adjusted_p in zip(paired, adjusted, strict=True):
        row["holm_adjusted_p"] = adjusted_p
        row["superiority_supported"] = bool(row["lower"] > 0 and adjusted_p < 0.05)

    length_results: list[dict[str, Any]] = []
    for length_bin in LENGTH_BINS:
        indices = [index for index, record in enumerate(records) if record.length_bin == length_bin]
        subset_records = [records[index] for index in indices]
        subset_predictions = {
            tool: [values[index] for index in indices] for tool, values in predictions.items()
        }
        subgroup_points, _ = point_and_indicators(subset_records, subset_predictions)
        for endpoint, subgroup_tools in subgroup_points.items():
            scalars = THREE_SCALARS if endpoint == "three_class" else BINARY_SCALARS
            for tool, subgroup_metrics in subgroup_tools.items():
                for metric in scalars:
                    length_results.append(
                        {
                            "length_bin": length_bin,
                            "endpoint": endpoint,
                            "tool": tool,
                            "metric": metric,
                            "estimate": subgroup_metrics[metric],
                            "n_records": len(indices),
                            "inference_status": "descriptive_only",
                        }
                    )
    return {
        "point_estimates": points,
        "confidence_intervals": intervals,
        "paired_co_primary": paired,
        "length_bin_metrics": length_results,
    }


def validate_authorization(path: Path, evaluator_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "prediction_freeze_sha256": EXPECTED_PREDICTION_FREEZE_SHA256,
        "comparison_contract_sha256": EXPECTED_COMPARISON_CONTRACT_SHA256,
        "sealed_label_map_sha256": EXPECTED_LABEL_SHA256,
        "evaluator_source_sha256": sha256_file(evaluator_path),
    }
    if (
        value.get("schema_version") != "mobiorigin-external-ground-truth-release-authorization-v1"
        or value.get("status") != "AUTHORIZED"
        or value.get("cohort_role") != "prospective_external_confirmatory"
    ):
        raise RuntimeError("External release authorization schema or status changed")
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise RuntimeError(f"External release authorization {field} changed")
    if (
        value.get("ground_truth_release_authorized") is not True
        or value.get("performance_evaluation_authorized") is not True
    ):
        raise RuntimeError("External ground-truth release remains prohibited")
    declared = str(value.get("authorization_sha256", ""))
    if canonical_hash(value, "authorization_sha256") != declared:
        raise RuntimeError("External release authorization identity is invalid")
    return value


def validate_frozen_inputs(
    labels: Path,
    prediction_freeze: Path,
    comparison_contract: Path,
    mobiorigin_predictions: Path,
    genomad_predictions: Path,
) -> None:
    contract = json.loads(comparison_contract.read_text(encoding="utf-8"))
    freeze = json.loads(prediction_freeze.read_text(encoding="utf-8"))
    if (
        contract.get("contract_sha256") != EXPECTED_COMPARISON_CONTRACT_SHA256
        or canonical_hash(contract, "contract_sha256") != EXPECTED_COMPARISON_CONTRACT_SHA256
    ):
        raise RuntimeError("External comparison contract identity changed")
    if (
        freeze.get("freeze_sha256") != EXPECTED_PREDICTION_FREEZE_SHA256
        or canonical_hash(freeze, "freeze_sha256") != EXPECTED_PREDICTION_FREEZE_SHA256
        or freeze.get("status") != "COMPLETE"
    ):
        raise RuntimeError("External prediction freeze identity changed")
    for path, expected, name in (
        (labels, EXPECTED_LABEL_SHA256, "sealed labels"),
        (mobiorigin_predictions, EXPECTED_MOBIORIGIN_SHA256, "MobiOrigin predictions"),
        (genomad_predictions, EXPECTED_GENOMAD_SHA256, "geNomad predictions"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen {name} identity changed")
    if oct(labels.stat().st_mode & 0o777) != "0o600":
        raise RuntimeError("Sealed label-map permissions changed")
    policy = contract.get("statistical_analysis", {})
    if (
        policy.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or policy.get("bootstrap_seed") != BOOTSTRAP_SEED
        or policy.get("unit") != "source_accession"
        or policy.get("co_primary_superiority_endpoints")
        != ["three_class_macro_f1", "plasmid_binary_f1"]
        or policy.get("co_primary_multiplicity") != "Holm across the two co-primary endpoints"
    ):
        raise RuntimeError("External statistical policy changed")


def atomic_tsv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def formatted(value: Any) -> Any:
    return f"{float(value):.12g}" if isinstance(value, (float, np.floating)) else value


def write_results(
    output: Path,
    records: Sequence[ExternalTruth],
    predictions: Mapping[str, Sequence[str]],
    analysis: Mapping[str, Any],
    authorization_sha256: str,
) -> None:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("External evaluation output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    point_rows: list[dict[str, Any]] = []
    for endpoint, tools in analysis["point_estimates"].items():
        scalars = THREE_SCALARS if endpoint == "three_class" else BINARY_SCALARS
        for tool, metrics in tools.items():
            for metric in scalars:
                point_rows.append(
                    {
                        "endpoint": endpoint,
                        "tool": tool,
                        "metric": metric,
                        "estimate": formatted(metrics[metric]),
                    }
                )
    atomic_tsv(
        output / "point_estimates.tsv", ["endpoint", "tool", "metric", "estimate"], point_rows
    )
    interval_rows = [
        {
            **row,
            "lower": formatted(row["lower"]),
            "upper": formatted(row["upper"]),
            "confidence_level": CONFIDENCE_LEVEL,
        }
        for row in analysis["confidence_intervals"]
    ]
    atomic_tsv(
        output / "confidence_intervals.tsv",
        ["endpoint", "tool", "metric", "lower", "upper", "confidence_level"],
        interval_rows,
    )
    paired_rows = []
    for original in analysis["paired_co_primary"]:
        row = dict(original)
        for field in (
            "mobiorigin",
            "genomad",
            "difference",
            "lower",
            "upper",
            "p_value",
            "holm_adjusted_p",
        ):
            row[field] = formatted(row[field])
        row["superiority_supported"] = str(row["superiority_supported"]).lower()
        paired_rows.append(row)
    atomic_tsv(
        output / "paired_co_primary_differences.tsv",
        [
            "endpoint",
            "metric",
            "preregistered_name",
            "mobiorigin",
            "genomad",
            "difference",
            "lower",
            "upper",
            "p_value",
            "holm_adjusted_p",
            "superiority_supported",
        ],
        paired_rows,
    )
    subgroup_rows = [
        {**row, "estimate": formatted(row["estimate"])} for row in analysis["length_bin_metrics"]
    ]
    atomic_tsv(
        output / "length_bin_descriptive_metrics.tsv",
        [
            "length_bin",
            "endpoint",
            "tool",
            "metric",
            "estimate",
            "n_records",
            "inference_status",
        ],
        subgroup_rows,
    )
    record_rows = [
        {
            "prediction_order": record.prediction_order,
            "opaque_contig_id": record.contig_id,
            "truth": record.truth,
            "length_bin": record.length_bin,
            "source_accession": record.source_cluster,
            "fragment_length_bp": record.length_bp,
            "pred_mobiorigin": predictions["mobiorigin"][index],
            "pred_genomad": predictions["genomad"][index],
        }
        for index, record in enumerate(records)
    ]
    atomic_tsv(
        output / "record_level_external_audit.tsv",
        [
            "prediction_order",
            "opaque_contig_id",
            "truth",
            "length_bin",
            "source_accession",
            "fragment_length_bp",
            "pred_mobiorigin",
            "pred_genomad",
        ],
        record_rows,
    )
    artifacts = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.glob("*.tsv"))
    ]
    manifest: dict[str, Any] = {
        "schema_version": "mobiorigin-external-comparison-results-v1",
        "status": "FROZEN",
        "cohort_role": "prospective_external_confirmatory",
        "records": len(records),
        "source_clusters": len({record.source_cluster for record in records}),
        "prediction_freeze_sha256": EXPECTED_PREDICTION_FREEZE_SHA256,
        "comparison_contract_sha256": EXPECTED_COMPARISON_CONTRACT_SHA256,
        "release_authorization_sha256": authorization_sha256,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "artifacts": artifacts,
    }
    manifest["results_freeze_sha256"] = canonical_hash(manifest, "results_freeze_sha256")
    atomic_json(output / "external_comparison_results_manifest.json", manifest)


def self_test() -> int:
    records = [
        ExternalTruth(f"x{i}", i, truth, "1k_to_lt2k", f"s{i}", 1200)
        for i, truth in enumerate(
            ("plasmid", "chromosome", "phage", "plasmid", "chromosome", "phage"), 1
        )
    ]
    predictions = {
        "mobiorigin": [record.truth for record in records],
        "genomad": [
            "chromosome",
            "chromosome",
            "phage",
            "plasmid",
            "unclassified",
            "chromosome",
        ],
    }
    first = evaluate(records, predictions, replicates=200, seed=17)
    second = evaluate(records, predictions, replicates=200, seed=17)
    if first["paired_co_primary"] != second["paired_co_primary"]:
        raise RuntimeError("Synthetic source-bootstrap evaluation is nondeterministic")
    if len(first["paired_co_primary"]) != 2:
        raise RuntimeError("Synthetic Holm family does not contain two co-primary endpoints")
    if not all(row["difference"] > 0 for row in first["paired_co_primary"]):
        raise RuntimeError("Synthetic paired direction was not reproduced")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        labels = root / "labels.tsv"
        with labels.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "prediction_order",
                    "opaque_contig_id",
                    "class",
                    "length_bin",
                    "source_accession",
                    "fragment_length_bp",
                ],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "prediction_order": record.prediction_order,
                        "opaque_contig_id": record.contig_id,
                        "class": record.truth,
                        "length_bin": record.length_bin,
                        "source_accession": record.source_cluster,
                        "fragment_length_bp": record.length_bp,
                    }
                )
        if load_truth(labels, expected_records=6) != records:
            raise RuntimeError("Synthetic external label schema was not reproduced")
    print("MobiOrigin external evaluator synthetic self-test: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prediction-freeze", type=Path, required=True)
    parser.add_argument("--comparison-contract", type=Path, required=True)
    parser.add_argument("--mobiorigin-predictions", type=Path, required=True)
    parser.add_argument("--genomad-predictions", type=Path, required=True)
    parser.add_argument("--release-authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cohort-role", choices=["prospective_external_confirmatory"], required=True
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    if effective == ["--self-test"]:
        return self_test()
    args = build_parser().parse_args(effective)
    validate_frozen_inputs(
        args.labels.resolve(),
        args.prediction_freeze.resolve(),
        args.comparison_contract.resolve(),
        args.mobiorigin_predictions.resolve(),
        args.genomad_predictions.resolve(),
    )
    authorization = validate_authorization(
        args.release_authorization.resolve(), Path(__file__).resolve()
    )
    # Protected payloads are opened only after all identities and authorization pass.
    records = load_truth(args.labels.resolve())
    identifiers = [record.contig_id for record in records]
    predictions = {
        "mobiorigin": load_mobiorigin(args.mobiorigin_predictions.resolve(), identifiers),
        "genomad": load_genomad(args.genomad_predictions.resolve(), identifiers),
    }
    analysis = evaluate(records, predictions)
    write_results(
        args.output_dir.resolve(),
        records,
        predictions,
        analysis,
        str(authorization["authorization_sha256"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
