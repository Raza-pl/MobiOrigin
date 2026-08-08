#!/usr/bin/env python3
"""Evaluate the frozen external confirmatory cohort after explicit label release.

This evaluator is deliberately separate from the development benchmark evaluator.
It validates every frozen identity and a one-time release authorization before it
parses either ground-truth labels or standardized predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

EXPECTED_RECORDS = 3000
EXPECTED_COHORT_CONTRACT_SHA256 = "15636187fd77f105e4b46b107f840553e90ef5ff7d738e00a85e65098a2e09ba"
EXPECTED_ENDPOINT_CONTRACT_SHA256 = (
    "20110ff24d9652546f58e1e2f2ec52e2ce3a4a5fcc64ab5e4cb034f8bad10ede"
)
EXPECTED_PREDICTION_FREEZE_SHA256 = (
    "e374613a7a2e1fc549d4d6d68b53eafb772b732655bb5418302deb11533ebc2c"
)
EXPECTED_SEALED_LABEL_MAP_SHA256 = (
    "c842a28507f1dce0ac83801c89ec68e3c4fff17ec4c24812084f44f95e51d4ec"
)
EXPECTED_STATISTICAL_CONTRACT_SHA256 = (
    "86b2359e281b5d01de39d41b7981fd29bcd6c1ee26c8f90fd72604b2156d2cf7"
)
EXPECTED_EVALUATOR_CONTRACT_SHA256 = (
    "aa8226b2b44037975016d5966c70f7944e004df0da95ed71de36fee137ad5843"
)
EXPECTED_BOOTSTRAP_REPLICATES = 10_000
EXPECTED_BOOTSTRAP_SEED = 20_260_808
EXPECTED_CONFIDENCE_LEVEL = 0.95

THREE_CLASSES = ("plasmid", "chromosome", "phage")
THREE_PREDICTIONS = (*THREE_CLASSES, "unclassified")
BINARY_TRUTH = ("plasmid", "non-plasmid")
BINARY_PREDICTIONS = (*BINARY_TRUTH, "unclassified")
LENGTH_BINS = (
    "1k_to_lt2k",
    "2k_to_lt5k",
    "5k_to_lt10k",
    "10k_to_lt50k",
    "50k_to_500k",
)
TOOL_DISPLAY_NAMES = {
    "plasflow2": "PlasFlow2",
    "genomad": "geNomad 1.12.0",
    "plasclass": "PlasClass",
    "plasflow_v1": "PlasFlow v1",
    "plasme": "PLASMe 1.1",
    "platon": "Platon 1.7",
}
THREE_CLASS_TOOLS = ("plasflow2", "genomad")
BINARY_TOOLS = (
    "plasflow2",
    "genomad",
    "plasclass",
    "plasflow_v1",
    "plasme",
    "platon",
)
PRIMARY_METRICS = {
    "three_class": ("macro_f1", "macro_recall_balanced_accuracy"),
    "plasmid_vs_non_plasmid": ("f1", "balanced_accuracy"),
}
THREE_SCALAR_METRICS = (
    "macro_f1",
    "macro_recall_balanced_accuracy",
    "accuracy",
    "multiclass_mcc",
    "prediction_coverage",
    "unclassified_fraction",
)
BINARY_SCALAR_METRICS = (
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
class TruthRecord:
    """One frozen label record used by the confirmatory evaluator."""

    contig_id: str
    prediction_order: int
    truth: str
    length_bin: str
    source_cluster: str
    length_bp: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def load_json_contract(
    path: Path,
    *,
    identity_field: str,
    expected_identity: str,
    description: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get(identity_field) != expected_identity:
        raise RuntimeError(f"{description} declared identity changed")
    if canonical_hash(value, identity_field) != expected_identity:
        raise RuntimeError(f"{description} canonical identity changed")
    return value


def _require_fields(fieldnames: Sequence[str] | None, required: set[str], name: str) -> None:
    observed = set(fieldnames or [])
    missing = sorted(required - observed)
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")


def load_truth_records(
    path: Path, *, expected_records: int = EXPECTED_RECORDS
) -> list[TruthRecord]:
    """Parse and validate the sealed label map after authorization."""

    records: list[TruthRecord] = []
    seen: set[str] = set()
    required = {
        "opaque_contig_id",
        "prediction_order",
        "class",
        "length_bin",
        "selected_primary_accession",
        "selected_fragment_length_bp",
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require_fields(reader.fieldnames, required, "sealed label map")
        for row_number, row in enumerate(reader, start=2):
            contig_id = row["opaque_contig_id"].strip()
            if not contig_id or contig_id in seen:
                raise ValueError(f"Invalid or duplicate label identifier at row {row_number}")
            seen.add(contig_id)
            try:
                prediction_order = int(row["prediction_order"])
                length_bp = int(row["selected_fragment_length_bp"])
            except ValueError as error:
                raise ValueError(f"Invalid numeric label field at row {row_number}") from error
            truth = row["class"].strip()
            length_bin = row["length_bin"].strip()
            source_cluster = row["selected_primary_accession"].strip()
            if truth not in THREE_CLASSES:
                raise ValueError(f"Invalid truth class at row {row_number}: {truth!r}")
            if length_bin not in LENGTH_BINS:
                raise ValueError(f"Invalid length bin at row {row_number}: {length_bin!r}")
            if not source_cluster:
                raise ValueError(f"Empty source cluster at row {row_number}")
            if length_bp <= 0:
                raise ValueError(f"Non-positive fragment length at row {row_number}")
            records.append(
                TruthRecord(
                    contig_id=contig_id,
                    prediction_order=prediction_order,
                    truth=truth,
                    length_bin=length_bin,
                    source_cluster=source_cluster,
                    length_bp=length_bp,
                )
            )
    if len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} label rows, observed {len(records)}")
    expected_order = list(range(1, expected_records + 1))
    observed_order = [record.prediction_order for record in records]
    if observed_order != expected_order:
        raise ValueError("Label rows are not in the frozen prediction order")
    return records


def load_predictions(
    path: Path,
    record_ids: Sequence[str],
    *,
    allowed_labels: set[str],
    tool: str,
) -> list[str]:
    """Read one standardized tool output and enforce exact cohort coverage/order."""

    predictions: list[str] = []
    identifiers: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require_fields(
            reader.fieldnames,
            {"contig_id", "predicted_label", "prediction_status"},
            f"{tool} predictions",
        )
        for row_number, row in enumerate(reader, start=2):
            identifier = row["contig_id"].strip()
            label = row["predicted_label"].strip()
            if label not in allowed_labels:
                raise ValueError(f"Invalid {tool} label at row {row_number}: {label!r}")
            if not row["prediction_status"].strip():
                raise ValueError(f"Empty {tool} prediction status at row {row_number}")
            identifiers.append(identifier)
            predictions.append(label)
    if identifiers != list(record_ids):
        raise ValueError(f"{tool} identifiers are incomplete, duplicated, or out of frozen order")
    return predictions


def binary_label(label: str) -> str:
    if label == "plasmid":
        return "plasmid"
    if label in {"chromosome", "phage", "non-plasmid"}:
        return "non-plasmid"
    if label == "unclassified":
        return "unclassified"
    raise ValueError(f"Cannot map label to frozen binary endpoint: {label!r}")


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(np.asarray(numerator, dtype=float), dtype=float)
    np.divide(numerator, denominator, out=result, where=np.asarray(denominator) != 0)
    return result


def _generalized_mcc(counts: np.ndarray) -> np.ndarray:
    """Coverage-aware MCC for rectangular truth-by-prediction matrices."""

    n_truth = counts.shape[-2]
    total = counts.sum(axis=(-2, -1))
    correct = sum(counts[..., index, index] for index in range(n_truth))
    truth_totals = counts.sum(axis=-1)
    prediction_totals = counts.sum(axis=-2)
    cross = (truth_totals * prediction_totals[..., :n_truth]).sum(axis=-1)
    numerator = correct * total - cross
    denominator = np.sqrt(
        np.maximum(total**2 - (prediction_totals**2).sum(axis=-1), 0)
        * np.maximum(total**2 - (truth_totals**2).sum(axis=-1), 0)
    )
    return _safe_divide(numerator, denominator)


def three_class_metrics_from_counts(counts: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    if counts.shape[-2:] != (3, 4):
        raise ValueError("Three-class counts must have shape (..., 3, 4)")
    total = counts.sum(axis=(-2, -1))
    true_totals = counts.sum(axis=-1)
    prediction_totals = counts.sum(axis=-2)
    true_positive = np.stack([counts[..., i, i] for i in range(3)], axis=-1)
    precision = _safe_divide(true_positive, prediction_totals[..., :3])
    recall = _safe_divide(true_positive, true_totals)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    classified = total - prediction_totals[..., 3]
    result: dict[str, np.ndarray] = {
        "macro_f1": f1.mean(axis=-1),
        "macro_recall_balanced_accuracy": recall.mean(axis=-1),
        "accuracy": _safe_divide(true_positive.sum(axis=-1), total),
        "multiclass_mcc": _generalized_mcc(counts),
        "prediction_coverage": _safe_divide(classified, total),
        "unclassified_fraction": _safe_divide(prediction_totals[..., 3], total),
    }
    for index, class_name in enumerate(THREE_CLASSES):
        result[f"precision_{class_name}"] = precision[..., index]
        result[f"recall_{class_name}"] = recall[..., index]
        result[f"f1_{class_name}"] = f1[..., index]
    return result


def binary_metrics_from_counts(counts: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    if counts.shape[-2:] != (2, 3):
        raise ValueError("Binary counts must have shape (..., 2, 3)")
    total = counts.sum(axis=(-2, -1))
    positive_total = counts[..., 0, :].sum(axis=-1)
    negative_total = counts[..., 1, :].sum(axis=-1)
    tp = counts[..., 0, 0]
    fp = counts[..., 1, 0]
    tn = counts[..., 1, 1]
    precision = _safe_divide(tp, tp + fp)
    sensitivity = _safe_divide(tp, positive_total)
    specificity = _safe_divide(tn, negative_total)
    f1 = _safe_divide(2 * precision * sensitivity, precision + sensitivity)
    unclassified = counts[..., :, 2].sum(axis=-1)
    return {
        "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mcc": _generalized_mcc(counts),
        "prediction_coverage": _safe_divide(total - unclassified, total),
        "unclassified_fraction": _safe_divide(unclassified, total),
    }


def _encode_counts(
    truth: Sequence[str],
    prediction: Sequence[str],
    truth_labels: Sequence[str],
    prediction_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    truth_lookup = {label: index for index, label in enumerate(truth_labels)}
    prediction_lookup = {label: index for index, label in enumerate(prediction_labels)}
    truth_codes = np.asarray([truth_lookup[label] for label in truth], dtype=np.int64)
    prediction_codes = np.asarray(
        [prediction_lookup[label] for label in prediction], dtype=np.int64
    )
    cell_codes = truth_codes * len(prediction_labels) + prediction_codes
    indicator = np.eye(len(truth_labels) * len(prediction_labels), dtype=np.int8)[cell_codes]
    counts = indicator.sum(axis=0).reshape(len(truth_labels), len(prediction_labels))
    return counts, indicator


def _point_metrics(
    records: Sequence[TruthRecord], predictions: Mapping[str, Sequence[str]]
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    truth_three = [record.truth for record in records]
    truth_binary = [binary_label(label) for label in truth_three]
    points: dict[str, dict[str, dict[str, float]]] = {
        "three_class": {},
        "plasmid_vs_non_plasmid": {},
    }
    counts_by_endpoint: dict[str, dict[str, np.ndarray]] = {
        "three_class": {},
        "plasmid_vs_non_plasmid": {},
    }
    indicators: dict[str, dict[str, np.ndarray]] = {
        "three_class": {},
        "plasmid_vs_non_plasmid": {},
    }
    for tool in THREE_CLASS_TOOLS:
        counts, indicator = _encode_counts(
            truth_three, predictions[tool], THREE_CLASSES, THREE_PREDICTIONS
        )
        metrics = three_class_metrics_from_counts(counts)
        points["three_class"][tool] = {name: float(value) for name, value in metrics.items()}
        counts_by_endpoint["three_class"][tool] = counts
        indicators["three_class"][tool] = indicator
    for tool in BINARY_TOOLS:
        mapped = [binary_label(label) for label in predictions[tool]]
        counts, indicator = _encode_counts(truth_binary, mapped, BINARY_TRUTH, BINARY_PREDICTIONS)
        metrics = binary_metrics_from_counts(counts)
        points["plasmid_vs_non_plasmid"][tool] = {
            name: float(value) for name, value in metrics.items()
        }
        counts_by_endpoint["plasmid_vs_non_plasmid"][tool] = counts
        indicators["plasmid_vs_non_plasmid"][tool] = indicator
    return points, counts_by_endpoint, indicators


def cluster_bootstrap(
    records: Sequence[TruthRecord],
    indicators: Mapping[str, Mapping[str, np.ndarray]],
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 128,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Run one shared source-cluster bootstrap for every endpoint and tool."""

    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    cluster_names = sorted({record.source_cluster for record in records})
    cluster_lookup = {name: index for index, name in enumerate(cluster_names)}
    record_clusters = np.asarray(
        [cluster_lookup[record.source_cluster] for record in records], dtype=np.int64
    )
    cluster_count = len(cluster_names)
    probability = np.full(cluster_count, 1.0 / cluster_count)
    rng = np.random.default_rng(seed)
    collected: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        endpoint: {tool: {} for tool in tools} for endpoint, tools in indicators.items()
    }
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        sampled_clusters = rng.multinomial(cluster_count, probability, size=size)
        record_weights = sampled_clusters[:, record_clusters]
        for endpoint, tools in indicators.items():
            for tool, indicator in tools.items():
                flat_counts = record_weights @ indicator
                if endpoint == "three_class":
                    counts = flat_counts.reshape(size, 3, 4)
                    metrics = three_class_metrics_from_counts(counts)
                else:
                    counts = flat_counts.reshape(size, 2, 3)
                    metrics = binary_metrics_from_counts(counts)
                for name, values in metrics.items():
                    collected[endpoint][tool].setdefault(name, []).append(values)
    return {
        endpoint: {
            tool: {name: np.concatenate(chunks) for name, chunks in metrics.items()}
            for tool, metrics in tools.items()
        }
        for endpoint, tools in collected.items()
    }


def percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def paired_bootstrap_p_value(differences: np.ndarray) -> float:
    replicates = len(differences)
    lower_tail = int(np.count_nonzero(differences <= 0))
    upper_tail = int(np.count_nonzero(differences >= 0))
    return min(1.0, 2.0 * (min(lower_tail, upper_tail) + 1) / (replicates + 1))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""

    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _subset_records(
    records: Sequence[TruthRecord],
    predictions: Mapping[str, Sequence[str]],
    indices: Sequence[int],
) -> tuple[list[TruthRecord], dict[str, list[str]]]:
    return (
        [records[index] for index in indices],
        {tool: [values[index] for index in indices] for tool, values in predictions.items()},
    )


def evaluate_records(
    records: Sequence[TruthRecord],
    predictions: Mapping[str, Sequence[str]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float = EXPECTED_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Evaluate validated in-memory records under the frozen statistical policy."""

    expected_tools = set(BINARY_TOOLS)
    if set(predictions) != expected_tools:
        raise ValueError(f"Prediction tool set changed: expected {sorted(expected_tools)}")
    if any(len(values) != len(records) for values in predictions.values()):
        raise ValueError("Prediction row count does not match truth records")
    points, counts, indicators = _point_metrics(records, predictions)
    bootstrap = cluster_bootstrap(
        records,
        indicators,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    intervals: list[dict[str, Any]] = []
    for endpoint, tools in bootstrap.items():
        for tool, metrics in tools.items():
            for metric, values in metrics.items():
                lower, upper = percentile_interval(values, confidence_level)
                intervals.append(
                    {
                        "endpoint": endpoint,
                        "tool": tool,
                        "metric": metric,
                        "lower": lower,
                        "upper": upper,
                    }
                )

    paired: list[dict[str, Any]] = []
    for endpoint, primary_metrics in PRIMARY_METRICS.items():
        comparator_tools = [tool for tool in bootstrap[endpoint] if tool != "plasflow2"]
        for metric in primary_metrics:
            family_rows: list[dict[str, Any]] = []
            for comparator in comparator_tools:
                differences = (
                    bootstrap[endpoint]["plasflow2"][metric]
                    - bootstrap[endpoint][comparator][metric]
                )
                lower, upper = percentile_interval(differences, confidence_level)
                family_rows.append(
                    {
                        "endpoint": endpoint,
                        "metric": metric,
                        "target": "plasflow2",
                        "comparator": comparator,
                        "target_estimate": points[endpoint]["plasflow2"][metric],
                        "comparator_estimate": points[endpoint][comparator][metric],
                        "difference": points[endpoint]["plasflow2"][metric]
                        - points[endpoint][comparator][metric],
                        "lower": lower,
                        "upper": upper,
                        "p_value": paired_bootstrap_p_value(differences),
                    }
                )
            adjusted = holm_adjust([float(row["p_value"]) for row in family_rows])
            for row, adjusted_p in zip(family_rows, adjusted, strict=True):
                row["holm_adjusted_p"] = adjusted_p
                row["superiority_supported"] = bool(
                    row["difference"] > 0 and row["lower"] > 0 and adjusted_p < 0.05
                )
                paired.append(row)

    subgroups: list[dict[str, Any]] = []
    for length_bin in LENGTH_BINS:
        indices = [index for index, record in enumerate(records) if record.length_bin == length_bin]
        subgroup_records, subgroup_predictions = _subset_records(records, predictions, indices)
        subgroup_points, _, _ = _point_metrics(subgroup_records, subgroup_predictions)
        for endpoint, subgroup_tools in subgroup_points.items():
            metric_names = (
                THREE_SCALAR_METRICS if endpoint == "three_class" else BINARY_SCALAR_METRICS
            )
            for tool, subgroup_metrics in subgroup_tools.items():
                for metric in metric_names:
                    subgroups.append(
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
        "counts": counts,
        "bootstrap": bootstrap,
        "confidence_intervals": intervals,
        "paired_differences": paired,
        "length_subgroups": subgroups,
    }


def validate_frozen_inputs(
    *,
    prediction_freeze_path: Path,
    statistical_contract_path: Path,
    endpoint_contract_path: Path,
    cohort_contract_path: Path,
    evaluator_contract_path: Path,
    labels_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate identities without parsing protected labels or predictions."""

    statistical = load_json_contract(
        statistical_contract_path,
        identity_field="contract_sha256",
        expected_identity=EXPECTED_STATISTICAL_CONTRACT_SHA256,
        description="statistical contract",
    )
    endpoint = load_json_contract(
        endpoint_contract_path,
        identity_field="contract_sha256",
        expected_identity=EXPECTED_ENDPOINT_CONTRACT_SHA256,
        description="endpoint contract",
    )
    cohort = load_json_contract(
        cohort_contract_path,
        identity_field="contract_sha256",
        expected_identity=EXPECTED_COHORT_CONTRACT_SHA256,
        description="cohort contract",
    )
    evaluator = load_json_contract(
        evaluator_contract_path,
        identity_field="contract_sha256",
        expected_identity=EXPECTED_EVALUATOR_CONTRACT_SHA256,
        description="evaluator contract",
    )
    freeze = load_json_contract(
        prediction_freeze_path,
        identity_field="prediction_freeze_sha256",
        expected_identity=EXPECTED_PREDICTION_FREEZE_SHA256,
        description="prediction freeze",
    )
    if statistical.get("uncertainty", {}).get("replicates") != EXPECTED_BOOTSTRAP_REPLICATES:
        raise RuntimeError("Frozen bootstrap replicate count changed")
    if statistical.get("uncertainty", {}).get("seed") != EXPECTED_BOOTSTRAP_SEED:
        raise RuntimeError("Frozen bootstrap seed changed")
    if endpoint.get("status") != "FROZEN" or cohort.get("status") != "FROZEN":
        raise RuntimeError("An upstream contract is not frozen")
    if evaluator.get("status") != "FROZEN":
        raise RuntimeError("Evaluator contract is not frozen")
    if sha256_file(labels_path) != EXPECTED_SEALED_LABEL_MAP_SHA256:
        raise RuntimeError("Sealed label-map identity changed")
    tools = freeze.get("tools")
    if not isinstance(tools, dict) or set(tools) != set(BINARY_TOOLS):
        raise RuntimeError("Prediction-freeze tool set changed")
    prediction_paths: dict[str, Path] = {}
    for tool in BINARY_TOOLS:
        state = tools[tool]
        if state.get("status") != "COMPLETE":
            raise RuntimeError(f"Frozen predictor is incomplete: {tool}")
        validation = state.get("validation", {})
        prediction_path = Path(str(validation.get("standardized_path", "")))
        if not prediction_path.is_absolute():
            prediction_path = prediction_freeze_path.parents[4] / prediction_path
        expected_hash = str(validation.get("standardized_sha256", ""))
        if not prediction_path.is_file() or sha256_file(prediction_path) != expected_hash:
            raise RuntimeError(f"Frozen standardized prediction identity changed: {tool}")
        prediction_paths[tool] = prediction_path
    return freeze, prediction_paths


def validate_release_authorization(
    path: Path,
    *,
    evaluator_path: Path,
) -> dict[str, Any]:
    """Validate the prospective release gate before protected content is parsed."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "nar-confirmatory-ground-truth-release-authorization-v1":
        raise RuntimeError("Ground-truth release authorization schema changed")
    if value.get("status") != "AUTHORIZED":
        raise RuntimeError("Ground-truth release is not authorized")
    if value.get("cohort_role") != "confirmatory":
        raise RuntimeError("Release authorization is not confirmatory-only")
    expected = {
        "prediction_freeze_sha256": EXPECTED_PREDICTION_FREEZE_SHA256,
        "statistical_contract_sha256": EXPECTED_STATISTICAL_CONTRACT_SHA256,
        "evaluator_contract_sha256": EXPECTED_EVALUATOR_CONTRACT_SHA256,
        "sealed_label_map_sha256": EXPECTED_SEALED_LABEL_MAP_SHA256,
        "evaluator_source_sha256": sha256_file(evaluator_path),
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise RuntimeError(f"Release authorization {field} changed")
    if value.get("ground_truth_performance_label_release_authorized") is not True:
        raise RuntimeError("Ground-truth label release remains prohibited")
    if value.get("performance_evaluation_authorized") is not True:
        raise RuntimeError("Confirmatory performance evaluation remains prohibited")
    declared = str(value.get("authorization_sha256", ""))
    if not declared or canonical_hash(value, "authorization_sha256") != declared:
        raise RuntimeError("Release authorization canonical identity is invalid")
    return value


def _atomic_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _format_float(value: Any) -> Any:
    return f"{float(value):.12g}" if isinstance(value, (float, np.floating)) else value


def write_results(
    output_dir: Path,
    records: Sequence[TruthRecord],
    predictions: Mapping[str, Sequence[str]],
    analysis: Mapping[str, Any],
    *,
    authorization_sha256: str,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Confirmatory evaluation output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    point_rows: list[dict[str, Any]] = []
    for endpoint, tools in analysis["point_estimates"].items():
        scalar_names = THREE_SCALAR_METRICS if endpoint == "three_class" else BINARY_SCALAR_METRICS
        for tool, metrics in tools.items():
            for metric in scalar_names:
                point_rows.append(
                    {
                        "endpoint": endpoint,
                        "tool": TOOL_DISPLAY_NAMES[tool],
                        "metric": metric,
                        "estimate": _format_float(metrics[metric]),
                        "primary": str(metric in PRIMARY_METRICS[endpoint]).lower(),
                    }
                )
    _atomic_tsv(
        output_dir / "point_estimates.tsv",
        ["endpoint", "tool", "metric", "estimate", "primary"],
        point_rows,
    )

    interval_rows = [
        {
            **row,
            "tool": TOOL_DISPLAY_NAMES[row["tool"]],
            "lower": _format_float(row["lower"]),
            "upper": _format_float(row["upper"]),
            "confidence_level": EXPECTED_CONFIDENCE_LEVEL,
        }
        for row in analysis["confidence_intervals"]
    ]
    _atomic_tsv(
        output_dir / "confidence_intervals.tsv",
        ["endpoint", "tool", "metric", "lower", "upper", "confidence_level"],
        interval_rows,
    )

    paired_rows = []
    for original in analysis["paired_differences"]:
        row = dict(original)
        row["target"] = TOOL_DISPLAY_NAMES[row["target"]]
        row["comparator"] = TOOL_DISPLAY_NAMES[row["comparator"]]
        for field in (
            "target_estimate",
            "comparator_estimate",
            "difference",
            "lower",
            "upper",
            "p_value",
            "holm_adjusted_p",
        ):
            row[field] = _format_float(row[field])
        row["superiority_supported"] = str(row["superiority_supported"]).lower()
        paired_rows.append(row)
    _atomic_tsv(
        output_dir / "paired_primary_differences.tsv",
        [
            "endpoint",
            "metric",
            "target",
            "comparator",
            "target_estimate",
            "comparator_estimate",
            "difference",
            "lower",
            "upper",
            "p_value",
            "holm_adjusted_p",
            "superiority_supported",
        ],
        paired_rows,
    )

    per_class_rows: list[dict[str, Any]] = []
    for tool, metrics in analysis["point_estimates"]["three_class"].items():
        for class_name in THREE_CLASSES:
            per_class_rows.append(
                {
                    "tool": TOOL_DISPLAY_NAMES[tool],
                    "class": class_name,
                    "precision": _format_float(metrics[f"precision_{class_name}"]),
                    "recall": _format_float(metrics[f"recall_{class_name}"]),
                    "f1": _format_float(metrics[f"f1_{class_name}"]),
                }
            )
    _atomic_tsv(
        output_dir / "three_class_per_class_metrics.tsv",
        ["tool", "class", "precision", "recall", "f1"],
        per_class_rows,
    )

    count_rows: list[dict[str, Any]] = []
    for endpoint, tools in analysis["counts"].items():
        truth_labels = THREE_CLASSES if endpoint == "three_class" else BINARY_TRUTH
        prediction_labels = THREE_PREDICTIONS if endpoint == "three_class" else BINARY_PREDICTIONS
        for tool, counts in tools.items():
            for truth_index, truth_label in enumerate(truth_labels):
                for prediction_index, prediction_label in enumerate(prediction_labels):
                    count_rows.append(
                        {
                            "endpoint": endpoint,
                            "tool": TOOL_DISPLAY_NAMES[tool],
                            "truth": truth_label,
                            "prediction": prediction_label,
                            "count": int(counts[truth_index, prediction_index]),
                        }
                    )
    _atomic_tsv(
        output_dir / "raw_confusion_counts.tsv",
        ["endpoint", "tool", "truth", "prediction", "count"],
        count_rows,
    )

    subgroup_rows = [
        {
            **row,
            "tool": TOOL_DISPLAY_NAMES[row["tool"]],
            "estimate": _format_float(row["estimate"]),
        }
        for row in analysis["length_subgroups"]
    ]
    _atomic_tsv(
        output_dir / "length_bin_descriptive_metrics.tsv",
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

    record_rows = []
    for index, record in enumerate(records):
        record_rows.append(
            {
                "opaque_contig_id": record.contig_id,
                "prediction_order": record.prediction_order,
                "truth": record.truth,
                "length_bin": record.length_bin,
                "selected_primary_accession": record.source_cluster,
                "selected_fragment_length_bp": record.length_bp,
                **{f"pred_{tool}": predictions[tool][index] for tool in BINARY_TOOLS},
            }
        )
    _atomic_tsv(
        output_dir / "record_level_confirmatory_audit.tsv",
        [
            "opaque_contig_id",
            "prediction_order",
            "truth",
            "length_bin",
            "selected_primary_accession",
            "selected_fragment_length_bp",
            *(f"pred_{tool}" for tool in BINARY_TOOLS),
        ],
        record_rows,
    )

    result_files = sorted(output_dir.glob("*.tsv"))
    artifacts = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in result_files
    ]
    manifest: dict[str, Any] = {
        "schema_version": "nar-confirmatory-evaluation-results-v1",
        "status": "FROZEN",
        "cohort_role": "confirmatory",
        "records": len(records),
        "source_clusters": len({record.source_cluster for record in records}),
        "prediction_freeze_sha256": EXPECTED_PREDICTION_FREEZE_SHA256,
        "statistical_contract_sha256": EXPECTED_STATISTICAL_CONTRACT_SHA256,
        "evaluator_contract_sha256": EXPECTED_EVALUATOR_CONTRACT_SHA256,
        "ground_truth_release_authorization_sha256": authorization_sha256,
        "bootstrap_replicates": EXPECTED_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
        "artifacts": artifacts,
    }
    manifest["results_freeze_sha256"] = canonical_hash(manifest, "results_freeze_sha256")
    _atomic_json(output_dir / "confirmatory_evaluation_results_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen external confirmatory cohort under its prospectively frozen "
            "source-cluster bootstrap and multiplicity policy. Labels remain inaccessible unless "
            "an exact release authorization is supplied."
        )
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prediction-freeze", type=Path, required=True)
    parser.add_argument("--statistical-contract", type=Path, required=True)
    parser.add_argument("--endpoint-contract", type=Path, required=True)
    parser.add_argument("--cohort-contract", type=Path, required=True)
    parser.add_argument("--evaluator-contract", type=Path, required=True)
    parser.add_argument("--release-authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort-role", choices=["confirmatory"], required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cohort_role != "confirmatory":
        raise RuntimeError("The dedicated evaluator only accepts cohort_role=confirmatory")
    _, prediction_paths = validate_frozen_inputs(
        prediction_freeze_path=args.prediction_freeze.resolve(),
        statistical_contract_path=args.statistical_contract.resolve(),
        endpoint_contract_path=args.endpoint_contract.resolve(),
        cohort_contract_path=args.cohort_contract.resolve(),
        evaluator_contract_path=args.evaluator_contract.resolve(),
        labels_path=args.labels.resolve(),
    )
    authorization = validate_release_authorization(
        args.release_authorization.resolve(), evaluator_path=Path(__file__).resolve()
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("Confirmatory evaluation output directory is not empty")

    # Protected content is not opened until every frozen identity and authorization passes.
    records = load_truth_records(args.labels.resolve())
    record_ids = [record.contig_id for record in records]
    predictions: dict[str, list[str]] = {}
    for tool in BINARY_TOOLS:
        allowed = set(THREE_PREDICTIONS if tool in THREE_CLASS_TOOLS else BINARY_PREDICTIONS)
        predictions[tool] = load_predictions(
            prediction_paths[tool], record_ids, allowed_labels=allowed, tool=tool
        )
    analysis = evaluate_records(
        records,
        predictions,
        bootstrap_replicates=EXPECTED_BOOTSTRAP_REPLICATES,
        bootstrap_seed=EXPECTED_BOOTSTRAP_SEED,
    )
    write_results(
        args.output_dir.resolve(),
        records,
        predictions,
        analysis,
        authorization_sha256=str(authorization["authorization_sha256"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
