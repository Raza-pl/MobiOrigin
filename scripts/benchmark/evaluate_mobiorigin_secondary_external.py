#!/usr/bin/env python3
"""Evaluate frozen post-hoc secondary external plasmid classifiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_RECORDS = 3000
EXPECTED_CONTRACT_SHA256 = (
    "7b113245479ac77634c5a3985324087c8fa91f192d5256954c47295a211da6c6"
)
EXPECTED_LABEL_SHA256 = (
    "6099e798b581028da63e11dacb5bc03eb16e0da86226d72fcc14bf575c4c0fd3"
)
EXPECTED_PREDICTION_SHA256 = {
    "mobiorigin": "d0b225830d7e89445c527ba4dd92843a4954d88b05eab153035ba30858bb789c",
    "plasclass": "0ffe93a65638dd9f59e2bc6795a3ffde06694ec18a40478f1989068ff2ff4c59",
    "plasflow_v1": "c929fae93ddc3468e49cdcda18d43f3ad2cd267cf01b93ddfc6c5bcfe59e28b9",
    "plasme": "560865153d1f5fd77acf981e6f17456dea8e8a2b86ce2fe24465c5b97ef24bbc",
    "platon": "ff190851f2f4444c9b9d529a80b260ed00eb7e9249c1c100343309e37890fbd6",
}
TOOLS = ("mobiorigin", "plasclass", "plasflow_v1", "plasme", "platon")
COMPARATORS = TOOLS[1:]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_822
VALID_TRUTH = {"chromosome", "plasmid", "phage"}
VALID_PREDICTIONS = {
    "chromosome",
    "plasmid",
    "phage",
    "non-plasmid",
    "unclassified",
}


@dataclass(frozen=True)
class TruthRecord:
    identifier: str
    order: int
    truth: str
    source_accession: str


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
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")


def load_truth(path: Path, expected: int = EXPECTED_RECORDS) -> list[TruthRecord]:
    records: list[TruthRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_fields(
            reader.fieldnames,
            {"prediction_order", "opaque_contig_id", "class", "source_accession"},
            "sealed label map",
        )
        for line, row in enumerate(reader, 2):
            identifier = row["opaque_contig_id"].strip()
            truth = row["class"].strip()
            source = row["source_accession"].strip()
            try:
                order = int(row["prediction_order"])
            except ValueError as error:
                raise ValueError(f"Invalid prediction order at row {line}") from error
            if not identifier or identifier in seen:
                raise ValueError(f"Invalid or duplicate truth identifier at row {line}")
            if truth not in VALID_TRUTH or not source:
                raise ValueError(f"Invalid biological truth at row {line}")
            seen.add(identifier)
            records.append(TruthRecord(identifier, order, truth, source))
    if len(records) != expected:
        raise ValueError(f"Expected {expected} truth rows, observed {len(records)}")
    if [record.order for record in records] != list(range(1, expected + 1)):
        raise ValueError("Truth rows are not in frozen prediction order")
    return records


def load_mobiorigin(path: Path, identifiers: Sequence[str]) -> list[str]:
    observed: list[str] = []
    predictions: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_fields(reader.fieldnames, {"sequence_id", "prediction"}, "MobiOrigin")
        for line, row in enumerate(reader, 2):
            identifier = row["sequence_id"].strip()
            label = row["prediction"].strip()
            if label not in VALID_PREDICTIONS:
                raise ValueError(f"Invalid MobiOrigin label at row {line}: {label!r}")
            observed.append(identifier)
            predictions.append(label)
    if observed != list(identifiers):
        raise ValueError("MobiOrigin identifiers are incomplete or out of order")
    return predictions


def load_secondary(path: Path, identifiers: Sequence[str], name: str) -> list[str]:
    observed: list[str] = []
    predictions: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_fields(
            reader.fieldnames,
            {"contig_id", "predicted_label", "prediction_status"},
            name,
        )
        for line, row in enumerate(reader, 2):
            identifier = row["contig_id"].strip()
            label = row["predicted_label"].strip()
            status = row["prediction_status"].strip()
            if label not in VALID_PREDICTIONS or not status:
                raise ValueError(f"Invalid {name} prediction at row {line}")
            observed.append(identifier)
            predictions.append(label)
    if observed != list(identifiers):
        raise ValueError(f"{name} identifiers are incomplete or out of order")
    return predictions


def binary_counts(truth: Sequence[str], predictions: Sequence[str]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "abstentions": 0}
    for expected, observed in zip(truth, predictions, strict=True):
        positive = expected == "plasmid"
        predicted_positive = observed == "plasmid"
        if observed == "unclassified":
            counts["abstentions"] += 1
        if positive and predicted_positive:
            counts["tp"] += 1
        elif positive:
            counts["fn"] += 1
        elif predicted_positive:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return counts


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(truth: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    counts = binary_counts(truth, predictions)
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    precision = safe_ratio(tp, tp + fp)
    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    f1 = safe_ratio(2 * precision * sensitivity, precision + sensitivity)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_ratio(tp * tn - fp * fn, denominator)
    total = len(truth)
    return {
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "mcc": mcc,
        "prediction_coverage": safe_ratio(total - counts["abstentions"], total),
        "unclassified_fraction": safe_ratio(counts["abstentions"], total),
    }


def grouped_bootstrap_indices(
    sources: Sequence[str], replicates: int, seed: int
) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(sources):
        groups[source].append(index)
    ordered = sorted(groups)
    members = [np.asarray(groups[source], dtype=np.int64) for source in ordered]
    rng = np.random.default_rng(seed)
    results: list[np.ndarray] = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(members), size=len(members))
        results.append(np.concatenate([members[int(value)] for value in sampled]))
    return results


def paired_bootstrap(
    truth: Sequence[str],
    sources: Sequence[str],
    predictions: Mapping[str, Sequence[str]],
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, float]]]:
    truth_array = np.asarray(truth, dtype=object)
    prediction_arrays = {
        name: np.asarray(values, dtype=object) for name, values in predictions.items()
    }
    distributions: dict[str, dict[str, list[float]]] = {
        comparator: {"f1": [], "balanced_accuracy": []}
        for comparator in COMPARATORS
    }
    for indices in grouped_bootstrap_indices(sources, replicates, seed):
        sampled_truth = truth_array[indices].tolist()
        sampled_mob = prediction_arrays["mobiorigin"][indices].tolist()
        mob_metrics = binary_metrics(sampled_truth, sampled_mob)
        for comparator in COMPARATORS:
            candidate = binary_metrics(
                sampled_truth, prediction_arrays[comparator][indices].tolist()
            )
            for endpoint in ("f1", "balanced_accuracy"):
                distributions[comparator][endpoint].append(
                    mob_metrics[endpoint] - candidate[endpoint]
                )
    p_values: dict[str, dict[str, float]] = {}
    for comparator, endpoint_values in distributions.items():
        p_values[comparator] = {}
        for endpoint, values in endpoint_values.items():
            array = np.asarray(values, dtype=np.float64)
            nonpositive = (float(np.count_nonzero(array <= 0)) + 1.0) / (
                len(array) + 1.0
            )
            nonnegative = (float(np.count_nonzero(array >= 0)) + 1.0) / (
                len(array) + 1.0
            )
            p_values[comparator][endpoint] = min(1.0, 2.0 * min(nonpositive, nonnegative))
    return distributions, p_values


def holm_adjust(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def evaluate_arrays(
    truth: Sequence[str],
    sources: Sequence[str],
    predictions: Mapping[str, Sequence[str]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if set(predictions) != set(TOOLS):
        raise ValueError("Frozen tool set changed")
    if not truth or len(truth) != len(sources):
        raise ValueError("Truth and source arrays are empty or misaligned")
    if any(len(values) != len(truth) for values in predictions.values()):
        raise ValueError("Prediction arrays are misaligned")
    points = {name: binary_metrics(truth, predictions[name]) for name in TOOLS}
    distributions, raw_p = paired_bootstrap(
        truth, sources, predictions, replicates, seed
    )
    family = {
        f"{comparator}:{endpoint}": value
        for comparator, endpoints in raw_p.items()
        for endpoint, value in endpoints.items()
    }
    adjusted = holm_adjust(family)
    comparisons: dict[str, Any] = {}
    for comparator in COMPARATORS:
        comparisons[comparator] = {}
        for endpoint in ("f1", "balanced_accuracy"):
            values = distributions[comparator][endpoint]
            low, high = percentile_interval(values)
            key = f"{comparator}:{endpoint}"
            difference = points["mobiorigin"][endpoint] - points[comparator][endpoint]
            comparisons[comparator][endpoint] = {
                "difference": difference,
                "confidence_interval_95": [low, high],
                "raw_p_value_two_sided": raw_p[comparator][endpoint],
                "holm_adjusted_p_value": adjusted[key],
                "exploratory_superiority": low > 0 and adjusted[key] < 0.05,
            }
    return {
        "analysis_role": "post_hoc_secondary_external_comparison",
        "records": len(truth),
        "point_metrics": points,
        "paired_comparisons": comparisons,
        "bootstrap": {
            "unit": "source_accession",
            "replicates": replicates,
            "seed": seed,
            "multiplicity": "Holm across 8 paired tests",
        },
    }


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
    if not isinstance(value, dict):
        raise RuntimeError("Evaluation authorization must be a JSON object")
    recorded = value.get("authorization_sha256")
    if not isinstance(recorded, str) or canonical_hash(value, "authorization_sha256") != recorded:
        raise RuntimeError("Evaluation authorization canonical identity is invalid")
    required = {
        "status": "AUTHORIZED",
        "analysis_role": "post_hoc_secondary_external_comparison",
        "evaluation_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "execution_passes_authorized": 1,
        "retrospective_tuning_authorized": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RuntimeError(f"Evaluation authorization mismatch: {key}")
    return value


def run_evaluation(arguments: argparse.Namespace) -> int:
    paths = {
        "mobiorigin": arguments.mobiorigin,
        "plasclass": arguments.plasclass,
        "plasflow_v1": arguments.plasflow_v1,
        "plasme": arguments.plasme,
        "platon": arguments.platon,
    }
    verify_authorization(arguments.authorization)
    if arguments.output_dir.exists():
        raise RuntimeError("Fresh output directory required")
    if sha256_file(arguments.labels) != EXPECTED_LABEL_SHA256:
        raise RuntimeError("Sealed label identity changed")
    for name, path in paths.items():
        if sha256_file(path) != EXPECTED_PREDICTION_SHA256[name]:
            raise RuntimeError(f"Frozen prediction identity changed: {name}")

    truth_records = load_truth(arguments.labels)
    identifiers = [record.identifier for record in truth_records]
    predictions = {
        "mobiorigin": load_mobiorigin(paths["mobiorigin"], identifiers),
        **{
            name: load_secondary(paths[name], identifiers, name)
            for name in COMPARATORS
        },
    }
    result = evaluate_arrays(
        [record.truth for record in truth_records],
        [record.source_accession for record in truth_records],
        predictions,
    )
    result["schema_version"] = "mobiorigin-secondary-external-evaluation-v1"
    result["evaluation_contract_sha256"] = EXPECTED_CONTRACT_SHA256
    result["authorization_sha256"] = json.loads(
        arguments.authorization.read_text(encoding="utf-8")
    )["authorization_sha256"]
    result["input_sha256"] = {
        "labels": EXPECTED_LABEL_SHA256,
        **EXPECTED_PREDICTION_SHA256,
    }
    result["result_sha256"] = canonical_hash(result, "result_sha256")

    arguments.output_dir.mkdir(parents=True)
    atomic_text(
        arguments.output_dir / "secondary_comparator_evaluation.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    rows: list[dict[str, Any]] = []
    for name in TOOLS:
        rows.append({"tool": name, **result["point_metrics"][name]})
    buffer = [
        "tool\tprecision\tsensitivity\tspecificity\tf1\tbalanced_accuracy\tmcc\tprediction_coverage\tunclassified_fraction"
    ]
    for row in rows:
        buffer.append(
            "\t".join(
                [str(row["tool"])]
                + [
                    f"{float(row[field]):.9f}"
                    for field in (
                        "precision",
                        "sensitivity",
                        "specificity",
                        "f1",
                        "balanced_accuracy",
                        "mcc",
                        "prediction_coverage",
                        "unclassified_fraction",
                    )
                ]
            )
        )
    atomic_text(arguments.output_dir / "secondary_comparator_metrics.tsv", "\n".join(buffer) + "\n")
    print("Secondary external comparison complete and frozen.")
    return 0


def self_test() -> int:
    truth = ["plasmid"] * 4 + ["chromosome"] * 4 + ["phage"] * 4
    sources = [f"source_{index:02d}" for index in range(12)]
    predictions = {
        "mobiorigin": ["plasmid"] * 4 + ["chromosome"] * 4 + ["phage"] * 4,
        "plasclass": ["plasmid"] * 3 + ["non-plasmid"] + ["non-plasmid"] * 8,
        "plasflow_v1": ["plasmid"] * 3 + ["unclassified"] + ["non-plasmid"] * 8,
        "plasme": ["plasmid"] * 2 + ["non-plasmid"] * 10,
        "platon": ["plasmid"] * 3 + ["non-plasmid"] + ["unclassified"] * 2 + ["non-plasmid"] * 6,
    }
    first = evaluate_arrays(truth, sources, predictions, replicates=200, seed=17)
    second = evaluate_arrays(truth, sources, predictions, replicates=200, seed=17)
    if first != second:
        raise RuntimeError("Synthetic evaluation is not deterministic")
    if first["point_metrics"]["mobiorigin"]["f1"] != 1.0:
        raise RuntimeError("Synthetic perfect MobiOrigin F1 was not reproduced")
    if len(first["paired_comparisons"]) != 4:
        raise RuntimeError("Synthetic comparator family was not reproduced")
    if first["point_metrics"]["platon"]["prediction_coverage"] != 10 / 12:
        raise RuntimeError("Synthetic abstention coverage was not reproduced")
    print("MobiOrigin secondary external evaluator synthetic self-test: PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--mobiorigin", type=Path)
    parser.add_argument("--plasclass", type=Path)
    parser.add_argument("--plasflow-v1", dest="plasflow_v1", type=Path)
    parser.add_argument("--plasme", type=Path)
    parser.add_argument("--platon", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    if not arguments.self_test:
        required = (
            "authorization",
            "labels",
            "mobiorigin",
            "plasclass",
            "plasflow_v1",
            "plasme",
            "platon",
            "output_dir",
        )
        missing = [name for name in required if getattr(arguments, name) is None]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")
    return arguments


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(self_test() if args.self_test else run_evaluation(args))
