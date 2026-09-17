#!/usr/bin/env python3
"""Bootstrap and paired tests for the MobiOrigin--geNomad comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASS_TO_CODE = {"chromosome": 0, "plasmid": 1, "phage": 2, "unclassified": 3}
SYSTEMS = (
    "mobiorigin_corrected",
    "genomad_calibrated",
    "genomad_uncalibrated",
)
METRICS = (
    "prediction_coverage",
    "macro_f1",
    "balanced_accuracy",
    "plasmid_precision",
    "plasmid_sensitivity",
    "plasmid_f1",
)
PRIMARY_METRICS = ("macro_f1", "plasmid_f1")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def encode(labels: list[str]) -> np.ndarray:
    unknown = sorted(set(labels).difference(CLASS_TO_CODE))
    if unknown:
        raise RuntimeError(f"Unsupported labels: {unknown}")
    return np.asarray([CLASS_TO_CODE[label] for label in labels], dtype=np.int8)


def metric_vector(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    f1_values: list[float] = []
    recalls: list[float] = []
    plasmid_precision = 0.0
    plasmid_recall = 0.0
    plasmid_f1 = 0.0
    for class_code in range(3):
        true_class = truth == class_code
        predicted_class = prediction == class_code
        tp = int(np.count_nonzero(true_class & predicted_class))
        fp = int(np.count_nonzero(~true_class & predicted_class))
        fn = int(np.count_nonzero(true_class & ~predicted_class))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        recalls.append(recall)
        if class_code == CLASS_TO_CODE["plasmid"]:
            plasmid_precision = precision
            plasmid_recall = recall
            plasmid_f1 = f1
    return np.asarray(
        [
            float(np.mean(prediction < CLASS_TO_CODE["unclassified"])),
            float(np.mean(f1_values)),
            float(np.mean(recalls)),
            plasmid_precision,
            plasmid_recall,
            plasmid_f1,
        ],
        dtype=float,
    )


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def exact_mcnemar(truth: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left_correct = left == truth
    right_correct = right == truth
    left_only = int(np.count_nonzero(left_correct & ~right_correct))
    right_only = int(np.count_nonzero(~left_correct & right_correct))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_correct_right_incorrect": left_only,
        "left_incorrect_right_correct": right_only,
        "discordant_records": discordant,
        "exact_two_sided_p": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted P values."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank_index in range(total):
        original_index = order[rank_index]
        running = max(running, p_values[original_index] * (total - rank_index))
        adjusted[original_index] = min(1.0, running)
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--mobiorigin-predictions", type=Path)
    args = parser.parse_args()

    if args.bootstrap < 1000 or args.permutations < 1000:
        raise RuntimeError("Use at least 1,000 bootstrap replicates and permutations")

    root = args.root.resolve()
    run_dir = args.run_dir.resolve()
    external = root / "evaluation/release_audit/mobiorigin_external_validation_20260817"
    truth_path = (
        external
        / "024_corrected_final_external_cohort_freeze"
        / "mobiorigin_external_validation_label_map.sealed.tsv"
    )
    prediction_paths = {
        "mobiorigin_corrected": (
            args.mobiorigin_predictions.resolve()
            if args.mobiorigin_predictions is not None
            else external
            / "029_dual_external_prediction_execution"
            / "mobiorigin_prediction_output/predictions.tsv"
        ),
        "genomad_calibrated": (
            external
            / "029_dual_external_prediction_execution"
            / "genomad_prediction_output/standardized_predictions.tsv"
        ),
        "genomad_uncalibrated": run_dir / "genomad_uncalibrated_standardized.tsv",
    }
    for path in (truth_path, *prediction_paths.values()):
        if not path.is_file():
            raise RuntimeError(f"Missing required input: {path}")

    truth_rows = read_tsv(truth_path)
    if len(truth_rows) != 3000:
        raise RuntimeError(f"Expected 3,000 records, observed {len(truth_rows)}")
    identifiers = [row["opaque_contig_id"] for row in truth_rows]
    truth = encode([row["class"] for row in truth_rows])

    predictions: dict[str, np.ndarray] = {}
    field_pairs = {
        "mobiorigin_corrected": ("sequence_id", "prediction"),
        "genomad_calibrated": ("contig_id", "predicted_label"),
        "genomad_uncalibrated": ("sequence_id", "predicted_label"),
    }
    for system in SYSTEMS:
        id_field, label_field = field_pairs[system]
        rows = read_tsv(prediction_paths[system])
        by_id = {row[id_field]: row[label_field] for row in rows}
        if len(by_id) != len(identifiers) or set(by_id) != set(identifiers):
            raise RuntimeError(f"Identifier mismatch for {system}")
        predictions[system] = encode([by_id[identifier] for identifier in identifiers])

    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(truth_rows):
        strata[(row["class"], row["length_bin"])].append(index)
    if len(strata) != 15 or any(len(indices) != 200 for indices in strata.values()):
        raise RuntimeError("Expected 15 balanced class-by-length strata of 200 records")

    rng = np.random.default_rng(args.seed)
    observed = {
        system: metric_vector(truth, prediction) for system, prediction in predictions.items()
    }
    boot = {system: np.empty((args.bootstrap, len(METRICS)), dtype=float) for system in SYSTEMS}
    stratum_arrays = [np.asarray(indices, dtype=np.int32) for indices in strata.values()]
    for replicate in range(args.bootstrap):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in stratum_arrays]
        )
        sampled_truth = truth[sample]
        for system in SYSTEMS:
            boot[system][replicate] = metric_vector(sampled_truth, predictions[system][sample])

    summary_rows: list[dict[str, object]] = []
    manuscript_rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        for metric_index, metric_name in enumerate(METRICS):
            lower, upper = percentile_interval(boot[system][:, metric_index])
            estimate = float(observed[system][metric_index])
            summary_rows.append(
                {
                    "system": system,
                    "metric": metric_name,
                    "estimate": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_replicates": args.bootstrap,
                }
            )
            manuscript_rows.append(
                {
                    "system": system,
                    "metric": metric_name,
                    "estimate_95ci": f"{estimate:.2f} ({lower:.2f}-{upper:.2f})",
                }
            )
    write_tsv(
        run_dir / "bootstrap_metric_summary.tsv",
        summary_rows,
        ["system", "metric", "estimate", "ci95_lower", "ci95_upper", "bootstrap_replicates"],
    )
    write_tsv(
        run_dir / "manuscript_ready_metrics.tsv",
        manuscript_rows,
        ["system", "metric", "estimate_95ci"],
    )

    pairs = (
        ("mobiorigin_corrected", "genomad_calibrated"),
        ("mobiorigin_corrected", "genomad_uncalibrated"),
        ("genomad_calibrated", "genomad_uncalibrated"),
    )
    test_rows: list[dict[str, object]] = []
    for left_name, right_name in pairs:
        left = predictions[left_name]
        right = predictions[right_name]
        for metric_name in PRIMARY_METRICS:
            metric_index = METRICS.index(metric_name)
            observed_difference = float(
                observed[left_name][metric_index] - observed[right_name][metric_index]
            )
            bootstrap_difference = (
                boot[left_name][:, metric_index] - boot[right_name][:, metric_index]
            )
            lower, upper = percentile_interval(bootstrap_difference)
            permuted = np.empty(args.permutations, dtype=float)
            for replicate in range(args.permutations):
                swap = rng.integers(0, 2, size=len(truth), dtype=np.int8).astype(bool)
                permuted_left = np.where(swap, right, left)
                permuted_right = np.where(swap, left, right)
                permuted[replicate] = (
                    metric_vector(truth, permuted_left)[metric_index]
                    - metric_vector(truth, permuted_right)[metric_index]
                )
            p_value = (1 + int(np.count_nonzero(np.abs(permuted) >= abs(observed_difference)))) / (
                args.permutations + 1
            )
            test_rows.append(
                {
                    "left_system": left_name,
                    "right_system": right_name,
                    "metric": metric_name,
                    "difference_left_minus_right": observed_difference,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "paired_permutation_p": p_value,
                    "confirmatory_family": (
                        left_name == "mobiorigin_corrected" and right_name == "genomad_calibrated"
                    ),
                    "holm_adjusted_p": "",
                    "permutations": args.permutations,
                }
            )
    confirmatory_rows = [row for row in test_rows if bool(row["confirmatory_family"])]
    adjusted = holm_adjust([float(row["paired_permutation_p"]) for row in confirmatory_rows])
    for row, adjusted_p in zip(confirmatory_rows, adjusted):
        row["holm_adjusted_p"] = adjusted_p
    write_tsv(
        run_dir / "paired_metric_tests.tsv",
        test_rows,
        [
            "left_system",
            "right_system",
            "metric",
            "difference_left_minus_right",
            "ci95_lower",
            "ci95_upper",
            "paired_permutation_p",
            "confirmatory_family",
            "holm_adjusted_p",
            "permutations",
        ],
    )

    mcnemar_rows: list[dict[str, object]] = []
    for left_name, right_name in pairs:
        result = exact_mcnemar(truth, predictions[left_name], predictions[right_name])
        mcnemar_rows.append({"left_system": left_name, "right_system": right_name, **result})
    write_tsv(
        run_dir / "paired_correctness_mcnemar.tsv",
        mcnemar_rows,
        [
            "left_system",
            "right_system",
            "left_correct_right_incorrect",
            "left_incorrect_right_correct",
            "discordant_records",
            "exact_two_sided_p",
        ],
    )

    report = {
        "status": "PASS",
        "records": len(truth),
        "design": "paired stratified bootstrap preserving 15 true-class-by-length cells",
        "confidence_interval": "two-sided percentile 95%",
        "paired_test": "record-wise label-swap permutation test",
        "multiple_testing": (
            "Holm adjustment across the two prespecified MobiOrigin-versus-calibrated-geNomad "
            "primary endpoints; calibration sensitivity comparisons remain descriptive"
        ),
        "bootstrap_replicates": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
        "systems": {
            system: {metric: float(value) for metric, value in zip(METRICS, observed[system])}
            for system in SYSTEMS
        },
    }
    (run_dir / "bootstrap_comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("===== BOOTSTRAP AND PAIRED COMPARISON =====")
    print(f"Records: {len(truth)}")
    print(f"Balanced strata: {len(strata)}")
    print(f"Bootstrap replicates: {args.bootstrap}")
    print(f"Paired permutations: {args.permutations}")
    for row in test_rows:
        print(
            f"{row['left_system']} vs {row['right_system']} | {row['metric']}: "
            f"delta={float(row['difference_left_minus_right']):.4f}, "
            f"95% CI={float(row['ci95_lower']):.4f} to {float(row['ci95_upper']):.4f}, "
            + (
                f"Holm-adjusted P={float(row['holm_adjusted_p']):.4g}"
                if row["holm_adjusted_p"] != ""
                else "descriptive sensitivity comparison"
            )
        )
    print(f"Results: {run_dir}")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
