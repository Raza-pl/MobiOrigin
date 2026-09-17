#!/usr/bin/env python3
"""Test whether the paired MobiOrigin--geNomad difference varies by length."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

LENGTH_BINS = (
    "1k_to_lt2k",
    "2k_to_lt5k",
    "5k_to_lt10k",
    "10k_to_lt50k",
    "50k_to_500k",
)
LENGTH_LABELS = ("1-<2 kb", "2-<5 kb", "5-<10 kb", "10-<50 kb", "50-500 kb")
CLASS_TO_CODE = {"chromosome": 0, "plasmid": 1, "phage": 2, "unclassified": 3}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def encode(labels: list[str]) -> np.ndarray:
    return np.asarray([CLASS_TO_CODE[label] for label in labels], dtype=np.int8)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    f1_values: list[float] = []
    plasmid_f1 = 0.0
    for class_code in range(3):
        true_class = truth == class_code
        predicted_class = prediction == class_code
        tp = int(np.count_nonzero(true_class & predicted_class))
        fp = int(np.count_nonzero(~true_class & predicted_class))
        fn = int(np.count_nonzero(true_class & ~predicted_class))
        denominator = 2 * tp + fp + fn
        f1 = 0.0 if denominator == 0 else 2 * tp / denominator
        f1_values.append(f1)
        if class_code == CLASS_TO_CODE["plasmid"]:
            plasmid_f1 = f1
    accuracy = float(np.mean(prediction == truth))
    return float(np.mean(f1_values)), plasmid_f1, accuracy


def interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def load_inputs(
    root: Path,
    mobiorigin_predictions: Path | None = None,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    external = root / "evaluation/release_audit/mobiorigin_external_validation_20260817"
    truth_path = (
        external
        / "024_corrected_final_external_cohort_freeze"
        / "mobiorigin_external_validation_label_map.sealed.tsv"
    )
    mobiorigin_path = (
        mobiorigin_predictions.resolve()
        if mobiorigin_predictions is not None
        else external
        / "029_dual_external_prediction_execution"
        / "mobiorigin_prediction_output/predictions.tsv"
    )
    genomad_path = (
        external
        / "029_dual_external_prediction_execution"
        / "genomad_prediction_output/standardized_predictions.tsv"
    )
    for path in (truth_path, mobiorigin_path, genomad_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required input: {path}")
    truth_rows = read_tsv(truth_path)
    identifiers = [row["opaque_contig_id"] for row in truth_rows]
    truth = encode([row["class"] for row in truth_rows])
    mob_rows = read_tsv(mobiorigin_path)
    geo_rows = read_tsv(genomad_path)
    mob_by_id = {row["sequence_id"]: row["prediction"] for row in mob_rows}
    geo_by_id = {row["contig_id"]: row["predicted_label"] for row in geo_rows}
    if (
        len(truth_rows) != 3000
        or set(mob_by_id) != set(identifiers)
        or set(geo_by_id) != set(identifiers)
    ):
        raise RuntimeError("External-cohort identity or completeness changed")
    mobiorigin = encode([mob_by_id[identifier] for identifier in identifiers])
    genomad = encode([geo_by_id[identifier] for identifier in identifiers])
    return truth_rows, truth, mobiorigin, genomad


def make_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 1900, 850
    top, panel_height = 130, 540
    panel_width = 700
    lefts = (150, 1050)

    def x(value: float, left: float) -> float:
        return left + 60 + value * (panel_width - 120) / 4.0

    def y_metric(value: float) -> float:
        return top + panel_height - (value - 0.35) / 0.65 * panel_height

    def y_delta(value: float) -> float:
        return top + panel_height - (value + 0.20) / 0.40 * panel_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.axis{stroke:#222;stroke-width:2}.grid{stroke:#ddd;stroke-width:1}</style>",
        '<text x="950" y="48" text-anchor="middle" font-size="32" font-weight="700">Comparative performance varies with sequence length</text>',
    ]
    # Panel A: paired system estimates.
    left = lefts[0]
    parts.extend(
        [
            '<text x="75" y="92" font-size="30" font-weight="700">A</text>',
            f'<text x="{left + panel_width / 2}" y="92" text-anchor="middle" font-size="25" font-weight="700">Macro-F1 by length</text>',
        ]
    )
    for tick in np.arange(0.4, 1.01, 0.1):
        yy = y_metric(float(tick))
        parts.append(
            f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{left + panel_width}" y2="{yy:.1f}"/>'
        )
        parts.append(
            f'<text x="{left - 18}" y="{yy + 7:.1f}" text-anchor="end" font-size="20">{tick:.1f}</text>'
        )
    for key, color, offset in (
        ("mobiorigin_macro_f1", "#0072B2", -10),
        ("genomad_macro_f1", "#D55E00", 10),
    ):
        coords = " ".join(
            f"{x(float(i), left) + offset:.1f},{y_metric(float(rows[i][key])):.1f}"
            for i in range(5)
        )
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="5"/>')
        for i in range(5):
            parts.append(
                f'<circle cx="{x(float(i), left) + offset:.1f}" cy="{y_metric(float(rows[i][key])):.1f}" r="8" fill="{color}"/>'
            )
    # Panel B: paired delta with bootstrap interval.
    left = lefts[1]
    parts.extend(
        [
            '<text x="975" y="92" font-size="30" font-weight="700">B</text>',
            f'<text x="{left + panel_width / 2}" y="92" text-anchor="middle" font-size="25" font-weight="700">Paired macro-F1 difference</text>',
        ]
    )
    for tick in np.arange(-0.2, 0.201, 0.05):
        yy = y_delta(float(tick))
        parts.append(
            f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{left + panel_width}" y2="{yy:.1f}"/>'
        )
        parts.append(
            f'<text x="{left - 18}" y="{yy + 7:.1f}" text-anchor="end" font-size="20">{tick:.2f}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{y_delta(0):.1f}" x2="{left + panel_width}" y2="{y_delta(0):.1f}" stroke="#222" stroke-width="3"/>'
    )
    for i, row in enumerate(rows):
        xx = x(float(i), left)
        low = y_delta(float(row["macro_f1_delta_ci95_lower"]))
        high = y_delta(float(row["macro_f1_delta_ci95_upper"]))
        yy = y_delta(float(row["macro_f1_delta"]))
        color = "#0072B2" if float(row["macro_f1_delta"]) >= 0 else "#D55E00"
        parts.append(
            f'<line x1="{xx:.1f}" y1="{low:.1f}" x2="{xx:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="5"/>'
        )
        parts.append(
            f'<line x1="{xx - 9:.1f}" y1="{low:.1f}" x2="{xx + 9:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="4"/>'
        )
        parts.append(
            f'<line x1="{xx - 9:.1f}" y1="{high:.1f}" x2="{xx + 9:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="4"/>'
        )
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="9" fill="{color}"/>')
    for left in lefts:
        parts.append(
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{left}" y1="{top + panel_height}" x2="{left + panel_width}" y2="{top + panel_height}"/>'
        )
        for i, label in enumerate(LENGTH_LABELS):
            xx = x(float(i), left)
            parts.append(
                f'<text x="{xx:.1f}" y="{top + panel_height + 36}" text-anchor="middle" font-size="18">{html.escape(label)}</text>'
            )
        parts.append(
            f'<text x="{left + panel_width / 2}" y="{top + panel_height + 78}" text-anchor="middle" font-size="23">Sequence length</text>'
        )
    parts.extend(
        [
            '<line x1="485" y1="770" x2="545" y2="770" stroke="#0072B2" stroke-width="5"/><text x="558" y="778" font-size="22">MobiOrigin v0.1.7</text>',
            '<line x1="850" y1="770" x2="910" y2="770" stroke="#D55E00" stroke-width="5"/><text x="923" y="778" font-size="22">geNomad calibrated</text>',
            '<text x="1400" y="778" text-anchor="middle" font-size="19">Positive difference favors MobiOrigin</text>',
            '<text x="950" y="826" text-anchor="middle" font-size="18" fill="#444">Intervals are paired, class-stratified bootstrap 95% confidence intervals within each prespecified length stratum.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


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
        raise RuntimeError("Use at least 1,000 replicates")

    root = args.root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    truth_rows, truth, mobiorigin, genomad = load_inputs(
        root,
        args.mobiorigin_predictions,
    )
    bin_indices = {
        length_bin: np.asarray(
            [index for index, row in enumerate(truth_rows) if row["length_bin"] == length_bin],
            dtype=np.int32,
        )
        for length_bin in LENGTH_BINS
    }
    if any(len(indices) != 600 for indices in bin_indices.values()):
        raise RuntimeError("Expected 600 records per length stratum")

    rng = np.random.default_rng(args.seed)
    result_rows: list[dict[str, object]] = []
    for length_bin, label in zip(LENGTH_BINS, LENGTH_LABELS):
        indices = bin_indices[length_bin]
        bin_truth = truth[indices]
        bin_mob = mobiorigin[indices]
        bin_geo = genomad[indices]
        mob_metrics = metrics(bin_truth, bin_mob)
        geo_metrics = metrics(bin_truth, bin_geo)
        bootstrap_delta = np.empty((args.bootstrap, 3), dtype=float)
        class_cells = [indices[truth[indices] == class_code] for class_code in range(3)]
        if any(len(cell) != 200 for cell in class_cells):
            raise RuntimeError(f"Unbalanced class cell in {length_bin}")
        for replicate in range(args.bootstrap):
            sample = np.concatenate(
                [rng.choice(cell, size=len(cell), replace=True) for cell in class_cells]
            )
            bootstrap_delta[replicate] = np.asarray(
                metrics(truth[sample], mobiorigin[sample])
            ) - np.asarray(metrics(truth[sample], genomad[sample]))
        macro_interval = interval(bootstrap_delta[:, 0])
        plasmid_interval = interval(bootstrap_delta[:, 1])
        accuracy_interval = interval(bootstrap_delta[:, 2])
        result_rows.append(
            {
                "length_bin": length_bin,
                "display_label": label,
                "records": len(indices),
                "mobiorigin_macro_f1": mob_metrics[0],
                "genomad_macro_f1": geo_metrics[0],
                "macro_f1_delta": mob_metrics[0] - geo_metrics[0],
                "macro_f1_delta_ci95_lower": macro_interval[0],
                "macro_f1_delta_ci95_upper": macro_interval[1],
                "mobiorigin_plasmid_f1": mob_metrics[1],
                "genomad_plasmid_f1": geo_metrics[1],
                "plasmid_f1_delta": mob_metrics[1] - geo_metrics[1],
                "plasmid_f1_delta_ci95_lower": plasmid_interval[0],
                "plasmid_f1_delta_ci95_upper": plasmid_interval[1],
                "mobiorigin_accuracy": mob_metrics[2],
                "genomad_accuracy": geo_metrics[2],
                "accuracy_delta": mob_metrics[2] - geo_metrics[2],
                "accuracy_delta_ci95_lower": accuracy_interval[0],
                "accuracy_delta_ci95_upper": accuracy_interval[1],
            }
        )
    write_tsv(run_dir / "length_stratified_paired_metrics.tsv", result_rows, list(result_rows[0]))

    # Global interaction: is the paired correctness advantage heterogeneous by length?
    paired_correctness = (mobiorigin == truth).astype(float) - (genomad == truth).astype(float)
    observed_bin_means = np.asarray(
        [float(np.mean(paired_correctness[bin_indices[length_bin]])) for length_bin in LENGTH_BINS]
    )
    observed_global = float(np.mean(paired_correctness))
    observed_statistic = float(600 * np.sum((observed_bin_means - observed_global) ** 2))
    length_codes = np.asarray([LENGTH_BINS.index(row["length_bin"]) for row in truth_rows])
    permuted_statistics = np.empty(args.permutations, dtype=float)
    class_indices = [np.flatnonzero(truth == class_code) for class_code in range(3)]
    for replicate in range(args.permutations):
        permuted_codes = length_codes.copy()
        for indices in class_indices:
            permuted_codes[indices] = rng.permutation(permuted_codes[indices])
        means = np.asarray(
            [float(np.mean(paired_correctness[permuted_codes == code])) for code in range(5)]
        )
        permuted_statistics[replicate] = 600 * np.sum((means - observed_global) ** 2)
    interaction_p = (1 + int(np.count_nonzero(permuted_statistics >= observed_statistic))) / (
        args.permutations + 1
    )
    interaction_row = {
        "test": "global_tool_by_length_interaction_on_paired_correctness",
        "statistic": observed_statistic,
        "permutation_p": interaction_p,
        "permutations": args.permutations,
        "stratification": "true class",
    }
    write_tsv(
        run_dir / "length_interaction_test.tsv",
        [interaction_row],
        list(interaction_row),
    )

    # Prespecified shortest-bin contrasts on the paired correctness advantage.
    shortest = bin_indices[LENGTH_BINS[0]]
    contrast_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for other_bin, other_label in zip(LENGTH_BINS[1:], LENGTH_LABELS[1:]):
        other = bin_indices[other_bin]
        observed = float(np.mean(paired_correctness[shortest]) - np.mean(paired_correctness[other]))
        combined = np.concatenate([shortest, other])
        null_values = np.empty(args.permutations, dtype=float)
        for replicate in range(args.permutations):
            first_parts: list[np.ndarray] = []
            second_parts: list[np.ndarray] = []
            for class_code in range(3):
                class_pool = combined[truth[combined] == class_code]
                shuffled = rng.permutation(class_pool)
                first_parts.append(shuffled[:200])
                second_parts.append(shuffled[200:])
            first = np.concatenate(first_parts)
            second = np.concatenate(second_parts)
            null_values[replicate] = float(
                np.mean(paired_correctness[first]) - np.mean(paired_correctness[second])
            )
        p_value = (1 + int(np.count_nonzero(np.abs(null_values) >= abs(observed)))) / (
            args.permutations + 1
        )
        raw_p_values.append(p_value)
        contrast_rows.append(
            {
                "reference_bin": LENGTH_BINS[0],
                "comparison_bin": other_bin,
                "comparison_label": other_label,
                "difference_in_accuracy_advantage": observed,
                "permutation_p": p_value,
                "holm_adjusted_p": 1.0,
                "permutations": args.permutations,
            }
        )
    for row, adjusted in zip(contrast_rows, holm_adjust(raw_p_values)):
        row["holm_adjusted_p"] = adjusted
    write_tsv(
        run_dir / "shortest_bin_interaction_contrasts.tsv", contrast_rows, list(contrast_rows[0])
    )

    contribution_rows: list[dict[str, object]] = []
    total_net = float(np.sum(paired_correctness))
    for length_bin, label in zip(LENGTH_BINS, LENGTH_LABELS):
        indices = bin_indices[length_bin]
        net = float(np.sum(paired_correctness[indices]))
        contribution_rows.append(
            {
                "length_bin": length_bin,
                "display_label": label,
                "mobiorigin_correct_minus_genomad_correct": int(net),
                "contribution_to_total_accuracy_difference": net / len(truth),
                "share_of_net_correct_call_advantage": net / total_net if total_net else 0.0,
            }
        )
    write_tsv(
        run_dir / "length_contribution_to_accuracy_difference.tsv",
        contribution_rows,
        list(contribution_rows[0]),
    )

    figure_path = run_dir / "Figure_S_length_interaction.svg"
    make_svg(figure_path, result_rows)
    report = {
        "status": "PASS",
        "records": len(truth),
        "length_strata": list(LENGTH_BINS),
        "bootstrap_replicates": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
        "global_interaction": interaction_row,
        "interpretation_boundary": "The global interaction tests heterogeneity of paired correctness, while macro-F1 and plasmid-F1 intervals are paired descriptive estimates within each length stratum.",
    }
    (run_dir / "length_interaction_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("===== TOOL-BY-LENGTH ANALYSIS =====")
    for row in result_rows:
        print(
            f"{row['display_label']}: macro-F1 delta={float(row['macro_f1_delta']):.4f} "
            f"(95% CI {float(row['macro_f1_delta_ci95_lower']):.4f} to "
            f"{float(row['macro_f1_delta_ci95_upper']):.4f})"
        )
    print(
        f"Global tool-by-length interaction: statistic={observed_statistic:.4f}, "
        f"permutation p={interaction_p:.6g}"
    )
    print(f"Figure: {figure_path}")
    print(f"Results: {run_dir}")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
