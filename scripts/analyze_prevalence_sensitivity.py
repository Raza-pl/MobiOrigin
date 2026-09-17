#!/usr/bin/env python3
"""Project plasmid-screening performance across assumed class prevalences."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SYSTEMS = (
    "mobiorigin_v0.1.7",
    "genomad_calibrated",
    "genomad_uncalibrated",
)
COLORS = {
    "mobiorigin_v0.1.7": "#0072B2",
    "genomad_calibrated": "#D55E00",
    "genomad_uncalibrated": "#666666",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plasmid_rates(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    true_positive_class = truth == "plasmid"
    predicted_positive = prediction == "plasmid"
    sensitivity = float(np.mean(predicted_positive[true_positive_class]))
    specificity = float(np.mean(~predicted_positive[~true_positive_class]))
    return sensitivity, specificity


def project(
    sensitivity: np.ndarray, specificity: np.ndarray, prevalence: float
) -> dict[str, np.ndarray]:
    true_positive = sensitivity * prevalence
    false_positive = (1.0 - specificity) * (1.0 - prevalence)
    false_negative = (1.0 - sensitivity) * prevalence
    true_negative = specificity * (1.0 - prevalence)
    ppv = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive, dtype=float),
        where=(true_positive + false_positive) > 0,
    )
    npv = np.divide(
        true_negative,
        true_negative + false_negative,
        out=np.zeros_like(true_negative, dtype=float),
        where=(true_negative + false_negative) > 0,
    )
    f1 = np.divide(
        2.0 * ppv * sensitivity,
        ppv + sensitivity,
        out=np.zeros_like(ppv, dtype=float),
        where=(ppv + sensitivity) > 0,
    )
    return {
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "positive_call_fraction": true_positive + false_positive,
        "false_positive_fraction": false_positive,
        "false_negative_fraction": false_negative,
    }


def interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def load_predictions(
    root: Path, run_dir: Path, mobiorigin_predictions: Path | None = None
) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    external = root / "evaluation/release_audit/mobiorigin_external_validation_20260817"
    truth_path = (
        external
        / "024_corrected_final_external_cohort_freeze"
        / "mobiorigin_external_validation_label_map.sealed.tsv"
    )
    paths = {
        "mobiorigin_v0.1.7": (
            mobiorigin_predictions.resolve()
            if mobiorigin_predictions is not None
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
    for path in (truth_path, *paths.values()):
        if not path.is_file():
            raise RuntimeError(f"Missing required input: {path}")
    truth_rows = read_tsv(truth_path)
    if len(truth_rows) != 3000:
        raise RuntimeError(f"Expected 3,000 records, observed {len(truth_rows)}")
    identifiers = [row["opaque_contig_id"] for row in truth_rows]
    fields = {
        "mobiorigin_v0.1.7": ("sequence_id", "prediction"),
        "genomad_calibrated": ("contig_id", "predicted_label"),
        "genomad_uncalibrated": ("sequence_id", "predicted_label"),
    }
    predictions: dict[str, np.ndarray] = {}
    for system in SYSTEMS:
        id_field, label_field = fields[system]
        rows = read_tsv(paths[system])
        by_id = {row[id_field]: row[label_field] for row in rows}
        if len(by_id) != len(identifiers) or set(by_id) != set(identifiers):
            raise RuntimeError(f"Identifier mismatch for {system}")
        predictions[system] = np.asarray([by_id[identifier] for identifier in identifiers])
    return truth_rows, predictions


def svg_figure(
    path: Path,
    prevalences: np.ndarray,
    point_curves: dict[str, dict[str, np.ndarray]],
    lower_curves: dict[str, dict[str, np.ndarray]],
    upper_curves: dict[str, dict[str, np.ndarray]],
) -> None:
    width, height = 1900, 850
    panel_width, panel_height = 720, 560
    lefts = (160, 1050)
    top = 115
    x_min, x_max = 0.01, 0.35
    y_min, y_max = 0.0, 1.0

    def sx(value: float, left: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * panel_width

    def sy(value: float) -> float:
        return top + panel_height - (value - y_min) / (y_max - y_min) * panel_height

    def points(xs: np.ndarray, ys: np.ndarray, left: float) -> str:
        return " ".join(f"{sx(float(x), left):.1f},{sy(float(y)):.1f}" for x, y in zip(xs, ys))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.axis{stroke:#222;stroke-width:2}.grid{stroke:#dddddd;stroke-width:1}.line{fill:none;stroke-width:5;stroke-linecap:round;stroke-linejoin:round}</style>",
        '<text x="950" y="48" text-anchor="middle" font-size="32" font-weight="700">Plasmid prevalence changes predictive value and F1</text>',
    ]
    for panel_index, (metric, title) in enumerate(
        (("ppv", "Expected plasmid precision (PPV)"), ("f1", "Expected plasmid F1"))
    ):
        left = lefts[panel_index]
        parts.append(
            f'<text x="{left - 75}" y="82" font-size="30" font-weight="700">{chr(65 + panel_index)}</text>'
        )
        parts.append(
            f'<text x="{left + panel_width / 2}" y="88" text-anchor="middle" font-size="25" font-weight="700">{title}</text>'
        )
        for y_tick in np.linspace(0, 1, 6):
            y = sy(float(y_tick))
            parts.append(
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + panel_width}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text x="{left - 18}" y="{y + 8:.1f}" text-anchor="end" font-size="21">{y_tick:.1f}</text>'
            )
        for x_tick in (0.01, 0.05, 0.10, 0.20, 0.30, 0.333):
            x = sx(x_tick, left)
            label = "33" if abs(x_tick - 0.333) < 0.001 else f"{100 * x_tick:.0f}"
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + panel_height}"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{top + panel_height + 34}" text-anchor="middle" font-size="21">{label}</text>'
            )
        benchmark_x = sx(1.0 / 3.0, left)
        parts.append(
            f'<line x1="{benchmark_x:.1f}" y1="{top}" x2="{benchmark_x:.1f}" y2="{top + panel_height}" stroke="#222" stroke-width="2" stroke-dasharray="8 8"/>'
        )
        if panel_index == 1:
            parts.append(
                f'<text x="{benchmark_x - 8:.1f}" y="{top + 28}" text-anchor="end" font-size="18">balanced cohort</text>'
            )
        for system in ("mobiorigin_v0.1.7", "genomad_calibrated"):
            polygon_x = np.concatenate([prevalences, prevalences[::-1]])
            polygon_y = np.concatenate(
                [lower_curves[system][metric], upper_curves[system][metric][::-1]]
            )
            parts.append(
                f'<polygon points="{points(polygon_x, polygon_y, left)}" fill="{COLORS[system]}" fill-opacity="0.12" stroke="none"/>'
            )
        for system in SYSTEMS:
            dash = ' stroke-dasharray="11 9"' if system == "genomad_uncalibrated" else ""
            width_value = 3 if system == "genomad_uncalibrated" else 5
            parts.append(
                f'<polyline class="line" points="{points(prevalences, point_curves[system][metric], left)}" stroke="{COLORS[system]}" stroke-width="{width_value}"{dash}/>'
            )
        parts.append(
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{left}" y1="{top + panel_height}" x2="{left + panel_width}" y2="{top + panel_height}"/>'
        )
        parts.append(
            f'<text x="{left + panel_width / 2}" y="{top + panel_height + 78}" text-anchor="middle" font-size="24">Assumed plasmid prevalence (%)</text>'
        )
    legend_y = 760
    labels = {
        "mobiorigin_v0.1.7": "MobiOrigin v0.1.7",
        "genomad_calibrated": "geNomad calibrated",
        "genomad_uncalibrated": "geNomad uncalibrated",
    }
    starts = (380, 790, 1240)
    for x, system in zip(starts, SYSTEMS):
        dash = ' stroke-dasharray="11 9"' if system == "genomad_uncalibrated" else ""
        parts.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 65}" y2="{legend_y}" stroke="{COLORS[system]}" stroke-width="5"{dash}/>'
        )
        parts.append(
            f'<text x="{x + 78}" y="{legend_y + 8}" font-size="22">{labels[system]}</text>'
        )
    parts.append(
        '<text x="950" y="822" text-anchor="middle" font-size="18" fill="#444">Projection assumes class-conditional sensitivity and specificity remain constant under prevalence shift; shaded regions are bootstrap 95% confidence intervals.</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--mobiorigin-predictions", type=Path)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        raise RuntimeError("Use at least 1,000 bootstrap replicates")

    root = args.root.resolve()
    run_dir = args.run_dir.resolve()
    truth_rows, predictions = load_predictions(root, run_dir, args.mobiorigin_predictions)
    truth = np.asarray([row["class"] for row in truth_rows])

    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(truth_rows):
        strata[(row["class"], row["length_bin"])].append(index)
    if len(strata) != 15 or any(len(indices) != 200 for indices in strata.values()):
        raise RuntimeError("Expected 15 class-by-length strata of 200 records")

    rng = np.random.default_rng(args.seed)
    rates = {system: plasmid_rates(truth, predictions[system]) for system in SYSTEMS}
    bootstrap_rates = {system: np.empty((args.bootstrap, 2), dtype=float) for system in SYSTEMS}
    stratum_arrays = [np.asarray(indices, dtype=np.int32) for indices in strata.values()]
    for replicate in range(args.bootstrap):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in stratum_arrays]
        )
        for system in SYSTEMS:
            bootstrap_rates[system][replicate] = plasmid_rates(
                truth[sample], predictions[system][sample]
            )

    prevalences = np.arange(0.01, 0.351, 0.01)
    point_curves: dict[str, dict[str, np.ndarray]] = {}
    lower_curves: dict[str, dict[str, np.ndarray]] = {}
    upper_curves: dict[str, dict[str, np.ndarray]] = {}
    output_rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        sensitivity, specificity = rates[system]
        point_curves[system] = {key: [] for key in project(np.asarray([sensitivity]), np.asarray([specificity]), 0.1)}  # type: ignore[assignment]
        lower_curves[system] = {key: [] for key in point_curves[system]}  # type: ignore[assignment]
        upper_curves[system] = {key: [] for key in point_curves[system]}  # type: ignore[assignment]
        boot_sensitivity = bootstrap_rates[system][:, 0]
        boot_specificity = bootstrap_rates[system][:, 1]
        for prevalence in prevalences:
            point = project(np.asarray([sensitivity]), np.asarray([specificity]), float(prevalence))
            boot_projected = project(boot_sensitivity, boot_specificity, float(prevalence))
            values: dict[str, float] = {}
            for metric in point:
                estimate = float(point[metric][0])
                lower, upper = interval(boot_projected[metric])
                point_curves[system][metric].append(estimate)  # type: ignore[union-attr]
                lower_curves[system][metric].append(lower)  # type: ignore[union-attr]
                upper_curves[system][metric].append(upper)  # type: ignore[union-attr]
                values[metric] = estimate
                values[f"{metric}_ci95_lower"] = lower
                values[f"{metric}_ci95_upper"] = upper
            output_rows.append(
                {
                    "system": system,
                    "assumed_plasmid_prevalence": float(prevalence),
                    "external_sensitivity": sensitivity,
                    "external_specificity": specificity,
                    **values,
                    "expected_false_positive_calls_per_1000": 1000.0
                    * values["false_positive_fraction"],
                    "expected_missed_plasmids_per_1000": 1000.0 * values["false_negative_fraction"],
                }
            )
        for curve_set in (point_curves, lower_curves, upper_curves):
            for metric in curve_set[system]:
                curve_set[system][metric] = np.asarray(curve_set[system][metric], dtype=float)

    fields = list(output_rows[0])
    write_tsv(run_dir / "plasmid_prevalence_sensitivity.tsv", output_rows, fields)
    key_values = {0.01, 0.05, 0.09, 0.10, 0.20, 0.33}
    key_rows = [
        row
        for row in output_rows
        if round(float(row["assumed_plasmid_prevalence"]), 2) in key_values
    ]
    write_tsv(run_dir / "plasmid_prevalence_key_scenarios.tsv", key_rows, fields)

    grid = np.linspace(0.0001, 0.50, 5000)
    crossover_rows: list[dict[str, object]] = []
    for left, right in (
        ("mobiorigin_v0.1.7", "genomad_calibrated"),
        ("mobiorigin_v0.1.7", "genomad_uncalibrated"),
    ):
        left_f1 = np.asarray(
            [
                project(np.asarray([rates[left][0]]), np.asarray([rates[left][1]]), float(p))["f1"][
                    0
                ]
                for p in grid
            ]
        )
        right_f1 = np.asarray(
            [
                project(np.asarray([rates[right][0]]), np.asarray([rates[right][1]]), float(p))[
                    "f1"
                ][0]
                for p in grid
            ]
        )
        difference = left_f1 - right_f1
        crossings = np.flatnonzero(np.signbit(difference[:-1]) != np.signbit(difference[1:]))
        crossover = float(grid[crossings[0]]) if len(crossings) else float("nan")
        crossover_rows.append(
            {
                "left_system": left,
                "right_system": right,
                "approximate_f1_crossover_prevalence": crossover,
            }
        )
    write_tsv(
        run_dir / "plasmid_f1_crossover.tsv",
        crossover_rows,
        ["left_system", "right_system", "approximate_f1_crossover_prevalence"],
    )

    figure_path = run_dir / "Figure_S_prevalence_sensitivity.svg"
    svg_figure(figure_path, prevalences, point_curves, lower_curves, upper_curves)
    report = {
        "status": "PASS",
        "records": len(truth_rows),
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "assumption": "Class-conditional sensitivity and specificity estimated on the balanced external cohort remain constant under prevalence shift.",
        "caution": "These are mathematical projections, not measured performance in a natural metagenome.",
        "external_binary_rates": {
            system: {"sensitivity": rates[system][0], "specificity": rates[system][1]}
            for system in SYSTEMS
        },
        "f1_crossovers": crossover_rows,
    }
    (run_dir / "prevalence_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("===== PLASMID PREVALENCE SENSITIVITY =====")
    for system in SYSTEMS:
        print(f"{system}: sensitivity={rates[system][0]:.4f}, specificity={rates[system][1]:.4f}")
    for row in crossover_rows:
        print(
            f"F1 crossover, {row['left_system']} vs {row['right_system']}: "
            f"{100 * float(row['approximate_f1_crossover_prevalence']):.2f}% plasmid prevalence"
        )
    print(f"Table: {run_dir / 'plasmid_prevalence_sensitivity.tsv'}")
    print(f"Key scenarios: {run_dir / 'plasmid_prevalence_key_scenarios.tsv'}")
    print(f"Figure: {figure_path}")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
