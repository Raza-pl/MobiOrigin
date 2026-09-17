#!/usr/bin/env python3
"""Audit the frozen plasmid-selective rejection rule by cohort and true class."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CLASS_NAMES = ("chromosome", "plasmid", "phage")
FEATURE_SYSTEMS = ("sequence", "fused")


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def class_metrics(
    truth: np.ndarray, prediction: np.ndarray, class_code: int
) -> tuple[float, float, float, float]:
    true_class = truth == class_code
    predicted_class = prediction == class_code
    tp = int(np.count_nonzero(true_class & predicted_class))
    fp = int(np.count_nonzero(~true_class & predicted_class))
    fn = int(np.count_nonzero(true_class & ~predicted_class))
    precision = safe_ratio(tp, tp + fp)
    sensitivity = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * sensitivity, precision + sensitivity)
    coverage = float(np.mean(prediction[true_class] < 3))
    return coverage, precision, sensitivity, f1


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = (
        z
        * np.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return float(center - half), float(center + half)


def make_svg(
    path: Path,
    audit_rows: list[dict[str, object]],
    curve_rows: list[dict[str, object]],
    threshold: float,
) -> None:
    width, height = 1880, 920
    colors = {"chromosome": "#0072B2", "plasmid": "#D55E00", "phage": "#009E73"}
    cohort_titles = {
        "locked_test": "Locked development test",
        "external": "Prospective external cohort",
    }
    systems = {"sequence": "Sequence only", "fused": "Fused features"}
    audit = {(str(r["cohort"]), str(r["feature_system"])): r for r in audit_rows}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.axis{stroke:#222;stroke-width:2}.grid{stroke:#ddd;stroke-width:1}</style>",
        '<text x="940" y="48" text-anchor="middle" font-size="32" font-weight="700">What the frozen selective rule rejects</text>',
        '<text x="940" y="82" text-anchor="middle" font-size="20">Class composition of rejected calls and plasmid precision–sensitivity trade-off</text>',
        '<text x="55" y="130" font-size="30" font-weight="700">A</text>',
        '<text x="1010" y="130" font-size="30" font-weight="700">B</text>',
        '<text x="450" y="130" text-anchor="middle" font-size="25" font-weight="700">Rejected calls by true class</text>',
        '<text x="1430" y="130" text-anchor="middle" font-size="25" font-weight="700">Threshold trade-off in fused features</text>',
    ]

    # Panel A: four stacked bars; bar height is count and fill is true-class composition.
    left, top, plot_w, plot_h = 120, 175, 720, 560
    max_rejected = max(int(r["rejected_calls"]) for r in audit_rows)
    max_axis = max(100, int(np.ceil(max_rejected / 25.0) * 25))
    for tick in range(0, max_axis + 1, max(25, max_axis // 5)):
        y = top + plot_h - tick / max_axis * plot_h
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text x="{left - 14}" y="{y + 7:.1f}" text-anchor="end" font-size="19">{tick}</text>'
        )
    positions = [left + 105, left + 275, left + 475, left + 645]
    combinations = [
        ("locked_test", "sequence"),
        ("locked_test", "fused"),
        ("external", "sequence"),
        ("external", "fused"),
    ]
    for x, (cohort, system) in zip(positions, combinations):
        row = audit[(cohort, system)]
        bottom = top + plot_h
        for class_name in CLASS_NAMES:
            count = int(row[f"rejected_true_{class_name}"])
            bar_h = count / max_axis * plot_h
            bottom -= bar_h
            parts.append(
                f'<rect x="{x - 45}" y="{bottom:.1f}" width="90" height="{bar_h:.1f}" fill="{colors[class_name]}"/>'
            )
        total = int(row["rejected_calls"])
        parts.append(
            f'<text x="{x}" y="{bottom - 12:.1f}" text-anchor="middle" font-size="19" font-weight="700">{total}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{top + plot_h + 34}" text-anchor="middle" font-size="18">{systems[system]}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{top + plot_h + 59}" text-anchor="middle" font-size="18">{cohort_titles[cohort].split()[0]}</text>'
        )
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>'
    )
    parts.append(
        f'<text x="36" y="{top + plot_h / 2}" text-anchor="middle" font-size="21" transform="rotate(-90 36 {top + plot_h / 2})">Rejected calls</text>'
    )

    # Panel B: fused threshold diagnostic; no optimization is performed.
    left2, plot_w2 = 1060, 700
    for tick in np.arange(0.45, 1.001, 0.05):
        y = top + plot_h - (tick - 0.45) / 0.55 * plot_h
        parts.append(
            f'<line class="grid" x1="{left2}" y1="{y:.1f}" x2="{left2 + plot_w2}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text x="{left2 - 14}" y="{y + 7:.1f}" text-anchor="end" font-size="19">{tick:.2f}</text>'
        )
    subsets: dict[str, list[dict[str, object]]] = {"locked_test": [], "external": []}
    for row in curve_rows:
        if row["feature_system"] == "fused":
            subsets[str(row["cohort"])].append(row)
    line_colors = {"locked_test": "#6A3D9A", "external": "#222222"}
    dashes = {"precision": "", "sensitivity": "10,7"}
    for cohort in ("locked_test", "external"):
        rows = sorted(subsets[cohort], key=lambda r: float(r["threshold"]))
        for metric in ("precision", "sensitivity"):
            points = []
            for row in rows:
                x = left2 + (float(row["threshold"]) + 0.05) / 0.55 * plot_w2
                y = top + plot_h - (float(row[f"plasmid_{metric}"]) - 0.45) / 0.55 * plot_h
                points.append(f"{x:.1f},{y:.1f}")
            dash = f' stroke-dasharray="{dashes[metric]}"' if dashes[metric] else ""
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{line_colors[cohort]}" stroke-width="4"{dash}/>'
            )
    threshold_x = left2 + (threshold + 0.05) / 0.55 * plot_w2
    parts.append(
        f'<line x1="{threshold_x:.1f}" y1="{top}" x2="{threshold_x:.1f}" y2="{top + plot_h}" stroke="#D55E00" stroke-width="3"/>'
    )
    parts.append(
        f'<text x="{threshold_x + 9:.1f}" y="{top + 25}" font-size="18" fill="#D55E00">Frozen threshold</text>'
    )
    for tick in np.arange(-0.05, 0.501, 0.05):
        x = left2 + (tick + 0.05) / 0.55 * plot_w2
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 34}" text-anchor="middle" font-size="18">{tick:.2f}</text>'
        )
    parts.append(f'<line class="axis" x1="{left2}" y1="{top}" x2="{left2}" y2="{top + plot_h}"/>')
    parts.append(
        f'<line class="axis" x1="{left2}" y1="{top + plot_h}" x2="{left2 + plot_w2}" y2="{top + plot_h}"/>'
    )
    parts.append(
        f'<text x="{left2 + plot_w2 / 2}" y="{top + plot_h + 70}" text-anchor="middle" font-size="21">Plasmid-margin threshold</text>'
    )
    parts.append(
        f'<text x="970" y="{top + plot_h / 2}" text-anchor="middle" font-size="21" transform="rotate(-90 970 {top + plot_h / 2})">Metric value</text>'
    )

    # Legends.
    legend_y = 850
    for index, class_name in enumerate(CLASS_NAMES):
        x = 100 + index * 220
        parts.append(
            f'<rect x="{x}" y="{legend_y - 20}" width="27" height="27" fill="{colors[class_name]}"/>'
        )
        parts.append(
            f'<text x="{x + 39}" y="{legend_y + 2}" font-size="20">True {class_name}</text>'
        )
    parts.append(
        '<line x1="1050" y1="835" x2="1115" y2="835" stroke="#222" stroke-width="4"/><text x="1127" y="842" font-size="19">Precision</text>'
    )
    parts.append(
        '<line x1="1280" y1="835" x2="1345" y2="835" stroke="#222" stroke-width="4" stroke-dasharray="10,7"/><text x="1357" y="842" font-size="19">Sensitivity</text>'
    )
    parts.append(
        '<line x1="1510" y1="820" x2="1575" y2="820" stroke="#6A3D9A" stroke-width="4"/><text x="1587" y="827" font-size="18">Locked</text>'
    )
    parts.append(
        '<line x1="1510" y1="852" x2="1575" y2="852" stroke="#222" stroke-width="4"/><text x="1587" y="859" font-size="18">External</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--factorial-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--external-fused-predictions", type=Path)
    parser.add_argument("--locked-fused-probabilities", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    factorial_dir = args.factorial_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_factorial_ablation as factorial

    cohorts = {
        "locked_test": factorial.load_locked(
            root,
            args.locked_fused_probabilities,
        ),
        "external": factorial.load_external(
            root,
            factorial_dir,
            args.external_fused_predictions,
        ),
    }
    classwise_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "status": "PASS",
        "scope": "post-hoc diagnostic of the frozen plasmid-selective rejection policy",
        "frozen_threshold": factorial.THRESHOLD,
        "threshold_retuned": False,
        "cohorts": {},
    }

    for cohort, payload in cohorts.items():
        truth = np.asarray(payload["truth"])
        probabilities = {
            "sequence": np.asarray(payload["sequence_probabilities"]),
            "fused": np.asarray(payload["fused_probabilities"]),
        }
        cohort_report: dict[str, object] = {}
        for feature_system in FEATURE_SYSTEMS:
            probability = probabilities[feature_system]
            forced = factorial.forced_labels(probability)
            selective = factorial.selective_labels(probability)
            predictions = {"forced": forced, "selective": selective}
            for policy, prediction in predictions.items():
                for class_code, class_name in enumerate(CLASS_NAMES):
                    coverage, precision, sensitivity, f1 = class_metrics(
                        truth, prediction, class_code
                    )
                    true_mask = truth == class_code
                    covered = int(np.count_nonzero(prediction[true_mask] < 3))
                    total = int(np.count_nonzero(true_mask))
                    low, high = wilson_interval(covered, total)
                    classwise_rows.append(
                        {
                            "cohort": cohort,
                            "feature_system": feature_system,
                            "decision_policy": policy,
                            "true_class": class_name,
                            "records": total,
                            "covered_records": covered,
                            "coverage": coverage,
                            "coverage_ci95_lower": low,
                            "coverage_ci95_upper": high,
                            "precision": precision,
                            "sensitivity": sensitivity,
                            "f1": f1,
                        }
                    )
                for true_code, true_name in enumerate(CLASS_NAMES):
                    for predicted_code, predicted_name in enumerate((*CLASS_NAMES, "unclassified")):
                        confusion_rows.append(
                            {
                                "cohort": cohort,
                                "feature_system": feature_system,
                                "decision_policy": policy,
                                "true_class": true_name,
                                "predicted_class": predicted_name,
                                "records": int(
                                    np.count_nonzero(
                                        (truth == true_code) & (prediction == predicted_code)
                                    )
                                ),
                            }
                        )

            rejected = selective == 3
            retained = ~rejected
            forced_correct = forced == truth
            rejected_count = int(np.count_nonzero(rejected))
            rejected_errors = int(np.count_nonzero(rejected & ~forced_correct))
            rejected_correct = int(np.count_nonzero(rejected & forced_correct))
            retained_errors = int(np.count_nonzero(retained & ~forced_correct))
            retained_count = int(np.count_nonzero(retained))
            rejected_error_rate = safe_ratio(rejected_errors, rejected_count)
            retained_error_rate = safe_ratio(retained_errors, retained_count)
            audit_row: dict[str, object] = {
                "cohort": cohort,
                "feature_system": feature_system,
                "records": len(truth),
                "rejected_calls": rejected_count,
                "rejection_rate": safe_ratio(rejected_count, len(truth)),
                "prevented_forced_errors": rejected_errors,
                "withheld_correct_calls": rejected_correct,
                "rejected_error_rate": rejected_error_rate,
                "retained_error_rate": retained_error_rate,
                "error_enrichment_ratio": safe_ratio(rejected_error_rate, retained_error_rate),
            }
            for class_code, class_name in enumerate(CLASS_NAMES):
                audit_row[f"rejected_true_{class_name}"] = int(
                    np.count_nonzero(rejected & (truth == class_code))
                )
            audit_rows.append(audit_row)
            cohort_report[feature_system] = dict(audit_row)

            # Descriptive threshold curve only; the frozen threshold is not optimized here.
            for threshold in np.linspace(-0.05, 0.50, 56):
                prediction = forced.copy()
                plasmid_score = probability[:, 1] - np.maximum(probability[:, 0], probability[:, 2])
                prediction[(prediction == 1) & (plasmid_score < threshold)] = 3
                coverage, precision, sensitivity, f1 = class_metrics(truth, prediction, 1)
                curve_rows.append(
                    {
                        "cohort": cohort,
                        "feature_system": feature_system,
                        "threshold": float(threshold),
                        "prediction_coverage": float(np.mean(prediction < 3)),
                        "plasmid_class_coverage": coverage,
                        "plasmid_precision": precision,
                        "plasmid_sensitivity": sensitivity,
                        "plasmid_f1": f1,
                    }
                )
        report["cohorts"][cohort] = cohort_report

    write_tsv(
        output_dir / "classwise_metrics_and_coverage.tsv",
        classwise_rows,
        [
            "cohort",
            "feature_system",
            "decision_policy",
            "true_class",
            "records",
            "covered_records",
            "coverage",
            "coverage_ci95_lower",
            "coverage_ci95_upper",
            "precision",
            "sensitivity",
            "f1",
        ],
    )
    write_tsv(
        output_dir / "confusion_matrix_long.tsv",
        confusion_rows,
        [
            "cohort",
            "feature_system",
            "decision_policy",
            "true_class",
            "predicted_class",
            "records",
        ],
    )
    write_tsv(
        output_dir / "rejection_audit.tsv",
        audit_rows,
        [
            "cohort",
            "feature_system",
            "records",
            "rejected_calls",
            "rejection_rate",
            "prevented_forced_errors",
            "withheld_correct_calls",
            "rejected_error_rate",
            "retained_error_rate",
            "error_enrichment_ratio",
            "rejected_true_chromosome",
            "rejected_true_plasmid",
            "rejected_true_phage",
        ],
    )
    write_tsv(
        output_dir / "threshold_diagnostic.tsv",
        curve_rows,
        [
            "cohort",
            "feature_system",
            "threshold",
            "prediction_coverage",
            "plasmid_class_coverage",
            "plasmid_precision",
            "plasmid_sensitivity",
            "plasmid_f1",
        ],
    )
    figure = output_dir / "Figure_S_selective_rejection_audit.svg"
    make_svg(figure, audit_rows, curve_rows, factorial.THRESHOLD)
    report["interpretation_boundary"] = (
        "Threshold curves are descriptive sensitivity diagnostics. They were not used "
        "to select or retune the frozen production threshold."
    )
    (output_dir / "selective_rejection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("===== SELECTIVE REJECTION AUDIT =====")
    for row in audit_rows:
        print(
            f"{row['cohort']} | {row['feature_system']}: "
            f"rejected {row['rejected_calls']}/{row['records']} "
            f"({float(row['rejection_rate']):.2%}); prevented "
            f"{row['prevented_forced_errors']} errors; withheld "
            f"{row['withheld_correct_calls']} correct calls; error enrichment "
            f"{float(row['error_enrichment_ratio']):.2f}x"
        )
    print(f"Class-wise table: {output_dir / 'classwise_metrics_and_coverage.tsv'}")
    print(f"Rejection audit: {output_dir / 'rejection_audit.tsv'}")
    print(f"Confusion matrices: {output_dir / 'confusion_matrix_long.tsv'}")
    print(f"Threshold diagnostic: {output_dir / 'threshold_diagnostic.tsv'}")
    print(f"Figure: {figure}")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
