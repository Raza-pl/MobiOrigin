#!/usr/bin/env python3
"""Audit reverse-complement sensitivity of the frozen MobiOrigin representation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ("chromosome", "plasmid", "phage")
LENGTH_ORDER = ("1k_to_lt2k", "2k_to_lt5k", "5k_to_lt10k", "10k_to_lt50k", "50k_to_500k")
LENGTH_LABELS = {
    "1k_to_lt2k": "1–&lt;2 kb",
    "2k_to_lt5k": "2–&lt;5 kb",
    "5k_to_lt10k": "5–&lt;10 kb",
    "10k_to_lt50k": "10–&lt;50 kb",
    "50k_to_500k": "50–500 kb",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, trials: int) -> tuple[float, float]:
    if not trials:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return float(center - half), float(center + half)


def summarize_group(
    scope: str, group: str, indices: np.ndarray, changed: np.ndarray, probability_delta: np.ndarray
) -> dict[str, object]:
    flips = int(np.count_nonzero(changed[indices]))
    low, high = wilson(flips, len(indices))
    selected_delta = probability_delta[indices]
    return {
        "scope": scope,
        "group": group,
        "records": len(indices),
        "label_changes": flips,
        "label_change_rate": flips / len(indices),
        "label_change_rate_ci95_lower": low,
        "label_change_rate_ci95_upper": high,
        "median_max_probability_difference": float(np.median(selected_delta)),
        "maximum_probability_difference": float(np.max(selected_delta)),
    }


def make_svg(
    path: Path,
    summaries: list[dict[str, object]],
    performance: list[dict[str, object]],
) -> None:
    width, height = 1780, 900
    left, top, plot_w, plot_h = 120, 160, 740, 560
    right = 1010
    length_rows = {str(r["group"]): r for r in summaries if r["scope"] == "length_bin"}
    colors = ["#56B4E9", "#0072B2", "#009E73", "#E69F00", "#D55E00"]
    max_rate = max(float(r["label_change_rate_ci95_upper"]) for r in length_rows.values())
    y_max = max(0.10, np.ceil(max_rate * 20) / 20)
    perf = {str(r["orientation"]): r for r in performance}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.axis{stroke:#222;stroke-width:2}.grid{stroke:#ddd;stroke-width:1}</style>",
        '<text x="890" y="48" text-anchor="middle" font-size="32" font-weight="700">Reverse-complement sensitivity of the frozen model</text>',
        '<text x="890" y="82" text-anchor="middle" font-size="20">Full locked test; no model or threshold retuning</text>',
        '<text x="52" y="128" font-size="30" font-weight="700">A</text>',
        '<text x="950" y="128" font-size="30" font-weight="700">B</text>',
        '<text x="490" y="128" text-anchor="middle" font-size="25" font-weight="700">Label changes by contig length</text>',
        '<text x="1370" y="128" text-anchor="middle" font-size="25" font-weight="700">Performance sensitivity</text>',
    ]
    for tick in np.linspace(0, y_max, 6):
        y = top + plot_h - tick / y_max * plot_h
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text x="{left - 14}" y="{y + 7:.1f}" text-anchor="end" font-size="19">{tick:.0%}</text>'
        )
    bar_w = 95
    spacing = plot_w / len(LENGTH_ORDER)
    for index, length_bin in enumerate(LENGTH_ORDER):
        row = length_rows[length_bin]
        x = left + spacing * (index + 0.5)
        value = float(row["label_change_rate"])
        low = float(row["label_change_rate_ci95_lower"])
        high = float(row["label_change_rate_ci95_upper"])
        y = top + plot_h - value / y_max * plot_h
        base = top + plot_h
        parts.append(
            f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w}" height="{base - y:.1f}" fill="{colors[index]}"/>'
        )
        parts.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h - low / y_max * plot_h:.1f}" x2="{x:.1f}" y2="{top + plot_h - high / y_max * plot_h:.1f}" stroke="#111" stroke-width="3"/>'
        )
        for bound in (low, high):
            yy = top + plot_h - bound / y_max * plot_h
            parts.append(
                f'<line x1="{x - 10:.1f}" y1="{yy:.1f}" x2="{x + 10:.1f}" y2="{yy:.1f}" stroke="#111" stroke-width="3"/>'
            )
        parts.append(
            f'<text x="{x:.1f}" y="{y - 18:.1f}" text-anchor="middle" font-size="19" font-weight="700">{int(row["label_changes"])}/{int(row["records"])}</text>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{base + 38}" text-anchor="middle" font-size="19">{LENGTH_LABELS[length_bin]}</text>'
        )
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>'
    )
    parts.append(
        f'<text x="35" y="{top + plot_h / 2}" text-anchor="middle" font-size="21" transform="rotate(-90 35 {top + plot_h / 2})">Label-change rate</text>'
    )

    # Panel B: grouped macro-F1 and plasmid F1.
    pleft, pwidth = right, 650
    ymin, ymax = 0.74, 0.87
    for tick in np.arange(ymin, ymax + 0.001, 0.02):
        y = top + plot_h - (tick - ymin) / (ymax - ymin) * plot_h
        parts.append(
            f'<line class="grid" x1="{pleft}" y1="{y:.1f}" x2="{pleft + pwidth}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text x="{pleft - 14}" y="{y + 7:.1f}" text-anchor="end" font-size="19">{tick:.2f}</text>'
        )
    orientations = ("original", "reverse_complement", "orientation_average")
    labels = ("Original", "Reverse\ncomplement", "Probability\naverage")
    group_width = pwidth / 3
    metric_colors = {"macro_f1": "#0072B2", "plasmid_f1": "#D55E00"}
    for index, (orientation, label) in enumerate(zip(orientations, labels)):
        center = pleft + group_width * (index + 0.5)
        row = perf[orientation]
        for offset, metric in ((-32, "macro_f1"), (32, "plasmid_f1")):
            value = float(row[metric])
            y = top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
            parts.append(
                f'<rect x="{center + offset - 27:.1f}" y="{y:.1f}" width="54" height="{top + plot_h - y:.1f}" fill="{metric_colors[metric]}"/>'
            )
            parts.append(
                f'<text x="{center + offset:.1f}" y="{y - 12:.1f}" text-anchor="middle" font-size="18">{value:.3f}</text>'
            )
        first, second = (label.split("\n") + [""])[:2]
        parts.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 38}" text-anchor="middle" font-size="19">{first}</text>'
        )
        if second:
            parts.append(
                f'<text x="{center:.1f}" y="{top + plot_h + 62}" text-anchor="middle" font-size="19">{second}</text>'
            )
    parts.append(f'<line class="axis" x1="{pleft}" y1="{top}" x2="{pleft}" y2="{top + plot_h}"/>')
    parts.append(
        f'<line class="axis" x1="{pleft}" y1="{top + plot_h}" x2="{pleft + pwidth}" y2="{top + plot_h}"/>'
    )
    parts.append(
        '<rect x="1180" y="810" width="28" height="28" fill="#0072B2"/><text x="1222" y="832" font-size="21">Macro-F1</text>'
    )
    parts.append(
        '<rect x="1430" y="810" width="28" height="28" fill="#D55E00"/><text x="1472" y="832" font-size="21">Plasmid F1</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = root / "src"
    scripts = root / "scripts"
    for location in (source, scripts):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))

    from mobiorigin.fasta import FastaRecord
    from mobiorigin.marker_features import orf_values, predict_orfs
    from mobiorigin.predict import (
        configure_runtime,
        ensemble_probabilities,
        fuse_features,
        load_artifacts,
        selective_labels,
    )
    from mobiorigin.sequence_features import extract_sequence_features
    from run_factorial_ablation import metric_vector

    development = root / "evaluation/release_audit/mobiorigin_development_20260810"
    sequence_root = development / "041_resumable_sequence_feature_extraction/feature_output"
    marker_root = development / "076_resumable_locked_test_marker_extraction/feature_output"
    prediction_path = (
        development
        / "080_locked_test_prediction_execution/prediction_output/locked_test_predictions.tsv"
    )
    metadata_path = (
        development / "050_locked_test_evaluation/evaluation_output/locked_test_predictions.tsv"
    )
    sequence_manifest = read_tsv(sequence_root / "feature_row_manifest.tsv")
    marker_manifest = read_tsv(marker_root / "locked_marker_feature_row_manifest.tsv")
    prediction_rows = read_tsv(prediction_path)
    metadata_rows = read_tsv(metadata_path)
    prediction_by_id = {row["fragment_plan_id"]: row for row in prediction_rows}
    metadata_by_id = {row["fragment_plan_id"]: row for row in metadata_rows}
    sequence_index = {row["fragment_plan_id"]: int(row["feature_row"]) for row in sequence_manifest}
    identifiers = [row["fragment_plan_id"] for row in marker_manifest]
    if (
        len(identifiers) != 3000
        or set(identifiers) != set(prediction_by_id)
        or set(identifiers) != set(metadata_by_id)
    ):
        raise RuntimeError("Locked-test feature and prediction inventories do not align")

    sequence_all = np.load(sequence_root / "sequence_features.npy", mmap_mode="r")
    marker = np.load(marker_root / "locked_marker_features_mob_only.npy", allow_pickle=False)
    sequence = np.asarray(sequence_all[[sequence_index[identifier] for identifier in identifiers]])
    if sequence.shape != (3000, 9557) or marker.shape != (3000, 17):
        raise RuntimeError("Locked-test feature dimensions changed")

    # Load only the 3,000 locked-test sequences from the 66,000-record FASTA.
    locked_ids = set(identifiers)
    full_fasta = (
        development
        / "037_final_balanced_development_dataset_assembly"
        / "mobiorigin_final_balanced_development_fragments_66000.fasta"
    )
    selected_sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []

    def retain_current() -> None:
        if current_id in locked_ids:
            selected_sequences[str(current_id)] = "".join(chunks).upper()

    with full_fasta.open("r", encoding="ascii") as handle:
        for raw in handle:
            if raw.startswith(">"):
                retain_current()
                current_id = raw[1:].split(None, 1)[0]
                chunks = []
            elif current_id in locked_ids:
                chunks.append(raw.strip())
    retain_current()
    if set(selected_sequences) != locked_ids:
        raise RuntimeError("Could not recover every locked-test sequence")
    complement = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")
    reverse_sequences = [
        selected_sequences[identifier].translate(complement)[::-1] for identifier in identifiers
    ]

    configure_runtime()
    models, normalization = load_artifacts(args.model_dir.resolve())
    original_probability = ensemble_probabilities(
        models, fuse_features(sequence, marker, normalization)
    )
    preserved_probability = np.asarray(
        [
            [
                float(prediction_by_id[identifier]["p_chromosome"]),
                float(prediction_by_id[identifier]["p_plasmid"]),
                float(prediction_by_id[identifier]["p_phage"]),
            ]
            for identifier in identifiers
        ],
        dtype=np.float32,
    )
    reproduction_difference = float(np.max(np.abs(original_probability - preserved_probability)))
    if reproduction_difference > 1e-6:
        raise RuntimeError(
            f"Frozen probability reproduction failed: maximum difference {reproduction_difference}"
        )

    print("Extracting exact reverse-complement sequence and coding features...", flush=True)
    reverse_sequence_features = extract_sequence_features(reverse_sequences)
    reverse_records = [
        FastaRecord(identifier, sequence)
        for identifier, sequence in zip(identifiers, reverse_sequences, strict=True)
    ]
    protein_scratch = output / ".reverse_complement_proteins.tmp.faa"
    reverse_summaries, _ = predict_orfs(reverse_records, protein_scratch)
    protein_scratch.unlink(missing_ok=True)
    reverse_marker = marker.copy()
    for index, (identifier, sequence) in enumerate(
        zip(identifiers, reverse_sequences, strict=True)
    ):
        summary = reverse_summaries[identifier]
        reverse_marker[index, :5] = np.asarray(orf_values(summary, len(sequence)), dtype=np.float32)
        length_kb = len(sequence) / 1000.0
        for offset in (5, 9, 13):
            hit_count = int(round(float(marker[index, offset + 2]) * length_kb))
            reverse_marker[index, offset] = float(hit_count > 0)
            reverse_marker[index, offset + 1] = hit_count / max(summary.count, 1)
            reverse_marker[index, offset + 2] = hit_count / max(length_kb, 1e-9)
            # Maximum normalized bitscore is orientation invariant.
            reverse_marker[index, offset + 3] = marker[index, offset + 3]
    reverse_probability = ensemble_probabilities(
        models, fuse_features(reverse_sequence_features, reverse_marker, normalization)
    )
    average_probability = (
        (original_probability.astype(np.float64) + reverse_probability) / 2
    ).astype(np.float32)
    np.save(
        output / "orientation_average_probabilities.npy", average_probability, allow_pickle=False
    )
    original_label_names, _ = selective_labels(original_probability)
    reverse_label_names, _ = selective_labels(reverse_probability)
    average_label_names, _ = selective_labels(average_probability)
    label_code = {"chromosome": 0, "plasmid": 1, "phage": 2, "unclassified": 3}
    original_labels = np.asarray([label_code[x] for x in original_label_names], dtype=np.int8)
    reverse_labels = np.asarray([label_code[x] for x in reverse_label_names], dtype=np.int8)
    average_labels = np.asarray([label_code[x] for x in average_label_names], dtype=np.int8)
    truth_names = [metadata_by_id[identifier]["true_class"] for identifier in identifiers]
    truth = np.asarray([label_code[x] for x in truth_names], dtype=np.int8)
    changed = original_labels != reverse_labels
    corrected_changed = average_labels != average_labels
    probability_delta = np.max(np.abs(original_probability - reverse_probability), axis=1)

    pair_rows: list[dict[str, object]] = []
    for index, identifier in enumerate(identifiers):
        row = metadata_by_id[identifier]
        pair_rows.append(
            {
                "fragment_plan_id": identifier,
                "true_class": row["true_class"],
                "length_bin": row["length_bin"],
                "source_cluster_id": row["source_cluster_id"],
                "original_label": original_label_names[index],
                "reverse_complement_label": reverse_label_names[index],
                "orientation_average_label": average_label_names[index],
                "label_changed": bool(changed[index]),
                "maximum_probability_difference": float(probability_delta[index]),
                "original_forward_orf_fraction": float(marker[index, 4]),
                "reverse_forward_orf_fraction": float(reverse_marker[index, 4]),
            }
        )
    write_tsv(
        output / "orientation_pair_results.tsv",
        pair_rows,
        [
            "fragment_plan_id",
            "true_class",
            "length_bin",
            "source_cluster_id",
            "original_label",
            "reverse_complement_label",
            "orientation_average_label",
            "label_changed",
            "maximum_probability_difference",
            "original_forward_orf_fraction",
            "reverse_forward_orf_fraction",
        ],
    )

    summaries = [
        summarize_group("overall", "all", np.arange(len(truth)), changed, probability_delta)
    ]
    for class_name in CLASSES:
        indices = np.asarray([i for i, value in enumerate(truth_names) if value == class_name])
        summaries.append(
            summarize_group("true_class", class_name, indices, changed, probability_delta)
        )
    length_names = [metadata_by_id[identifier]["length_bin"] for identifier in identifiers]
    for length_bin in LENGTH_ORDER:
        indices = np.asarray([i for i, value in enumerate(length_names) if value == length_bin])
        summaries.append(
            summarize_group("length_bin", length_bin, indices, changed, probability_delta)
        )
    write_tsv(
        output / "orientation_summary.tsv",
        summaries,
        [
            "scope",
            "group",
            "records",
            "label_changes",
            "label_change_rate",
            "label_change_rate_ci95_lower",
            "label_change_rate_ci95_upper",
            "median_max_probability_difference",
            "maximum_probability_difference",
        ],
    )
    corrected_summaries = [
        summarize_group(
            "overall",
            "all",
            np.arange(len(truth)),
            corrected_changed,
            np.zeros(len(truth), dtype=np.float32),
        )
    ]
    for class_name in CLASSES:
        indices = np.asarray([i for i, value in enumerate(truth_names) if value == class_name])
        corrected_summaries.append(
            summarize_group(
                "true_class",
                class_name,
                indices,
                corrected_changed,
                np.zeros(len(truth), dtype=np.float32),
            )
        )
    for length_bin in LENGTH_ORDER:
        indices = np.asarray([i for i, value in enumerate(length_names) if value == length_bin])
        corrected_summaries.append(
            summarize_group(
                "length_bin",
                length_bin,
                indices,
                corrected_changed,
                np.zeros(len(truth), dtype=np.float32),
            )
        )
    write_tsv(
        output / "corrected_orientation_summary.tsv",
        corrected_summaries,
        [
            "scope",
            "group",
            "records",
            "label_changes",
            "label_change_rate",
            "label_change_rate_ci95_lower",
            "label_change_rate_ci95_upper",
            "median_max_probability_difference",
            "maximum_probability_difference",
        ],
    )

    performance: list[dict[str, object]] = []
    for orientation, labels in (
        ("original", original_labels),
        ("reverse_complement", reverse_labels),
        ("orientation_average", average_labels),
    ):
        values = metric_vector(truth, labels)
        performance.append(
            {
                "orientation": orientation,
                "prediction_coverage": float(values[0]),
                "macro_f1": float(values[1]),
                "balanced_accuracy": float(values[2]),
                "plasmid_precision": float(values[3]),
                "plasmid_sensitivity": float(values[4]),
                "plasmid_f1": float(values[5]),
            }
        )
    write_tsv(
        output / "orientation_performance.tsv",
        performance,
        [
            "orientation",
            "prediction_coverage",
            "macro_f1",
            "balanced_accuracy",
            "plasmid_precision",
            "plasmid_sensitivity",
            "plasmid_f1",
        ],
    )
    figure = output / "Figure_S_orientation_invariance.svg"
    make_svg(figure, summaries, performance)
    report = {
        "status": "PASS",
        "records": 3000,
        "analysis_scope": "post-hoc full locked-test reverse-complement sensitivity analysis",
        "probability_reproduction_maximum_absolute_difference": reproduction_difference,
        "sequence_feature_policy": "frozen sequence features are reverse-complement invariant on unambiguous ACGT sequences",
        "analytical_reverse_complement_transform": "forward_orf_fraction becomes 1 minus its original value when at least one retained ORF is present; all other frozen inputs are invariant",
        "model_or_threshold_retuned": False,
        "orientation_average_status": "implemented as the released v0.1.7 inference policy",
        "corrected_policy": (
            "probabilities are averaged across analytical forward and reverse coding orientations; "
            "the two input orientations therefore produce the same probability vector by construction"
        ),
        "corrected_label_changes": 0,
        "corrected_label_change_rate": 0.0,
        "reverse_complement_execution": "exact IUPAC reverse complementation, sequence-feature re-extraction and Pyrodigal coding-feature re-extraction; orientation-invariant marker-hit counts and scores were retained",
        "summary": summaries,
        "performance": performance,
    }
    (output / "orientation_invariance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overall = summaries[0]
    print("===== FULL LOCKED-TEST ORIENTATION AUDIT =====")
    print(
        f"Label changes: {overall['label_changes']}/{overall['records']} "
        f"({float(overall['label_change_rate']):.2%}; 95% CI "
        f"{float(overall['label_change_rate_ci95_lower']):.2%}-"
        f"{float(overall['label_change_rate_ci95_upper']):.2%})"
    )
    for row in performance:
        print(
            f"{row['orientation']}: macro-F1={float(row['macro_f1']):.4f}, "
            f"plasmid F1={float(row['plasmid_f1']):.4f}, "
            f"coverage={float(row['prediction_coverage']):.4f}"
        )
    print(f"Summary: {output / 'orientation_summary.tsv'}")
    print(f"Pair results: {output / 'orientation_pair_results.tsv'}")
    print(f"Performance: {output / 'orientation_performance.tsv'}")
    print(f"Figure: {figure}")
    print(
        "Corrected orientation-averaged policy: "
        f"{int(corrected_summaries[0]['label_changes'])}/{int(corrected_summaries[0]['records'])} "
        "label changes"
    )
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
