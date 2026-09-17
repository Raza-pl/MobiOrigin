#!/usr/bin/env python3
"""Evaluate sequence/fusion and forced/selective MobiOrigin factorial arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = (
    "sequence_forced",
    "sequence_selective",
    "fused_forced",
    "fused_selective",
)
METRICS = (
    "prediction_coverage",
    "macro_f1",
    "balanced_accuracy",
    "plasmid_precision",
    "plasmid_sensitivity",
    "plasmid_f1",
)
THRESHOLD = 0.19835489988327026
MODEL_HASHES = {
    "seed_20260810/model_state.pt": "c4e664864a182ed4a161c5faa8799cd2bbafd417596bdcab0c6ad4ead38cd01a",
    "seed_20260811/model_state.pt": "91e701e1adee0a2e19ab318326f5d29ea75f88fd748f5f2a114d72a823d2237c",
    "seed_20260812/model_state.pt": "755280e3797089a74ae621672629c16e9893a8e91b3a1da4ae194e6c71a0f771",
}
EXPECTED_EXTERNAL_FASTA_SHA256 = "9d4a251b99bdb42624010489d55fc067a560c810d496bc830d88f227c7697588"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(labels: list[str]) -> np.ndarray:
    mapping = {"chromosome": 0, "plasmid": 1, "phage": 2, "unclassified": 3}
    unknown = sorted(set(labels).difference(mapping))
    if unknown:
        raise RuntimeError(f"Unsupported labels: {unknown}")
    return np.asarray([mapping[label] for label in labels], dtype=np.int8)


def forced_labels(probabilities: np.ndarray) -> np.ndarray:
    return np.argmax(probabilities, axis=1).astype(np.int8)


def selective_labels(probabilities: np.ndarray) -> np.ndarray:
    labels = forced_labels(probabilities)
    plasmid_score = probabilities[:, 1] - np.maximum(probabilities[:, 0], probabilities[:, 2])
    labels[(labels == 1) & (plasmid_score < THRESHOLD)] = 3
    return labels


def metric_vector(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    f1_values: list[float] = []
    recalls: list[float] = []
    plasmid_values = (0.0, 0.0, 0.0)
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
        if class_code == 1:
            plasmid_values = (precision, recall, f1)
    return np.asarray(
        [
            float(np.mean(prediction < 3)),
            float(np.mean(f1_values)),
            float(np.mean(recalls)),
            *plasmid_values,
        ],
        dtype=float,
    )


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def load_or_generate_external_sequence_probabilities(
    root: Path, output_dir: Path, fasta: Path, expected_ids: list[str]
) -> np.ndarray:
    probabilities_path = output_dir / "external_sequence_only_probabilities.npy"
    metadata_path = output_dir / "external_sequence_only_probability_metadata.json"
    expected_metadata = {
        "fasta_sha256": EXPECTED_EXTERNAL_FASTA_SHA256,
        "sequence_model_sha256": MODEL_HASHES,
        "records": len(expected_ids),
        "feature_dimensions": 9557,
        "ensemble_seeds": [20260810, 20260811, 20260812],
    }
    if probabilities_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        probabilities = np.load(probabilities_path, allow_pickle=False)
        if (
            all(metadata.get(key) == value for key, value in expected_metadata.items())
            and metadata.get("probabilities_sha256") == sha256_file(probabilities_path)
            and probabilities.shape == (len(expected_ids), 3)
            and probabilities.dtype == np.float32
        ):
            print("Reusing verified external sequence-only probabilities.", flush=True)
            return probabilities
        raise RuntimeError("Existing external sequence-only probability cache is incompatible")

    if sha256_file(fasta) != EXPECTED_EXTERNAL_FASTA_SHA256:
        raise RuntimeError("External FASTA identity changed")
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from mobiorigin.fasta import read_fasta
    from mobiorigin.model import load_model
    from mobiorigin.predict import configure_runtime, ensemble_probabilities
    from mobiorigin.sequence_features import extract_sequence_features

    records = read_fasta(fasta)
    observed_ids = [record.identifier for record in records]
    if observed_ids != expected_ids or not all(record.supported for record in records):
        raise RuntimeError("External FASTA order, identifiers, or supported lengths changed")
    model_root = (
        root
        / "evaluation/release_audit/mobiorigin_development_20260810"
        / "045_resumable_sequence_only_training/training_output"
    )
    models = []
    for relative, expected_hash in MODEL_HASHES.items():
        model_path = model_root / relative
        if not model_path.is_file() or sha256_file(model_path) != expected_hash:
            raise RuntimeError(f"Sequence-only model identity changed: {model_path}")
        models.append(load_model(model_path, input_dim=9557))
    print(
        f"Extracting 9,557 sequence features for {len(records):,} external records...", flush=True
    )
    configure_runtime()
    features = extract_sequence_features([record.sequence for record in records])
    if features.shape != (len(records), 9557) or features.dtype != np.float32:
        raise RuntimeError("External sequence feature matrix is invalid")
    print("Running the frozen three-seed sequence-only ensemble...", flush=True)
    probabilities = ensemble_probabilities(models, features)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = probabilities_path.with_suffix(".npy.part")
    with temporary.open("wb") as handle:
        np.save(handle, probabilities, allow_pickle=False)
    os.replace(temporary, probabilities_path)
    metadata = {
        **expected_metadata,
        "probabilities_sha256": sha256_file(probabilities_path),
        "labels_accessed_for_model_or_threshold_selection": False,
        "status": "FROZEN",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return probabilities


def load_external(
    root: Path,
    output_dir: Path,
    corrected_fused_predictions: Path | None = None,
) -> dict[str, object]:
    external = root / "evaluation/release_audit/mobiorigin_external_validation_20260817"
    truth_path = (
        external
        / "024_corrected_final_external_cohort_freeze"
        / "mobiorigin_external_validation_label_map.sealed.tsv"
    )
    fasta = (
        external
        / "024_corrected_final_external_cohort_freeze"
        / "mobiorigin_external_validation_cohort_3000.fasta"
    )
    fused_path = (
        corrected_fused_predictions.resolve()
        if corrected_fused_predictions is not None
        else external
        / "029_dual_external_prediction_execution"
        / "mobiorigin_prediction_output/predictions.tsv"
    )
    rows = read_tsv(truth_path)
    identifiers = [row["opaque_contig_id"] for row in rows]
    sequence_probabilities = load_or_generate_external_sequence_probabilities(
        root, output_dir, fasta, identifiers
    )
    fused_rows = read_tsv(fused_path)
    fused_by_id = {row["sequence_id"]: row for row in fused_rows}
    if len(rows) != 3000 or set(fused_by_id) != set(identifiers):
        raise RuntimeError("External fused predictions do not match the frozen cohort")
    keep = np.asarray(
        [
            index
            for index, identifier in enumerate(identifiers)
            if int(fused_by_id[identifier].get("non_acgt_bases", "0")) == 0
        ],
        dtype=np.int32,
    )
    fused_probabilities = np.asarray(
        [
            [
                float(fused_by_id[identifier]["p_chromosome"]),
                float(fused_by_id[identifier]["p_plasmid"]),
                float(fused_by_id[identifier]["p_phage"]),
            ]
            for identifier in identifiers
        ],
        dtype=np.float32,
    )
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    retained_rows = [rows[int(index)] for index in keep]
    for index, row in enumerate(retained_rows):
        strata[(row["class"], row["length_bin"])].append(index)
    if len(strata) != 15 or any(not indices for indices in strata.values()):
        raise RuntimeError("External class-by-length strata are incomplete")
    return {
        "truth": encode([row["class"] for row in retained_rows]),
        "sequence_probabilities": sequence_probabilities[keep],
        "fused_probabilities": fused_probabilities[keep],
        "resampling_units": [np.asarray(indices, dtype=np.int32) for indices in strata.values()],
        "resampling_method": "record bootstrap within true-class-by-length strata",
    }


def load_locked(root: Path, corrected_probabilities: Path | None = None) -> dict[str, object]:
    development = root / "evaluation/release_audit/mobiorigin_development_20260810"
    sequence_path = (
        development / "050_locked_test_evaluation/evaluation_output/locked_test_predictions.tsv"
    )
    fused_path = (
        development
        / "080_locked_test_prediction_execution/prediction_output/locked_test_predictions.tsv"
    )
    sequence_rows = read_tsv(sequence_path)
    fused_rows = read_tsv(fused_path)
    fused_by_id = {row["fragment_plan_id"]: row for row in fused_rows}
    identifiers = [row["fragment_plan_id"] for row in sequence_rows]
    if len(sequence_rows) != 3000 or set(fused_by_id) != set(identifiers):
        raise RuntimeError("Locked-test prediction inventories do not align")
    sequence_probabilities = np.asarray(
        [
            [
                float(row["ensemble_chromosome_probability"]),
                float(row["ensemble_plasmid_probability"]),
                float(row["ensemble_phage_probability"]),
            ]
            for row in sequence_rows
        ],
        dtype=np.float32,
    )
    fused_probabilities = np.asarray(
        [
            [
                float(fused_by_id[identifier]["p_chromosome"]),
                float(fused_by_id[identifier]["p_plasmid"]),
                float(fused_by_id[identifier]["p_phage"]),
            ]
            for identifier in identifiers
        ],
        dtype=np.float32,
    )
    if corrected_probabilities is not None:
        fused_probabilities = np.load(corrected_probabilities.resolve(), allow_pickle=False)
        if fused_probabilities.shape != (3000, 3) or fused_probabilities.dtype != np.float32:
            raise RuntimeError("Corrected locked-test probabilities are incompatible")
    clusters: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(sequence_rows):
        clusters[(row["true_class"], row["source_cluster_id"])].append(index)
    class_clusters: dict[str, list[np.ndarray]] = defaultdict(list)
    for (class_name, _cluster), indices in clusters.items():
        class_clusters[class_name].append(np.asarray(indices, dtype=np.int32))
    if set(class_clusters) != {"chromosome", "plasmid", "phage"}:
        raise RuntimeError("Locked-test source-cluster structure changed")
    return {
        "truth": encode([row["true_class"] for row in sequence_rows]),
        "sequence_probabilities": sequence_probabilities,
        "fused_probabilities": fused_probabilities,
        "resampling_units": class_clusters,
        "resampling_method": "source-cluster bootstrap within true class",
    }


def arm_predictions(sequence: np.ndarray, fused: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "sequence_forced": forced_labels(sequence),
        "sequence_selective": selective_labels(sequence),
        "fused_forced": forced_labels(fused),
        "fused_selective": selective_labels(fused),
    }


def sample_indices(cohort: str, units: object, rng: np.random.Generator) -> np.ndarray:
    if cohort == "external":
        cells = units
        assert isinstance(cells, list)
        return np.concatenate([rng.choice(cell, size=len(cell), replace=True) for cell in cells])
    class_clusters = units
    assert isinstance(class_clusters, dict)
    sampled: list[np.ndarray] = []
    for class_name in ("chromosome", "plasmid", "phage"):
        clusters = class_clusters[class_name]
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        sampled.extend(clusters[int(index)] for index in chosen)
    return np.concatenate(sampled)


def make_svg(path: Path, metric_rows: list[dict[str, object]]) -> None:
    width, height = 1900, 900
    panel_lefts = (140, 1040)
    panel_width, panel_height, top = 700, 570, 130
    arm_labels = ("Sequence\nforced", "Sequence\nselective", "Fused\nforced", "Fused\nselective")
    colors = {"macro_f1": "#0072B2", "plasmid_f1": "#D55E00"}
    row_map = {(str(row["cohort"]), str(row["arm"])): row for row in metric_rows}

    def sy(value: float) -> float:
        return top + panel_height - (value - 0.55) / 0.40 * panel_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.axis{stroke:#222;stroke-width:2}.grid{stroke:#ddd;stroke-width:1}</style>",
        '<text x="950" y="48" text-anchor="middle" font-size="32" font-weight="700">Feature fusion and selective rejection have separable effects</text>',
    ]
    for panel_index, cohort in enumerate(("locked_test", "external")):
        left = panel_lefts[panel_index]
        title = (
            "Locked development test" if cohort == "locked_test" else "Prospective external cohort"
        )
        parts.append(
            f'<text x="{left - 65}" y="91" font-size="30" font-weight="700">{chr(65 + panel_index)}</text>'
        )
        parts.append(
            f'<text x="{left + panel_width / 2}" y="91" text-anchor="middle" font-size="25" font-weight="700">{title}</text>'
        )
        for tick in np.arange(0.55, 0.951, 0.05):
            yy = sy(float(tick))
            parts.append(
                f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{left + panel_width}" y2="{yy:.1f}"/>'
            )
            parts.append(
                f'<text x="{left - 16}" y="{yy + 7:.1f}" text-anchor="end" font-size="19">{tick:.2f}</text>'
            )
        group_width = panel_width / 4
        for arm_index, (arm, label) in enumerate(zip(ARMS, arm_labels)):
            center = left + group_width * (arm_index + 0.5)
            row = row_map[(cohort, arm)]
            for offset, metric in ((-29, "macro_f1"), (29, "plasmid_f1")):
                value = float(row[metric])
                low = float(row[f"{metric}_ci95_lower"])
                high = float(row[f"{metric}_ci95_upper"])
                x = center + offset
                base_y = sy(0.55)
                value_y = sy(value)
                parts.append(
                    f'<rect x="{x - 24:.1f}" y="{value_y:.1f}" width="48" height="{base_y - value_y:.1f}" fill="{colors[metric]}"/>'
                )
                parts.append(
                    f'<line x1="{x:.1f}" y1="{sy(low):.1f}" x2="{x:.1f}" y2="{sy(high):.1f}" stroke="#111" stroke-width="3"/>'
                )
                parts.append(
                    f'<line x1="{x - 8:.1f}" y1="{sy(low):.1f}" x2="{x + 8:.1f}" y2="{sy(low):.1f}" stroke="#111" stroke-width="3"/>'
                )
                parts.append(
                    f'<line x1="{x - 8:.1f}" y1="{sy(high):.1f}" x2="{x + 8:.1f}" y2="{sy(high):.1f}" stroke="#111" stroke-width="3"/>'
                )
            first, second = label.split("\n")
            parts.append(
                f'<text x="{center:.1f}" y="{top + panel_height + 37}" text-anchor="middle" font-size="20">{first}</text>'
            )
            parts.append(
                f'<text x="{center:.1f}" y="{top + panel_height + 62}" text-anchor="middle" font-size="20">{second}</text>'
            )
        parts.append(
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{left}" y1="{top + panel_height}" x2="{left + panel_width}" y2="{top + panel_height}"/>'
        )
    parts.extend(
        [
            '<rect x="600" y="805" width="28" height="28" fill="#0072B2"/><text x="642" y="827" font-size="22">Macro-F1</text>',
            '<rect x="920" y="805" width="28" height="28" fill="#D55E00"/><text x="962" y="827" font-size="22">Plasmid F1</text>',
            '<text x="950" y="875" text-anchor="middle" font-size="18" fill="#444">Selective arms apply the same frozen plasmid-margin threshold (0.19835); error bars show paired bootstrap 95% confidence intervals.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--external-fused-predictions", type=Path)
    parser.add_argument("--locked-fused-probabilities", type=Path)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        raise RuntimeError("Use at least 1,000 bootstrap replicates")
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading the locked-test factorial inputs...", flush=True)
    cohorts = {"locked_test": load_locked(root, args.locked_fused_probabilities)}
    print("Preparing the external factorial inputs...", flush=True)
    cohorts["external"] = load_external(root, output_dir, args.external_fused_predictions)
    rng = np.random.default_rng(args.seed)

    metric_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    report_cohorts: dict[str, object] = {}
    for cohort_name, payload in cohorts.items():
        truth = payload["truth"]
        assert isinstance(truth, np.ndarray)
        arm_labels = arm_predictions(
            np.asarray(payload["sequence_probabilities"]),
            np.asarray(payload["fused_probabilities"]),
        )
        observed = {arm: metric_vector(truth, labels) for arm, labels in arm_labels.items()}
        boot = {arm: np.empty((args.bootstrap, len(METRICS)), dtype=float) for arm in ARMS}
        for replicate in range(args.bootstrap):
            sample = sample_indices(cohort_name, payload["resampling_units"], rng)
            sampled_truth = truth[sample]
            for arm in ARMS:
                boot[arm][replicate] = metric_vector(sampled_truth, arm_labels[arm][sample])
        for arm in ARMS:
            row: dict[str, object] = {"cohort": cohort_name, "arm": arm, "records": len(truth)}
            for metric_index, metric_name in enumerate(METRICS):
                low, high = interval(boot[arm][:, metric_index])
                row[metric_name] = float(observed[arm][metric_index])
                row[f"{metric_name}_ci95_lower"] = low
                row[f"{metric_name}_ci95_upper"] = high
            metric_rows.append(row)
        effects = {
            "fusion_effect_forced": ("fused_forced", "sequence_forced"),
            "fusion_effect_selective": ("fused_selective", "sequence_selective"),
            "selection_effect_sequence": ("sequence_selective", "sequence_forced"),
            "selection_effect_fused": ("fused_selective", "fused_forced"),
        }
        for effect_name, (left, right) in effects.items():
            for metric_index, metric_name in enumerate(METRICS):
                observed_difference = float(
                    observed[left][metric_index] - observed[right][metric_index]
                )
                bootstrap_difference = boot[left][:, metric_index] - boot[right][:, metric_index]
                low, high = interval(bootstrap_difference)
                effect_rows.append(
                    {
                        "cohort": cohort_name,
                        "effect": effect_name,
                        "left_arm": left,
                        "right_arm": right,
                        "metric": metric_name,
                        "difference": observed_difference,
                        "ci95_lower": low,
                        "ci95_upper": high,
                        "bootstrap_replicates": args.bootstrap,
                    }
                )
        for metric_index, metric_name in enumerate(METRICS):
            interaction = (
                observed["fused_selective"][metric_index]
                - observed["fused_forced"][metric_index]
                - observed["sequence_selective"][metric_index]
                + observed["sequence_forced"][metric_index]
            )
            bootstrap_interaction = (
                boot["fused_selective"][:, metric_index]
                - boot["fused_forced"][:, metric_index]
                - boot["sequence_selective"][:, metric_index]
                + boot["sequence_forced"][:, metric_index]
            )
            low, high = interval(bootstrap_interaction)
            effect_rows.append(
                {
                    "cohort": cohort_name,
                    "effect": "fusion_by_selection_interaction",
                    "left_arm": "factorial_difference_in_differences",
                    "right_arm": "",
                    "metric": metric_name,
                    "difference": float(interaction),
                    "ci95_lower": low,
                    "ci95_upper": high,
                    "bootstrap_replicates": args.bootstrap,
                }
            )
        report_cohorts[cohort_name] = {
            "resampling_method": payload["resampling_method"],
            "arms": {
                arm: {metric: float(value) for metric, value in zip(METRICS, observed[arm])}
                for arm in ARMS
            },
        }

    metric_fields = ["cohort", "arm", "records"]
    for metric_name in METRICS:
        metric_fields.extend(
            [metric_name, f"{metric_name}_ci95_lower", f"{metric_name}_ci95_upper"]
        )
    write_tsv(output_dir / "factorial_ablation_metrics.tsv", metric_rows, metric_fields)
    write_tsv(
        output_dir / "factorial_ablation_effects.tsv",
        effect_rows,
        [
            "cohort",
            "effect",
            "left_arm",
            "right_arm",
            "metric",
            "difference",
            "ci95_lower",
            "ci95_upper",
            "bootstrap_replicates",
        ],
    )
    manuscript_rows: list[dict[str, object]] = []
    for row in metric_rows:
        manuscript_rows.append(
            {
                "cohort": row["cohort"],
                "arm": row["arm"],
                "coverage": f"{float(row['prediction_coverage']):.2f}",
                "macro_f1_95ci": (
                    f"{float(row['macro_f1']):.2f} "
                    f"({float(row['macro_f1_ci95_lower']):.2f}-{float(row['macro_f1_ci95_upper']):.2f})"
                ),
                "plasmid_precision_95ci": (
                    f"{float(row['plasmid_precision']):.2f} "
                    f"({float(row['plasmid_precision_ci95_lower']):.2f}-{float(row['plasmid_precision_ci95_upper']):.2f})"
                ),
                "plasmid_sensitivity_95ci": (
                    f"{float(row['plasmid_sensitivity']):.2f} "
                    f"({float(row['plasmid_sensitivity_ci95_lower']):.2f}-{float(row['plasmid_sensitivity_ci95_upper']):.2f})"
                ),
                "plasmid_f1_95ci": (
                    f"{float(row['plasmid_f1']):.2f} "
                    f"({float(row['plasmid_f1_ci95_lower']):.2f}-{float(row['plasmid_f1_ci95_upper']):.2f})"
                ),
            }
        )
    write_tsv(
        output_dir / "factorial_ablation_manuscript_table.tsv",
        manuscript_rows,
        [
            "cohort",
            "arm",
            "coverage",
            "macro_f1_95ci",
            "plasmid_precision_95ci",
            "plasmid_sensitivity_95ci",
            "plasmid_f1_95ci",
        ],
    )
    figure = output_dir / "Figure_S_factorial_ablation.svg"
    make_svg(figure, metric_rows)
    report = {
        "status": "PASS",
        "analysis_scope": "post-hoc mechanistic factorial analysis using frozen models and one fixed production threshold",
        "factor_sequence_features": "sequence-only models versus fused sequence-plus-17-marker-feature models",
        "factor_decision_policy": "forced three-class argmax versus fixed plasmid-selective rejection",
        "selective_threshold": THRESHOLD,
        "threshold_retuned": False,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "cohorts": report_cohorts,
        "interpretation_boundary": "This isolates the complete 17-feature fusion block from selective rejection. It does not separate five coding-structure features from 12 MOB-suite marker features.",
    }
    (output_dir / "factorial_ablation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("===== FACTORIAL ABLATION =====")
    for row in manuscript_rows:
        print(
            f"{row['cohort']} | {row['arm']}: macro-F1 {row['macro_f1_95ci']}; "
            f"plasmid F1 {row['plasmid_f1_95ci']}; coverage {row['coverage']}"
        )
    print(f"Table: {output_dir / 'factorial_ablation_manuscript_table.tsv'}")
    print(f"Effects: {output_dir / 'factorial_ablation_effects.tsv'}")
    print(f"Figure: {figure}")
    print(f"Report: {output_dir / 'factorial_ablation_report.json'}")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
