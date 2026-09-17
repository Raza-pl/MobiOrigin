#!/usr/bin/env python3
"""Controlled end-to-end IUPAC-N stress test for MobiOrigin v0.1.7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASS_NAMES = ("chromosome", "plasmid", "phage")
PATTERNS = ("dispersed", "contiguous_block")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(master_seed: int, identifier: str, pattern: str, fraction: float) -> int:
    material = f"{master_seed}|{identifier}|{pattern}|{fraction:.8f}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def inject_n(
    sequence: str,
    fraction: float,
    pattern: str,
    master_seed: int,
    identifier: str,
) -> tuple[str, int]:
    count = max(1, int(round(len(sequence) * fraction)))
    count = min(count, len(sequence))
    rng = np.random.default_rng(stable_seed(master_seed, identifier, pattern, fraction))
    values = np.frombuffer(sequence.encode("ascii"), dtype="S1").copy()
    if pattern == "dispersed":
        indices = rng.choice(len(values), size=count, replace=False)
    elif pattern == "contiguous_block":
        start = int(rng.integers(0, len(values) - count + 1))
        indices = np.arange(start, start + count)
    else:
        raise RuntimeError(f"Unsupported ambiguity pattern: {pattern}")
    values[indices] = b"N"
    return values.tobytes().decode("ascii"), count


def wilson(successes: int, trials: int) -> tuple[float, float]:
    if not trials:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return float(center - half), float(center + half)


def encode(labels: list[str]) -> np.ndarray:
    mapping = {"chromosome": 0, "plasmid": 1, "phage": 2, "unclassified": 3}
    return np.asarray([mapping[label] for label in labels], dtype=np.int8)


def make_svg(path: Path, summaries: list[dict[str, object]]) -> None:
    width, height = 1780, 900
    lefts = (130, 1010)
    top, plot_w, plot_h = 160, 700, 560
    colors = {"dispersed": "#0072B2", "contiguous_block": "#D55E00"}
    positive = [row for row in summaries if float(row["injected_fraction"]) > 0]
    fractions = sorted({float(row["injected_fraction"]) for row in positive})
    max_flip = max(float(row["label_change_rate_ci95_upper"]) for row in positive)
    flip_max = max(0.10, float(np.ceil(max_flip * 20) / 20))
    f1_values = [float(row["macro_f1"]) for row in summaries]
    f1_min = max(0.0, np.floor((min(f1_values) - 0.03) * 20) / 20)
    f1_max = min(1.0, np.ceil((max(f1_values) + 0.03) * 20) / 20)
    if f1_max - f1_min < 0.10:
        f1_min = max(0.0, f1_min - 0.05)
        f1_max = min(1.0, f1_max + 0.05)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.axis{stroke:#222;stroke-width:2}.grid{stroke:#ddd;stroke-width:1}</style>",
        '<text x="890" y="48" text-anchor="middle" font-size="32" font-weight="700">Controlled ambiguous-base stress test</text>',
        '<text x="890" y="82" text-anchor="middle" font-size="20">Stratified external subset; frozen models, databases and threshold</text>',
        '<text x="55" y="128" font-size="30" font-weight="700">A</text>',
        '<text x="935" y="128" font-size="30" font-weight="700">B</text>',
        '<text x="480" y="128" text-anchor="middle" font-size="25" font-weight="700">Prediction stability</text>',
        '<text x="1360" y="128" text-anchor="middle" font-size="25" font-weight="700">Multiclass performance</text>',
    ]
    for panel, (ymin, ymax, title) in enumerate(
        ((0.0, flip_max, "Label-change rate"), (f1_min, f1_max, "Macro-F1"))
    ):
        left = lefts[panel]
        for tick in np.linspace(ymin, ymax, 6):
            y = top + plot_h - (tick - ymin) / (ymax - ymin) * plot_h
            parts.append(
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>'
            )
            text = f"{tick:.0%}" if panel == 0 else f"{tick:.2f}"
            parts.append(
                f'<text x="{left - 14}" y="{y + 7:.1f}" text-anchor="end" font-size="19">{text}</text>'
            )
        for pattern in PATTERNS:
            rows = sorted(
                [row for row in summaries if row["pattern"] == pattern],
                key=lambda row: float(row["injected_fraction"]),
            )
            points: list[str] = []
            for row in rows:
                fraction = float(row["injected_fraction"])
                x = left + fraction / max(fractions) * plot_w
                value = float(row["label_change_rate"] if panel == 0 else row["macro_f1"])
                y = top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
                points.append(f"{x:.1f},{y:.1f}")
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{colors[pattern]}"/>')
                if panel == 0 and fraction > 0:
                    low = float(row["label_change_rate_ci95_lower"])
                    high = float(row["label_change_rate_ci95_upper"])
                    y1 = top + plot_h - low / ymax * plot_h
                    y2 = top + plot_h - high / ymax * plot_h
                    parts.append(
                        f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{colors[pattern]}" stroke-width="2"/>'
                    )
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[pattern]}" stroke-width="4"/>'
            )
        for fraction in [0.0, *fractions]:
            x = left + fraction / max(fractions) * plot_w
            parts.append(
                f'<text x="{x:.1f}" y="{top + plot_h + 38}" text-anchor="middle" font-size="18">{fraction:.1%}</text>'
            )
        parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
        parts.append(
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>'
        )
        parts.append(
            f'<text x="{left + plot_w / 2}" y="{top + plot_h + 76}" text-anchor="middle" font-size="21">Bases replaced with N</text>'
        )
        parts.append(
            f'<text x="{left - 92}" y="{top + plot_h / 2}" text-anchor="middle" font-size="21" transform="rotate(-90 {left - 92} {top + plot_h / 2})">{title}</text>'
        )
    parts.append(
        '<line x1="625" y1="840" x2="690" y2="840" stroke="#0072B2" stroke-width="5"/><text x="706" y="847" font-size="21">Dispersed Ns</text>'
    )
    parts.append(
        '<line x1="990" y1="840" x2="1055" y2="840" stroke="#D55E00" stroke-width="5"/><text x="1071" y="847" font-size="21">Contiguous N block</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--database-dir", required=True, type=Path)
    parser.add_argument("--diamond", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--records-per-stratum", type=int, default=20)
    parser.add_argument("--fractions", default="0.001,0.005,0.01,0.02,0.05")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    fractions = sorted({float(value) for value in args.fractions.split(",")})
    if not fractions or fractions[0] <= 0 or fractions[-1] > 0.10:
        raise RuntimeError("Ambiguity fractions must be greater than 0 and no more than 0.10")
    if args.records_per_stratum < 1 or args.records_per_stratum > 200:
        raise RuntimeError("records-per-stratum must be between 1 and 200")

    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = root / "src"
    scripts = root / "scripts"
    for location in (source, scripts):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    os.environ["MOBIORIGIN_DIAMOND"] = str(args.diamond.resolve())
    from mobiorigin.predict import predict
    from run_factorial_ablation import metric_vector

    external = root / "evaluation/release_audit/mobiorigin_external_validation_20260817"
    freeze = external / "024_corrected_final_external_cohort_freeze"
    fasta_path = freeze / "mobiorigin_external_validation_cohort_3000.fasta"
    truth_path = freeze / "mobiorigin_external_validation_label_map.sealed.tsv"
    original_prediction_path = (
        external
        / "029_dual_external_prediction_execution"
        / "mobiorigin_prediction_output/predictions.tsv"
    )
    truth_rows = read_tsv(truth_path)
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in truth_rows:
        strata[(row["class"], row["length_bin"])].append(row)
    if len(strata) != 15 or any(len(rows) != 200 for rows in strata.values()):
        raise RuntimeError("External class-by-length structure changed")
    selection_rng = np.random.default_rng(args.seed)
    selected_rows: list[dict[str, str]] = []
    for key in sorted(strata):
        rows = strata[key]
        chosen = selection_rng.choice(len(rows), size=args.records_per_stratum, replace=False)
        selected_rows.extend(rows[int(index)] for index in sorted(chosen))
    selected_ids = [row["opaque_contig_id"] for row in selected_rows]
    selected_set = set(selected_ids)

    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []

    def retain_current() -> None:
        if current_id in selected_set:
            sequences[str(current_id)] = "".join(chunks).upper()

    with fasta_path.open("r", encoding="ascii") as handle:
        for raw in handle:
            if raw.startswith(">"):
                retain_current()
                current_id = raw[1:].split(None, 1)[0]
                chunks = []
            elif current_id in selected_set:
                chunks.append(raw.strip())
    retain_current()
    if set(sequences) != selected_set:
        raise RuntimeError("Could not recover every selected external sequence")

    original_by_id = {
        row["sequence_id"]: row
        for row in read_tsv(original_prediction_path)
        if row["sequence_id"] in selected_set
    }
    if set(original_by_id) != selected_set:
        raise RuntimeError("Original predictions do not cover the selected subset")
    truth_names = [row["class"] for row in selected_rows]
    truth = encode(truth_names)
    original_labels = [original_by_id[identifier]["prediction"] for identifier in selected_ids]
    original_label_codes = encode(original_labels)
    original_probabilities = np.asarray(
        [
            [
                float(original_by_id[identifier]["p_chromosome"]),
                float(original_by_id[identifier]["p_plasmid"]),
                float(original_by_id[identifier]["p_phage"]),
            ]
            for identifier in selected_ids
        ],
        dtype=np.float32,
    )
    selection_rows = [
        {
            "selection_order": index + 1,
            "opaque_contig_id": row["opaque_contig_id"],
            "true_class": row["class"],
            "length_bin": row["length_bin"],
            "length_bp": row["fragment_length_bp"],
        }
        for index, row in enumerate(selected_rows)
    ]
    write_tsv(
        output / "selected_external_subset.tsv",
        selection_rows,
        ["selection_order", "opaque_contig_id", "true_class", "length_bin", "length_bp"],
    )

    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    stratum_rows: list[dict[str, object]] = []
    baseline_metric = metric_vector(truth, original_label_codes)
    for pattern in PATTERNS:
        summary_rows.append(
            {
                "pattern": pattern,
                "injected_fraction": 0.0,
                "records": len(selected_ids),
                "label_changes": 0,
                "label_change_rate": 0.0,
                "label_change_rate_ci95_lower": 0.0,
                "label_change_rate_ci95_upper": 0.0,
                "median_max_probability_difference": 0.0,
                "p95_max_probability_difference": 0.0,
                "prediction_coverage": float(baseline_metric[0]),
                "macro_f1": float(baseline_metric[1]),
                "balanced_accuracy": float(baseline_metric[2]),
                "plasmid_precision": float(baseline_metric[3]),
                "plasmid_sensitivity": float(baseline_metric[4]),
                "plasmid_f1": float(baseline_metric[5]),
            }
        )

    inputs = output / "temporary_inputs"
    runs = output / "condition_runs"
    inputs.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)
    for pattern in PATTERNS:
        for fraction in fractions:
            tag = f"{pattern}_{fraction:.4f}".replace(".", "p")
            condition_fasta = inputs / f"{tag}.fasta"
            mutation_counts: dict[str, int] = {}
            with condition_fasta.open("w", encoding="ascii") as handle:
                for identifier in selected_ids:
                    mutated, count = inject_n(
                        sequences[identifier], fraction, pattern, args.seed, identifier
                    )
                    mutation_counts[identifier] = count
                    handle.write(f">{identifier}\n{mutated}\n")
            condition_hash = file_sha256(condition_fasta)
            run_dir = runs / tag
            reusable = False
            if (run_dir / "predictions.tsv").is_file() and (run_dir / "provenance.json").is_file():
                provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
                reusable = provenance.get("input_fasta_sha256") == condition_hash
            if not reusable:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                print(
                    f"Running {pattern} ambiguity at {fraction:.1%} "
                    f"({len(selected_ids)} records)...",
                    flush=True,
                )
                predict(
                    input_fasta=condition_fasta,
                    output_dir=run_dir,
                    database_dir=args.database_dir.resolve(),
                    model_dir=args.model_dir.resolve(),
                    threads=args.threads,
                    progress=lambda message: print(f"  {message}", flush=True),
                )
            else:
                print(f"Reusing verified condition: {pattern} {fraction:.1%}", flush=True)
            condition_fasta.unlink(missing_ok=True)
            result_rows = read_tsv(run_dir / "predictions.tsv")
            result_by_id = {row["sequence_id"]: row for row in result_rows}
            if set(result_by_id) != selected_set:
                raise RuntimeError(f"Condition output inventory changed: {tag}")
            labels = [result_by_id[identifier]["prediction"] for identifier in selected_ids]
            label_codes = encode(labels)
            probabilities = np.asarray(
                [
                    [
                        float(result_by_id[identifier]["p_chromosome"]),
                        float(result_by_id[identifier]["p_plasmid"]),
                        float(result_by_id[identifier]["p_phage"]),
                    ]
                    for identifier in selected_ids
                ],
                dtype=np.float32,
            )
            changed = label_codes != original_label_codes
            probability_delta = np.max(np.abs(probabilities - original_probabilities), axis=1)
            metrics = metric_vector(truth, label_codes)
            changes = int(np.count_nonzero(changed))
            low, high = wilson(changes, len(changed))
            summary_rows.append(
                {
                    "pattern": pattern,
                    "injected_fraction": fraction,
                    "records": len(selected_ids),
                    "label_changes": changes,
                    "label_change_rate": changes / len(changed),
                    "label_change_rate_ci95_lower": low,
                    "label_change_rate_ci95_upper": high,
                    "median_max_probability_difference": float(np.median(probability_delta)),
                    "p95_max_probability_difference": float(np.percentile(probability_delta, 95)),
                    "prediction_coverage": float(metrics[0]),
                    "macro_f1": float(metrics[1]),
                    "balanced_accuracy": float(metrics[2]),
                    "plasmid_precision": float(metrics[3]),
                    "plasmid_sensitivity": float(metrics[4]),
                    "plasmid_f1": float(metrics[5]),
                }
            )
            for index, row in enumerate(selected_rows):
                identifier = selected_ids[index]
                pair_rows.append(
                    {
                        "pattern": pattern,
                        "injected_fraction": fraction,
                        "opaque_contig_id": identifier,
                        "true_class": row["class"],
                        "length_bin": row["length_bin"],
                        "length_bp": row["fragment_length_bp"],
                        "bases_replaced_with_n": mutation_counts[identifier],
                        "original_label": original_labels[index],
                        "perturbed_label": labels[index],
                        "label_changed": bool(changed[index]),
                        "maximum_probability_difference": float(probability_delta[index]),
                    }
                )
            for class_name in CLASS_NAMES:
                for length_bin in sorted({row["length_bin"] for row in selected_rows}):
                    indices = np.asarray(
                        [
                            index
                            for index, row in enumerate(selected_rows)
                            if row["class"] == class_name and row["length_bin"] == length_bin
                        ],
                        dtype=np.int32,
                    )
                    stratum_changes = int(np.count_nonzero(changed[indices]))
                    s_low, s_high = wilson(stratum_changes, len(indices))
                    stratum_rows.append(
                        {
                            "pattern": pattern,
                            "injected_fraction": fraction,
                            "true_class": class_name,
                            "length_bin": length_bin,
                            "records": len(indices),
                            "label_changes": stratum_changes,
                            "label_change_rate": stratum_changes / len(indices),
                            "label_change_rate_ci95_lower": s_low,
                            "label_change_rate_ci95_upper": s_high,
                            "median_max_probability_difference": float(
                                np.median(probability_delta[indices])
                            ),
                        }
                    )

    summary_fields = [
        "pattern",
        "injected_fraction",
        "records",
        "label_changes",
        "label_change_rate",
        "label_change_rate_ci95_lower",
        "label_change_rate_ci95_upper",
        "median_max_probability_difference",
        "p95_max_probability_difference",
        "prediction_coverage",
        "macro_f1",
        "balanced_accuracy",
        "plasmid_precision",
        "plasmid_sensitivity",
        "plasmid_f1",
    ]
    write_tsv(output / "ambiguity_stress_summary.tsv", summary_rows, summary_fields)
    write_tsv(
        output / "ambiguity_stress_by_stratum.tsv",
        stratum_rows,
        [
            "pattern",
            "injected_fraction",
            "true_class",
            "length_bin",
            "records",
            "label_changes",
            "label_change_rate",
            "label_change_rate_ci95_lower",
            "label_change_rate_ci95_upper",
            "median_max_probability_difference",
        ],
    )
    write_tsv(
        output / "ambiguity_pair_results.tsv",
        pair_rows,
        [
            "pattern",
            "injected_fraction",
            "opaque_contig_id",
            "true_class",
            "length_bin",
            "length_bp",
            "bases_replaced_with_n",
            "original_label",
            "perturbed_label",
            "label_changed",
            "maximum_probability_difference",
        ],
    )
    figure = output / "Figure_S_ambiguity_stress.svg"
    make_svg(figure, summary_rows)
    report = {
        "status": "PASS",
        "analysis_scope": "post-hoc end-to-end controlled ambiguity stress test",
        "cohort": "stratified random subset of the prospective external cohort",
        "records": len(selected_ids),
        "records_per_true_class_by_length_stratum": args.records_per_stratum,
        "selection_seed": args.seed,
        "patterns": list(PATTERNS),
        "injected_fractions": fractions,
        "model_database_or_threshold_retuned": False,
        "interpretation_boundary": "This is a synthetic robustness diagnostic. It does not estimate the prevalence or distribution of ambiguous bases in a target metagenome population.",
        "summary": summary_rows,
    }
    (output / "ambiguity_stress_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("===== AMBIGUITY STRESS TEST =====")
    for row in summary_rows:
        if float(row["injected_fraction"]) > 0:
            print(
                f"{row['pattern']} {float(row['injected_fraction']):.1%}: "
                f"label changes {int(row['label_changes'])}/{int(row['records'])} "
                f"({float(row['label_change_rate']):.2%}); "
                f"macro-F1 {float(row['macro_f1']):.4f}"
            )
    print(f"Summary: {output / 'ambiguity_stress_summary.tsv'}")
    print(f"Strata: {output / 'ambiguity_stress_by_stratum.tsv'}")
    print(f"Pairs: {output / 'ambiguity_pair_results.tsv'}")
    print(f"Figure: {figure}")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
