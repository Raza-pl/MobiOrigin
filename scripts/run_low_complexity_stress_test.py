#!/usr/bin/env python3
"""End-to-end low-complexity stress test for MobiOrigin v0.1.7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PATTERNS = ("homopolymer_A", "dinucleotide_AT", "local_12mer_tandem")
CLASS_NAMES = ("chromosome", "plasmid", "phage")


def stable_rng(seed: int, identifier: str, pattern: str, fraction: float) -> np.random.Generator:
    material = f"{seed}|{identifier}|{pattern}|{fraction:.8f}".encode()
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(value)


def replace_with_low_complexity(
    sequence: str,
    pattern: str,
    fraction: float,
    seed: int,
    identifier: str,
) -> tuple[str, int, int]:
    count = max(1, min(len(sequence), int(round(len(sequence) * fraction))))
    rng = stable_rng(seed, identifier, pattern, fraction)
    start = int(rng.integers(0, len(sequence) - count + 1))
    if pattern == "homopolymer_A":
        replacement = "A" * count
    elif pattern == "dinucleotide_AT":
        replacement = ("AT" * ((count + 1) // 2))[:count]
    elif pattern == "local_12mer_tandem":
        motif_length = min(12, len(sequence))
        motif_start = int(rng.integers(0, len(sequence) - motif_length + 1))
        motif = sequence[motif_start : motif_start + motif_length]
        replacement = (motif * ((count + motif_length - 1) // motif_length))[:count]
    else:
        raise RuntimeError(f"Unsupported pattern: {pattern}")
    return sequence[:start] + replacement + sequence[start + count :], start, count


def normalized_kmer_entropy(sequence: str, k: int = 4) -> float:
    if len(sequence) < k:
        return 0.0
    mapping = np.full(256, -1, dtype=np.int16)
    for character, value in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
        mapping[ord(character)] = value
    encoded = mapping[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    windows = np.lib.stride_tricks.sliding_window_view(encoded, k)
    valid = np.all(windows >= 0, axis=1)
    if not np.any(valid):
        return 0.0
    powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    identifiers = windows[valid].astype(np.int64) @ powers
    counts = np.bincount(identifiers, minlength=4**k).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(4**k))


def make_svg(path: Path, summary: list[dict[str, object]]) -> None:
    width, height = 1800, 900
    lefts = (130, 1020)
    top, plot_w, plot_h = 160, 700, 550
    colors = {
        "homopolymer_A": "#D55E00",
        "dinucleotide_AT": "#CC79A7",
        "local_12mer_tandem": "#0072B2",
    }
    labels = {
        "homopolymer_A": "Homopolymer A",
        "dinucleotide_AT": "AT repeat",
        "local_12mer_tandem": "Local 12-mer tandem repeat",
    }
    positive = [row for row in summary if float(row["replaced_fraction"]) > 0]
    fractions = sorted({float(row["replaced_fraction"]) for row in positive})
    max_fraction = max(fractions)
    max_flip = max(float(row["label_change_rate_ci95_upper"]) for row in positive)
    flip_max = max(0.10, float(np.ceil(max_flip * 20) / 20))
    f1_values = [float(row["macro_f1"]) for row in summary]
    f1_min = max(0.0, float(np.floor((min(f1_values) - 0.03) * 20) / 20))
    f1_max = min(1.0, float(np.ceil((max(f1_values) + 0.03) * 20) / 20))
    if f1_max - f1_min < 0.10:
        f1_min = max(0.0, f1_min - 0.05)
        f1_max = min(1.0, f1_max + 0.05)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.axis{stroke:#222;stroke-width:2}.grid{stroke:#ddd;stroke-width:1}</style>",
        '<text x="900" y="48" text-anchor="middle" font-size="32" font-weight="700">Low-complexity sequence stress test</text>',
        '<text x="900" y="82" text-anchor="middle" font-size="20">Same stratified external subset; frozen models, databases and threshold</text>',
        '<text x="55" y="128" font-size="30" font-weight="700">A</text>',
        '<text x="945" y="128" font-size="30" font-weight="700">B</text>',
        '<text x="480" y="128" text-anchor="middle" font-size="25" font-weight="700">Prediction stability</text>',
        '<text x="1370" y="128" text-anchor="middle" font-size="25" font-weight="700">Multiclass performance</text>',
    ]
    panels = ((0.0, flip_max, "Label-change rate"), (f1_min, f1_max, "Macro-F1"))
    for panel, (ymin, ymax, ytitle) in enumerate(panels):
        left = lefts[panel]
        for tick in np.linspace(ymin, ymax, 6):
            y = top + plot_h - (tick - ymin) / (ymax - ymin) * plot_h
            parts.append(
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>'
            )
            tick_text = f"{tick:.0%}" if panel == 0 else f"{tick:.2f}"
            parts.append(
                f'<text x="{left - 14}" y="{y + 7:.1f}" text-anchor="end" font-size="19">{tick_text}</text>'
            )
        for pattern in PATTERNS:
            rows = sorted(
                [row for row in summary if row["pattern"] == pattern],
                key=lambda row: float(row["replaced_fraction"]),
            )
            points: list[str] = []
            for row in rows:
                fraction = float(row["replaced_fraction"])
                x = left + fraction / max_fraction * plot_w
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
            x = left + fraction / max_fraction * plot_w
            parts.append(
                f'<text x="{x:.1f}" y="{top + plot_h + 38}" text-anchor="middle" font-size="18">{fraction:.0%}</text>'
            )
        parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
        parts.append(
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>'
        )
        parts.append(
            f'<text x="{left + plot_w / 2}" y="{top + plot_h + 76}" text-anchor="middle" font-size="21">Sequence replaced by low-complexity tract</text>'
        )
        parts.append(
            f'<text x="{left - 92}" y="{top + plot_h / 2}" text-anchor="middle" font-size="21" transform="rotate(-90 {left - 92} {top + plot_h / 2})">{ytitle}</text>'
        )
    legend_x = (360, 760, 1150)
    for x, pattern in zip(legend_x, PATTERNS):
        parts.append(
            f'<line x1="{x}" y1="835" x2="{x + 65}" y2="835" stroke="{colors[pattern]}" stroke-width="5"/>'
        )
        parts.append(f'<text x="{x + 80}" y="842" font-size="20">{labels[pattern]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--selected-subset", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--database-dir", required=True, type=Path)
    parser.add_argument("--diamond", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--records-per-stratum", type=int, default=20)
    parser.add_argument("--fractions", default="0.01,0.05,0.10,0.20")
    parser.add_argument("--patterns", default=",".join(PATTERNS))
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    fractions = sorted({float(value) for value in args.fractions.split(",")})
    patterns = tuple(value.strip() for value in args.patterns.split(",") if value.strip())
    if any(pattern not in PATTERNS for pattern in patterns):
        raise RuntimeError("Unsupported low-complexity pattern")
    if not fractions or fractions[0] <= 0 or fractions[-1] > 0.50:
        raise RuntimeError("Fractions must be greater than 0 and no more than 0.50")

    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for location in (root / "src", root / "scripts"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    os.environ["MOBIORIGIN_DIAMOND"] = str(args.diamond.resolve())
    from mobiorigin.predict import predict
    from run_ambiguity_stress_test import (
        encode,
        file_sha256,
        read_tsv,
        wilson,
        write_tsv,
    )
    from run_factorial_ablation import metric_vector

    selected_all = read_tsv(args.selected_subset.resolve())
    by_stratum: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected_all:
        by_stratum[(row["true_class"], row["length_bin"])].append(row)
    if len(by_stratum) != 15 or any(
        len(rows) < args.records_per_stratum for rows in by_stratum.values()
    ):
        raise RuntimeError("Selected subset does not contain the requested class-by-length balance")
    selected_rows: list[dict[str, str]] = []
    for key in sorted(by_stratum):
        selected_rows.extend(by_stratum[key][: args.records_per_stratum])
    selected_ids = [row["opaque_contig_id"] for row in selected_rows]
    selected_set = set(selected_ids)

    external = root / "evaluation/release_audit/mobiorigin_external_validation_20260817"
    fasta_path = (
        external
        / "024_corrected_final_external_cohort_freeze"
        / "mobiorigin_external_validation_cohort_3000.fasta"
    )
    original_path = (
        external
        / "029_dual_external_prediction_execution"
        / "mobiorigin_prediction_output/predictions.tsv"
    )
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
        raise RuntimeError("Could not recover every selected sequence")

    original_by_id = {
        row["sequence_id"]: row
        for row in read_tsv(original_path)
        if row["sequence_id"] in selected_set
    }
    if set(original_by_id) != selected_set:
        raise RuntimeError("Original prediction inventory is incomplete")
    truth = encode([row["true_class"] for row in selected_rows])
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
    original_entropy = np.asarray([normalized_kmer_entropy(sequences[x]) for x in selected_ids])
    baseline_metrics = metric_vector(truth, original_label_codes)

    summary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    stratum_rows: list[dict[str, object]] = []
    for pattern in patterns:
        summary_rows.append(
            {
                "pattern": pattern,
                "replaced_fraction": 0.0,
                "records": len(selected_ids),
                "label_changes": 0,
                "label_change_rate": 0.0,
                "label_change_rate_ci95_lower": 0.0,
                "label_change_rate_ci95_upper": 0.0,
                "median_max_probability_difference": 0.0,
                "median_normalized_4mer_entropy": float(np.median(original_entropy)),
                "prediction_coverage": float(baseline_metrics[0]),
                "macro_f1": float(baseline_metrics[1]),
                "balanced_accuracy": float(baseline_metrics[2]),
                "plasmid_precision": float(baseline_metrics[3]),
                "plasmid_sensitivity": float(baseline_metrics[4]),
                "plasmid_f1": float(baseline_metrics[5]),
            }
        )

    inputs = output / "temporary_inputs"
    runs = output / "condition_runs"
    inputs.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)
    for pattern in patterns:
        for fraction in fractions:
            tag = f"{pattern}_{fraction:.4f}".replace(".", "p")
            fasta = inputs / f"{tag}.fasta"
            positions: dict[str, int] = {}
            replaced_counts: dict[str, int] = {}
            entropy: dict[str, float] = {}
            with fasta.open("w", encoding="ascii") as handle:
                for identifier in selected_ids:
                    mutated, start, count = replace_with_low_complexity(
                        sequences[identifier], pattern, fraction, args.seed, identifier
                    )
                    positions[identifier] = start
                    replaced_counts[identifier] = count
                    entropy[identifier] = normalized_kmer_entropy(mutated)
                    handle.write(f">{identifier}\n{mutated}\n")
            fasta_hash = file_sha256(fasta)
            run_dir = runs / tag
            reusable = False
            if (run_dir / "predictions.tsv").is_file() and (run_dir / "provenance.json").is_file():
                provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
                reusable = provenance.get("input_fasta_sha256") == fasta_hash
            if not reusable:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                print(
                    f"Running {pattern} at {fraction:.0%} ({len(selected_ids)} records)...",
                    flush=True,
                )
                predict(
                    input_fasta=fasta,
                    output_dir=run_dir,
                    database_dir=args.database_dir.resolve(),
                    model_dir=args.model_dir.resolve(),
                    threads=args.threads,
                    progress=lambda message: print(f"  {message}", flush=True),
                )
            else:
                print(f"Reusing verified condition: {pattern} {fraction:.0%}", flush=True)
            fasta.unlink(missing_ok=True)
            result_by_id = {
                row["sequence_id"]: row for row in read_tsv(run_dir / "predictions.tsv")
            }
            if set(result_by_id) != selected_set:
                raise RuntimeError(f"Condition output inventory changed: {tag}")
            labels = [result_by_id[identifier]["prediction"] for identifier in selected_ids]
            codes = encode(labels)
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
            changed = codes != original_label_codes
            probability_delta = np.max(np.abs(probabilities - original_probabilities), axis=1)
            metrics = metric_vector(truth, codes)
            changes = int(np.count_nonzero(changed))
            low, high = wilson(changes, len(changed))
            summary_rows.append(
                {
                    "pattern": pattern,
                    "replaced_fraction": fraction,
                    "records": len(selected_ids),
                    "label_changes": changes,
                    "label_change_rate": changes / len(changed),
                    "label_change_rate_ci95_lower": low,
                    "label_change_rate_ci95_upper": high,
                    "median_max_probability_difference": float(np.median(probability_delta)),
                    "median_normalized_4mer_entropy": float(
                        np.median([entropy[x] for x in selected_ids])
                    ),
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
                        "replaced_fraction": fraction,
                        "opaque_contig_id": identifier,
                        "true_class": row["true_class"],
                        "length_bin": row["length_bin"],
                        "length_bp": row["length_bp"],
                        "replacement_start_0based": positions[identifier],
                        "replaced_bases": replaced_counts[identifier],
                        "original_entropy": float(original_entropy[index]),
                        "perturbed_entropy": entropy[identifier],
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
                            i
                            for i, row in enumerate(selected_rows)
                            if row["true_class"] == class_name and row["length_bin"] == length_bin
                        ],
                        dtype=np.int32,
                    )
                    count = int(np.count_nonzero(changed[indices]))
                    s_low, s_high = wilson(count, len(indices))
                    stratum_rows.append(
                        {
                            "pattern": pattern,
                            "replaced_fraction": fraction,
                            "true_class": class_name,
                            "length_bin": length_bin,
                            "records": len(indices),
                            "label_changes": count,
                            "label_change_rate": count / len(indices),
                            "label_change_rate_ci95_lower": s_low,
                            "label_change_rate_ci95_upper": s_high,
                        }
                    )

    summary_fields = [
        "pattern",
        "replaced_fraction",
        "records",
        "label_changes",
        "label_change_rate",
        "label_change_rate_ci95_lower",
        "label_change_rate_ci95_upper",
        "median_max_probability_difference",
        "median_normalized_4mer_entropy",
        "prediction_coverage",
        "macro_f1",
        "balanced_accuracy",
        "plasmid_precision",
        "plasmid_sensitivity",
        "plasmid_f1",
    ]
    write_tsv(output / "low_complexity_summary.tsv", summary_rows, summary_fields)
    write_tsv(
        output / "low_complexity_by_stratum.tsv",
        stratum_rows,
        [
            "pattern",
            "replaced_fraction",
            "true_class",
            "length_bin",
            "records",
            "label_changes",
            "label_change_rate",
            "label_change_rate_ci95_lower",
            "label_change_rate_ci95_upper",
        ],
    )
    write_tsv(
        output / "low_complexity_pair_results.tsv",
        pair_rows,
        [
            "pattern",
            "replaced_fraction",
            "opaque_contig_id",
            "true_class",
            "length_bin",
            "length_bp",
            "replacement_start_0based",
            "replaced_bases",
            "original_entropy",
            "perturbed_entropy",
            "original_label",
            "perturbed_label",
            "label_changed",
            "maximum_probability_difference",
        ],
    )
    figure = output / "Figure_S_low_complexity_stress.svg"
    # The full production run always uses all three patterns.
    if patterns == PATTERNS:
        make_svg(figure, summary_rows)
    report = {
        "status": "PASS",
        "analysis_scope": "post-hoc end-to-end synthetic low-complexity sensitivity analysis",
        "cohort": "same stratified external subset used for the ambiguity stress test",
        "records": len(selected_ids),
        "patterns": list(patterns),
        "replaced_fractions": fractions,
        "model_database_or_threshold_retuned": False,
        "complexity_measure": "normalized Shannon entropy of observed 4-mers",
        "interpretation_boundary": "These deliberately synthetic perturbations probe sensitivity. They do not estimate the frequency or form of low-complexity sequence in real assemblies, and the patterns also alter local base composition and coding potential.",
        "summary": summary_rows,
    }
    (output / "low_complexity_stress_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("===== LOW-COMPLEXITY STRESS TEST =====")
    for row in summary_rows:
        if float(row["replaced_fraction"]) > 0:
            print(
                f"{row['pattern']} {float(row['replaced_fraction']):.0%}: "
                f"label changes {int(row['label_changes'])}/{int(row['records'])} "
                f"({float(row['label_change_rate']):.2%}); "
                f"macro-F1 {float(row['macro_f1']):.4f}"
            )
    print(f"Summary: {output / 'low_complexity_summary.tsv'}")
    print(f"Strata: {output / 'low_complexity_by_stratum.tsv'}")
    print(f"Pairs: {output / 'low_complexity_pair_results.tsv'}")
    if figure.is_file():
        print(f"Figure: {figure}")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
