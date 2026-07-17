#!/usr/bin/env python3
"""Run the complete locked validation and calibration sequence for a candidate."""

# The repository root is added before importing sibling script modules.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_classifier_errors import analyze_errors
from scripts.calibrate_model import calibrate_model
from scripts.evaluate_fasta_model import evaluate_fasta_model, evaluate_fasta_models
from scripts.evaluate_feature_rows import evaluate_feature_rows


def run_validation(
    candidate_dir: Path,
    dataset_dir: Path,
    benchmark_dir: Path,
    *,
    baseline_model: Path | None = None,
    phage_manifest: Path | None = None,
    tiers: tuple[str, ...] = ("tier1", "tier2"),
) -> dict[str, object]:
    """Calibrate on validation only, then run every locked evaluation."""

    model_path = candidate_dir / "mlp_v2.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Candidate model is not ready: {model_path}")
    features_path = dataset_dir / "features.npy"
    ids_path = dataset_dir / "seq_ids.txt"
    split_manifest = candidate_dir / "split_manifest.tsv"
    evaluation_dir = candidate_dir / "evaluation"

    validation_dir = evaluation_dir / "validation_uncalibrated"
    evaluate_feature_rows(
        features_path,
        model_path,
        split_manifest,
        validation_dir,
        split="validation",
        ids_path=ids_path,
    )

    calibrated_model = candidate_dir / "mlp_v2_calibrated.pt"
    calibration_report = candidate_dir / "calibration.json"
    calibrate_model(
        validation_dir / "predictions.npz",
        model_path,
        calibrated_model,
        calibration_report,
    )

    internal_test_metrics = evaluate_feature_rows(
        features_path,
        calibrated_model,
        split_manifest,
        evaluation_dir / "internal_test_calibrated",
        split="test",
        ids_path=ids_path,
    )
    if phage_manifest is None:
        phage_manifest = benchmark_dir / "locked_phage_rows.tsv"
    locked_phage_metrics = evaluate_feature_rows(
        features_path,
        calibrated_model,
        phage_manifest,
        evaluation_dir / f"{phage_manifest.stem}_calibrated",
    )

    tier_directories = {
        "tier1": benchmark_dir / "tier1" / "all_species",
        "tier2": benchmark_dir / "tier2_metagenome",
    }
    unknown_tiers = set(tiers) - set(tier_directories)
    if unknown_tiers:
        raise ValueError(f"Unknown benchmark tiers: {sorted(unknown_tiers)}")
    tier_results: dict[str, object] = {}
    for tier_name in tiers:
        tier_directory = tier_directories[tier_name]
        tier_out = evaluation_dir / f"{tier_name}_candidate"
        if baseline_model is None:
            metrics = evaluate_fasta_model(
                tier_directory / "input.fasta",
                tier_directory / "labels.tsv",
                calibrated_model,
                tier_out,
            )
            paired_results = {"candidate": metrics}
        else:
            baseline_out = evaluation_dir / f"{tier_name}_production_baseline"
            paired_results = evaluate_fasta_models(
                tier_directory / "input.fasta",
                tier_directory / "labels.tsv",
                {
                    "candidate": calibrated_model,
                    "production_baseline": baseline_model,
                },
                {
                    "candidate": tier_out,
                    "production_baseline": baseline_out,
                },
            )
            metrics = paired_results["candidate"]
        analyze_errors(
            tier_out / "predictions.tsv",
            tier_out / "error_analysis",
        )
        tier_results[tier_name] = metrics

        if baseline_model is not None:
            baseline_metrics = paired_results["production_baseline"]
            analyze_errors(
                baseline_out / "predictions.tsv",
                baseline_out / "error_analysis",
            )
            tier_results[f"{tier_name}_production_baseline"] = baseline_metrics

    summary: dict[str, object] = {
        "candidate_model": str(model_path),
        "calibrated_model": str(calibrated_model),
        "calibration_report": str(calibration_report),
        "baseline_model": str(baseline_model) if baseline_model else None,
        "phage_manifest": str(phage_manifest),
        "internal_test": internal_test_metrics,
        "locked_phage": locked_phage_metrics,
        "evaluated_tiers": list(tiers),
        "tier_results": {
            name: {
                "argmax": result["argmax"],
                "production_thresholds": result["production_thresholds"],
            }
            for name, result in tier_results.items()
        },
    }
    (evaluation_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    stage_name = "-".join(tiers) if tiers else "internal-only"
    (evaluation_dir / f"validation_summary_{stage_name}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--phage-manifest", type=Path)
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=["tier1", "tier2"],
        default=["tier1", "tier2"],
    )
    args = parser.parse_args()

    summary = run_validation(
        args.candidate_dir,
        args.dataset_dir,
        args.benchmark_dir,
        baseline_model=args.baseline_model,
        phage_manifest=args.phage_manifest,
        tiers=tuple(args.tiers),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
