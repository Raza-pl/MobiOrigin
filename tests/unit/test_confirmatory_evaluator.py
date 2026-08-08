"""Synthetic, label-blind tests for the dedicated confirmatory evaluator."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark import evaluate_confirmatory as evaluator


def _records() -> list[evaluator.TruthRecord]:
    return [
        evaluator.TruthRecord("r1", 1, "plasmid", "1k_to_lt2k", "A", 1000),
        evaluator.TruthRecord("r2", 2, "chromosome", "1k_to_lt2k", "A", 1200),
        evaluator.TruthRecord("r3", 3, "phage", "2k_to_lt5k", "B", 2500),
        evaluator.TruthRecord("r4", 4, "plasmid", "2k_to_lt5k", "B", 3000),
        evaluator.TruthRecord("r5", 5, "chromosome", "5k_to_lt10k", "C", 6000),
        evaluator.TruthRecord("r6", 6, "phage", "5k_to_lt10k", "C", 7000),
    ]


def _perfect_predictions() -> dict[str, list[str]]:
    native = ["plasmid", "chromosome", "phage", "plasmid", "chromosome", "phage"]
    binary = ["plasmid", "non-plasmid", "non-plasmid", "plasmid", "non-plasmid", "non-plasmid"]
    return {
        "plasflow2": list(native),
        "genomad": list(native),
        "plasclass": list(binary),
        "plasflow_v1": list(binary),
        "plasme": list(binary),
        "platon": list(binary),
    }


def _canonical(value: dict[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_three_class_perfect_metrics() -> None:
    counts = np.diag([2, 3, 4, 0])[:3, :]
    metrics = evaluator.three_class_metrics_from_counts(counts)
    for name in (
        "macro_f1",
        "macro_recall_balanced_accuracy",
        "accuracy",
        "multiclass_mcc",
        "prediction_coverage",
    ):
        assert float(metrics[name]) == pytest.approx(1.0)
    assert float(metrics["unclassified_fraction"]) == 0.0


def test_three_class_abstentions_are_retained_and_penalized() -> None:
    counts = np.asarray([[0, 0, 0, 2], [0, 0, 0, 2], [0, 0, 0, 2]])
    metrics = evaluator.three_class_metrics_from_counts(counts)
    assert float(metrics["macro_f1"]) == 0.0
    assert float(metrics["macro_recall_balanced_accuracy"]) == 0.0
    assert float(metrics["accuracy"]) == 0.0
    assert float(metrics["multiclass_mcc"]) == 0.0
    assert float(metrics["prediction_coverage"]) == 0.0
    assert float(metrics["unclassified_fraction"]) == 1.0


def test_binary_abstentions_penalize_both_class_denominators() -> None:
    counts = np.asarray([[0, 0, 3], [0, 0, 4]])
    metrics = evaluator.binary_metrics_from_counts(counts)
    assert float(metrics["sensitivity"]) == 0.0
    assert float(metrics["specificity"]) == 0.0
    assert float(metrics["balanced_accuracy"]) == 0.0
    assert float(metrics["f1"]) == 0.0
    assert float(metrics["mcc"]) == 0.0
    assert float(metrics["prediction_coverage"]) == 0.0


def test_binary_mapping_is_frozen() -> None:
    assert evaluator.binary_label("plasmid") == "plasmid"
    assert evaluator.binary_label("chromosome") == "non-plasmid"
    assert evaluator.binary_label("phage") == "non-plasmid"
    assert evaluator.binary_label("non-plasmid") == "non-plasmid"
    assert evaluator.binary_label("unclassified") == "unclassified"
    with pytest.raises(ValueError, match="Cannot map"):
        evaluator.binary_label("virus")


def test_holm_adjustment_preserves_original_order() -> None:
    assert evaluator.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_paired_bootstrap_p_value_uses_plus_one_correction() -> None:
    assert evaluator.paired_bootstrap_p_value(np.zeros(10)) == 1.0
    assert evaluator.paired_bootstrap_p_value(np.ones(9)) == pytest.approx(0.2)


def test_shared_cluster_bootstrap_is_reproducible_and_paired() -> None:
    first = evaluator.evaluate_records(
        _records(),
        _perfect_predictions(),
        bootstrap_replicates=64,
        bootstrap_seed=17,
    )
    second = evaluator.evaluate_records(
        _records(),
        _perfect_predictions(),
        bootstrap_replicates=64,
        bootstrap_seed=17,
    )
    first_values = first["bootstrap"]["three_class"]["plasflow2"]["macro_f1"]
    second_values = second["bootstrap"]["three_class"]["plasflow2"]["macro_f1"]
    assert np.array_equal(first_values, second_values)
    three_class_rows = [
        row
        for row in first["paired_differences"]
        if row["endpoint"] == "three_class" and row["metric"] == "macro_f1"
    ]
    assert len(three_class_rows) == 1
    assert three_class_rows[0]["difference"] == 0.0
    assert three_class_rows[0]["lower"] == 0.0
    assert three_class_rows[0]["upper"] == 0.0
    assert three_class_rows[0]["holm_adjusted_p"] == 1.0
    assert three_class_rows[0]["superiority_supported"] is False


def test_cluster_bootstrap_samples_whole_cluster_multiplicities() -> None:
    records = [
        evaluator.TruthRecord("a1", 1, "plasmid", "1k_to_lt2k", "A", 1000),
        evaluator.TruthRecord("a2", 2, "chromosome", "1k_to_lt2k", "A", 1000),
    ]
    _, _, indicators = evaluator._point_metrics(
        records,
        {
            "plasflow2": ["plasmid", "chromosome"],
            "genomad": ["plasmid", "chromosome"],
            "plasclass": ["plasmid", "non-plasmid"],
            "plasflow_v1": ["plasmid", "non-plasmid"],
            "plasme": ["plasmid", "non-plasmid"],
            "platon": ["plasmid", "non-plasmid"],
        },
    )
    bootstrap = evaluator.cluster_bootstrap(records, indicators, replicates=8, seed=3)
    assert np.all(bootstrap["plasmid_vs_non_plasmid"]["plasflow2"]["balanced_accuracy"] == 1.0)


def test_load_truth_records_enforces_schema_order_and_clusters(tmp_path: Path) -> None:
    labels = tmp_path / "labels.tsv"
    fields = [
        "opaque_contig_id",
        "prediction_order",
        "class",
        "length_bin",
        "selected_primary_accession",
        "selected_fragment_length_bp",
    ]
    with labels.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "opaque_contig_id": "opaque1",
                "prediction_order": 1,
                "class": "plasmid",
                "length_bin": "1k_to_lt2k",
                "selected_primary_accession": "SRC1",
                "selected_fragment_length_bp": 1500,
            }
        )
    records = evaluator.load_truth_records(labels, expected_records=1)
    assert records[0].source_cluster == "SRC1"

    text = labels.read_text().replace("SRC1", "")
    labels.write_text(text)
    with pytest.raises(ValueError, match="Empty source cluster"):
        evaluator.load_truth_records(labels, expected_records=1)


def test_load_predictions_requires_complete_frozen_order(tmp_path: Path) -> None:
    path = tmp_path / "predictions.tsv"
    path.write_text(
        "contig_id\tpredicted_label\tprediction_status\n"
        "r2\tplasmid\tcalled\n"
        "r1\tunclassified\tabstained\n"
    )
    with pytest.raises(ValueError, match="frozen order"):
        evaluator.load_predictions(
            path,
            ["r1", "r2"],
            allowed_labels=set(evaluator.THREE_PREDICTIONS),
            tool="synthetic",
        )


def test_release_authorization_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "evaluator.py"
    source.write_text("# frozen evaluator\n")
    authorization: dict[str, object] = {
        "schema_version": "nar-confirmatory-ground-truth-release-authorization-v1",
        "status": "AUTHORIZED",
        "cohort_role": "confirmatory",
        "prediction_freeze_sha256": evaluator.EXPECTED_PREDICTION_FREEZE_SHA256,
        "statistical_contract_sha256": evaluator.EXPECTED_STATISTICAL_CONTRACT_SHA256,
        "evaluator_contract_sha256": evaluator.EXPECTED_EVALUATOR_CONTRACT_SHA256,
        "sealed_label_map_sha256": evaluator.EXPECTED_SEALED_LABEL_MAP_SHA256,
        "evaluator_source_sha256": evaluator.sha256_file(source),
        "ground_truth_performance_label_release_authorized": True,
        "performance_evaluation_authorized": True,
    }
    authorization["authorization_sha256"] = _canonical(authorization, "authorization_sha256")
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(authorization))
    observed = evaluator.validate_release_authorization(path, evaluator_path=source)
    assert observed["status"] == "AUTHORIZED"

    authorization["cohort_role"] = "development"
    authorization["authorization_sha256"] = _canonical(authorization, "authorization_sha256")
    path.write_text(json.dumps(authorization))
    with pytest.raises(RuntimeError, match="confirmatory-only"):
        evaluator.validate_release_authorization(path, evaluator_path=source)


def test_write_results_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    (output / "existing.txt").write_text("protected")
    with pytest.raises(RuntimeError, match="not empty"):
        evaluator.write_results(
            output,
            _records(),
            _perfect_predictions(),
            {},
            authorization_sha256="synthetic",
        )


def test_write_results_creates_complete_synthetic_artifact_set(tmp_path: Path) -> None:
    records = _records()
    predictions = _perfect_predictions()
    analysis = evaluator.evaluate_records(
        records,
        predictions,
        bootstrap_replicates=16,
        bootstrap_seed=5,
    )
    output = tmp_path / "results"
    evaluator.write_results(
        output,
        records,
        predictions,
        analysis,
        authorization_sha256="synthetic-authorization",
    )
    expected = {
        "point_estimates.tsv",
        "confidence_intervals.tsv",
        "paired_primary_differences.tsv",
        "three_class_per_class_metrics.tsv",
        "raw_confusion_counts.tsv",
        "length_bin_descriptive_metrics.tsv",
        "record_level_confirmatory_audit.tsv",
        "confirmatory_evaluation_results_manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "confirmatory_evaluation_results_manifest.json").read_text())
    assert manifest["status"] == "FROZEN"
    assert manifest["records"] == len(records)
    assert len(manifest["artifacts"]) == 7


def test_main_authorizes_before_parsing_protected_content() -> None:
    source = inspect.getsource(evaluator.main)
    frozen_index = source.index("validate_frozen_inputs(")
    authorization_index = source.index("validate_release_authorization(")
    truth_index = source.index("load_truth_records(")
    prediction_index = source.index("load_predictions(")
    assert frozen_index < authorization_index < truth_index < prediction_index


def test_tracked_evaluator_contract_has_valid_canonical_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/benchmark/contracts/confirmatory_evaluator_contract_v1.json"
    value = json.loads(path.read_text())
    assert value["contract_sha256"] == evaluator.EXPECTED_EVALUATOR_CONTRACT_SHA256
    assert _canonical(value, "contract_sha256") == evaluator.EXPECTED_EVALUATOR_CONTRACT_SHA256
    assert value["authorization"]["authorization_required_before_label_parse"] is True


def test_cli_help_does_not_require_protected_inputs() -> None:
    parser = evaluator.build_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--help"])
    assert exit_info.value.code == 0
