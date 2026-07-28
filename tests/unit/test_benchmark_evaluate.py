"""Tests for benchmark run-integrity handling."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.benchmark.evaluate import (
    _parse_genomad,
    _parse_mobrecon,
    _parse_plasclass,
    _parse_plasflow2,
    _parse_plasflow_v1,
    _parse_rfplasmid,
    evaluate,
)


def _write_labels(path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "contig_id",
                "true_label",
                "length",
                "length_tier",
                "source_accession",
                "molecule_type",
                "taxon",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "contig_id": "p1",
                "true_label": "plasmid",
                "length": 5000,
                "length_tier": "2-5 kb",
                "source_accession": "P1",
                "molecule_type": "plasmid",
                "taxon": "test",
            }
        )
        writer.writerow(
            {
                "contig_id": "c1",
                "true_label": "chromosome",
                "length": 5000,
                "length_tier": "2-5 kb",
                "source_accession": "C1",
                "molecule_type": "chromosome",
                "taxon": "test",
            }
        )


def test_attempted_failed_tools_remain_in_denominator_as_unclassified(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    out = tmp_path / "eval"
    labels = tmp_path / "labels.tsv"
    (results / "plasflow2").mkdir(parents=True)
    _write_labels(labels)

    (results / "plasflow2" / "all_predictions.tsv").write_text(
        "contig_id\tlabel\n" "p1\tplasmid\n" "c1\tchromosome\n"
    )
    (results / "timing.tsv").write_text(
        "tool\twallclock_sec\tstatus\n"
        "plasflow2\t1\tok\n"
        "genomad\t1\tfailed\n"
        "plasclass\t1\tfailed\n"
        "rfplasmid\t1\tfailed\n"
        "mobrecon\t1\tfailed\n"
    )

    evaluate(results, labels, out)

    with (out / "metrics_overall.tsv").open() as fh:
        metrics = {row["tool"]: row for row in csv.DictReader(fh, delimiter="\t")}
    assert set(metrics) == {
        "plasflow2",
        "genomad",
        "plasclass",
        "rfplasmid",
        "mobrecon",
    }
    assert metrics["plasflow2"]["f1"] == "1.0"
    for tool in ("genomad", "plasclass", "rfplasmid", "mobrecon"):
        assert metrics[tool]["fn"] == "1"
        assert metrics[tool]["tn"] == "1"
        assert metrics[tool]["n_unclassified"] == "2"
        assert metrics[tool]["prediction_coverage"] == "0.0"

    with (out / "tool_status.tsv").open() as fh:
        statuses = {row["tool"]: row for row in csv.DictReader(fh, delimiter="\t")}
    assert statuses["plasflow2"]["available"] == "true"
    assert statuses["genomad"]["available"] == "false"
    assert statuses["genomad"]["included_in_metrics"] == "true"
    assert "failed" in statuses["genomad"]["reason"]


def test_mobrecon_standardized_output_preserves_abstentions(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = tmp_path / "eval"
    labels = tmp_path / "labels.tsv"
    (results / "mobrecon").mkdir(parents=True)
    _write_labels(labels)

    (results / "mobrecon" / "standardized_predictions.tsv").write_text(
        "contig_id\tpredicted_label\tprediction_status\n"
        "p1\tunclassified\tmissing_output\n"
        "c1\tchromosome\tcalled\n"
    )
    (results / "timing.tsv").write_text("tool\twallclock_sec\tstatus\n" "mobrecon\t1\tok\n")

    evaluate(results, labels, out)

    with (out / "metrics_overall.tsv").open() as fh:
        metrics = list(csv.DictReader(fh, delimiter="\t"))
    assert len(metrics) == 1
    assert metrics[0] == {
        "tool": "mobrecon",
        "precision": "0.0",
        "recall": "0.0",
        "specificity": "1.0",
        "balanced_accuracy": "0.5",
        "f1": "0.0",
        "mcc": "0.0",
        "tp": "0",
        "fp": "0",
        "tn": "1",
        "fn": "1",
        "n_unclassified": "1",
        "unclassified_fraction": "0.5",
        "prediction_coverage": "0.5",
        "n_total": "2",
        "n_plasmid": "1",
    }

    with (out / "per_contig.tsv").open() as fh:
        predictions = {
            row["contig_id"]: row["pred_mobrecon"] for row in csv.DictReader(fh, delimiter="\t")
        }
    assert predictions == {"p1": "unclassified", "c1": "chromosome"}

    with (out / "confusion.tsv").open() as fh:
        confusion = list(csv.DictReader(fh, delimiter="\t"))
    assert confusion[0]["n_unclassified"] == "1"


def test_tools_with_no_output_or_recorded_attempt_are_excluded(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = tmp_path / "eval"
    labels = tmp_path / "labels.tsv"
    (results / "plasflow2").mkdir(parents=True)
    _write_labels(labels)
    (results / "plasflow2" / "all_predictions.tsv").write_text(
        "contig_id\tlabel\np1\tplasmid\nc1\tchromosome\n"
    )

    evaluate(results, labels, out)

    with (out / "metrics_overall.tsv").open() as fh:
        metrics = list(csv.DictReader(fh, delimiter="\t"))
    assert [row["tool"] for row in metrics] == ["plasflow2"]

    with (out / "tool_status.tsv").open() as fh:
        statuses = {row["tool"]: row for row in csv.DictReader(fh, delimiter="\t")}
    assert statuses["genomad"]["available"] == "false"
    assert statuses["genomad"]["included_in_metrics"] == "false"
    assert "not attempted" in statuses["genomad"]["reason"]


def test_mobrecon_raw_fallback_canonicalizes_description_headers(
    tmp_path: Path,
) -> None:
    mobrecon_dir = tmp_path / "mobrecon"
    mobrecon_dir.mkdir()
    (mobrecon_dir / "contig_report.txt").write_text(
        "contig_id\tmolecule_type\n"
        "p1 circular plasmid description\tplasmid\n"
        "c1 chromosome description\tchromosome\n"
    )

    assert _parse_mobrecon(tmp_path) == {
        "p1": "plasmid",
        "c1": "chromosome",
    }


def test_full_coverage_parsers_preserve_unclassified_calls(tmp_path: Path) -> None:
    (tmp_path / "plasflow2").mkdir()
    (tmp_path / "plasflow2" / "all_predictions.tsv").write_text(
        "contig_id\tlabel\np1\tunclassified\n"
    )
    (tmp_path / "rfplasmid").mkdir()
    (tmp_path / "rfplasmid" / "outputRFPlasmid.txt").write_text(
        "seqname\tprediction\np1\tUncertain\n"
    )

    assert _parse_plasflow2(tmp_path) == {"p1": "unclassified"}
    assert _parse_rfplasmid(tmp_path) == {"p1": "unclassified"}


def test_genomad_standardized_output_preserves_all_labels(
    tmp_path: Path,
) -> None:
    genomad_dir = tmp_path / "genomad"
    genomad_dir.mkdir()
    (genomad_dir / "standardized_predictions.tsv").write_text(
        "contig_id\tpredicted_label\tprediction_status\n"
        "p1\tplasmid\tcalled_plasmid\n"
        "v1\tphage\tcalled_phage\n"
        "c1\tchromosome\tnot_detected\n"
        "u1\tunclassified\tambiguous_dual_call\n"
    )

    assert _parse_genomad(tmp_path) == {
        "p1": "plasmid",
        "v1": "phage",
        "c1": "chromosome",
        "u1": "unclassified",
    }


def test_genomad_standardized_output_rejects_duplicates(
    tmp_path: Path,
) -> None:
    genomad_dir = tmp_path / "genomad"
    genomad_dir.mkdir()
    (genomad_dir / "standardized_predictions.tsv").write_text(
        "contig_id\tpredicted_label\tprediction_status\n"
        "p1\tplasmid\tcalled_plasmid\n"
        "p1\tchromosome\tnot_detected\n"
    )

    with pytest.raises(ValueError, match="Duplicate geNomad"):
        _parse_genomad(tmp_path)


def test_incomplete_standardized_genomad_output_abstains(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    output = tmp_path / "evaluation"
    labels = tmp_path / "labels.tsv"
    genomad_dir = results / "genomad"
    genomad_dir.mkdir(parents=True)
    _write_labels(labels)

    (genomad_dir / "standardized_predictions.tsv").write_text(
        "contig_id\tpredicted_label\tprediction_status\n" "p1\tplasmid\tcalled_plasmid\n"
    )
    (results / "timing.tsv").write_text("tool\twallclock_sec\tstatus\n" "genomad\t1\tok\n")

    evaluate(results, labels, output)

    with (output / "per_contig.tsv").open() as handle:
        predictions = {
            row["contig_id"]: row["pred_genomad"] for row in csv.DictReader(handle, delimiter="\t")
        }
    assert predictions == {
        "p1": "plasmid",
        "c1": "unclassified",
    }

    with (output / "tool_status.tsv").open() as handle:
        tool_status = {row["tool"]: row for row in csv.DictReader(handle, delimiter="\t")}
    assert tool_status["genomad"]["available"] == "false"
    assert "incomplete output" in tool_status["genomad"]["reason"]


def test_plasclass_standardized_output_preserves_binary_semantics(
    tmp_path: Path,
) -> None:
    plasclass_dir = tmp_path / "plasclass"
    plasclass_dir.mkdir()

    (plasclass_dir / "standardized_predictions.tsv").write_text(
        "contig_id\tpredicted_label\tprediction_status\t"
        "plasmid_score\tdecision_threshold\tsource_tool\t"
        "source_version\n"
        "p1\tplasmid\tcalled_plasmid\t0.75\t0.5\t"
        "PlasClass\t0.1\n"
        "c1\tnon-plasmid\tcalled_non_plasmid\t0.25\t0.5\t"
        "PlasClass\t0.1\n"
        "u1\tunclassified\tmissing_output\t\t0.5\t"
        "PlasClass\t0.1\n"
    )

    assert _parse_plasclass(tmp_path) == {
        "p1": "plasmid",
        "c1": "non-plasmid",
        "u1": "unclassified",
    }


def test_plasclass_standardized_output_rejects_duplicates(
    tmp_path: Path,
) -> None:
    plasclass_dir = tmp_path / "plasclass"
    plasclass_dir.mkdir()

    (plasclass_dir / "standardized_predictions.tsv").write_text(
        "contig_id\tpredicted_label\tprediction_status\t"
        "plasmid_score\tdecision_threshold\tsource_tool\t"
        "source_version\n"
        "p1\tplasmid\tcalled_plasmid\t0.75\t0.5\t"
        "PlasClass\t0.1\n"
        "p1\tnon-plasmid\tcalled_non_plasmid\t0.25\t0.5\t"
        "PlasClass\t0.1\n"
    )

    with pytest.raises(
        ValueError,
        match="Duplicate PlasClass",
    ):
        _parse_plasclass(tmp_path)


def test_plasclass_standardized_output_rejects_score_label_mismatch(
    tmp_path: Path,
) -> None:
    plasclass_dir = tmp_path / "plasclass"
    plasclass_dir.mkdir()

    (plasclass_dir / "standardized_predictions.tsv").write_text(
        "contig_id\tpredicted_label\tprediction_status\t"
        "plasmid_score\tdecision_threshold\tsource_tool\t"
        "source_version\n"
        "p1\tnon-plasmid\tcalled_non_plasmid\t0.90\t0.5\t"
        "PlasClass\t0.1\n"
    )

    with pytest.raises(
        ValueError,
        match="score and label are inconsistent",
    ):
        _parse_plasclass(tmp_path)


def test_plasclass_abstentions_remain_in_primary_metrics(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    output = tmp_path / "evaluation"
    labels = tmp_path / "labels.tsv"
    plasclass_dir = results / "plasclass"
    plasclass_dir.mkdir(parents=True)
    _write_labels(labels)

    (plasclass_dir / "standardized_predictions.tsv").write_text(
        "contig_id\tpredicted_label\tprediction_status\t"
        "plasmid_score\tdecision_threshold\tsource_tool\t"
        "source_version\n"
        "p1\tunclassified\tmissing_output\t\t0.5\t"
        "PlasClass\t0.1\n"
        "c1\tnon-plasmid\tcalled_non_plasmid\t0.20\t0.5\t"
        "PlasClass\t0.1\n"
    )
    (results / "timing.tsv").write_text("tool\twallclock_sec\tstatus\n" "plasclass\t1\tok\n")

    evaluate(results, labels, output)

    with (output / "metrics_overall.tsv").open() as handle:
        metrics = list(csv.DictReader(handle, delimiter="\t"))

    assert len(metrics) == 1
    assert metrics[0]["tool"] == "plasclass"
    assert metrics[0]["tp"] == "0"
    assert metrics[0]["fp"] == "0"
    assert metrics[0]["tn"] == "1"
    assert metrics[0]["fn"] == "1"
    assert metrics[0]["n_unclassified"] == "1"
    assert metrics[0]["prediction_coverage"] == "0.5"

    with (output / "per_contig.tsv").open() as handle:
        predictions = {
            row["contig_id"]: row["pred_plasclass"]
            for row in csv.DictReader(handle, delimiter="\t")
        }

    assert predictions == {
        "p1": "unclassified",
        "c1": "non-plasmid",
    }


def test_incomplete_standardized_plasclass_output_abstains(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    output = tmp_path / "evaluation"
    labels = tmp_path / "labels.tsv"
    plasclass_dir = results / "plasclass"
    plasclass_dir.mkdir(parents=True)
    _write_labels(labels)

    (plasclass_dir / "standardized_predictions.tsv").write_text(
        "contig_id\tpredicted_label\tprediction_status\t"
        "plasmid_score\tdecision_threshold\tsource_tool\t"
        "source_version\n"
        "p1\tplasmid\tcalled_plasmid\t0.75\t0.5\t"
        "PlasClass\t0.1\n"
    )
    (results / "timing.tsv").write_text("tool\twallclock_sec\tstatus\n" "plasclass\t1\tok\n")

    evaluate(results, labels, output)

    with (output / "per_contig.tsv").open() as handle:
        predictions = {
            row["contig_id"]: row["pred_plasclass"]
            for row in csv.DictReader(handle, delimiter="\t")
        }

    assert predictions == {
        "p1": "plasmid",
        "c1": "unclassified",
    }

    with (output / "tool_status.tsv").open() as handle:
        tool_status = {row["tool"]: row for row in csv.DictReader(handle, delimiter="\t")}

    assert tool_status["plasclass"]["available"] == "false"
    assert tool_status["plasclass"]["included_in_metrics"] == "true"
    assert "incomplete output" in tool_status["plasclass"]["reason"]


PLASFLOW_V1_DIGEST = "sha256:e69acee3233010dbf5a5245620252bf5" "b9bde930ad5546473ec496992995a7da"

PLASFLOW_V1_FIELDS = [
    "contig_id",
    "raw_label",
    "predicted_label",
    "prediction_status",
    "plasmid_probability",
    "chromosome_probability",
    "max_class_probability",
    "decision_threshold",
    "source_tool",
    "source_version",
    "container_digest",
]


def _write_plasflow_v1_standardized(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PLASFLOW_V1_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _plasflow_v1_row(
    contig_id: str,
    raw_label: str,
    label: str,
    prediction_status: str,
    plasmid_probability: str,
    chromosome_probability: str,
    max_class_probability: str,
) -> dict[str, str]:
    return {
        "contig_id": contig_id,
        "raw_label": raw_label,
        "predicted_label": label,
        "prediction_status": prediction_status,
        "plasmid_probability": plasmid_probability,
        "chromosome_probability": chromosome_probability,
        "max_class_probability": max_class_probability,
        "decision_threshold": "0.7",
        "source_tool": "PlasFlow",
        "source_version": "1.1",
        "container_digest": PLASFLOW_V1_DIGEST,
    }


def test_plasflow_v1_standardized_output_preserves_native_semantics(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "plasflow_v1"
    tool_dir.mkdir()

    _write_plasflow_v1_standardized(
        tool_dir / "standardized_predictions.tsv",
        [
            _plasflow_v1_row(
                "p1",
                "plasmid.Proteobacteria",
                "plasmid",
                "called_plasmid",
                "0.9",
                "0.1",
                "0.8",
            ),
            _plasflow_v1_row(
                "c1",
                "chromosome.Firmicutes",
                "non-plasmid",
                "called_non_plasmid",
                "0.1",
                "0.9",
                "0.8",
            ),
            _plasflow_v1_row(
                "u1",
                "unclassified.unclassified",
                "unclassified",
                "native_abstention",
                "0.55",
                "0.45",
                "0.3",
            ),
            _plasflow_v1_row(
                "m1",
                "",
                "unclassified",
                "missing_output",
                "",
                "",
                "",
            ),
        ],
    )

    assert _parse_plasflow_v1(tmp_path) == {
        "p1": "plasmid",
        "c1": "non-plasmid",
        "u1": "unclassified",
        "m1": "unclassified",
    }


def test_plasflow_v1_standardized_output_rejects_duplicates(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "plasflow_v1"
    tool_dir.mkdir()
    row = _plasflow_v1_row(
        "p1",
        "plasmid.Proteobacteria",
        "plasmid",
        "called_plasmid",
        "0.9",
        "0.1",
        "0.8",
    )

    _write_plasflow_v1_standardized(
        tool_dir / "standardized_predictions.tsv",
        [row, row],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate PlasFlow v1",
    ):
        _parse_plasflow_v1(tmp_path)


def test_plasflow_v1_rejects_label_status_mismatch(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "plasflow_v1"
    tool_dir.mkdir()

    _write_plasflow_v1_standardized(
        tool_dir / "standardized_predictions.tsv",
        [
            _plasflow_v1_row(
                "p1",
                "plasmid.Proteobacteria",
                "plasmid",
                "called_non_plasmid",
                "0.9",
                "0.1",
                "0.8",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="status is inconsistent",
    ):
        _parse_plasflow_v1(tmp_path)


def test_plasflow_v1_rejects_invalid_aggregate_probabilities(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "plasflow_v1"
    tool_dir.mkdir()

    _write_plasflow_v1_standardized(
        tool_dir / "standardized_predictions.tsv",
        [
            _plasflow_v1_row(
                "p1",
                "plasmid.Proteobacteria",
                "plasmid",
                "called_plasmid",
                "0.9",
                "0.2",
                "0.8",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="do not sum to one",
    ):
        _parse_plasflow_v1(tmp_path)


def test_plasflow_v1_missing_output_rejects_scores(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "plasflow_v1"
    tool_dir.mkdir()

    _write_plasflow_v1_standardized(
        tool_dir / "standardized_predictions.tsv",
        [
            _plasflow_v1_row(
                "m1",
                "",
                "unclassified",
                "missing_output",
                "0.5",
                "",
                "",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="must not contain probabilities",
    ):
        _parse_plasflow_v1(tmp_path)


def test_plasflow_v1_native_abstentions_remain_in_metrics(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    output = tmp_path / "evaluation"
    labels = tmp_path / "labels.tsv"
    tool_dir = results / "plasflow_v1"
    tool_dir.mkdir(parents=True)
    _write_labels(labels)

    _write_plasflow_v1_standardized(
        tool_dir / "standardized_predictions.tsv",
        [
            _plasflow_v1_row(
                "p1",
                "unclassified.unclassified",
                "unclassified",
                "native_abstention",
                "0.6",
                "0.4",
                "0.3",
            ),
            _plasflow_v1_row(
                "c1",
                "chromosome.Firmicutes",
                "non-plasmid",
                "called_non_plasmid",
                "0.1",
                "0.9",
                "0.8",
            ),
        ],
    )
    (results / "timing.tsv").write_text("tool\twallclock_sec\tstatus\n" "plasflow_v1\t1\tok\n")

    evaluate(results, labels, output)

    with (output / "metrics_overall.tsv").open() as handle:
        metrics = list(csv.DictReader(handle, delimiter="\t"))

    assert len(metrics) == 1
    assert metrics[0]["tool"] == "plasflow_v1"
    assert metrics[0]["fn"] == "1"
    assert metrics[0]["tn"] == "1"
    assert metrics[0]["n_unclassified"] == "1"
    assert metrics[0]["prediction_coverage"] == "0.5"

    with (output / "per_contig.tsv").open() as handle:
        predictions = {
            row["contig_id"]: row["pred_plasflow_v1"]
            for row in csv.DictReader(
                handle,
                delimiter="\t",
            )
        }

    assert predictions == {
        "p1": "unclassified",
        "c1": "non-plasmid",
    }
