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
    _parse_plasme,
    _parse_platon,
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


PLASME_STANDARDIZED_FIELDS = [
    "contig_id",
    "input_header",
    "length",
    "raw_candidate_present",
    "raw_order",
    "raw_identity",
    "raw_coverage",
    "raw_plasme_score",
    "raw_overlap",
    "raw_positive_fasta_present",
    "predicted_label",
    "prediction_status",
    "identity_threshold",
    "coverage_threshold",
    "probability_threshold",
    "source_tool",
    "source_version",
    "source_commit",
    "container_image_id",
]


def _plasme_standardized_row(
    contig_id: str,
    *,
    label: str,
    candidate_present: bool = True,
    identity: str = "0.1",
    coverage: str = "0.1",
    score: str = "0.5",
) -> dict[str, str]:
    positive = label == "plasmid"

    return {
        "contig_id": contig_id,
        "input_header": contig_id,
        "length": "1000",
        "raw_candidate_present": ("true" if candidate_present else "false"),
        "raw_order": "1" if candidate_present else "",
        "raw_identity": identity if candidate_present else "",
        "raw_coverage": coverage if candidate_present else "",
        "raw_plasme_score": score if candidate_present else "",
        "raw_overlap": "0" if candidate_present else "",
        "raw_positive_fasta_present": ("true" if positive else "false"),
        "predicted_label": label,
        "prediction_status": ("called_plasmid" if positive else "called_non_plasmid"),
        "identity_threshold": "0.9",
        "coverage_threshold": "0.9",
        "probability_threshold": "0.5",
        "source_tool": "PLASMe",
        "source_version": "1.1",
        "source_commit": ("ef0409bad9c8c9ee5d66d90812bf56b345d8dd1d"),
        "container_image_id": (
            "sha256:fbc29e53cf4b331f328241da0e7a835c" "84a50e8aa51a6baf94931aa43559f9a7"
        ),
    }


def _write_plasme_standardized(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PLASME_STANDARDIZED_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_plasme_standardized_output_preserves_binary_semantics(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "plasme"
    tool_dir.mkdir()

    _write_plasme_standardized(
        tool_dir / "standardized_predictions.tsv",
        [
            _plasme_standardized_row(
                "alignment_positive",
                label="plasmid",
                identity="0.9",
                coverage="0.9",
                score="-1",
            ),
            _plasme_standardized_row(
                "transformer_positive",
                label="plasmid",
                identity="0.1",
                coverage="0.1",
                score="0.500001",
            ),
            _plasme_standardized_row(
                "threshold_negative",
                label="non-plasmid",
                identity="0.899",
                coverage="1",
                score="0.5",
            ),
            _plasme_standardized_row(
                "no_candidate",
                label="non-plasmid",
                candidate_present=False,
            ),
        ],
    )

    assert _parse_plasme(tmp_path) == {
        "alignment_positive": "plasmid",
        "transformer_positive": "plasmid",
        "threshold_negative": "non-plasmid",
        "no_candidate": "non-plasmid",
    }


def test_plasme_parser_rejects_duplicate_identifiers(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "plasme"
    tool_dir.mkdir()
    row = _plasme_standardized_row(
        "duplicate",
        label="non-plasmid",
    )

    _write_plasme_standardized(
        tool_dir / "standardized_predictions.tsv",
        [row, row],
    )

    with pytest.raises(ValueError, match="Duplicate PLASMe"):
        _parse_plasme(tmp_path)


def test_plasme_parser_rejects_nonbinary_relabeling(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "plasme"
    tool_dir.mkdir()
    row = _plasme_standardized_row(
        "contig",
        label="non-plasmid",
    )
    row["predicted_label"] = "chromosome"

    _write_plasme_standardized(
        tool_dir / "standardized_predictions.tsv",
        [row],
    )

    with pytest.raises(ValueError, match="must remain binary"):
        _parse_plasme(tmp_path)


def test_plasme_parser_rejects_threshold_semantic_mismatch(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "plasme"
    tool_dir.mkdir()

    _write_plasme_standardized(
        tool_dir / "standardized_predictions.tsv",
        [
            _plasme_standardized_row(
                "threshold_boundary",
                label="plasmid",
                identity="0.1",
                coverage="0.1",
                score="0.5",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="positive-FASTA flag disagrees",
    ):
        _parse_plasme(tmp_path)


def test_plasme_parser_rejects_frozen_identity_mismatch(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "plasme"
    tool_dir.mkdir()
    row = _plasme_standardized_row(
        "contig",
        label="non-plasmid",
    )
    row["source_version"] = "different"

    _write_plasme_standardized(
        tool_dir / "standardized_predictions.tsv",
        [row],
    )

    with pytest.raises(ValueError, match="source_version"):
        _parse_plasme(tmp_path)


def test_incomplete_plasme_output_is_retained_as_abstention(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    output = tmp_path / "evaluation"
    labels = tmp_path / "labels.tsv"
    tool_dir = results / "plasme"
    tool_dir.mkdir(parents=True)
    _write_labels(labels)

    _write_plasme_standardized(
        tool_dir / "standardized_predictions.tsv",
        [
            _plasme_standardized_row(
                "p1",
                label="plasmid",
                identity="0.9",
                coverage="0.9",
                score="-1",
            )
        ],
    )
    (results / "timing.tsv").write_text("tool\twallclock_sec\tstatus\n" "plasme\t1\tok\n")

    evaluate(results, labels, output)

    with (output / "metrics_overall.tsv").open() as handle:
        metrics = list(csv.DictReader(handle, delimiter="\t"))

    assert len(metrics) == 1
    assert metrics[0]["tool"] == "plasme"
    assert metrics[0]["tp"] == "1"
    assert metrics[0]["tn"] == "1"
    assert metrics[0]["n_unclassified"] == "1"
    assert metrics[0]["prediction_coverage"] == "0.5"

    with (output / "per_contig.tsv").open() as handle:
        predictions = {
            row["contig_id"]: row["pred_plasme"] for row in csv.DictReader(handle, delimiter="\t")
        }

    assert predictions == {
        "p1": "plasmid",
        "c1": "unclassified",
    }

    with (output / "tool_status.tsv").open() as handle:
        statuses = {row["tool"]: row for row in csv.DictReader(handle, delimiter="\t")}

    assert statuses["plasme"]["available"] == "false"
    assert statuses["plasme"]["included_in_metrics"] == "true"
    assert "incomplete output" in statuses["plasme"]["reason"]


PLATON_STANDARDIZED_FIELDS = [
    "contig_id",
    "input_header",
    "length",
    "raw_tool_contig_id",
    "raw_native_label",
    "predicted_label",
    "prediction_status",
    "plasmid_score",
    "decision_threshold",
    "rds",
    "is_circular",
    "inc_types",
    "replication_hit_count",
    "mobilization_hit_count",
    "orit_hit_count",
    "conjugation_hit_count",
    "amr_hit_count",
    "rrna_hit_count",
    "reference_plasmid_hit_count",
    "source_tool",
    "source_version",
    "mode",
    "metagenome_mode",
]


def _platon_standardized_row(
    contig_id: str,
    *,
    label: str,
    prediction_status: str | None = None,
    length: int = 1_500,
) -> dict[str, str]:
    if prediction_status is None:
        prediction_status = {
            "plasmid": "called_plasmid",
            "non-plasmid": "called_non_plasmid",
            "unclassified": "missing_output",
        }[label]

    row = {field: "" for field in PLATON_STANDARDIZED_FIELDS}
    row.update(
        {
            "contig_id": contig_id,
            "input_header": contig_id,
            "length": str(length),
            "predicted_label": label,
            "prediction_status": prediction_status,
            "source_tool": "Platon",
            "source_version": "1.7",
            "mode": "accuracy",
            "metagenome_mode": "true",
        }
    )

    if prediction_status == "called_plasmid":
        row.update(
            {
                "raw_tool_contig_id": contig_id,
                "raw_native_label": "plasmid",
                "rds": "1.25",
                "is_circular": "false",
                "inc_types": "[]",
                "replication_hit_count": "0",
                "mobilization_hit_count": "0",
                "orit_hit_count": "0",
                "conjugation_hit_count": "0",
                "amr_hit_count": "0",
                "rrna_hit_count": "0",
                "reference_plasmid_hit_count": "0",
            }
        )
    elif prediction_status == "called_non_plasmid":
        row.update(
            {
                "raw_tool_contig_id": contig_id,
                "raw_native_label": "chromosome",
            }
        )

    return row


def _write_platon_standardized(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PLATON_STANDARDIZED_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_platon_parser_preserves_binary_and_abstention_semantics(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "platon"
    tool_dir.mkdir()
    _write_platon_standardized(
        tool_dir / "standardized_predictions.tsv",
        [
            _platon_standardized_row("p1", label="plasmid"),
            _platon_standardized_row("c1", label="non-plasmid"),
            _platon_standardized_row(
                "short",
                label="unclassified",
                prediction_status="unsupported_length",
                length=999,
            ),
            _platon_standardized_row(
                "missing",
                label="unclassified",
                prediction_status="missing_output",
            ),
        ],
    )

    assert _parse_platon(tmp_path) == {
        "p1": "plasmid",
        "c1": "non-plasmid",
        "short": "unclassified",
        "missing": "unclassified",
    }


def test_platon_parser_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    tool_dir = tmp_path / "platon"
    tool_dir.mkdir()
    row = _platon_standardized_row("duplicate", label="plasmid")
    _write_platon_standardized(
        tool_dir / "standardized_predictions.tsv",
        [row, row],
    )

    with pytest.raises(ValueError, match="Duplicate Platon"):
        _parse_platon(tmp_path)


@pytest.mark.parametrize("invalid_label", ["chromosome", "phage"])
def test_platon_parser_rejects_nonbinary_relabeling(
    tmp_path: Path,
    invalid_label: str,
) -> None:
    tool_dir = tmp_path / "platon"
    tool_dir.mkdir()
    row = _platon_standardized_row("contig", label="non-plasmid")
    row["predicted_label"] = invalid_label
    _write_platon_standardized(
        tool_dir / "standardized_predictions.tsv",
        [row],
    )

    with pytest.raises(ValueError, match="predicted_label"):
        _parse_platon(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_tool", "Different"),
        ("source_version", "1.6"),
        ("mode", "sensitivity"),
        ("metagenome_mode", "false"),
    ],
)
def test_platon_parser_rejects_frozen_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    tool_dir = tmp_path / "platon"
    tool_dir.mkdir()
    row = _platon_standardized_row("contig", label="plasmid")
    row[field] = value
    _write_platon_standardized(
        tool_dir / "standardized_predictions.tsv",
        [row],
    )

    with pytest.raises(ValueError, match=field):
        _parse_platon(tmp_path)


@pytest.mark.parametrize("field", ["plasmid_score", "decision_threshold"])
def test_platon_parser_rejects_probability_or_adapter_threshold(
    tmp_path: Path,
    field: str,
) -> None:
    tool_dir = tmp_path / "platon"
    tool_dir.mkdir()
    row = _platon_standardized_row("contig", label="plasmid")
    row[field] = "0.5"
    _write_platon_standardized(
        tool_dir / "standardized_predictions.tsv",
        [row],
    )

    with pytest.raises(ValueError, match="calibrated plasmid probability"):
        _parse_platon(tmp_path)


@pytest.mark.parametrize(
    ("label", "field", "value", "message"),
    [
        ("plasmid", "rds", "", "RDS required"),
        ("non-plasmid", "rds", "1.0", "evidence must be blank"),
    ],
)
def test_platon_parser_rejects_invalid_evidence_semantics(
    tmp_path: Path,
    label: str,
    field: str,
    value: str,
    message: str,
) -> None:
    tool_dir = tmp_path / "platon"
    tool_dir.mkdir()
    row = _platon_standardized_row("contig", label=label)
    row[field] = value
    _write_platon_standardized(
        tool_dir / "standardized_predictions.tsv",
        [row],
    )

    with pytest.raises(ValueError, match=message):
        _parse_platon(tmp_path)


@pytest.mark.parametrize(
    "row",
    [
        _platon_standardized_row(
            "supported",
            label="unclassified",
            prediction_status="unsupported_length",
            length=1_500,
        ),
        _platon_standardized_row(
            "too_long",
            label="plasmid",
            prediction_status="called_plasmid",
            length=500_001,
        ),
    ],
)
def test_platon_parser_rejects_length_status_mismatch(
    tmp_path: Path,
    row: dict[str, str],
) -> None:
    tool_dir = tmp_path / "platon"
    tool_dir.mkdir()
    _write_platon_standardized(
        tool_dir / "standardized_predictions.tsv",
        [row],
    )

    with pytest.raises(ValueError, match="unsupported-length semantics"):
        _parse_platon(tmp_path)


def test_incomplete_platon_output_is_retained_as_abstention(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    output = tmp_path / "evaluation"
    labels = tmp_path / "labels.tsv"
    tool_dir = results / "platon"
    tool_dir.mkdir(parents=True)
    _write_labels(labels)

    _write_platon_standardized(
        tool_dir / "standardized_predictions.tsv",
        [_platon_standardized_row("p1", label="plasmid", length=5_000)],
    )
    (results / "timing.tsv").write_text("tool\twallclock_sec\tstatus\nplaton\t1\tok\n")

    evaluate(results, labels, output)

    with (output / "metrics_overall.tsv").open() as handle:
        metrics = list(csv.DictReader(handle, delimiter="\t"))

    assert len(metrics) == 1
    assert metrics[0]["tool"] == "platon"
    assert metrics[0]["tp"] == "1"
    assert metrics[0]["tn"] == "1"
    assert metrics[0]["n_unclassified"] == "1"
    assert metrics[0]["prediction_coverage"] == "0.5"

    with (output / "per_contig.tsv").open() as handle:
        predictions = {
            row["contig_id"]: row["pred_platon"] for row in csv.DictReader(handle, delimiter="\t")
        }

    assert predictions == {
        "p1": "plasmid",
        "c1": "unclassified",
    }

    with (output / "tool_status.tsv").open() as handle:
        statuses = {row["tool"]: row for row in csv.DictReader(handle, delimiter="\t")}

    assert statuses["platon"]["available"] == "false"
    assert statuses["platon"]["included_in_metrics"] == "true"
    assert "incomplete output" in statuses["platon"]["reason"]
