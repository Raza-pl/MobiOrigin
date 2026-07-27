"""Tests for benchmark run-integrity handling."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.benchmark.evaluate import (
    _parse_mobrecon,
    _parse_plasflow2,
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
