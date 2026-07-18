"""Tests for benchmark run-integrity handling."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.benchmark.evaluate import evaluate


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


def test_failed_tools_are_excluded_instead_of_scored_as_negatives(tmp_path) -> None:
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
        metrics = list(csv.DictReader(fh, delimiter="\t"))
    assert [row["tool"] for row in metrics] == ["plasflow2"]
    assert metrics[0]["f1"] == "1.0"

    with (out / "tool_status.tsv").open() as fh:
        statuses = {row["tool"]: row for row in csv.DictReader(fh, delimiter="\t")}
    assert statuses["plasflow2"]["available"] == "true"
    assert statuses["genomad"]["available"] == "false"
    assert "failed" in statuses["genomad"]["reason"]
