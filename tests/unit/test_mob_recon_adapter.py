"""Regression tests for the frozen MOB-recon benchmark adapter."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.benchmark.adapters.mob_recon import adapt_mob_recon


def write_fasta(path: Path, headers: list[str]) -> None:
    path.write_text("".join(f">{header}\nACGTACGT\n" for header in headers))


def write_report(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    fields = [
        "contig_id",
        "molecule_type",
        "primary_cluster_id",
        "secondary_cluster_id",
        "rep_type(s)",
        "relaxase_type(s)",
        "mash_nearest_neighbor",
        "mash_neighbor_distance",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_output(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_description_bearing_ids_are_normalized(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    report = tmp_path / "contig_report.txt"
    output = tmp_path / "standardized.tsv"
    metadata = tmp_path / "metadata.json"

    write_fasta(
        fasta,
        [
            "p1 complete plasmid description",
            "c1 chromosome description",
        ],
    )
    write_report(
        report,
        [
            {
                "contig_id": "p1 complete plasmid description",
                "molecule_type": "plasmid",
                "primary_cluster_id": "AA001",
            },
            {
                "contig_id": "c1 chromosome description",
                "molecule_type": "chromosome",
            },
        ],
    )

    result = adapt_mob_recon(
        fasta,
        report,
        output,
        metadata,
    )
    rows = read_output(output)

    assert [row["contig_id"] for row in rows] == ["p1", "c1"]
    assert [row["predicted_label"] for row in rows] == [
        "plasmid",
        "chromosome",
    ]
    assert rows[0]["raw_tool_contig_id"] == ("p1 complete plasmid description")
    assert result["standardized_rows"] == 2
    assert result["continuous_score_available"] is False
    assert metadata.is_file()


def test_missing_rows_remain_in_denominator_as_unclassified(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    report = tmp_path / "contig_report.txt"
    output = tmp_path / "standardized.tsv"

    write_fasta(fasta, ["p1", "missing1"])
    write_report(
        report,
        [{"contig_id": "p1", "molecule_type": "plasmid"}],
    )

    result = adapt_mob_recon(fasta, report, output)
    rows = read_output(output)

    assert len(rows) == 2
    assert rows[1]["contig_id"] == "missing1"
    assert rows[1]["predicted_label"] == "unclassified"
    assert rows[1]["prediction_status"] == "missing_output"
    assert result["missing_output_ids"] == ["missing1"]


def test_unsupported_molecule_type_is_unclassified(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    report = tmp_path / "contig_report.txt"
    output = tmp_path / "standardized.tsv"

    write_fasta(fasta, ["x1"])
    write_report(
        report,
        [{"contig_id": "x1", "molecule_type": "unknown"}],
    )

    result = adapt_mob_recon(fasta, report, output)
    row = read_output(output)[0]

    assert row["predicted_label"] == "unclassified"
    assert row["prediction_status"] == "unsupported_molecule_type"
    assert result["unsupported_molecule_type_ids"] == ["x1"]


def test_fasta_canonical_identifier_collision_is_rejected(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    report = tmp_path / "contig_report.txt"
    output = tmp_path / "standardized.tsv"

    write_fasta(fasta, ["dup first", "dup second"])
    write_report(report, [])

    with pytest.raises(
        ValueError,
        match="Canonical FASTA identifier collision",
    ):
        adapt_mob_recon(fasta, report, output)


def test_report_canonical_identifier_collision_is_rejected(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    report = tmp_path / "contig_report.txt"
    output = tmp_path / "standardized.tsv"

    write_fasta(fasta, ["dup first"])
    write_report(
        report,
        [
            {
                "contig_id": "dup first",
                "molecule_type": "plasmid",
            },
            {
                "contig_id": "dup second",
                "molecule_type": "chromosome",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate canonical MOB-recon identifier",
    ):
        adapt_mob_recon(fasta, report, output)


def test_report_ids_absent_from_input_are_rejected(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    report = tmp_path / "contig_report.txt"
    output = tmp_path / "standardized.tsv"

    write_fasta(fasta, ["known"])
    write_report(
        report,
        [
            {
                "contig_id": "known",
                "molecule_type": "chromosome",
            },
            {
                "contig_id": "unexpected",
                "molecule_type": "plasmid",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="identifiers absent from the input",
    ):
        adapt_mob_recon(fasta, report, output)


def test_output_is_deterministic_and_follows_input_order(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    report = tmp_path / "contig_report.txt"
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"

    write_fasta(fasta, ["c2", "p1", "c1"])
    write_report(
        report,
        [
            {"contig_id": "p1", "molecule_type": "plasmid"},
            {"contig_id": "c1", "molecule_type": "chromosome"},
            {"contig_id": "c2", "molecule_type": "chromosome"},
        ],
    )

    adapt_mob_recon(fasta, report, first)
    adapt_mob_recon(fasta, report, second)

    assert first.read_bytes() == second.read_bytes()
    assert [row["contig_id"] for row in read_output(first)] == ["c2", "p1", "c1"]
