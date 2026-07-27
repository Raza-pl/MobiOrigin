"""Regression tests for the frozen geNomad comparator adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.benchmark.adapters.genomad import adapt_genomad

PLASMID_FIELDS = [
    "seq_name",
    "length",
    "plasmid_score",
    "fdr",
    "n_hallmarks",
    "marker_enrichment",
]
VIRUS_FIELDS = [
    "seq_name",
    "length",
    "virus_score",
    "fdr",
    "n_hallmarks",
    "marker_enrichment",
]
SCORE_FIELDS = [
    "seq_name",
    "chromosome_score",
    "plasmid_score",
    "virus_score",
]


def write_tsv(
    file_path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    with file_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_output(file_path: Path) -> list[dict[str, str]]:
    with file_path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def make_standard_case(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    fasta = tmp_path / "input.fasta"
    plasmids = tmp_path / "plasmids.tsv"
    viruses = tmp_path / "viruses.tsv"
    scores = tmp_path / "scores.tsv"
    output = tmp_path / "standardized.tsv"
    metadata = tmp_path / "metadata.json"

    fasta.write_text(
        ">p1 plasmid description\nAAAA\n"
        ">v1 virus description\nAAAA\n"
        ">b1 ambiguous description\nAAAA\n"
        ">c1 chromosome description\nAAAA\n"
    )

    write_tsv(
        plasmids,
        PLASMID_FIELDS,
        [
            {
                "seq_name": "p1",
                "length": "4",
                "plasmid_score": "0.95",
                "fdr": "0.01",
                "n_hallmarks": "2",
                "marker_enrichment": "3.5",
            },
            {
                "seq_name": "b1",
                "length": "4",
                "plasmid_score": "0.80",
                "fdr": "0.05",
                "n_hallmarks": "1",
                "marker_enrichment": "1.5",
            },
        ],
    )

    write_tsv(
        viruses,
        VIRUS_FIELDS,
        [
            {
                "seq_name": "v1",
                "length": "4",
                "virus_score": "0.97",
                "fdr": "0.01",
                "n_hallmarks": "3",
                "marker_enrichment": "4.5",
            },
            {
                "seq_name": "b1",
                "length": "4",
                "virus_score": "0.85",
                "fdr": "0.03",
                "n_hallmarks": "1",
                "marker_enrichment": "2.0",
            },
        ],
    )

    write_tsv(
        scores,
        SCORE_FIELDS,
        [
            {
                "seq_name": "p1",
                "chromosome_score": "0.03",
                "plasmid_score": "0.95",
                "virus_score": "0.02",
            },
            {
                "seq_name": "v1",
                "chromosome_score": "0.02",
                "plasmid_score": "0.01",
                "virus_score": "0.97",
            },
            {
                "seq_name": "b1",
                "chromosome_score": "0.05",
                "plasmid_score": "0.50",
                "virus_score": "0.45",
            },
            {
                "seq_name": "c1",
                "chromosome_score": "0.90",
                "plasmid_score": "0.06",
                "virus_score": "0.04",
            },
        ],
    )

    return fasta, plasmids, viruses, scores, output, metadata


def test_complete_classification_semantics_and_order(
    tmp_path: Path,
) -> None:
    files = make_standard_case(tmp_path)

    result = adapt_genomad(
        input_fasta=files[0],
        plasmid_summary=files[1],
        virus_summary=files[2],
        calibrated_scores=files[3],
        output_path=files[4],
        metadata_output=files[5],
    )

    rows = read_output(files[4])

    assert [row["contig_id"] for row in rows] == [
        "p1",
        "v1",
        "b1",
        "c1",
    ]
    assert [row["predicted_label"] for row in rows] == [
        "plasmid",
        "phage",
        "unclassified",
        "chromosome",
    ]
    assert [row["prediction_status"] for row in rows] == [
        "called_plasmid",
        "called_phage",
        "ambiguous_dual_call",
        "not_detected",
    ]
    assert rows[0]["input_header"] == "p1 plasmid description"
    assert rows[3]["chromosome_score"] == "0.9"
    assert result["standardized_rows"] == 4
    assert result["contract_sha256"] == (
        "cd5f4cbab615d931c35470df24640950c489e4d8424c523ed54eedecf35bfdee"
    )

    saved_metadata = json.loads(files[5].read_text())
    assert saved_metadata["label_counts"] == {
        "chromosome": 1,
        "phage": 1,
        "plasmid": 1,
        "unclassified": 1,
    }


def test_provirus_children_map_to_parent_and_are_aggregated(
    tmp_path: Path,
) -> None:
    fasta, plasmids, viruses, scores, output, metadata = make_standard_case(tmp_path)

    fasta.write_text(">host1 parent sequence\nAAAA\n")
    write_tsv(plasmids, PLASMID_FIELDS, [])
    write_tsv(
        viruses,
        VIRUS_FIELDS,
        [
            {
                "seq_name": "host1|provirus_1_100",
                "length": "100",
                "virus_score": "0.80",
                "fdr": "0.04",
                "n_hallmarks": "2",
                "marker_enrichment": "3.0",
            },
            {
                "seq_name": "host1|provirus_200_400",
                "length": "201",
                "virus_score": "0.95",
                "fdr": "0.01",
                "n_hallmarks": "4",
                "marker_enrichment": "5.0",
            },
        ],
    )
    write_tsv(
        scores,
        SCORE_FIELDS,
        [
            {
                "seq_name": "host1",
                "chromosome_score": "0.10",
                "plasmid_score": "0.05",
                "virus_score": "0.85",
            }
        ],
    )

    adapt_genomad(
        fasta,
        plasmids,
        viruses,
        scores,
        output,
        metadata,
    )
    row = read_output(output)[0]

    assert row["predicted_label"] == "phage"
    assert row["raw_virus_ids"] == ("host1|provirus_1_100;host1|provirus_200_400")
    assert row["virus_fdr"] == "0.01"
    assert row["virus_n_hallmarks"] == "4"
    assert row["virus_marker_enrichment"] == "5"


def test_rejects_canonical_fasta_identifier_collision(
    tmp_path: Path,
) -> None:
    files = make_standard_case(tmp_path)
    files[0].write_text(
        ">duplicate first description\nAAAA\n" ">duplicate second description\nAAAA\n"
    )

    with pytest.raises(ValueError, match="collision"):
        adapt_genomad(
            files[0],
            files[1],
            files[2],
            files[3],
            files[4],
        )


def test_rejects_unknown_genomad_identifier(tmp_path: Path) -> None:
    files = make_standard_case(tmp_path)
    write_tsv(
        files[1],
        PLASMID_FIELDS,
        [
            {
                "seq_name": "not_in_input",
                "length": "100",
                "plasmid_score": "0.9",
                "fdr": "0.01",
                "n_hallmarks": "1",
                "marker_enrichment": "2",
            }
        ],
    )

    with pytest.raises(ValueError, match="absent from input"):
        adapt_genomad(
            files[0],
            files[1],
            files[2],
            files[3],
            files[4],
        )


def test_rejects_nonfinite_or_out_of_range_scores(
    tmp_path: Path,
) -> None:
    files = make_standard_case(tmp_path)
    write_tsv(
        files[3],
        SCORE_FIELDS,
        [
            {
                "seq_name": "p1",
                "chromosome_score": "0.1",
                "plasmid_score": "nan",
                "virus_score": "0.1",
            }
        ],
    )

    with pytest.raises(ValueError, match="Non-finite"):
        adapt_genomad(
            files[0],
            files[1],
            files[2],
            files[3],
            files[4],
        )


def test_rejects_missing_required_summary_columns(
    tmp_path: Path,
) -> None:
    files = make_standard_case(tmp_path)
    write_tsv(
        files[1],
        ["seq_name", "plasmid_score"],
        [{"seq_name": "p1", "plasmid_score": "0.9"}],
    )

    with pytest.raises(ValueError, match="missing columns"):
        adapt_genomad(
            files[0],
            files[1],
            files[2],
            files[3],
            files[4],
        )


def test_rejects_duplicate_raw_summary_identifier(
    tmp_path: Path,
) -> None:
    files = make_standard_case(tmp_path)
    duplicate = {
        "seq_name": "v1",
        "length": "4",
        "virus_score": "0.9",
        "fdr": "0.01",
        "n_hallmarks": "1",
        "marker_enrichment": "2",
    }
    write_tsv(files[2], VIRUS_FIELDS, [duplicate, duplicate])

    with pytest.raises(ValueError, match="Duplicate geNomad"):
        adapt_genomad(
            files[0],
            files[1],
            files[2],
            files[3],
            files[4],
        )
