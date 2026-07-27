from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.benchmark.adapters.plasclass import (
    CONTRACT_SHA256,
    DECISION_THRESHOLD,
    adapt_plasclass,
    canonical_id,
    load_fasta_records,
    load_raw_scores,
    model_scale,
)


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_canonical_id_uses_first_header_token() -> None:
    assert canonical_id("contig_1 description text") == "contig_1"


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (0, 1_000),
        (1_000, 1_000),
        (5_500, 1_000),
        (5_501, 10_000),
        (55_000, 10_000),
        (55_001, 100_000),
        (300_000, 100_000),
        (300_001, 500_000),
    ],
)
def test_model_scale_boundaries(length: int, expected: int) -> None:
    assert model_scale(length) == expected


def test_adapter_preserves_input_order_and_binary_semantics(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    scores = tmp_path / "raw.tsv"
    output = tmp_path / "standardized.tsv"
    metadata_path = tmp_path / "metadata.json"

    write_fasta(
        fasta,
        [
            ("contig_a full description", "A" * 1_000),
            ("contig_b another description", "C" * 6_000),
            ("contig_c", "G" * 10_000),
        ],
    )
    scores.write_text(
        "contig_c\t0.5000000000\n" "contig_a raw description\t0.4999999999\n" "contig_b\t0.95\n"
    )

    metadata = adapt_plasclass(
        input_fasta=fasta,
        raw_scores=scores,
        output_path=output,
        metadata_output=metadata_path,
    )
    rows = read_tsv(output)

    assert [row["contig_id"] for row in rows] == [
        "contig_a",
        "contig_b",
        "contig_c",
    ]
    assert [row["predicted_label"] for row in rows] == [
        "non-plasmid",
        "plasmid",
        "plasmid",
    ]
    assert [row["prediction_status"] for row in rows] == [
        "called_non_plasmid",
        "called_plasmid",
        "called_plasmid",
    ]
    assert [row["model_scale"] for row in rows] == [
        "1000",
        "10000",
        "10000",
    ]
    assert rows[0]["input_header"] == "contig_a full description"
    assert rows[0]["raw_tool_contig_id"] == ("contig_a raw description")
    assert rows[0]["decision_threshold"] == "0.5"
    assert rows[0]["source_tool"] == "PlasClass"
    assert rows[0]["source_version"] == "0.1"

    assert metadata["contract_sha256"] == CONTRACT_SHA256
    assert metadata["decision_threshold"] == DECISION_THRESHOLD
    assert metadata["threshold_tuning_allowed"] is False
    assert metadata["complete_output"] is True
    assert metadata["runner_success_allowed"] is True
    assert metadata["input_sequences"] == 3
    assert metadata["raw_score_rows"] == 3
    assert metadata["standardized_rows"] == 3
    assert metadata["label_counts"] == {
        "non-plasmid": 1,
        "plasmid": 2,
    }
    assert metadata["missing_output_ids"] == []
    assert json.loads(metadata_path.read_text()) == metadata


def test_missing_output_becomes_explicit_abstention(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    scores = tmp_path / "raw.tsv"
    output = tmp_path / "standardized.tsv"

    write_fasta(
        fasta,
        [
            ("present", "A" * 20),
            ("missing", "C" * 20),
        ],
    )
    scores.write_text("present\t0.9\n")

    metadata = adapt_plasclass(
        input_fasta=fasta,
        raw_scores=scores,
        output_path=output,
    )
    rows = read_tsv(output)

    assert len(rows) == 2
    assert rows[1]["contig_id"] == "missing"
    assert rows[1]["predicted_label"] == "unclassified"
    assert rows[1]["prediction_status"] == "missing_output"
    assert rows[1]["plasmid_score"] == ""
    assert metadata["complete_output"] is False
    assert metadata["runner_success_allowed"] is False
    assert metadata["missing_output_ids"] == ["missing"]


def test_extra_output_identifier_is_rejected(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    scores = tmp_path / "raw.tsv"

    write_fasta(fasta, [("input_1", "A" * 20)])
    scores.write_text("input_1\t0.8\nextra_1\t0.2\n")

    with pytest.raises(
        ValueError,
        match="identifiers absent from the input FASTA",
    ):
        adapt_plasclass(
            input_fasta=fasta,
            raw_scores=scores,
            output_path=tmp_path / "output.tsv",
        )


def test_duplicate_canonical_score_identifier_is_rejected(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "raw.tsv"
    scores.write_text("contig_1 first description\t0.8\n" "contig_1 second description\t0.7\n")

    with pytest.raises(
        ValueError,
        match="Duplicate canonical PlasClass identifier",
    ):
        load_raw_scores(scores)


@pytest.mark.parametrize(
    "raw_score",
    [
        "nan",
        "inf",
        "-inf",
        "-0.01",
        "1.01",
        "not-a-number",
        "",
    ],
)
def test_invalid_scores_are_rejected(
    tmp_path: Path,
    raw_score: str,
) -> None:
    scores = tmp_path / "raw.tsv"
    scores.write_text(f"contig_1\t{raw_score}\n")

    with pytest.raises(
        ValueError,
        match=(
            "Missing PlasClass score"
            "|Invalid PlasClass score"
            "|Non-finite PlasClass score"
            "|outside"
        ),
    ):
        load_raw_scores(scores)


def test_malformed_raw_row_is_rejected(tmp_path: Path) -> None:
    scores = tmp_path / "raw.tsv"
    scores.write_text("contig_1,0.8\n")

    with pytest.raises(
        ValueError,
        match="expected exactly two tab-separated columns",
    ):
        load_raw_scores(scores)


def test_fasta_canonical_collision_is_rejected(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    write_fasta(
        fasta,
        [
            ("same first description", "AAAA"),
            ("same second description", "CCCC"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Canonical FASTA identifier collision",
    ):
        load_fasta_records(fasta)


def test_empty_fasta_sequence_is_rejected(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">empty\n")

    with pytest.raises(ValueError, match="has no sequence"):
        load_fasta_records(fasta)


def test_direct_cli_execution(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    scores = tmp_path / "raw.tsv"
    output = tmp_path / "standardized.tsv"
    metadata = tmp_path / "metadata.json"

    write_fasta(fasta, [("contig_1 description", "ACGT" * 10)])
    scores.write_text("contig_1\t0.75\n")

    script = Path(__file__).parents[2] / "scripts" / "benchmark" / "adapters" / "plasclass.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-fasta",
            str(fasta),
            "--raw-scores",
            str(scores),
            "--output",
            str(output),
            "--metadata-output",
            str(metadata),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert metadata.is_file()

    rows = read_tsv(output)
    assert len(rows) == 1
    assert rows[0]["contig_id"] == "contig_1"
    assert rows[0]["predicted_label"] == "plasmid"

    recorded = json.loads(metadata.read_text())
    assert recorded["contract_sha256"] == CONTRACT_SHA256
    assert recorded["production_workflow_component"] is False
    assert recorded["models_deserialized_by_adapter"] is False
