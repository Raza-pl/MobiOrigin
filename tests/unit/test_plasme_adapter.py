"""Tests for the frozen manuscript-only PLASMe adapter."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.benchmark.adapters.plasme import (
    CANDIDATE_FIELDS,
    CONTAINER_IMAGE_ID,
    CONTRACT_SHA256,
    COVERAGE_THRESHOLD,
    IDENTITY_THRESHOLD,
    PROBABILITY_THRESHOLD,
    SOURCE_COMMIT,
    adapt_plasme,
    canonical_id,
    load_candidate_rows,
    parse_plasme_score,
    parse_unit_interval,
    recomputed_positive,
)

EXPECTED_CONTRACT_SHA256 = "735407cb3b7d91200ec9ca9643336c981060c735fae89d0db08fb1fa2bcc98fc"
EXPECTED_IMAGE_ID = "sha256:fbc29e53cf4b331f328241da0e7a835c84a50e8aa51a6baf94931aa43559f9a7"
EXPECTED_SOURCE_COMMIT = "ef0409bad9c8c9ee5d66d90812bf56b345d8dd1d"


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def write_candidates(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for row in rows:
            external_row = dict(row)
            external_row["PLASMe"] = external_row.pop("plasme_score")
            writer.writerow(external_row)


def candidate(
    query: str,
    *,
    order: int = 1,
    identity: object = 0.0,
    coverage: object = 0.0,
    plasme_score: object = -1,
    overlap: object = 0.0,
) -> dict[str, object]:
    return {
        "query": query,
        "order": order,
        "identity": identity,
        "coverage": coverage,
        "plasme_score": plasme_score,
        "overlap": overlap,
    }


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def prepare_paths(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    return (
        tmp_path / "input.fasta",
        tmp_path / "positive.fasta",
        tmp_path / "candidates.csv",
        tmp_path / "standardized.tsv",
        tmp_path / "metadata.json",
    )


def test_frozen_identity_constants() -> None:
    assert CONTRACT_SHA256 == EXPECTED_CONTRACT_SHA256
    assert len(CONTRACT_SHA256) == 64
    assert CONTAINER_IMAGE_ID == EXPECTED_IMAGE_ID
    assert SOURCE_COMMIT == EXPECTED_SOURCE_COMMIT
    assert IDENTITY_THRESHOLD == 0.9
    assert COVERAGE_THRESHOLD == 0.9
    assert PROBABILITY_THRESHOLD == 0.5


def test_canonical_id_uses_first_header_token() -> None:
    assert canonical_id("contig_1 descriptive text") == "contig_1"


def test_complete_binary_normalization_and_input_order(tmp_path: Path) -> None:
    input_fasta, positive_fasta, candidates, output, metadata_path = prepare_paths(tmp_path)
    write_fasta(
        input_fasta,
        [
            ("p_alignment complete header", "AAAA"),
            ("p_transformer another header", "CCCC"),
            ("c_negative retained negative", "GGGG"),
            ("c_missing no candidate", "TTTT"),
        ],
    )
    write_fasta(
        positive_fasta,
        [
            ("p_alignment", "AAAA"),
            ("p_transformer", "CCCC"),
        ],
    )
    write_candidates(
        candidates,
        [
            candidate(
                "p_alignment",
                order=1,
                identity=0.9,
                coverage=0.9,
                plasme_score=-1,
            ),
            candidate(
                "p_transformer",
                order=2,
                identity=0.2,
                coverage=0.2,
                plasme_score=0.5000001,
            ),
            candidate(
                "c_negative",
                order=3,
                identity=0.899,
                coverage=1.0,
                plasme_score=0.5,
            ),
        ],
    )

    metadata = adapt_plasme(
        input_fasta=input_fasta,
        positive_fasta=positive_fasta,
        candidate_csv=candidates,
        output_path=output,
        metadata_output=metadata_path,
    )

    rows = read_tsv(output)
    assert [row["contig_id"] for row in rows] == [
        "p_alignment",
        "p_transformer",
        "c_negative",
        "c_missing",
    ]
    assert [row["predicted_label"] for row in rows] == [
        "plasmid",
        "plasmid",
        "non-plasmid",
        "non-plasmid",
    ]
    assert all(row["predicted_label"] in {"plasmid", "non-plasmid"} for row in rows)
    assert rows[0]["input_header"] == "p_alignment complete header"
    assert rows[0]["raw_candidate_present"] == "true"
    assert rows[3]["raw_candidate_present"] == "false"
    assert rows[3]["prediction_status"] == "called_non_plasmid"
    assert float(rows[0]["identity_threshold"]) == 0.9
    assert float(rows[0]["coverage_threshold"]) == 0.9
    assert float(rows[0]["probability_threshold"]) == 0.5
    assert rows[0]["container_image_id"] == EXPECTED_IMAGE_ID

    saved_metadata = json.loads(metadata_path.read_text())
    assert saved_metadata == metadata
    assert metadata["contract_sha256"] == EXPECTED_CONTRACT_SHA256
    assert metadata["complete_input_coverage"] is True
    assert metadata["three_class_claim"] is False
    assert metadata["label_counts"] == {
        "non-plasmid": 2,
        "plasmid": 2,
    }


def test_transformer_threshold_is_strictly_greater_than_half() -> None:
    assert (
        recomputed_positive(
            candidate(
                "x",
                identity=0.0,
                coverage=0.0,
                plasme_score=0.5,
            )
        )
        is False
    )
    assert (
        recomputed_positive(
            candidate(
                "x",
                identity=0.0,
                coverage=0.0,
                plasme_score=0.5000001,
            )
        )
        is True
    )


def test_alignment_thresholds_are_inclusive() -> None:
    assert (
        recomputed_positive(
            candidate(
                "x",
                identity=0.9,
                coverage=0.9,
                plasme_score=-1,
            )
        )
        is True
    )


@pytest.mark.parametrize(
    ("identity", "coverage"),
    [
        (0.899999, 1.0),
        (1.0, 0.899999),
    ],
)
def test_alignment_fails_below_either_threshold(
    identity: float,
    coverage: float,
) -> None:
    assert (
        recomputed_positive(
            candidate(
                "x",
                identity=identity,
                coverage=coverage,
                plasme_score=-1,
            )
        )
        is False
    )


def test_empty_positive_fasta_is_valid(tmp_path: Path) -> None:
    input_fasta, positive_fasta, candidates, output, metadata_path = prepare_paths(tmp_path)
    write_fasta(input_fasta, [("negative", "AAAA")])
    positive_fasta.write_text("")
    write_candidates(
        candidates,
        [
            candidate(
                "negative",
                identity=0.1,
                coverage=0.1,
                plasme_score=0.5,
            )
        ],
    )

    adapt_plasme(
        input_fasta=input_fasta,
        positive_fasta=positive_fasta,
        candidate_csv=candidates,
        output_path=output,
        metadata_output=metadata_path,
    )

    rows = read_tsv(output)
    assert rows[0]["predicted_label"] == "non-plasmid"


@pytest.mark.parametrize(
    "value",
    ["nan", "inf", "-inf", "-0.5", "1.1", "-2"],
)
def test_plasme_score_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_plasme_score(value, raw_identifier="contig")


@pytest.mark.parametrize(
    "value",
    ["nan", "inf", "-0.1", "1.1"],
)
def test_unit_interval_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_unit_interval(
            value,
            field="identity",
            raw_identifier="contig",
        )


def test_candidate_schema_is_exact(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "query,order,identity,coverage,plasme_score,overlap,unexpected\n" "x,1,0,0,-1,0,value\n"
    )

    with pytest.raises(ValueError, match="columns"):
        load_candidate_rows(candidates)


def test_duplicate_candidate_canonical_ids_are_rejected(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates.csv"
    write_candidates(
        candidates,
        [
            candidate("x", order=1),
            candidate("x duplicate header", order=2),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_candidate_rows(candidates)


def test_extra_candidate_identifier_is_rejected(tmp_path: Path) -> None:
    input_fasta, positive_fasta, candidates, output, _ = prepare_paths(tmp_path)
    write_fasta(input_fasta, [("input", "AAAA")])
    positive_fasta.write_text("")
    write_candidates(candidates, [candidate("unexpected")])

    with pytest.raises(ValueError, match="absent from the input FASTA"):
        adapt_plasme(
            input_fasta=input_fasta,
            positive_fasta=positive_fasta,
            candidate_csv=candidates,
            output_path=output,
        )


def test_duplicate_input_canonical_ids_are_rejected(
    tmp_path: Path,
) -> None:
    input_fasta, positive_fasta, candidates, output, _ = prepare_paths(tmp_path)
    write_fasta(
        input_fasta,
        [
            ("same first header", "AAAA"),
            ("same second header", "CCCC"),
        ],
    )
    positive_fasta.write_text("")
    write_candidates(candidates, [])

    with pytest.raises(ValueError, match="Duplicate"):
        adapt_plasme(
            input_fasta=input_fasta,
            positive_fasta=positive_fasta,
            candidate_csv=candidates,
            output_path=output,
        )


def test_extra_positive_identifier_is_rejected(tmp_path: Path) -> None:
    input_fasta, positive_fasta, candidates, output, _ = prepare_paths(tmp_path)
    write_fasta(input_fasta, [("input", "AAAA")])
    write_fasta(positive_fasta, [("unexpected", "AAAA")])
    write_candidates(candidates, [])

    with pytest.raises(ValueError, match="absent from the input FASTA"):
        adapt_plasme(
            input_fasta=input_fasta,
            positive_fasta=positive_fasta,
            candidate_csv=candidates,
            output_path=output,
        )


def test_positive_sequence_mismatch_is_rejected(tmp_path: Path) -> None:
    input_fasta, positive_fasta, candidates, output, _ = prepare_paths(tmp_path)
    write_fasta(input_fasta, [("x", "AAAA")])
    write_fasta(positive_fasta, [("x", "TTTT")])
    write_candidates(
        candidates,
        [
            candidate(
                "x",
                identity=0.9,
                coverage=0.9,
                plasme_score=-1,
            )
        ],
    )

    with pytest.raises(ValueError, match="sequence"):
        adapt_plasme(
            input_fasta=input_fasta,
            positive_fasta=positive_fasta,
            candidate_csv=candidates,
            output_path=output,
        )


def test_official_positive_output_disagreement_is_rejected(
    tmp_path: Path,
) -> None:
    input_fasta, positive_fasta, candidates, output, _ = prepare_paths(tmp_path)
    write_fasta(input_fasta, [("x", "AAAA")])
    positive_fasta.write_text("")
    write_candidates(
        candidates,
        [
            candidate(
                "x",
                identity=0.9,
                coverage=0.9,
                plasme_score=-1,
            )
        ],
    )

    with pytest.raises(ValueError, match="disagrees"):
        adapt_plasme(
            input_fasta=input_fasta,
            positive_fasta=positive_fasta,
            candidate_csv=candidates,
            output_path=output,
        )


def test_empty_input_is_rejected(tmp_path: Path) -> None:
    input_fasta, positive_fasta, candidates, output, _ = prepare_paths(tmp_path)
    input_fasta.write_text("")
    positive_fasta.write_text("")
    write_candidates(candidates, [])

    with pytest.raises(ValueError):
        adapt_plasme(
            input_fasta=input_fasta,
            positive_fasta=positive_fasta,
            candidate_csv=candidates,
            output_path=output,
        )


def test_cli_writes_standardized_output_and_metadata(
    tmp_path: Path,
) -> None:
    input_fasta, positive_fasta, candidates, output, metadata_path = prepare_paths(tmp_path)
    write_fasta(input_fasta, [("negative control", "AAAA")])
    positive_fasta.write_text("")
    write_candidates(
        candidates,
        [
            candidate(
                "negative",
                identity=0.0,
                coverage=0.0,
                plasme_score=0.5,
            )
        ],
    )

    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/adapters/plasme.py",
            "--input-fasta",
            str(input_fasta),
            "--positive-fasta",
            str(positive_fasta),
            "--candidate-csv",
            str(candidates),
            "--output",
            str(output),
            "--metadata-output",
            str(metadata_path),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert read_tsv(output)[0]["predicted_label"] == "non-plasmid"
    cli_metadata = json.loads(metadata_path.read_text())
    assert cli_metadata["contract_sha256"] == EXPECTED_CONTRACT_SHA256
    assert cli_metadata["complete_input_coverage"] is True
    assert cli_metadata["production_workflow_component"] is False


def test_cli_help_declares_manuscript_only_scope() -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/adapters/plasme.py",
            "--help",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "manuscript benchmark" in result.stdout
    assert "not part of the PlasFlow2 prediction workflow" in result.stdout
