"""Regression tests for the frozen PlasFlow v1.1 adapter."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.benchmark.adapters.plasflow_v1 import (
    CHROMOSOME_FIELDS,
    CONTAINER_DIGEST,
    CONTRACT_SHA256,
    DECISION_THRESHOLD,
    OUTPUT_FIELDS,
    PLASMID_FIELDS,
    PROBABILITY_FIELDS,
    RAW_METADATA_FIELDS,
    adapt_plasflow_v1,
    canonical_id,
    load_fasta_records,
    load_raw_predictions,
)


def write_fasta(
    path: Path,
    records: list[tuple[str, str]],
) -> None:
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def probability_vector(
    class_id: int,
    peak: float,
) -> list[float]:
    remainder = (1.0 - peak) / (len(PROBABILITY_FIELDS) - 1)
    values = [remainder] * len(PROBABILITY_FIELDS)
    values[class_id] = peak
    return values


def raw_row(
    index: int,
    contig_name: str,
    length: int,
    class_id: int,
    label: str,
    peak: float = 0.8,
) -> dict[str, str]:
    values = probability_vector(class_id, peak)

    row = {
        "": str(index),
        "contig_id": str(index),
        "contig_name": contig_name,
        "contig_length": str(length),
        "id": str(class_id),
        "label": label,
    }

    row.update(
        {
            field: f"{value:.17g}"
            for field, value in zip(
                PROBABILITY_FIELDS,
                values,
            )
        }
    )
    return row


def write_raw_predictions(
    path: Path,
    rows: list[dict[str, str]],
    extra_fields: list[str] | None = None,
) -> None:
    fields = RAW_METADATA_FIELDS + PROBABILITY_FIELDS
    fields.extend(extra_fields or [])

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def standard_records() -> list[tuple[str, str]]:
    return [
        ("plasmid_1 description", "A" * 1000),
        ("chromosome_1 description", "C" * 2000),
        ("uncertain_1 description", "G" * 3000),
    ]


def standard_raw_rows() -> list[dict[str, str]]:
    return [
        raw_row(
            0,
            "plasmid_1",
            1000,
            26,
            "plasmid.Proteobacteria",
            0.8,
        ),
        raw_row(
            1,
            "chromosome_1",
            2000,
            8,
            "chromosome.Firmicutes",
            0.8,
        ),
        raw_row(
            2,
            "uncertain_1",
            3000,
            23,
            "unclassified.unclassified",
            0.4,
        ),
    ]


def test_frozen_contract_constants() -> None:
    assert CONTRACT_SHA256 == ("7e24ac29aedb6f0b106e421302cb1ee147f076c96c24d299f9b9d55d1b42c3bf")
    assert CONTAINER_DIGEST == (
        "sha256:e69acee3233010dbf5a5245620252bf5b9bde930ad5546473ec496992995a7da"
    )
    assert DECISION_THRESHOLD == 0.7
    assert len(PROBABILITY_FIELDS) == 28
    assert len(CHROMOSOME_FIELDS) == 18
    assert len(PLASMID_FIELDS) == 10


def test_canonical_id_uses_first_header_token() -> None:
    assert canonical_id("contig_1 description text") == "contig_1"


def test_adapter_preserves_order_and_native_binary_semantics(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    raw = tmp_path / "predictions.tsv"
    output = tmp_path / "standardized.tsv"
    metadata_path = tmp_path / "metadata.json"

    write_fasta(fasta, standard_records())
    write_raw_predictions(raw, standard_raw_rows())

    metadata = adapt_plasflow_v1(
        input_fasta=fasta,
        raw_predictions=raw,
        output_path=output,
        metadata_output=metadata_path,
    )

    rows = read_tsv(output)

    assert list(rows[0]) == OUTPUT_FIELDS
    assert [row["contig_id"] for row in rows] == [
        "plasmid_1",
        "chromosome_1",
        "uncertain_1",
    ]
    assert [row["predicted_label"] for row in rows] == [
        "plasmid",
        "non-plasmid",
        "unclassified",
    ]
    assert [row["prediction_status"] for row in rows] == [
        "called_plasmid",
        "called_non_plasmid",
        "native_abstention",
    ]
    assert [row["raw_label"] for row in rows] == [
        "plasmid.Proteobacteria",
        "chromosome.Firmicutes",
        "unclassified.unclassified",
    ]

    assert metadata["complete_output"] is True
    assert metadata["runner_success_allowed"] is True
    assert metadata["label_counts"] == {
        "non-plasmid": 1,
        "plasmid": 1,
        "unclassified": 1,
    }
    assert metadata["native_phage_class_available"] is False
    assert metadata["three_class_ranking_allowed"] is False

    saved_metadata = json.loads(metadata_path.read_text())
    assert saved_metadata == metadata


def test_native_aggregate_plasmid_label_is_preserved(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    raw = tmp_path / "predictions.tsv"
    output = tmp_path / "standardized.tsv"

    write_fasta(fasta, [("aggregate_plasmid", "A" * 1200)])
    write_raw_predictions(
        raw,
        [
            raw_row(
                0,
                "aggregate_plasmid",
                1200,
                23,
                "plasmid.unclassified",
                0.4,
            )
        ],
    )

    adapt_plasflow_v1(fasta, raw, output)
    row = read_tsv(output)[0]

    assert row["raw_label"] == "plasmid.unclassified"
    assert row["predicted_label"] == "plasmid"
    assert row["prediction_status"] == "called_plasmid"


def test_missing_output_is_retained_as_abstention(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    raw = tmp_path / "predictions.tsv"
    output = tmp_path / "standardized.tsv"

    write_fasta(fasta, standard_records())
    write_raw_predictions(raw, standard_raw_rows()[:2])

    metadata = adapt_plasflow_v1(fasta, raw, output)
    rows = read_tsv(output)

    missing = rows[2]
    assert missing["contig_id"] == "uncertain_1"
    assert missing["predicted_label"] == "unclassified"
    assert missing["prediction_status"] == "missing_output"
    assert missing["plasmid_probability"] == ""
    assert metadata["complete_output"] is False
    assert metadata["runner_success_allowed"] is False
    assert metadata["missing_output_ids"] == ["uncertain_1"]


def test_extra_raw_identifier_is_rejected(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    raw = tmp_path / "predictions.tsv"
    output = tmp_path / "standardized.tsv"

    write_fasta(fasta, [("plasmid_1", "A" * 1000)])
    rows = standard_raw_rows()[:1]
    rows.append(
        raw_row(
            1,
            "not_in_input",
            1000,
            26,
            "plasmid.Proteobacteria",
        )
    )
    write_raw_predictions(raw, rows)

    with pytest.raises(
        ValueError,
        match="identifiers absent from the input FASTA",
    ):
        adapt_plasflow_v1(fasta, raw, output)


def test_fasta_identifier_collision_is_rejected(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "collision.fasta"
    write_fasta(
        fasta,
        [
            ("same first description", "A" * 20),
            ("same second description", "C" * 20),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Canonical FASTA identifier collision",
    ):
        load_fasta_records(fasta)


def test_duplicate_raw_identifier_is_rejected(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "predictions.tsv"
    row = raw_row(
        0,
        "duplicate",
        1000,
        26,
        "plasmid.Proteobacteria",
    )
    second = dict(row)
    second[""] = "1"
    second["contig_id"] = "1"

    write_raw_predictions(raw, [row, second])

    with pytest.raises(
        ValueError,
        match="Duplicate canonical PlasFlow v1 identifier",
    ):
        load_raw_predictions(raw)


def test_nonfinite_probability_is_rejected(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "predictions.tsv"
    row = raw_row(
        0,
        "bad_probability",
        1000,
        26,
        "plasmid.Proteobacteria",
    )
    row[PLASMID_FIELDS[-1]] = "nan"
    write_raw_predictions(raw, [row])

    with pytest.raises(ValueError, match="Non-finite probability"):
        load_raw_predictions(raw)


def test_probability_sum_is_validated(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "predictions.tsv"
    row = raw_row(
        0,
        "bad_sum",
        1000,
        26,
        "plasmid.Proteobacteria",
    )
    row[CHROMOSOME_FIELDS[0]] = "0"
    write_raw_predictions(raw, [row])

    with pytest.raises(
        ValueError,
        match="do not sum to one",
    ):
        load_raw_predictions(raw)


def test_class_id_must_match_probability_argmax(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "predictions.tsv"
    row = raw_row(
        0,
        "bad_class",
        1000,
        26,
        "plasmid.Proteobacteria",
    )
    row["id"] = "8"
    write_raw_predictions(raw, [row])

    with pytest.raises(
        ValueError,
        match="inconsistent with the probability argmax",
    ):
        load_raw_predictions(raw)


def test_invalid_native_label_is_rejected(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "predictions.tsv"
    row = raw_row(
        0,
        "bad_label",
        1000,
        26,
        "phage.Proteobacteria",
    )
    write_raw_predictions(raw, [row])

    with pytest.raises(
        ValueError,
        match="Invalid final native label",
    ):
        load_raw_predictions(raw)


def test_unexpected_raw_column_is_rejected(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "predictions.tsv"
    row = raw_row(
        0,
        "extra_column",
        1000,
        26,
        "plasmid.Proteobacteria",
    )
    row["unexpected"] = "value"
    write_raw_predictions(
        raw,
        [row],
        extra_fields=["unexpected"],
    )

    with pytest.raises(
        ValueError,
        match="unexpected columns",
    ):
        load_raw_predictions(raw)


def test_length_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    raw = tmp_path / "predictions.tsv"
    output = tmp_path / "standardized.tsv"

    write_fasta(fasta, [("length_test", "A" * 1000)])
    write_raw_predictions(
        raw,
        [
            raw_row(
                0,
                "length_test",
                999,
                26,
                "plasmid.Proteobacteria",
            )
        ],
    )

    with pytest.raises(ValueError, match="Length mismatch"):
        adapt_plasflow_v1(fasta, raw, output)


def test_cli_help_declares_manuscript_scope() -> None:
    script = Path(__file__).parents[2] / "scripts" / "benchmark" / "adapters" / "plasflow_v1.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    normalized_help = " ".join(completed.stdout.split())

    assert completed.returncode == 0
    assert "manuscript-only binary comparator schema" in normalized_help
    assert "not part of the PlasFlow2 prediction workflow" in normalized_help
