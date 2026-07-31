"""Tests for the frozen manuscript-only Platon adapter."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.benchmark.adapters.platon import (
    CONTAINER_IMAGE_ID,
    CONTRACT_SHA256,
    MAXIMUM_LENGTH,
    METAGENOME_MODE,
    MINIMUM_LENGTH,
    MODE,
    OUTPUT_FIELDS,
    RAW_TSV_FIELDS,
    SOURCE_COMMIT,
    TOOL_NAME,
    TOOL_VERSION,
    adapt_platon,
    canonical_id,
    load_raw_json,
    load_raw_tsv,
)

EXPECTED_CONTRACT_SHA256 = "b8add8c173bdd049f750133cdfd6bef2d8be42b0315c2bcf66e4182079aa6e96"
EXPECTED_IMAGE_ID = "sha256:74d96300053a9ce3d4f10bbb935b20631e1d8547c1df632d5f05b178eb2cbbf6"
EXPECTED_SOURCE_COMMIT = "3cce2dd295348e25be4b5bd64f3622c2603d6ba0"

ADAPTER_PATH = Path("scripts/benchmark/adapters/platon.py")


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    """Write simple test FASTA records."""

    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def make_json_record(
    contig_id: str,
    sequence: str,
    *,
    rds: float = 12.34,
    circular: bool = False,
    orf_count: int = 1,
    replication_count: int = 0,
    mobilization_count: int = 0,
    orit_count: int = 0,
    conjugation_count: int = 0,
    amr_count: int = 0,
    rrna_count: int = 0,
    plasmid_hit_count: int = 0,
) -> dict[str, Any]:
    """Create one structurally valid native JSON record."""

    return {
        "id": contig_id,
        "length": len(sequence),
        "sequence": sequence,
        "orfs": {
            str(index): {
                "id": str(index),
                "start": 1,
                "end": 3,
                "strand": "+",
                "product": "test",
                "protein_id": "test",
                "score": 1.0,
            }
            for index in range(1, orf_count + 1)
        },
        "is_circular": circular,
        "inc_types": [],
        "amr_hits": [{} for _ in range(amr_count)],
        "mobilization_hits": [{} for _ in range(mobilization_count)],
        "orit_hits": [{} for _ in range(orit_count)],
        "replication_hits": [{} for _ in range(replication_count)],
        "conjugation_hits": [{} for _ in range(conjugation_count)],
        "rrnas": [{} for _ in range(rrna_count)],
        "plasmid_hits": [{} for _ in range(plasmid_hit_count)],
        "coverage": 0,
        "protein_score": rds,
    }


def make_tsv_row(record: dict[str, Any]) -> dict[str, object]:
    """Create the matching official TSV row for a JSON record."""

    return {
        "ID": record["id"],
        "Length": record["length"],
        "Coverage": "NA",
        "# ORFs": len(record["orfs"]),
        "RDS": f"{float(record['protein_score']):.1f}",
        "Circular": "yes" if record["is_circular"] else "no",
        "Inc Type(s)": 0,
        "# Replication": len(record["replication_hits"]),
        "# Mobilization": len(record["mobilization_hits"]),
        "# OriT": len(record["orit_hits"]),
        "# Conjugation": len(record["conjugation_hits"]),
        "# AMRs": len(record["amr_hits"]),
        "# rRNAs": len(record["rrnas"]),
        "# Plasmid Hits": len(record["plasmid_hits"]),
    }


def write_json(
    path: Path,
    records: dict[str, dict[str, Any]],
) -> None:
    path.write_text(json.dumps(records) + "\n")


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RAW_TSV_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def prepare_valid_case(
    tmp_path: Path,
) -> dict[str, Path]:
    """Create one plasmid and one chromosome native partition."""

    paths = {
        "input": tmp_path / "input.fasta",
        "plasmid": tmp_path / "native.plasmid.fasta",
        "chromosome": tmp_path / "native.chromosome.fasta",
        "json": tmp_path / "native.json",
        "tsv": tmp_path / "native.tsv",
        "output": tmp_path / "standardized.tsv",
        "metadata": tmp_path / "metadata.json",
    }

    plasmid_sequence = "A" * 1200
    chromosome_sequence = "C" * 1300

    write_fasta(
        paths["input"],
        [
            ("plasmid_1 complete header", plasmid_sequence),
            ("chromosome_1 complete header", chromosome_sequence),
        ],
    )
    write_fasta(
        paths["plasmid"],
        [("plasmid_1", plasmid_sequence)],
    )
    write_fasta(
        paths["chromosome"],
        [("chromosome_1", chromosome_sequence)],
    )

    record = make_json_record(
        "plasmid_1",
        plasmid_sequence,
        rds=12.34,
        circular=True,
        orf_count=2,
        replication_count=1,
        mobilization_count=1,
        conjugation_count=2,
        plasmid_hit_count=1,
    )

    write_json(paths["json"], {"plasmid_1": record})
    write_tsv(paths["tsv"], [make_tsv_row(record)])

    return paths


def run_adapter(paths: dict[str, Path]) -> dict[str, Any]:
    return adapt_platon(
        input_fasta=paths["input"],
        plasmid_fasta=paths["plasmid"],
        chromosome_fasta=paths["chromosome"],
        raw_json=paths["json"],
        raw_tsv=paths["tsv"],
        output_path=paths["output"],
        metadata_output=paths["metadata"],
    )


def test_frozen_identity_constants() -> None:
    assert TOOL_NAME == "Platon"
    assert TOOL_VERSION == "1.7"
    assert CONTRACT_SHA256 == EXPECTED_CONTRACT_SHA256
    assert CONTAINER_IMAGE_ID == EXPECTED_IMAGE_ID
    assert SOURCE_COMMIT == EXPECTED_SOURCE_COMMIT
    assert MODE == "accuracy"
    assert METAGENOME_MODE is True
    assert MINIMUM_LENGTH == 1000
    assert MAXIMUM_LENGTH == 500000


def test_canonical_id_uses_first_header_token() -> None:
    assert canonical_id("contig_1 descriptive text") == "contig_1"


def test_complete_normalization_preserves_input_order(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)
    metadata = run_adapter(paths)
    rows = read_tsv(paths["output"])

    assert [row["contig_id"] for row in rows] == [
        "plasmid_1",
        "chromosome_1",
    ]
    assert rows[0]["input_header"] == "plasmid_1 complete header"
    assert rows[1]["input_header"] == "chromosome_1 complete header"

    assert rows[0]["raw_native_label"] == "plasmid"
    assert rows[0]["predicted_label"] == "plasmid"
    assert rows[0]["prediction_status"] == "called_plasmid"

    assert rows[1]["raw_native_label"] == "chromosome"
    assert rows[1]["predicted_label"] == "non-plasmid"
    assert rows[1]["prediction_status"] == "called_non_plasmid"

    assert rows[0]["plasmid_score"] == ""
    assert rows[0]["decision_threshold"] == ""
    assert float(rows[0]["rds"]) == pytest.approx(12.34)
    assert rows[0]["is_circular"] == "true"
    assert rows[0]["replication_hit_count"] == "1"
    assert rows[0]["mobilization_hit_count"] == "1"
    assert rows[0]["conjugation_hit_count"] == "2"
    assert rows[0]["reference_plasmid_hit_count"] == "1"

    assert rows[1]["rds"] == ""
    assert rows[1]["is_circular"] == ""
    assert rows[1]["plasmid_score"] == ""

    assert metadata["runner_success_allowed"] is True
    assert metadata["complete_supported_partition"] is True
    assert metadata["calibrated_plasmid_probability_available"] is False
    assert metadata["rds_treated_as_probability"] is False
    assert metadata["confirmed_chromosome_claim"] is False
    assert metadata["native_phage_class_available"] is False
    assert metadata["three_class_claim"] is False
    assert metadata["label_counts"] == {
        "non-plasmid": 1,
        "plasmid": 1,
    }

    assert json.loads(paths["metadata"].read_text()) == metadata


def test_output_columns_match_frozen_contract(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)
    run_adapter(paths)

    with paths["output"].open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == OUTPUT_FIELDS


def test_unsupported_lengths_become_abstentions(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)

    original = [
        ("plasmid_1", "A" * 1200),
        ("chromosome_1", "C" * 1300),
        ("too_short", "G" * 999),
        ("too_long", "T" * 500001),
    ]
    write_fasta(paths["input"], original)

    metadata = run_adapter(paths)
    rows = {row["contig_id"]: row for row in read_tsv(paths["output"])}

    assert rows["too_short"]["predicted_label"] == "unclassified"
    assert rows["too_short"]["prediction_status"] == "unsupported_length"
    assert rows["too_long"]["predicted_label"] == "unclassified"
    assert rows["too_long"]["prediction_status"] == "unsupported_length"
    assert metadata["unsupported_length_ids"] == [
        "too_long",
        "too_short",
    ]
    assert metadata["runner_success_allowed"] is True


def test_missing_supported_output_is_retained_as_abstention(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)

    write_fasta(
        paths["input"],
        [
            ("plasmid_1", "A" * 1200),
            ("chromosome_1", "C" * 1300),
            ("missing_1", "G" * 1400),
        ],
    )

    metadata = run_adapter(paths)
    rows = {row["contig_id"]: row for row in read_tsv(paths["output"])}

    assert rows["missing_1"]["predicted_label"] == "unclassified"
    assert rows["missing_1"]["prediction_status"] == "missing_output"
    assert metadata["missing_supported_output_ids"] == ["missing_1"]
    assert metadata["complete_supported_partition"] is False
    assert metadata["runner_success_allowed"] is False


def test_all_native_chromosome_output_is_supported(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)

    write_fasta(paths["plasmid"], [])
    write_fasta(
        paths["chromosome"],
        [
            ("plasmid_1", "A" * 1200),
            ("chromosome_1", "C" * 1300),
        ],
    )
    write_json(paths["json"], {})
    write_tsv(paths["tsv"], [])

    metadata = run_adapter(paths)
    assert metadata["label_counts"] == {"non-plasmid": 2}
    assert metadata["raw_json_records"] == 0
    assert metadata["raw_tsv_rows"] == 0


def test_plasmid_and_chromosome_overlap_is_rejected(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)
    write_fasta(
        paths["chromosome"],
        [
            ("chromosome_1", "C" * 1300),
            ("plasmid_1", "A" * 1200),
        ],
    )

    with pytest.raises(ValueError, match="both native FASTAs"):
        run_adapter(paths)


def test_extra_plasmid_identifier_is_rejected(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)
    write_fasta(
        paths["plasmid"],
        [
            ("plasmid_1", "A" * 1200),
            ("unexpected", "G" * 1200),
        ],
    )

    with pytest.raises(
        ValueError,
        match="plasmid FASTA contains identifiers absent",
    ):
        run_adapter(paths)


def test_extra_chromosome_identifier_is_rejected(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)
    write_fasta(
        paths["chromosome"],
        [
            ("chromosome_1", "C" * 1300),
            ("unexpected", "G" * 1200),
        ],
    )

    with pytest.raises(
        ValueError,
        match="chromosome FASTA contains identifiers absent",
    ):
        run_adapter(paths)


@pytest.mark.parametrize("native_name", ["plasmid", "chromosome"])
def test_native_fasta_sequence_mismatch_is_rejected(
    tmp_path: Path,
    native_name: str,
) -> None:
    paths = prepare_valid_case(tmp_path)

    if native_name == "plasmid":
        write_fasta(paths["plasmid"], [("plasmid_1", "T" * 1200)])
    else:
        write_fasta(
            paths["chromosome"],
            [("chromosome_1", "T" * 1300)],
        )

    with pytest.raises(ValueError, match="sequence does not match input"):
        run_adapter(paths)


def test_duplicate_canonical_input_identifier_is_rejected(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)
    write_fasta(
        paths["input"],
        [
            ("same first", "A" * 1200),
            ("same second", "C" * 1300),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate canonical input"):
        run_adapter(paths)


def test_json_identifiers_must_match_plasmid_fasta(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)
    record = make_json_record("other", "A" * 1200)
    write_json(paths["json"], {"other": record})

    with pytest.raises(
        ValueError,
        match="JSON identifiers differ",
    ):
        run_adapter(paths)


def test_tsv_identifiers_must_match_plasmid_fasta(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)
    record = make_json_record("other", "A" * 1200)
    write_tsv(paths["tsv"], [make_tsv_row(record)])

    with pytest.raises(
        ValueError,
        match="TSV identifiers differ",
    ):
        run_adapter(paths)


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"x": {}, "x": {}}\n')

    with pytest.raises(ValueError, match="Duplicate key"):
        load_raw_json(path)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text("{not-json")

    with pytest.raises(ValueError, match="Malformed Platon JSON"):
        load_raw_json(path)


def test_missing_json_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    record = make_json_record("x", "A" * 1200)
    del record["protein_score"]
    write_json(path, {"x": record})

    with pytest.raises(ValueError, match="missing fields"):
        load_raw_json(path)


@pytest.mark.parametrize(
    "invalid_rds",
    [float("nan"), float("inf"), "not-a-number"],
)
def test_nonfinite_or_invalid_json_rds_is_rejected(
    tmp_path: Path,
    invalid_rds: object,
) -> None:
    path = tmp_path / "invalid-rds.json"
    record = make_json_record("x", "A" * 1200)
    record["protein_score"] = invalid_rds
    write_json(path, {"x": record})

    with pytest.raises(ValueError, match="Platon RDS"):
        load_raw_json(path)


def test_json_sequence_length_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "length.json"
    record = make_json_record("x", "A" * 1200)
    record["length"] = 1201
    write_json(path, {"x": record})

    with pytest.raises(ValueError, match="sequence length mismatch"):
        load_raw_json(path)


def test_unexpected_tsv_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad-header.tsv"
    path.write_text("ID\tLength\nx\t1200\n")

    with pytest.raises(ValueError, match="Unexpected Platon TSV fields"):
        load_raw_tsv(path)


def test_duplicate_tsv_identifier_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.tsv"
    record = make_json_record("x", "A" * 1200)
    row = make_tsv_row(record)
    write_tsv(path, [row, row])

    with pytest.raises(ValueError, match="Duplicate canonical Platon TSV"):
        load_raw_tsv(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("# ORFs", "-1", "# ORFs"),
        ("Circular", "maybe", "circularity"),
        ("Coverage", "not-a-number", "coverage"),
        ("RDS", "nan", "TSV RDS"),
    ],
)
def test_invalid_tsv_values_are_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    path = tmp_path / "invalid.tsv"
    record = make_json_record("x", "A" * 1200)
    row = make_tsv_row(record)
    row[field] = value
    write_tsv(path, [row])

    with pytest.raises(ValueError, match=message):
        load_raw_tsv(path)


def test_json_tsv_rds_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)
    record = make_json_record("plasmid_1", "A" * 1200, rds=12.34)
    row = make_tsv_row(record)
    row["RDS"] = "99.9"
    write_tsv(paths["tsv"], [row])

    with pytest.raises(ValueError, match="RDS mismatch"):
        run_adapter(paths)


def test_json_tsv_count_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)

    record = json.loads(paths["json"].read_text())["plasmid_1"]
    row = make_tsv_row(record)
    row["# ORFs"] = 99
    write_tsv(paths["tsv"], [row])

    with pytest.raises(ValueError, match="# ORFs mismatch"):
        run_adapter(paths)


def test_json_tsv_circularity_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)

    record = json.loads(paths["json"].read_text())["plasmid_1"]
    row = make_tsv_row(record)
    row["Circular"] = "no"
    write_tsv(paths["tsv"], [row])

    with pytest.raises(ValueError, match="circularity mismatch"):
        run_adapter(paths)


def test_native_call_for_unsupported_length_is_rejected(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)
    sequence = "A" * 999

    write_fasta(
        paths["input"],
        [
            ("plasmid_1", sequence),
            ("chromosome_1", "C" * 1300),
        ],
    )
    write_fasta(paths["plasmid"], [("plasmid_1", sequence)])

    record = make_json_record("plasmid_1", sequence)
    write_json(paths["json"], {"plasmid_1": record})
    write_tsv(paths["tsv"], [make_tsv_row(record)])

    with pytest.raises(
        ValueError,
        match="unsupported-length inputs",
    ):
        run_adapter(paths)


def test_cli_writes_standardized_output_and_metadata(
    tmp_path: Path,
) -> None:
    paths = prepare_valid_case(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(ADAPTER_PATH),
            "--input-fasta",
            str(paths["input"]),
            "--plasmid-fasta",
            str(paths["plasmid"]),
            "--chromosome-fasta",
            str(paths["chromosome"]),
            "--raw-json",
            str(paths["json"]),
            "--raw-tsv",
            str(paths["tsv"]),
            "--output",
            str(paths["output"]),
            "--metadata-output",
            str(paths["metadata"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert paths["output"].is_file()
    assert paths["metadata"].is_file()

    printed = json.loads(completed.stdout)
    saved = json.loads(paths["metadata"].read_text())

    assert printed == saved
    assert saved["contract_sha256"] == EXPECTED_CONTRACT_SHA256
    assert saved["runner_success_allowed"] is True
    assert saved["production_workflow_component"] is False


def test_cli_help_states_manuscript_scope() -> None:
    completed = subprocess.run(
        [sys.executable, str(ADAPTER_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "manuscript benchmark" in completed.stdout
    assert "not part of the PlasFlow2 prediction workflow" in completed.stdout
