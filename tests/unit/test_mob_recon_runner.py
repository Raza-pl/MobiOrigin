"""Regression tests for the frozen MOB-recon publication runner."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from scripts.benchmark.runners.mob_recon import (
    build_command,
    database_fingerprint,
    load_checksum_table,
    parse_resource_usage,
    prepare_output_directory,
    standardize_output,
    verify_database,
)


def write_checksum_table(
    table_file: Path,
    database_directory: Path,
    filenames: list[str],
) -> dict[str, str]:
    checksums: dict[str, str] = {}

    with table_file.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "size_bytes", "sha256"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for filename in filenames:
            database_file = database_directory / filename
            checksum = hashlib.sha256(database_file.read_bytes()).hexdigest()
            checksums[filename] = checksum
            writer.writerow(
                {
                    "filename": filename,
                    "size_bytes": database_file.stat().st_size,
                    "sha256": checksum,
                }
            )

    return checksums


def test_checksum_table_and_database_verification(
    tmp_path: Path,
) -> None:
    database_directory = tmp_path / "database"
    database_directory.mkdir()
    (database_directory / "a.txt").write_text("alpha")
    (database_directory / "b.txt").write_text("beta")

    table_file = tmp_path / "checksums.tsv"
    expected_hashes = write_checksum_table(
        table_file,
        database_directory,
        ["a.txt", "b.txt"],
    )
    expected_fingerprint = database_fingerprint(expected_hashes)

    expected = load_checksum_table(table_file)
    verification = verify_database(
        database_directory,
        expected,
        expected_fingerprint,
    )

    assert verification["valid"] is True
    assert verification["expected_files"] == 2
    assert verification["verified_files"] == 2
    assert verification["mismatches"] == []


def test_database_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    database_directory = tmp_path / "database"
    database_directory.mkdir()
    database_file = database_directory / "asset.txt"
    database_file.write_text("original")

    table_file = tmp_path / "checksums.tsv"
    expected_hashes = write_checksum_table(
        table_file,
        database_directory,
        ["asset.txt"],
    )
    expected = load_checksum_table(table_file)

    database_file.write_text("tampered")

    verification = verify_database(
        database_directory,
        expected,
        database_fingerprint(expected_hashes),
    )

    assert verification["valid"] is False
    assert any("SHA-256 mismatch" in message for message in verification["mismatches"])


def test_unsafe_checksum_filename_is_rejected(
    tmp_path: Path,
) -> None:
    table_file = tmp_path / "checksums.tsv"
    table_file.write_text("filename\tsize_bytes\tsha256\n" "../escape\t1\t" + ("0" * 64) + "\n")

    with pytest.raises(ValueError, match="Unsafe database filename"):
        load_checksum_table(table_file)


def test_parse_macos_resource_usage() -> None:
    stderr_text = """
        12.50 real         10.25 user         1.50 sys
           149352448  maximum resident set size
           110517008  peak memory footprint
    """

    metrics = parse_resource_usage(stderr_text)

    assert metrics["time_reported_real_seconds"] == 12.5
    assert metrics["user_seconds"] == 10.25
    assert metrics["system_seconds"] == 1.5
    assert metrics["peak_rss_bytes"] == 149352448
    assert metrics["peak_memory_footprint_bytes"] == 110517008


def test_build_command_uses_frozen_wrappers_and_explicit_paths(
    tmp_path: Path,
) -> None:
    command = build_command(
        mob_executable=tmp_path / "env" / "bin" / "mob_recon",
        input_fasta=tmp_path / "input.fasta",
        raw_output_directory=tmp_path / "raw",
        database_directory=tmp_path / "database",
        threads=4,
        sample_id="sample_1",
    )

    assert command[:4] == [
        "/usr/bin/caffeinate",
        "-i",
        "/usr/bin/time",
        "-l",
    ]
    assert "--database_directory" in command
    assert "--num_threads" in command
    assert command[command.index("--num_threads") + 1] == "4"
    assert command[command.index("--sample_id") + 1] == "sample_1"


def test_existing_output_directory_is_rejected(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "existing"
    output_directory.mkdir()

    with pytest.raises(
        FileExistsError,
        match="Refusing to overwrite",
    ):
        prepare_output_directory(output_directory)


def test_missing_raw_report_becomes_complete_unclassified_output(
    tmp_path: Path,
) -> None:
    input_fasta = tmp_path / "input.fasta"
    input_fasta.write_text(">contig_1 description\nACGT\n" ">contig_2\nTGCA\n")
    raw_output = tmp_path / "raw"
    output_directory = tmp_path / "output"
    output_directory.mkdir()

    report_was_emitted, metadata = standardize_output(
        input_fasta,
        raw_output,
        output_directory,
    )

    assert report_was_emitted is False
    assert metadata["standardized_rows"] == 2
    assert metadata["label_counts"] == {"unclassified": 2}
    assert metadata["status_counts"] == {"missing_output": 2}

    with (output_directory / "standardized_predictions.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert [row["contig_id"] for row in rows] == [
        "contig_1",
        "contig_2",
    ]
    assert {row["predicted_label"] for row in rows} == {"unclassified"}
