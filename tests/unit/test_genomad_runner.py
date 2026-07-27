from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.benchmark.runners import genomad as runner


def write_fasta(path: Path) -> None:
    path.write_text(
        ">contig_1 first sequence\n" "ACGTACGT\n" ">contig_2 second sequence\n" "AAAACCCC\n"
    )


def build_synthetic_database(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, str]]:
    database = tmp_path / "genomad_db"
    database.mkdir()

    checksums: dict[str, str] = {}
    for index in range(27):
        relative_name = f"database/file_{index:02d}.bin"
        database_file = database / relative_name
        database_file.parent.mkdir(parents=True, exist_ok=True)
        database_file.write_bytes(f"database-{index}\n".encode())
        checksums[relative_name] = hashlib.sha256(database_file.read_bytes()).hexdigest()

    table = tmp_path / "database_checksums.tsv"
    table.write_text("".join(f"{checksums[name]}\t{name}\n" for name in sorted(checksums)))
    return database, table, checksums


def test_load_database_checksum_table_requires_exact_frozen_shape(
    tmp_path: Path,
) -> None:
    _, table, expected = build_synthetic_database(tmp_path)

    observed = runner.load_database_checksum_table(table)

    assert observed == expected
    assert len(observed) == 27
    assert (
        runner.database_fingerprint(observed)
        == hashlib.sha256(
            "".join(f"{expected[name]}\t{name}\n" for name in sorted(expected)).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "relative_name",
    [
        "../escape.bin",
        "/absolute/path.bin",
    ],
)
def test_database_checksum_table_rejects_unsafe_paths(
    tmp_path: Path,
    relative_name: str,
) -> None:
    _, table, _ = build_synthetic_database(tmp_path)
    lines = table.read_text().splitlines()
    checksum = lines[0].split("\t", 1)[0]
    lines[0] = f"{checksum}\t{relative_name}"
    table.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="Unsafe database path"):
        runner.load_database_checksum_table(table)


def test_database_checksum_table_rejects_duplicate_paths(
    tmp_path: Path,
) -> None:
    _, table, _ = build_synthetic_database(tmp_path)
    lines = table.read_text().splitlines()
    lines[1] = lines[0]
    table.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="Duplicate database path"):
        runner.load_database_checksum_table(table)


def test_verify_database_detects_file_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _, checksums = build_synthetic_database(tmp_path)
    expected_fingerprint = runner.database_fingerprint(checksums)
    monkeypatch.setattr(
        runner,
        "EXPECTED_DATABASE_FINGERPRINT",
        expected_fingerprint,
    )

    initial = runner.verify_database(database, checksums)
    assert initial["valid"] is True
    assert initial["verified_files"] == 27

    tampered = database / sorted(checksums)[0]
    tampered.write_bytes(b"tampered\n")

    result = runner.verify_database(database, checksums)

    assert result["valid"] is False
    assert any("SHA-256 mismatch" in value for value in result["mismatches"])
    assert any("Aggregate database fingerprint" in value for value in result["mismatches"])


def test_fasta_statistics_counts_records_and_bases(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    write_fasta(fasta)

    assert runner.fasta_statistics(fasta) == {
        "sequences": 2,
        "bases": 16,
    }


def test_get_tool_version_accepts_only_frozen_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="geNomad, version 1.12.0\n",
            stderr="",
        ),
    )

    assert runner.get_tool_version(Path("/fake/genomad")) == "1.12.0"


def test_get_tool_version_rejects_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="geNomad, version 1.13.0\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        runner.get_tool_version(Path("/fake/genomad"))


def test_build_command_matches_preregistered_execution_contract(
    tmp_path: Path,
) -> None:
    command = runner.build_command(
        executable=Path("/env/bin/genomad"),
        input_fasta=tmp_path / "input.fasta",
        raw_output_directory=tmp_path / "raw",
        database_directory=tmp_path / "db",
        threads=4,
    )

    assert command == [
        "/usr/bin/caffeinate",
        "-i",
        "/usr/bin/time",
        "-l",
        "/env/bin/genomad",
        "end-to-end",
        str(tmp_path / "input.fasta"),
        str(tmp_path / "raw"),
        str(tmp_path / "db"),
        "--threads",
        "4",
        "--splits",
        "8",
        "--enable-score-calibration",
        "--composition",
        "auto",
        "--force-auto",
    ]
    assert "--threshold" not in command


def test_build_command_rejects_invalid_thread_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        runner.build_command(
            executable=Path("/env/bin/genomad"),
            input_fasta=tmp_path / "input.fasta",
            raw_output_directory=tmp_path / "raw",
            database_directory=tmp_path / "db",
            threads=0,
        )


def test_prepare_output_directory_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    runner.prepare_output_directory(output)
    assert output.is_dir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner.prepare_output_directory(output)


def test_discover_genomad_outputs_requires_unique_files(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    summary = raw / "sample_summary"
    classification = raw / "sample_aggregated_classification"
    summary.mkdir(parents=True)
    classification.mkdir(parents=True)

    plasmid = summary / "sample_plasmid_summary.tsv"
    virus = summary / "sample_virus_summary.tsv"
    calibrated = classification / "sample_calibrated_aggregated_classification.tsv"

    plasmid.write_text("seq_name\n")
    virus.write_text("seq_name\n")
    calibrated.write_text("seq_name\n")

    assert runner.discover_genomad_outputs(raw) == {
        "plasmid_summary": plasmid,
        "virus_summary": virus,
        "calibrated_scores": calibrated,
    }

    (summary / "duplicate_plasmid_summary.tsv").write_text("seq_name\n")
    with pytest.raises(RuntimeError, match="found 2"):
        runner.discover_genomad_outputs(raw)


def test_failed_run_emits_complete_unclassified_output(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    output = tmp_path / "output"
    output.mkdir()
    write_fasta(fasta)

    metadata = runner.write_failed_standardized_output(
        input_fasta=fasta,
        output_directory=output,
        failure_reason="synthetic failure",
    )

    with (output / "standardized_predictions.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert [row["contig_id"] for row in rows] == [
        "contig_1",
        "contig_2",
    ]
    assert {row["predicted_label"] for row in rows} == {"unclassified"}
    assert {row["prediction_status"] for row in rows} == {"tool_failed"}
    assert metadata["standardized_rows"] == 2
    assert metadata["successful_tool_output"] is False
    assert metadata["confirmatory_tuning"] is False

    saved_metadata = json.loads((output / "adapter_metadata.json").read_text())
    assert saved_metadata == metadata


def test_parse_resource_usage_supports_macos_time_output() -> None:
    stderr = (
        "12.50 real 20.25 user 3.75 sys\n"
        "123456 maximum resident set size\n"
        "234567 peak memory footprint\n"
    )

    assert runner.parse_resource_usage(stderr) == {
        "time_reported_real_seconds": 12.5,
        "user_seconds": 20.25,
        "system_seconds": 3.75,
        "peak_rss_bytes": 123456,
        "peak_memory_footprint_bytes": 234567,
    }


def test_package_inventory_reads_conda_metadata(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "env"
    metadata_dir = prefix / "conda-meta"
    metadata_dir.mkdir(parents=True)
    package_file = metadata_dir / "genomad.json"
    package_file.write_text(
        json.dumps(
            {
                "name": "genomad",
                "version": "1.12.0",
                "build": "pyhdfd78af_0",
            }
        )
    )

    records = runner.package_inventory(prefix)

    assert len(records) == 1
    assert records[0]["name"] == "genomad"
    assert records[0]["version"] == "1.12.0"
    assert records[0]["metadata_sha256"] == runner.sha256_file(package_file)


def configure_synthetic_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    fasta = tmp_path / "input.fasta"
    write_fasta(fasta)

    environment_prefix = tmp_path / "environment"
    executable = environment_prefix / "bin" / "genomad"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")

    database, checksum_table, checksums = build_synthetic_database(tmp_path)

    scope_contract = tmp_path / "scope_contract.txt"
    scope_contract.write_text("manuscript-only geNomad comparator contract\n")

    adapter_file = Path(runner.__file__).resolve().parents[1] / "adapters" / "genomad.py"

    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        runner,
        "EXPECTED_DATABASE_FINGERPRINT",
        runner.database_fingerprint(checksums),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_DATABASE_TABLE_SHA256",
        runner.sha256_file(checksum_table),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_SCOPE_CONTRACT_SHA256",
        runner.sha256_file(scope_contract),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_ADAPTER_SHA256",
        runner.sha256_file(adapter_file),
    )
    monkeypatch.setattr(
        runner,
        "get_tool_version",
        lambda executable_path: runner.TOOL_VERSION,
    )

    return {
        "fasta": fasta,
        "environment_prefix": environment_prefix,
        "executable": executable,
        "database": database,
        "checksum_table": checksum_table,
        "scope_contract": scope_contract,
        "output": tmp_path / "output",
    }


def write_synthetic_standardized_output(
    input_fasta: Path,
    output_directory: Path,
) -> dict[str, object]:
    records = runner.load_fasta_headers(input_fasta)
    standardized = output_directory / "standardized_predictions.tsv"

    rows = []
    for index, (contig_id, input_header) in enumerate(records):
        row = {field: "" for field in runner.OUTPUT_FIELDS}
        row.update(
            {
                "contig_id": contig_id,
                "input_header": input_header,
                "predicted_label": ("plasmid" if index == 0 else "chromosome"),
                "prediction_status": ("called_plasmid" if index == 0 else "not_detected"),
                "source_tool": runner.TOOL_NAME,
                "source_version": runner.TOOL_VERSION,
            }
        )
        rows.append(row)

    with standardized.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=runner.OUTPUT_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata: dict[str, object] = {
        "schema_version": "nar-comparator-adapter-v1",
        "scope": runner.SCOPE,
        "standardized_rows": len(rows),
        "label_counts": {
            "plasmid": 1,
            "chromosome": 1,
        },
        "successful_tool_output": True,
        "confirmatory_tuning": False,
    }
    (output_directory / "adapter_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def test_execute_command_returns_child_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 4321
        returncode = 0

        def communicate(
            self,
            timeout: int | None = None,
        ) -> tuple[str, str]:
            observed["timeout"] = timeout
            return "tool stdout\n", "tool stderr\n"

    def fake_popen(
        command: list[str],
        **kwargs: object,
    ) -> FakeProcess:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    returncode, timed_out, stdout, stderr = runner.execute_command(
        command=["tool", "--flag"],
        environment={"PATH": "/env/bin"},
        working_directory=tmp_path,
        timeout_seconds=30,
    )

    assert returncode == 0
    assert timed_out is False
    assert stdout == "tool stdout\n"
    assert stderr == "tool stderr\n"
    assert observed["timeout"] == 30
    assert observed["command"] == ["tool", "--flag"]

    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["shell"] if "shell" in kwargs else True
    assert kwargs["start_new_session"] is True


def test_execute_command_terminates_process_group_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    class TimeoutProcess:
        pid = 9876
        returncode = -15

        def __init__(self) -> None:
            self.calls = 0

        def communicate(
            self,
            timeout: int | None = None,
        ) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(
                    cmd=["genomad"],
                    timeout=timeout or 1,
                )
            return "partial stdout\n", "partial stderr\n"

    process = TimeoutProcess()
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda process_id, signal_number: signals.append((process_id, signal_number)),
    )

    returncode, timed_out, stdout, stderr = runner.execute_command(
        command=["genomad"],
        environment={},
        working_directory=tmp_path,
        timeout_seconds=1,
    )

    assert returncode == 124
    assert timed_out is True
    assert stdout == "partial stdout\n"
    assert "partial stderr" in stderr
    assert "Runner timeout after 1 seconds" in stderr
    assert signals == [(9876, runner.signal.SIGTERM)]


def test_standardization_failure_becomes_complete_abstention(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    raw = tmp_path / "missing_raw_output"
    output = tmp_path / "output"
    output.mkdir()
    write_fasta(fasta)

    succeeded, metadata, discovered, error = runner.standardize_genomad_output(
        input_fasta=fasta,
        raw_output_directory=raw,
        output_directory=output,
    )

    assert succeeded is False
    assert discovered == {}
    assert "RuntimeError" in error
    assert metadata["successful_tool_output"] is False

    with (output / "standardized_predictions.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(rows) == 2
    assert {row["predicted_label"] for row in rows} == {"unclassified"}
    assert {row["prediction_status"] for row in rows} == {"tool_failed"}


def test_successful_run_writes_complete_provenance_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = configure_synthetic_run(tmp_path, monkeypatch)

    def fake_execute(
        command: list[str],
        environment: dict[str, str],
        working_directory: Path,
        timeout_seconds: int,
    ) -> tuple[int, bool, str, str]:
        raw_output = Path(command[command.index("end-to-end") + 2])
        raw_output.mkdir(parents=True)
        (raw_output / "synthetic_artifact.txt").write_text("synthetic geNomad output\n")
        assert environment["CONDA_PREFIX"] == str(paths["environment_prefix"])
        assert working_directory == Path.cwd().resolve()
        assert timeout_seconds == 120
        return (
            0,
            False,
            "geNomad completed\n",
            ("2.50 real 4.00 user 0.50 sys\n" "123456 maximum resident set size\n"),
        )

    def fake_standardize(
        input_fasta: Path,
        raw_output_directory: Path,
        output_directory: Path,
    ) -> tuple[
        bool,
        dict[str, object],
        dict[str, str],
        str,
    ]:
        assert raw_output_directory.is_dir()
        metadata = write_synthetic_standardized_output(
            input_fasta,
            output_directory,
        )
        return (
            True,
            metadata,
            {"synthetic": str(raw_output_directory)},
            "",
        )

    monkeypatch.setattr(runner, "execute_command", fake_execute)
    monkeypatch.setattr(
        runner,
        "standardize_genomad_output",
        fake_standardize,
    )

    manifest = runner.run_genomad(
        input_fasta=paths["fasta"],
        output_directory=paths["output"],
        environment_prefix=paths["environment_prefix"],
        database_directory=paths["database"],
        checksum_table=paths["checksum_table"],
        scope_contract=paths["scope_contract"],
        threads=4,
        timeout_seconds=120,
    )

    assert manifest["run_status"] == "ok"
    assert manifest["scope"] == runner.SCOPE
    assert manifest["production_workflow_component"] is False
    assert manifest["confirmatory_tuning"] is False
    assert manifest["execution"]["shell"] is False
    assert manifest["execution"]["threads"] == 4
    assert manifest["execution"]["splits"] == 8
    assert manifest["execution"]["score_calibration"] is True
    assert manifest["execution"]["custom_thresholds"] is False
    assert manifest["execution"]["timed_out"] is False
    assert manifest["database"]["pre_run"]["valid"] is True
    assert manifest["database"]["post_run"]["valid"] is True
    assert manifest["adapter"]["succeeded"] is True

    saved = json.loads((paths["output"] / "run_manifest.json").read_text())
    assert saved["run_status"] == "ok"
    assert saved["input"]["sequences"] == 2
    assert saved["input"]["bases"] == 16

    for required in [
        "command.json",
        "environment.json",
        "stdout.log",
        "stderr.log",
        "standardized_predictions.tsv",
        "adapter_metadata.json",
        "run_manifest.json",
    ]:
        assert (paths["output"] / required).is_file()


def test_failed_tool_run_is_preserved_as_abstention_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = configure_synthetic_run(tmp_path, monkeypatch)

    def fake_failed_execute(
        command: list[str],
        environment: dict[str, str],
        working_directory: Path,
        timeout_seconds: int,
    ) -> tuple[int, bool, str, str]:
        raw_output = Path(command[command.index("end-to-end") + 2])
        raw_output.mkdir(parents=True)
        return 7, False, "", "synthetic tool failure\n"

    monkeypatch.setattr(
        runner,
        "execute_command",
        fake_failed_execute,
    )

    manifest = runner.run_genomad(
        input_fasta=paths["fasta"],
        output_directory=paths["output"],
        environment_prefix=paths["environment_prefix"],
        database_directory=paths["database"],
        checksum_table=paths["checksum_table"],
        scope_contract=paths["scope_contract"],
        threads=2,
        timeout_seconds=60,
    )

    assert manifest["run_status"] == "failed"
    assert manifest["execution"]["returncode"] == 7
    assert manifest["adapter"]["succeeded"] is False
    assert any("geNomad return code: 7" in value for value in manifest["failure_reasons"])
    assert any("adapter failure" in value for value in manifest["failure_reasons"])

    with (paths["output"] / "standardized_predictions.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(rows) == 2
    assert {row["predicted_label"] for row in rows} == {"unclassified"}
    assert {row["prediction_status"] for row in rows} == {"tool_failed"}


def test_post_run_database_tampering_fails_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = configure_synthetic_run(tmp_path, monkeypatch)

    def tampering_execute(
        command: list[str],
        environment: dict[str, str],
        working_directory: Path,
        timeout_seconds: int,
    ) -> tuple[int, bool, str, str]:
        raw_output = Path(command[command.index("end-to-end") + 2])
        raw_output.mkdir(parents=True)
        first_database_file = sorted(
            path for path in paths["database"].rglob("*") if path.is_file()
        )[0]
        first_database_file.write_bytes(b"post-run tampering\n")
        return 0, False, "", ""

    def fake_standardize(
        input_fasta: Path,
        raw_output_directory: Path,
        output_directory: Path,
    ) -> tuple[
        bool,
        dict[str, object],
        dict[str, str],
        str,
    ]:
        metadata = write_synthetic_standardized_output(
            input_fasta,
            output_directory,
        )
        return True, metadata, {}, ""

    monkeypatch.setattr(
        runner,
        "execute_command",
        tampering_execute,
    )
    monkeypatch.setattr(
        runner,
        "standardize_genomad_output",
        fake_standardize,
    )

    manifest = runner.run_genomad(
        input_fasta=paths["fasta"],
        output_directory=paths["output"],
        environment_prefix=paths["environment_prefix"],
        database_directory=paths["database"],
        checksum_table=paths["checksum_table"],
        scope_contract=paths["scope_contract"],
    )

    assert manifest["run_status"] == "failed"
    assert manifest["database"]["pre_run"]["valid"] is True
    assert manifest["database"]["post_run"]["valid"] is False
    assert "post-run database verification failed" in (manifest["failure_reasons"])


def test_runner_remains_outside_production_workflow() -> None:
    project_root = Path(__file__).resolve().parents[2]
    production_sources = project_root / "src" / "plasflow2"

    forbidden_imports = []
    for source_file in production_sources.rglob("*.py"):
        text = source_file.read_text()
        if (
            "scripts.benchmark" in text
            or "benchmark.runners.genomad" in text
            or "benchmark.adapters.genomad" in text
        ):
            forbidden_imports.append(str(source_file.relative_to(project_root)))

    assert forbidden_imports == []

    pyproject = (project_root / "pyproject.toml").read_text()
    assert "scripts.benchmark.runners.genomad" not in pyproject
    assert "scripts.benchmark.adapters.genomad" not in pyproject

    assert runner.SCOPE == ("manuscript-only comparative benchmarking")


def test_runner_help_works_by_direct_path_from_external_directory(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    runner_path = project_root / "scripts" / "benchmark" / "runners" / "genomad.py"

    completed = subprocess.run(
        [sys.executable, str(runner_path), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "manuscript-only geNomad comparator" in completed.stdout
    assert "not part of the PlasFlow2 prediction workflow" in (completed.stdout)
    assert completed.stderr == ""
