"""Tests for the frozen manuscript-only PLASMe runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.benchmark.runners import plasme as runner

PROJECT_ROOT = Path(__file__).parents[2]
TRACKED_CONTRACT = (
    PROJECT_ROOT / "scripts" / "benchmark" / "contracts" / "plasme_runner_contract_v1.json"
)
SCOPE_CONTRACT = (
    PROJECT_ROOT
    / "evaluation"
    / "release_audit"
    / "nar_benchmark_20260727"
    / "03_protocol_freeze"
    / "08d_protocol_freeze_report.txt"
)


def load_contract() -> dict[str, Any]:
    """Load the canonical tracked runner contract."""

    payload = json.loads(TRACKED_CONTRACT.read_text())
    assert isinstance(payload, dict)
    return payload


def write_fasta(path: Path) -> None:
    """Write one valid synthetic input record."""

    path.write_text(">p1 synthetic record\nAAAA\n")


def successful_tool_result() -> dict[str, Any]:
    """Return one synthetic successful execution result."""

    return {
        "started_at": "2026-07-29T00:00:00Z",
        "finished_at": "2026-07-29T00:00:01Z",
        "wallclock_seconds": 1.0,
        "return_code": 0,
        "timed_out": False,
        "interrupted": False,
        "user_seconds": 0.2,
        "system_seconds": 0.1,
        "peak_rss_bytes_when_available": 1024,
    }


def install_success_mocks(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    """Install a complete no-Docker successful runner boundary."""

    image = {
        "id": runner.IMAGE_ID,
        "os": "linux",
        "architecture": "amd64",
    }
    snapshot = {
        "file_count": 44,
        "total_bytes": 12345,
        "content_sha256": "database-content",
    }

    monkeypatch.setattr(
        runner,
        "validate_runner_contract",
        lambda path: contract,
    )
    monkeypatch.setattr(
        runner,
        "validate_adapter_identity",
        lambda path: runner.ADAPTER_SHA256,
    )
    monkeypatch.setattr(
        runner,
        "validate_scope_contract",
        lambda path, supplied_contract: "scope-sha256",
    )
    monkeypatch.setattr(
        runner,
        "find_docker_binary",
        lambda: Path("/usr/bin/docker"),
    )
    monkeypatch.setattr(
        runner,
        "current_host_docker_context",
        lambda docker: "unchanged-host-context",
    )
    monkeypatch.setattr(
        runner,
        "inspect_frozen_image",
        lambda docker: dict(image),
    )
    monkeypatch.setattr(
        runner,
        "validate_database_foundation",
        lambda database, supplied_contract: {
            "directory": str(database),
            "transformer_model_count": 36,
        },
    )
    monkeypatch.setattr(
        runner,
        "verify_database_snapshot",
        lambda database, supplied_contract: dict(snapshot),
    )
    monkeypatch.setattr(
        runner,
        "remove_container",
        lambda docker, name: {
            "return_code": 0,
            "already_absent": True,
            "stderr": "",
        },
    )

    def fake_execute(
        command: list[str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> dict[str, Any]:
        stdout_path.write_text("synthetic PLASMe stdout\n")
        stderr_path.write_text("")
        raw = stdout_path.parent / "raw"
        temporary = raw / "temp"
        temporary.mkdir(parents=True)
        (raw / "predicted_plasmids.fasta").write_text(">p1\nAAAA\n")
        (temporary / "PLASMe_candidate.csv").write_text(
            "order,query,identity,coverage,PLASMe,overlap\n" "1,p1,0.95,0.95,0.8,1\n"
        )
        return successful_tool_result()

    monkeypatch.setattr(runner, "execute_tool", fake_execute)

    def fake_adapter(
        input_fasta: Path,
        positive_fasta: Path,
        candidate_csv: Path,
        output_path: Path,
        metadata_output: Path | None = None,
    ) -> dict[str, Any]:
        assert input_fasta.is_file()
        assert positive_fasta.is_file()
        assert candidate_csv.is_file()

        output_path.write_text(
            "contig_id\tpredicted_label\tprediction_status\n" "p1\tplasmid\tcalled_plasmid\n"
        )
        metadata = {
            "status": "PASS",
            "rows": 1,
            "contract_sha256": runner.ADAPTER_CONTRACT_SHA256,
        }

        if metadata_output is not None:
            metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

        return metadata

    monkeypatch.setattr(runner, "adapt_plasme", fake_adapter)


def test_tracked_contract_is_valid() -> None:
    contract = runner.validate_runner_contract(TRACKED_CONTRACT)

    assert contract["contract_sha256"] == (runner.RUNNER_CONTRACT_SHA256)
    assert contract["publication_parameters"]["threads"] == 8
    assert contract["publication_parameters"]["timeout_seconds"] == 172800
    assert contract["production_workflow_component"] is False


def test_contract_tampering_is_rejected(tmp_path: Path) -> None:
    contract = load_contract()
    contract["publication_parameters"]["threads"] = 7
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(contract))

    with pytest.raises(
        runner.RunnerContractError,
        match="content hash mismatch",
    ):
        runner.validate_runner_contract(tampered)


@pytest.mark.parametrize(
    "cohort_role",
    ["training", "tuning", "test", "", "Confirmatory"],
)
def test_unknown_cohort_roles_are_rejected(
    cohort_role: str,
) -> None:
    with pytest.raises(
        runner.RunnerContractError,
        match="Invalid cohort role",
    ):
        runner.validate_cohort_role(cohort_role)


@pytest.mark.parametrize(
    "cohort_role",
    ["development", "confirmatory"],
)
def test_frozen_cohort_roles_are_accepted(
    cohort_role: str,
) -> None:
    runner.validate_cohort_role(cohort_role)


def test_fasta_inventory_records_identity_and_size(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">p1 first record\n" "AAAA\n" ">c1 second record\n" "CC\n" "CC\n")

    inventory = runner.inventory_fasta(fasta)

    assert inventory["sequence_count"] == 2
    assert inventory["base_count"] == 8
    assert inventory["sha256"] == runner.sha256_file(fasta)


def test_fasta_inventory_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "duplicate.fasta"
    fasta.write_text(">p1 first\nAAAA\n>p1 second\nCCCC\n")

    with pytest.raises(
        runner.RunnerContractError,
        match="Duplicate canonical",
    ):
        runner.inventory_fasta(fasta)


def test_fasta_inventory_rejects_empty_sequences(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "empty.fasta"
    fasta.write_text(">p1\n>p2\nAAAA\n")

    with pytest.raises(
        runner.RunnerContractError,
        match="has no sequence",
    ):
        runner.inventory_fasta(fasta)


def test_existing_output_directory_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(
        runner.RunnerContractError,
        match="may not be overwritten",
    ):
        runner.prepare_output_directory(output)


def test_new_output_directory_is_created(
    tmp_path: Path,
) -> None:
    output = tmp_path / "new-output"

    resolved = runner.prepare_output_directory(output)

    assert resolved == output.resolve()
    assert output.is_dir()


def test_candidate_header_contract(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.csv"
    candidate.write_text("order,query,identity,coverage,PLASMe,overlap\n")

    runner.validate_candidate_header(candidate)

    candidate.write_text("query,identity,coverage,PLASMe\n")

    with pytest.raises(
        runner.RunnerContractError,
        match="Unexpected PLASMe candidate columns",
    ):
        runner.validate_candidate_header(candidate)


def test_scope_contract_identity_is_exact() -> None:
    contract = load_contract()

    digest = runner.validate_scope_contract(
        SCOPE_CONTRACT,
        contract,
    )

    assert digest == contract["protocol_contract"]["sha256"]


def test_build_container_command_is_frozen(
    tmp_path: Path,
) -> None:
    contract = load_contract()
    input_fasta = tmp_path / "input.fasta"
    database = tmp_path / "database"
    output = tmp_path / "output"

    command = runner.build_container_command(
        Path("/usr/bin/docker"),
        input_fasta,
        database,
        output,
        "plasme-test-container",
        contract,
    )

    assert command[:3] == [
        "/usr/bin/docker",
        "--context",
        runner.DOCKER_CONTEXT,
    ]
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "linux/amd64" in command
    assert runner.IMAGE_TAG in command
    assert "shell=True" not in " ".join(command)

    input_mount = next(value for value in command if "target=/work/input.fasta" in value)
    database_mount = next(value for value in command if "target=/opt/plasme/DB" in value)
    output_mount = next(value for value in command if "target=/work/raw" in value)

    assert input_mount.endswith(",readonly")
    assert database_mount.endswith(",readonly")
    assert not output_mount.endswith(",readonly")
    assert "target=/work," not in output_mount
    assert not any(value.endswith("target=/work") for value in command)

    entrypoint_index = command.index("--entrypoint")
    image_index = command.index(runner.IMAGE_TAG)
    effective_command = [
        command[entrypoint_index + 1],
        *command[image_index + 1 :],
    ]

    assert effective_command == (contract["command_contract"]["official_command"])
    assert effective_command == [
        "/opt/conda/envs/plasme/bin/python",
        "/opt/plasme/PLASMe.py",
        "/work/input.fasta",
        "/work/raw/predicted_plasmids.fasta",
        "--mode",
        "balance",
        "--thread",
        "8",
        "--temp",
        "/work/raw/temp",
    ]


def test_output_mount_cannot_shadow_input_mount(
    tmp_path: Path,
) -> None:
    command = runner.build_container_command(
        Path("/usr/bin/docker"),
        tmp_path / "input.fasta",
        tmp_path / "database",
        tmp_path / "output",
        "plasme-mount-test",
        load_contract(),
    )

    mount_values = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]

    assert any("target=/work/input.fasta,readonly" in value for value in mount_values)
    assert any("target=/work/raw" in value for value in mount_values)
    assert not any(value.endswith("target=/work") for value in mount_values)


def test_output_argument_is_a_new_fasta_file() -> None:
    contract = load_contract()
    command = contract["command_contract"]
    official = command["official_command"]

    assert official[3] == ("/work/raw/predicted_plasmids.fasta")
    assert command["raw_output_container_path"] == ("/work/raw/predicted_plasmids.fasta")
    assert command["output_argument_type"] == "FASTA file"
    assert command["output_path_must_not_exist_before_run"] is True
    assert command["writable_container_directory"] == ("/work/raw")
    assert command["temporary_container_path"] == ("/work/raw/temp")
    assert command["temporary_container_path"] != command["raw_output_container_path"]


def test_expected_database_files_exclude_removed_runtime_inputs() -> None:
    contract = load_contract()

    expected = runner.expected_database_files(contract)

    assert "plsdb.zip" not in expected
    assert "plsdb_Mar30.fna" not in expected
    assert "plsdb_Mar30.fna.aa" not in expected
    assert "plas_chrom_thres.csv" in expected
    assert "trans_model/unified.pt" in expected
    assert "plsdb_Mar30.dmnd" in expected


def test_database_snapshot_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "database"
    database.mkdir()
    payload = database / "asset.bin"
    payload.write_bytes(b"verified database asset\n")
    expected_sha256 = hashlib.sha256(payload.read_bytes()).hexdigest()

    monkeypatch.setattr(
        runner,
        "expected_database_files",
        lambda contract: {
            "asset.bin": {
                "bytes": payload.stat().st_size,
                "sha256": expected_sha256,
            }
        },
    )

    snapshot = runner.verify_database_snapshot(database, {})

    assert snapshot["file_count"] == 1
    assert snapshot["total_bytes"] == payload.stat().st_size

    payload.write_bytes(b"tampered database asset\n")

    with pytest.raises(
        runner.RunnerContractError,
        match="size mismatch|SHA-256 mismatch",
    ):
        runner.verify_database_snapshot(database, {})


def test_database_snapshot_rejects_unexpected_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "database"
    database.mkdir()
    expected_file = database / "expected.bin"
    expected_file.write_bytes(b"expected\n")
    (database / "unexpected.bin").write_bytes(b"unexpected\n")

    monkeypatch.setattr(
        runner,
        "expected_database_files",
        lambda contract: {
            "expected.bin": {
                "bytes": expected_file.stat().st_size,
                "sha256": hashlib.sha256(expected_file.read_bytes()).hexdigest(),
            }
        },
    )

    with pytest.raises(
        runner.RunnerContractError,
        match="Unexpected runtime database files",
    ):
        runner.verify_database_snapshot(database, {})


def test_standardized_output_requires_one_row_per_input(
    tmp_path: Path,
) -> None:
    output = tmp_path / "standardized.tsv"
    output.write_text("contig_id\tpredicted_label\n" "p1\tplasmid\n")

    runner.validate_standardized_count(output, 1)

    with pytest.raises(
        runner.RunnerContractError,
        match="row count mismatch",
    ):
        runner.validate_standardized_count(output, 2)


def test_artifact_checksums_cover_retained_files(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    nested = output / "raw"
    nested.mkdir(parents=True)
    (output / "runner_provenance.json").write_text("{}\n")
    (nested / "result.tsv").write_text("result\n")

    runner.write_artifact_checksums(output)

    checksum_file = output / "artifact_checksums.sha256"
    lines = checksum_file.read_text().splitlines()

    assert len(lines) == 2
    assert any(line.endswith("  raw/result.tsv") for line in lines)
    assert any(line.endswith("  runner_provenance.json") for line in lines)
    assert not any(line.endswith("artifact_checksums.sha256") for line in lines)


def test_execute_tool_success_captures_resources(
    tmp_path: Path,
) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    result = runner.execute_tool(
        [
            sys.executable,
            "-c",
            "print('synthetic runner child')",
        ],
        stdout,
        stderr,
    )

    assert result["return_code"] == 0
    assert result["timed_out"] is False
    assert result["interrupted"] is False
    assert result["wallclock_seconds"] >= 0
    assert "synthetic runner child" in stdout.read_text()


def test_execute_tool_timeout_terminates_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    monkeypatch.setattr(runner, "TIMEOUT_SECONDS", 0.02)

    result = runner.execute_tool(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        stdout,
        stderr,
    )

    assert result["return_code"] == 124
    assert result["timed_out"] is True
    assert result["interrupted"] is False


def test_remove_container_uses_frozen_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_capture(
        command: list[str],
        *,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        captured.extend(command)
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Error: No such container",
        )

    monkeypatch.setattr(runner, "run_capture", fake_capture)

    result = runner.remove_container(
        Path("/usr/bin/docker"),
        "plasme-test",
    )

    assert captured[:3] == [
        "/usr/bin/docker",
        "--context",
        runner.DOCKER_CONTEXT,
    ]
    assert captured[-3:] == [
        "rm",
        "--force",
        "plasme-test",
    ]
    assert result["already_absent"] is True


def test_successful_runner_writes_complete_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract()
    install_success_mocks(monkeypatch, contract)
    input_fasta = tmp_path / "input.fasta"
    output = tmp_path / "output"
    write_fasta(input_fasta)

    provenance = runner.run_plasme(
        input_fasta=input_fasta,
        output_directory=output,
        database_directory=tmp_path / "database",
        docker_context=runner.DOCKER_CONTEXT,
        scope_contract=SCOPE_CONTRACT,
        cohort_role="development",
    )

    assert provenance["status"] == "PASS"
    assert provenance["cohort_role"] == "development"
    assert provenance["confirmatory_data_accessed"] is False
    assert provenance["errors"] == []

    required = [
        "command.json",
        "stdout.log",
        "stderr.log",
        "resource_usage.json",
        "standardized_predictions.tsv",
        "adapter_metadata.json",
        "runner_provenance.json",
        "artifact_checksums.sha256",
        "raw/predicted_plasmids.fasta",
        "raw/temp/PLASMe_candidate.csv",
    ]

    for relative_name in required:
        assert (output / relative_name).is_file(), relative_name

    command = json.loads((output / "command.json").read_text())["argv"]
    assert "--network" in command
    assert "--read-only" in command


def test_confirmatory_role_is_recorded_without_tuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract()
    install_success_mocks(monkeypatch, contract)
    input_fasta = tmp_path / "input.fasta"
    output = tmp_path / "output"
    write_fasta(input_fasta)

    provenance = runner.run_plasme(
        input_fasta=input_fasta,
        output_directory=output,
        database_directory=tmp_path / "database",
        docker_context=runner.DOCKER_CONTEXT,
        scope_contract=SCOPE_CONTRACT,
        cohort_role="confirmatory",
    )

    assert provenance["status"] == "PASS"
    assert provenance["confirmatory_data_accessed"] is True
    assert provenance["confirmatory_tuning_allowed"] is False


def test_nonzero_tool_status_retains_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract()
    install_success_mocks(monkeypatch, contract)
    input_fasta = tmp_path / "input.fasta"
    output = tmp_path / "output"
    write_fasta(input_fasta)

    def failed_execute(
        command: list[str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> dict[str, Any]:
        stdout_path.write_text("")
        stderr_path.write_text("synthetic failure\n")
        result = successful_tool_result()
        result["return_code"] = 9
        return result

    monkeypatch.setattr(runner, "execute_tool", failed_execute)

    with pytest.raises(
        runner.RunnerContractError,
        match="nonzero status 9",
    ):
        runner.run_plasme(
            input_fasta=input_fasta,
            output_directory=output,
            database_directory=tmp_path / "database",
            docker_context=runner.DOCKER_CONTEXT,
            scope_contract=SCOPE_CONTRACT,
            cohort_role="development",
        )

    provenance = json.loads((output / "runner_provenance.json").read_text())

    assert provenance["status"] == "FAIL"
    assert any("nonzero status 9" in error for error in provenance["errors"])
    assert (output / "stderr.log").is_file()
    assert (output / "resource_usage.json").is_file()
    assert (output / "artifact_checksums.sha256").is_file()


def test_changed_image_identity_causes_runner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract()
    install_success_mocks(monkeypatch, contract)
    input_fasta = tmp_path / "input.fasta"
    output = tmp_path / "output"
    write_fasta(input_fasta)

    images = iter(
        [
            {
                "id": runner.IMAGE_ID,
                "os": "linux",
                "architecture": "amd64",
            },
            {
                "id": "sha256:changed",
                "os": "linux",
                "architecture": "amd64",
            },
        ]
    )
    monkeypatch.setattr(
        runner,
        "inspect_frozen_image",
        lambda docker: next(images),
    )

    with pytest.raises(
        runner.RunnerContractError,
        match="image identity changed",
    ):
        runner.run_plasme(
            input_fasta=input_fasta,
            output_directory=output,
            database_directory=tmp_path / "database",
            docker_context=runner.DOCKER_CONTEXT,
            scope_contract=SCOPE_CONTRACT,
            cohort_role="development",
        )

    provenance = json.loads((output / "runner_provenance.json").read_text())
    assert provenance["status"] == "FAIL"


def test_changed_database_identity_causes_runner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract()
    install_success_mocks(monkeypatch, contract)
    input_fasta = tmp_path / "input.fasta"
    output = tmp_path / "output"
    write_fasta(input_fasta)

    snapshots = iter(
        [
            {
                "file_count": 44,
                "total_bytes": 12345,
                "content_sha256": "before",
            },
            {
                "file_count": 44,
                "total_bytes": 12345,
                "content_sha256": "after",
            },
        ]
    )
    monkeypatch.setattr(
        runner,
        "verify_database_snapshot",
        lambda database, supplied_contract: next(snapshots),
    )

    with pytest.raises(
        runner.RunnerContractError,
        match="database identity changed",
    ):
        runner.run_plasme(
            input_fasta=input_fasta,
            output_directory=output,
            database_directory=tmp_path / "database",
            docker_context=runner.DOCKER_CONTEXT,
            scope_contract=SCOPE_CONTRACT,
            cohort_role="development",
        )

    provenance = json.loads((output / "runner_provenance.json").read_text())
    assert provenance["status"] == "FAIL"
