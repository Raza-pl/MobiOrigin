"""Tests for the frozen manuscript-only Platon runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.benchmark.runners import platon as runner


def _contract_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmark/contracts/platon_runner_contract_v1.json"
    )


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{header}\n{sequence}\n" for header, sequence in records))


def _tool_result(return_code: int = 0) -> dict[str, Any]:
    return {
        "started_at": "2026-07-31T00:00:00Z",
        "finished_at": "2026-07-31T00:00:01Z",
        "wallclock_seconds": 1.0,
        "return_code": return_code,
        "timed_out": return_code == 124,
        "interrupted": False,
        "user_seconds": 0.1,
        "system_seconds": 0.1,
        "peak_rss_bytes_when_available": 1,
    }


def _configure_mocked_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tool_return_code: int = 0,
    adapter_return_code: int = 0,
) -> tuple[Path, Path, Path, Path]:
    input_fasta = tmp_path / "input.fasta"
    output_directory = tmp_path / "output"
    database_directory = tmp_path / "database"
    scope_contract = tmp_path / "scope.txt"
    database_directory.mkdir()
    scope_contract.write_text("scope\n")
    _write_fasta(input_fasta, [("contig description", "A" * 1_200)])

    contract = {"protocol_contract": {"sha256": runner.sha256_file(scope_contract)}}
    image = {"id": runner.IMAGE_ID, "os": "linux", "architecture": "amd64"}
    database = {
        "manifest_sha256": runner.DATABASE_MANIFEST_SHA256,
        "verified_files": 31,
        "verified_bytes": 100,
    }

    monkeypatch.setattr(runner, "validate_runner_contract", lambda path: contract)
    monkeypatch.setattr(runner, "validate_adapter_identity", lambda path: runner.ADAPTER_SHA256)
    monkeypatch.setattr(runner, "find_docker_binary", lambda: Path("/bin/echo"))
    monkeypatch.setattr(runner, "current_host_docker_context", lambda binary: "default")
    monkeypatch.setattr(runner, "inspect_frozen_image", lambda binary: dict(image))
    monkeypatch.setattr(
        runner,
        "verify_database_manifest",
        lambda database_path, frozen_contract: dict(database),
    )
    monkeypatch.setattr(
        runner,
        "execute_container",
        lambda command, stdout_path, stderr_path: _tool_result(tool_return_code),
    )
    monkeypatch.setattr(
        runner,
        "validate_required_native_outputs",
        lambda raw_directory: {
            "plasmid_fasta": 1,
            "chromosome_fasta": 1,
            "json": 1,
            "tsv": 1,
            "log": 1,
        },
    )

    def fake_adapter(
        adapter_path: Path,
        adapter_input: Path,
        raw_directory: Path,
        adapter_output: Path,
    ) -> dict[str, Any]:
        if adapter_return_code == 0:
            (adapter_output / "standardized_predictions.tsv").write_text(
                "contig_id\tpredicted_label\ncontig\tplasmid\n"
            )
            (adapter_output / "adapter_metadata.json").write_text("{}\n")
        return {
            "argv": ["adapter"],
            "return_code": adapter_return_code,
            "stdout_path": "adapter_stdout.log",
            "stderr_path": "adapter_stderr.log",
        }

    monkeypatch.setattr(runner, "run_adapter", fake_adapter)
    monkeypatch.setattr(runner, "validate_standardized_count", lambda path, count: None)
    monkeypatch.setattr(
        runner,
        "remove_container",
        lambda binary, name: {
            "return_code": 0,
            "already_absent": True,
            "stderr": "",
        },
    )
    return input_fasta, output_directory, database_directory, scope_contract


def test_runner_contract_identity_and_frozen_parameters() -> None:
    contract = runner.validate_runner_contract(_contract_path())
    assert contract["contract_sha256"] == runner.RUNNER_CONTRACT_SHA256
    assert contract["publication_parameters"]["mode"] == "accuracy"
    assert contract["publication_parameters"]["metagenome_mode"] is True
    assert contract["publication_parameters"]["threads"] == 4
    assert contract["publication_parameters"]["timeout_seconds"] == 172_800
    assert contract["publication_parameters"]["threshold_tuning_allowed"] is False


def test_runner_contract_rejects_tampering(tmp_path: Path) -> None:
    contract = json.loads(_contract_path().read_text())
    contract["publication_parameters"]["threads"] = 5
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(runner.RunnerContractError, match="content hash mismatch"):
        runner.validate_runner_contract(path)


def test_adapter_identity_is_frozen() -> None:
    adapter = Path(__file__).resolve().parents[2] / "scripts/benchmark/adapters/platon.py"
    assert runner.validate_adapter_identity(adapter) == runner.ADAPTER_SHA256


def test_adapter_identity_rejects_changed_source(tmp_path: Path) -> None:
    changed = tmp_path / "platon.py"
    changed.write_text("CONTRACT_SHA256 = 'changed'\n")
    with pytest.raises(runner.RunnerContractError, match="SHA-256 mismatch"):
        runner.validate_adapter_identity(changed)


@pytest.mark.parametrize("cohort_role", ["development", "confirmatory"])
def test_frozen_cohort_roles_are_accepted(cohort_role: str) -> None:
    runner.validate_cohort_role(cohort_role)


def test_unknown_cohort_role_is_rejected() -> None:
    with pytest.raises(runner.RunnerContractError, match="Invalid cohort role"):
        runner.validate_cohort_role("tuning")


def test_fasta_inventory_records_supported_and_unsupported_lengths(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    _write_fasta(
        fasta,
        [
            ("supported description", "A" * 1_000),
            ("short", "C" * 999),
            ("long", "G" * 500_001),
        ],
    )
    inventory = runner.inventory_fasta(fasta)
    assert inventory["sequence_count"] == 3
    assert inventory["base_count"] == 502_000
    assert inventory["unsupported_length_count"] == 2


def test_fasta_inventory_rejects_canonical_collision(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    _write_fasta(fasta, [("same one", "A"), ("same two", "C")])
    with pytest.raises(runner.RunnerContractError, match="Duplicate canonical"):
        runner.inventory_fasta(fasta)


def test_fasta_inventory_rejects_empty_record(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">empty\n")
    with pytest.raises(runner.RunnerContractError, match="no sequence"):
        runner.inventory_fasta(fasta)


def test_output_directory_overwrite_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(runner.RunnerContractError, match="already exists"):
        runner.prepare_output_directory(output)


def test_container_command_preserves_frozen_isolation(tmp_path: Path) -> None:
    input_fasta = tmp_path / "input.fasta"
    database = tmp_path / "database"
    output = tmp_path / "output"
    input_fasta.write_text(">x\nA\n")
    database.mkdir()
    output.mkdir()
    command = runner.build_container_command(
        Path("/opt/homebrew/bin/docker"),
        input_fasta,
        database,
        output,
        "platon-test",
    )
    assert command[:3] == [
        "/opt/homebrew/bin/docker",
        "--context",
        runner.DOCKER_CONTEXT,
    ]
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "--platform" in command
    assert command[command.index("--platform") + 1] == "linux/amd64"
    assert runner.IMAGE_REFERENCE in command
    assert command[-13:] == [
        "/usr/local/bin/platon",
        "--db",
        "/database",
        "--output",
        "/work/raw",
        "--prefix",
        "platon",
        "--mode",
        "accuracy",
        "--meta",
        "--threads",
        "4",
        "/work/input.fasta",
    ]
    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert any(value.endswith("dst=/work/input.fasta,readonly") for value in mounts)
    assert any(value.endswith("dst=/database,readonly") for value in mounts)
    assert any(value.endswith("dst=/work") and "readonly" not in value for value in mounts)


def test_image_inspection_rejects_changed_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps([{"Id": "sha256:changed", "Os": "linux", "Architecture": "amd64"}]),
        stderr="",
    )
    monkeypatch.setattr(runner, "run_capture", lambda command: completed)
    with pytest.raises(runner.RunnerContractError, match="image ID mismatch"):
        runner.inspect_frozen_image(Path("/bin/docker"))


def test_scope_contract_hash_is_verified(tmp_path: Path) -> None:
    scope = tmp_path / "scope.txt"
    scope.write_text("frozen\n")
    contract = {"protocol_contract": {"sha256": runner.sha256_file(scope)}}
    assert runner.validate_scope_contract(scope, contract) == runner.sha256_file(scope)
    scope.write_text("changed\n")
    with pytest.raises(runner.RunnerContractError, match="SHA-256 mismatch"):
        runner.validate_scope_contract(scope, contract)


def test_native_output_contract_requires_every_file(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in (
        "platon.plasmid.fasta",
        "platon.chromosome.fasta",
        "platon.json",
        "platon.tsv",
    ):
        (raw / name).write_text("")
    with pytest.raises(runner.RunnerContractError, match="output is missing"):
        runner.validate_required_native_outputs(raw)


def test_standardized_count_requires_one_row_per_input(tmp_path: Path) -> None:
    table = tmp_path / "standardized.tsv"
    table.write_text("contig_id\tpredicted_label\none\tplasmid\n")
    runner.validate_standardized_count(table, 1)
    with pytest.raises(runner.RunnerContractError, match="row count mismatch"):
        runner.validate_standardized_count(table, 2)


def test_artifact_checksum_manifest_is_complete(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "two.txt").write_text("two\n")
    runner.write_artifact_checksums(tmp_path)
    lines = (tmp_path / "artifact_checksums.sha256").read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("  nested/two.txt")
    assert lines[1].endswith("  one.txt")


@pytest.mark.parametrize("cohort_role", ["development", "confirmatory"])
def test_successful_mocked_run_records_frozen_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cohort_role: str,
) -> None:
    input_fasta, output, database, scope = _configure_mocked_run(monkeypatch, tmp_path)
    provenance = runner.run_platon(
        input_fasta=input_fasta,
        output_directory=output,
        database_directory=database,
        docker_context=runner.DOCKER_CONTEXT,
        scope_contract=scope,
        cohort_role=cohort_role,
    )
    assert provenance["status"] == "PASS"
    assert provenance["cohort_role"] == cohort_role
    assert provenance["confirmatory_tuning_performed"] is False
    assert provenance["publication_parameters"] == {
        "mode": "accuracy",
        "metagenome_mode": True,
        "characterize_mode": False,
        "threads": 4,
        "timeout_seconds": 172_800,
    }
    assert (output / "runner_provenance.json").is_file()
    assert (output / "artifact_checksums.sha256").is_file()


def test_nonzero_tool_failure_retains_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_fasta, output, database, scope = _configure_mocked_run(
        monkeypatch, tmp_path, tool_return_code=2
    )
    with pytest.raises(runner.RunnerContractError, match="nonzero status 2"):
        runner.run_platon(
            input_fasta=input_fasta,
            output_directory=output,
            database_directory=database,
            docker_context=runner.DOCKER_CONTEXT,
            scope_contract=scope,
            cohort_role="development",
        )
    provenance = json.loads((output / "runner_provenance.json").read_text())
    assert provenance["status"] == "FAIL"
    assert provenance["tool_result"]["return_code"] == 2
    assert (output / "artifact_checksums.sha256").is_file()


def test_adapter_failure_retains_raw_execution_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_fasta, output, database, scope = _configure_mocked_run(
        monkeypatch, tmp_path, adapter_return_code=1
    )
    with pytest.raises(runner.RunnerContractError, match="adapter returned status 1"):
        runner.run_platon(
            input_fasta=input_fasta,
            output_directory=output,
            database_directory=database,
            docker_context=runner.DOCKER_CONTEXT,
            scope_contract=scope,
            cohort_role="development",
        )
    provenance = json.loads((output / "runner_provenance.json").read_text())
    assert provenance["tool_result"]["return_code"] == 0
    assert provenance["adapter_result"]["return_code"] == 1


@pytest.mark.parametrize(
    ("changed_component", "expected_message"),
    [
        ("image", "image identity changed"),
        ("database", "Database identity changed"),
        ("context", "Docker context changed"),
    ],
)
def test_post_run_identity_changes_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_component: str,
    expected_message: str,
) -> None:
    input_fasta, output, database, scope = _configure_mocked_run(monkeypatch, tmp_path)

    if changed_component == "image":
        images = iter(
            [
                {"id": runner.IMAGE_ID, "os": "linux", "architecture": "amd64"},
                {"id": "changed", "os": "linux", "architecture": "amd64"},
            ]
        )
        monkeypatch.setattr(runner, "inspect_frozen_image", lambda binary: next(images))
    elif changed_component == "database":
        databases = iter(
            [
                {
                    "manifest_sha256": runner.DATABASE_MANIFEST_SHA256,
                    "verified_files": 31,
                    "verified_bytes": 100,
                },
                {
                    "manifest_sha256": "changed",
                    "verified_files": 31,
                    "verified_bytes": 100,
                },
            ]
        )
        monkeypatch.setattr(
            runner,
            "verify_database_manifest",
            lambda path, contract: next(databases),
        )
    else:
        contexts = iter(["default", "changed"])
        monkeypatch.setattr(runner, "current_host_docker_context", lambda binary: next(contexts))

    with pytest.raises(runner.RunnerContractError, match=expected_message):
        runner.run_platon(
            input_fasta=input_fasta,
            output_directory=output,
            database_directory=database,
            docker_context=runner.DOCKER_CONTEXT,
            scope_contract=scope,
            cohort_role="development",
        )


def test_changed_docker_context_argument_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.RunnerContractError, match="Docker context"):
        runner.run_platon(
            input_fasta=tmp_path / "input.fasta",
            output_directory=tmp_path / "output",
            database_directory=tmp_path / "database",
            docker_context="default",
            scope_contract=tmp_path / "scope.txt",
            cohort_role="development",
        )


def test_runner_help_works_from_external_directory(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    runner_path = project_root / "scripts/benchmark/runners/platon.py"
    completed = subprocess.run(
        [sys.executable, str(runner_path), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    normalized = " ".join(completed.stdout.split())
    assert completed.returncode == 0
    assert "manuscript-only Platon 1.7 comparator" in normalized
    assert "not part of the PlasFlow2 prediction workflow" in normalized
    assert completed.stderr == ""


def test_runner_is_isolated_from_production_sources() -> None:
    project_root = Path(__file__).resolve().parents[2]
    production = project_root / "src/plasflow2"
    violations = []
    for path in production.rglob("*.py"):
        text = path.read_text()
        if "benchmark.runners.platon" in text or "benchmark.adapters.platon" in text:
            violations.append(str(path.relative_to(project_root)))
    assert violations == []
