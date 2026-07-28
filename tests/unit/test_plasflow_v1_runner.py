from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.benchmark.runners import plasflow_v1 as runner

CONTRACT_PATH = runner.PROJECT_ROOT / runner.CONTRACT_RELATIVE_PATH


def test_runner_contract_is_authentic() -> None:
    contract = runner.validate_runner_contract(CONTRACT_PATH)

    assert contract["contract_sha256"] == runner.RUNNER_CONTRACT_SHA256
    assert contract["production_workflow_component"] is False
    assert contract["cohort_dependence"]["full_frozen_cohort_single_invocation_required"] is True


def test_tampered_runner_contract_is_rejected(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    contract["publication_parameters"]["threshold"] = 0.6
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(contract))

    with pytest.raises(
        runner.RunnerContractError,
        match="content hash mismatch",
    ):
        runner.validate_runner_contract(tampered)


def test_frozen_adapter_identity_is_authentic() -> None:
    adapter = runner.PROJECT_ROOT / runner.ADAPTER_RELATIVE_PATH

    assert runner.validate_adapter_identity(adapter) == runner.ADAPTER_SHA256


def test_invalid_context_is_rejected_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner,
        "find_docker_binary",
        lambda: pytest.fail("Docker discovery must not occur"),
    )

    with pytest.raises(
        runner.RunnerContractError,
        match="Docker context must equal",
    ):
        runner.execute_frozen_run(
            input_fasta=tmp_path / "unused.fasta",
            output_directory=tmp_path / "unused-output",
            scope_contract=CONTRACT_PATH,
            docker_context="default",
            cohort_role="development",
        )


def test_invalid_cohort_role_is_rejected() -> None:
    with pytest.raises(
        runner.RunnerContractError,
        match="Invalid cohort role",
    ):
        runner.validate_cohort_role("tuning")


def test_fasta_inventory_records_identity_and_order(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">contig_b description\n" "ACGT\n" ">contig_a\n" "AACCGG\n")

    inventory = runner.inventory_fasta(fasta)

    assert inventory["sequence_count"] == 2
    assert inventory["base_count"] == 10
    assert inventory["identifiers"] == ["contig_b", "contig_a"]
    assert inventory["sha256"] == runner.sha256_file(fasta)


def test_fasta_identifier_collisions_are_rejected(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "collision.fasta"
    fasta.write_text(">same first\n" "AAAA\n" ">same second\n" "CCCC\n")

    with pytest.raises(
        runner.RunnerContractError,
        match="identifier collision",
    ):
        runner.inventory_fasta(fasta)


def test_nonempty_output_directory_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite\n")

    with pytest.raises(
        runner.RunnerContractError,
        match="new or empty",
    ):
        runner.prepare_output_directory(output)


def test_input_staging_is_byte_identical(tmp_path: Path) -> None:
    fasta = tmp_path / "source.fasta"
    fasta.write_bytes(b">a description\nACGT\n")
    output = tmp_path / "output"
    output.mkdir()

    staged = runner.stage_input(
        fasta,
        output,
        runner.sha256_file(fasta),
    )

    assert staged.read_bytes() == fasta.read_bytes()
    assert runner.sha256_file(staged) == runner.sha256_file(fasta)


def test_frozen_distribution_has_no_batch_or_cache_interface() -> None:
    contract = runner.validate_runner_contract(CONTRACT_PATH)
    parameters = contract["publication_parameters"]
    cohort = contract["cohort_dependence"]

    assert parameters["native_batch_size_argument_supported"] is False
    assert parameters["native_cache_files_produced"] is False
    assert cohort["native_batching_supported"] is False
    assert cohort["native_cache_files_produced"] is False
    assert cohort["full_cohort_loaded_for_kmer_counting"] is True


def test_frozen_command_has_no_network_and_no_sharding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_is_file = Path.is_file

    def mocked_is_file(file_path: Path) -> bool:
        if str(file_path) in {
            "/usr/bin/caffeinate",
            "/usr/bin/time",
        }:
            return True
        return original_is_file(file_path)

    monkeypatch.setattr(Path, "is_file", mocked_is_file)

    command = runner.build_tool_command(
        Path("/mock/docker"),
        tmp_path.resolve(),
        "plasflow-v1-test",
    )

    assert command[0:4] == [
        "/usr/bin/caffeinate",
        "-i",
        "/usr/bin/time",
        "-l",
    ]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--platform") + 1] == "linux/amd64"
    assert command[command.index("--threshold") + 1] == "0.7"
    assert "--batch_size" not in command
    assert command.count(runner.IMAGE_REFERENCE) == 1
    assert "--shard" not in command
    assert "--threads" not in command


def test_missing_container_cleanup_is_successful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=["docker"],
        returncode=1,
        stdout="",
        stderr="Error response from daemon: No such container: test",
    )
    monkeypatch.setattr(
        runner,
        "run_capture",
        lambda *args, **kwargs: completed,
    )

    result = runner.cleanup_container(
        Path("/mock/docker"),
        "test",
    )

    assert result["already_absent"] is True
    assert result["successful"] is True


def test_timeout_terminates_complete_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "TIMEOUT_SECONDS", 0.05)

    result = runner.execute_tool_process(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
        ],
        tmp_path / "stdout.log",
        tmp_path / "stderr.log",
    )

    assert result["timed_out"] is True
    assert result["returncode"] != 0
    assert result["termination"]["sigterm_sent"] is True


def _configure_mock_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int,
    timed_out: bool = False,
    complete_output: bool = True,
) -> None:
    image = {
        "id": "sha256:mock-image",
        "repo_digests": ["quay.io/biocontainers/plasflow@" + runner.IMAGE_DIGEST],
        "os": "linux",
        "architecture": "amd64",
    }

    monkeypatch.setattr(
        runner,
        "validate_adapter_identity",
        lambda adapter_path: runner.ADAPTER_SHA256,
    )
    monkeypatch.setattr(
        runner,
        "find_docker_binary",
        lambda: Path("/mock/docker"),
    )
    monkeypatch.setattr(
        runner,
        "current_host_docker_context",
        lambda docker_binary: "desktop-default",
    )
    monkeypatch.setattr(
        runner,
        "container_inventory",
        lambda docker_binary: [],
    )
    monkeypatch.setattr(
        runner,
        "inspect_frozen_image",
        lambda docker_binary: dict(image),
    )
    monkeypatch.setattr(
        runner,
        "build_tool_command",
        lambda docker_binary, output_directory, container_name: [
            "/mock/docker",
            "--context",
            runner.DOCKER_CONTEXT,
            "run",
            "--network",
            "none",
            runner.IMAGE_REFERENCE,
        ],
    )
    monkeypatch.setattr(
        runner,
        "cleanup_container",
        lambda docker_binary, container_name: {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "already_absent": False,
            "successful": True,
        },
    )

    def fake_execute(
        command: list[str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> dict[str, Any]:
        stdout_path.write_text("mock execution\n")
        stderr_path.write_text("")
        output = stdout_path.parent

        if returncode == 0:
            (output / "raw_predictions.tsv").write_text("mock table\n")
            for suffix in (
                "_plasmids.fasta",
                "_chromosomes.fasta",
                "_unclassified.fasta",
            ):
                (output / f"raw_predictions.tsv{suffix}").write_text("")
        return {
            "returncode": returncode,
            "timed_out": timed_out,
            "received_signal": None,
            "termination": {},
            "wallclock_seconds": 0.01,
            "user_seconds_delta": 0.0,
            "system_seconds_delta": 0.0,
            "maximum_resident_set_size_raw": 1,
        }

    monkeypatch.setattr(
        runner,
        "execute_tool_process",
        fake_execute,
    )

    def fake_adapter(
        *,
        input_fasta: Path,
        raw_predictions: Path,
        output_path: Path,
        metadata_output: Path,
    ) -> dict[str, Any]:
        inventory = runner.inventory_fasta(input_fasta)
        count = inventory["sequence_count"]
        output_path.write_text(
            "contig_id\tpredicted_label\tprediction_status\n"
            + "".join(
                f"{identifier}\tunclassified\tnative_abstention\n"
                for identifier in inventory["identifiers"]
            )
        )
        metadata = {
            "complete_output": complete_output,
            "runner_success_allowed": complete_output,
            "standardized_rows": count,
            "raw_prediction_rows": count,
        }
        runner.write_json(metadata_output, metadata)
        return metadata

    monkeypatch.setattr(
        runner,
        "adapt_plasflow_v1",
        fake_adapter,
    )


@pytest.mark.parametrize(
    ("cohort_role", "confirmatory_accessed"),
    [
        ("development", False),
        ("confirmatory", True),
    ],
)
def test_mocked_complete_run_records_frozen_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cohort_role: str,
    confirmatory_accessed: bool,
) -> None:
    _configure_mock_runtime(
        monkeypatch,
        returncode=0,
    )

    fasta = tmp_path / f"{cohort_role}.fasta"
    fasta.write_text(">first\n" "ACGT\n" ">second description\n" "AACCGG\n")
    output = tmp_path / f"{cohort_role}-output"

    provenance = runner.execute_frozen_run(
        input_fasta=fasta,
        output_directory=output,
        scope_contract=CONTRACT_PATH,
        docker_context=runner.DOCKER_CONTEXT,
        cohort_role=cohort_role,
    )

    assert provenance["status"] == "PASS"
    assert provenance["overall_status"] == 0
    assert provenance["production_workflow_component"] is False
    assert provenance["cohort_role"] == cohort_role
    assert provenance["confirmatory_data_accessed"] is confirmatory_accessed
    assert provenance["single_cohort_invocation"] is True
    assert provenance["external_sharding"] is False
    assert provenance["input"]["sequence_count"] == 2
    assert provenance["staged_input"]["matches_original"] is True
    assert provenance["container_inventory_restored"] is True
    assert provenance["host_docker_context_unchanged"] is True
    assert (output / "artifact_checksums.sha256").is_file()
    assert (output / "command.json").is_file()


def test_nonzero_execution_retains_failure_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_mock_runtime(
        monkeypatch,
        returncode=9,
    )

    fasta = tmp_path / "failure.fasta"
    fasta.write_text(">failure\nACGT\n")
    output = tmp_path / "failure-output"

    with pytest.raises(
        runner.RunnerContractError,
        match="returned nonzero status 9",
    ):
        runner.execute_frozen_run(
            input_fasta=fasta,
            output_directory=output,
            scope_contract=CONTRACT_PATH,
            docker_context=runner.DOCKER_CONTEXT,
            cohort_role="development",
        )

    provenance = json.loads((output / "runner_provenance.json").read_text())

    assert provenance["status"] == "FAIL"
    assert provenance["overall_status"] == 1
    assert provenance["container_cleanup"]["successful"] is True
    assert any("returned nonzero status 9" in error for error in provenance["errors"])
    assert (output / "resource_usage.txt").is_file()
    assert (output / "artifact_checksums.sha256").is_file()


def test_incomplete_adapter_output_fails_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_mock_runtime(
        monkeypatch,
        returncode=0,
        complete_output=False,
    )

    fasta = tmp_path / "incomplete.fasta"
    fasta.write_text(">incomplete\nACGT\n")
    output = tmp_path / "incomplete-output"

    with pytest.raises(
        runner.RunnerContractError,
        match="Adapter reported incomplete output",
    ):
        runner.execute_frozen_run(
            input_fasta=fasta,
            output_directory=output,
            scope_contract=CONTRACT_PATH,
            docker_context=runner.DOCKER_CONTEXT,
            cohort_role="development",
        )

    provenance = json.loads((output / "runner_provenance.json").read_text())

    assert provenance["status"] == "FAIL"
    assert any("Adapter reported incomplete output" in error for error in provenance["errors"])


def test_direct_help_preserves_manuscript_scope() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(runner.__file__).resolve()),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "manuscript-only" in completed.stdout
    assert "never externally sharded" in completed.stdout
    assert "not part of the PlasFlow2 prediction workflow" in completed.stdout
