"""Tests for the frozen manuscript-only PlasClass runner."""

from __future__ import annotations

import csv
import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.benchmark.runners import plasclass as runner

PROJECT_ROOT = Path(__file__).parents[2]
RUNNER_SOURCE = PROJECT_ROOT / "scripts" / "benchmark" / "runners" / "plasclass.py"
TRACKED_CONTRACT = (
    PROJECT_ROOT / "scripts" / "benchmark" / "contracts" / "plasclass_runner_contract_v1_1.json"
)


def write_fasta(path: Path) -> None:
    path.write_text(
        ">p1 plasmid description\n"
        "A" * 100 + "\n"
        ">c1 chromosome description\n" + "C" * 100 + "\n"
    )


def make_fake_environment(path: Path) -> None:
    bin_directory = path / "bin"
    bin_directory.mkdir(parents=True)
    (bin_directory / "python").write_text("")
    (bin_directory / "classify_fasta.py").write_text("#!/usr/bin/env python\n")


def valid_environment_result() -> dict[str, Any]:
    return {
        "valid": True,
        "environment_prefix": "/synthetic",
        "versions": {},
        "required_versions": {},
        "official_script": "/synthetic/classify_fasta.py",
        "official_script_sha256": "synthetic",
        "expected_official_script_sha256": "synthetic",
        "adapter_sha256": runner.EXPECTED_ADAPTER_SHA256,
        "expected_adapter_sha256": (runner.EXPECTED_ADAPTER_SHA256),
        "verified_model_assets": 8,
        "model_assets": [],
        "mismatches": [],
    }


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_tracked_runner_contract_is_valid() -> None:
    contract = runner.load_runner_contract(TRACKED_CONTRACT)

    assert contract["contract_sha256"] == (runner.EXPECTED_RUNNER_CONTRACT_SHA256)
    assert contract["schema_version"] == ("nar-plasclass-runner-contract-v1.1")
    assert contract["publication_parameters"]["processes"] == 10
    assert contract["publication_parameters"]["decision_threshold"] == 0.5
    assert contract["input_contract"]["cohort_role_required"] is True


def test_contract_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    contract = json.loads(TRACKED_CONTRACT.read_text())
    contract["scope"] = "tampered"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(contract))

    with pytest.raises(
        ValueError,
        match="canonical SHA-256 mismatch",
    ):
        runner.load_runner_contract(tampered)


def test_rehashed_foreign_contract_is_rejected(
    tmp_path: Path,
) -> None:
    contract = json.loads(TRACKED_CONTRACT.read_text())
    contract["publication_parameters"]["processes"] = 9
    contract["contract_sha256"] = runner.canonical_contract_hash(contract)
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps(contract))

    with pytest.raises(
        ValueError,
        match="does not match the frozen",
    ):
        runner.load_runner_contract(foreign)


@pytest.mark.parametrize(
    "relative_name",
    [
        "bin/classify_fasta.py",
        "lib/python3.7/site-packages/plasclass/data/m1000",
        "single_name",
    ],
)
def test_safe_relative_path_accepts_safe_paths(
    relative_name: str,
) -> None:
    assert runner.safe_relative_path(relative_name) == Path(relative_name)


@pytest.mark.parametrize(
    "relative_name",
    [
        "/absolute/file",
        "../escape",
        "safe/../../escape",
        "",
    ],
)
def test_safe_relative_path_rejects_unsafe_paths(
    relative_name: str,
) -> None:
    with pytest.raises(ValueError, match="Unsafe contract path"):
        runner.safe_relative_path(relative_name)


def test_build_command_is_frozen(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "environment"
    make_fake_environment(environment)
    input_fasta = tmp_path / "input.fasta"
    raw_scores = tmp_path / "raw_scores.tsv"

    command = runner.build_command(
        environment_prefix=environment,
        input_fasta=input_fasta,
        raw_scores=raw_scores,
    )

    assert command == [
        "/usr/bin/caffeinate",
        "-i",
        "/usr/bin/time",
        "-l",
        str(environment / "bin" / "python"),
        str(environment / "bin" / "classify_fasta.py"),
        "-f",
        str(input_fasta),
        "-o",
        str(raw_scores),
        "-p",
        "10",
    ]


def test_existing_output_directory_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(
        FileExistsError,
        match="Refusing to overwrite",
    ):
        runner.prepare_output_directory(output)


def test_resource_usage_parses_macos_time() -> None:
    parsed = runner.parse_resource_usage(
        "  12.50 real  8.25 user  1.75 sys\n"
        "  123456 maximum resident set size\n"
        "  654321 peak memory footprint\n"
    )

    assert parsed == {
        "time_reported_real_seconds": 12.5,
        "user_seconds": 8.25,
        "system_seconds": 1.75,
        "peak_rss_bytes": 123456,
        "peak_memory_footprint_bytes": 654321,
    }


def test_failure_abstentions_cover_every_input(
    tmp_path: Path,
) -> None:
    output = tmp_path / "standardized.tsv"
    records: list[dict[str, str | int]] = [
        {
            "contig_id": "p1",
            "input_header": "p1 description",
            "length": 100,
            "model_scale": 1000,
        },
        {
            "contig_id": "c1",
            "input_header": "c1 description",
            "length": 6000,
            "model_scale": 10000,
        },
    ]

    runner.write_failure_abstentions(records, output)
    rows = read_tsv(output)

    assert [row["contig_id"] for row in rows] == [
        "p1",
        "c1",
    ]
    assert {row["predicted_label"] for row in rows} == {"unclassified"}
    assert {row["prediction_status"] for row in rows} == {"missing_output"}
    assert all(row["plasmid_score"] == "" for row in rows)


def test_checksum_manifest_excludes_itself(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")

    runner.write_artifact_checksums(tmp_path)
    checksum_file = tmp_path / "artifact_checksums.sha256"
    lines = checksum_file.read_text().splitlines()

    assert len(lines) == 2
    assert lines[0].endswith("  a.txt")
    assert lines[1].endswith("  b.txt")
    assert not any("artifact_checksums.sha256" in line for line in lines)


def test_environment_verification_reports_missing_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "environment"
    make_fake_environment(environment)
    contract = json.loads(TRACKED_CONTRACT.read_text())

    monkeypatch.setattr(
        runner,
        "runtime_probe",
        lambda unused: dict(contract["environment_contract"]["required_versions"]),
    )

    result = runner.verify_environment(
        environment,
        contract,
    )

    assert result["valid"] is False
    assert result["verified_model_assets"] == 8
    assert any("missing" in problem for problem in result["mismatches"])


@pytest.mark.parametrize(
    ("cohort_role", "confirmatory_accessed"),
    [
        ("development", False),
        ("confirmatory", True),
    ],
)
def test_successful_synthetic_run_records_truthful_cohort_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cohort_role: str,
    confirmatory_accessed: bool,
) -> None:
    environment = tmp_path / "environment"
    make_fake_environment(environment)

    input_fasta = tmp_path / "input.fasta"
    write_fasta(input_fasta)
    output = tmp_path / f"output_{cohort_role}"

    monkeypatch.setattr(
        runner,
        "verify_environment",
        lambda unused_prefix, unused_contract: (valid_environment_result()),
    )

    def fake_run_process(
        command: list[str],
    ) -> dict[str, Any]:
        raw_path = Path(command[command.index("-o") + 1])
        raw_path.write_text("p1\t0.90\n" "c1\t0.10\n")
        return {
            "returncode": 0,
            "timed_out": False,
            "termination_signal": "",
            "wallclock_seconds": 1.25,
            "stdout": "synthetic stdout\n",
            "stderr": "synthetic stderr\n",
            "resource_usage": {},
        }

    monkeypatch.setattr(
        runner,
        "run_process",
        fake_run_process,
    )

    return_code = runner.run_benchmark(
        input_fasta=input_fasta,
        output_directory=output,
        environment_prefix=environment,
        scope_contract=TRACKED_CONTRACT,
        cohort_role=cohort_role,
    )

    assert return_code == 0

    standardized = read_tsv(output / "standardized_predictions.tsv")
    assert [row["predicted_label"] for row in standardized] == ["plasmid", "non-plasmid"]

    provenance = json.loads((output / "runner_provenance.json").read_text())
    command_record = json.loads((output / "command.json").read_text())

    assert provenance["success"] is True
    assert provenance["cohort_role"] == cohort_role
    assert provenance["confirmatory_data_accessed"] is confirmatory_accessed
    assert provenance["confirmatory_tuning"] is False
    assert command_record["cohort_role"] == cohort_role
    assert command_record["confirmatory_tuning"] is False
    assert (output / "artifact_checksums.sha256").is_file()


def test_failed_tool_run_writes_complete_abstentions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "environment"
    make_fake_environment(environment)
    input_fasta = tmp_path / "input.fasta"
    write_fasta(input_fasta)
    output = tmp_path / "failed_output"

    monkeypatch.setattr(
        runner,
        "verify_environment",
        lambda unused_prefix, unused_contract: (valid_environment_result()),
    )
    monkeypatch.setattr(
        runner,
        "run_process",
        lambda unused_command: {
            "returncode": 2,
            "timed_out": False,
            "termination_signal": "",
            "wallclock_seconds": 0.5,
            "stdout": "",
            "stderr": "synthetic failure",
            "resource_usage": {},
        },
    )

    return_code = runner.run_benchmark(
        input_fasta=input_fasta,
        output_directory=output,
        environment_prefix=environment,
        scope_contract=TRACKED_CONTRACT,
        cohort_role="confirmatory",
    )

    assert return_code == 1

    rows = read_tsv(output / "standardized_predictions.tsv")
    assert len(rows) == 2
    assert {row["predicted_label"] for row in rows} == {"unclassified"}

    provenance = json.loads((output / "runner_provenance.json").read_text())
    metadata = json.loads((output / "adapter_metadata.json").read_text())

    assert provenance["success"] is False
    assert provenance["cohort_role"] == "confirmatory"
    assert provenance["confirmatory_data_accessed"] is True
    assert provenance["confirmatory_tuning"] is False
    assert metadata["failure_abstentions_written"] is True
    assert any("return code" in message for message in provenance["errors"])


def test_invalid_cohort_role_is_rejected_before_output(
    tmp_path: Path,
) -> None:
    input_fasta = tmp_path / "input.fasta"
    write_fasta(input_fasta)
    output = tmp_path / "output"

    with pytest.raises(
        ValueError,
        match="cohort_role",
    ):
        runner.run_benchmark(
            input_fasta=input_fasta,
            output_directory=output,
            environment_prefix=tmp_path / "environment",
            scope_contract=TRACKED_CONTRACT,
            cohort_role="hidden-test",
        )

    assert not output.exists()


def test_timeout_sends_sigterm_to_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 1234
        returncode = -signal.SIGTERM

        def __init__(self) -> None:
            self.calls = 0

        def communicate(
            self,
            timeout: int | None = None,
        ) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(
                    cmd=["plasclass"],
                    timeout=timeout,
                )
            return "stdout", "stderr"

    fake_process = FakeProcess()
    sent_signals: list[tuple[int, signal.Signals]] = []

    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *unused_args, **unused_kwargs: fake_process,
    )
    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda process_id, sent_signal: sent_signals.append((process_id, sent_signal)),
    )

    result = runner.run_process(["plasclass"])

    assert result["timed_out"] is True
    assert result["termination_signal"] == "SIGTERM"
    assert sent_signals == [
        (1234, signal.SIGTERM),
    ]


def test_unresponsive_timeout_escalates_to_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 4321
        returncode = -signal.SIGKILL

        def __init__(self) -> None:
            self.calls = 0

        def communicate(
            self,
            timeout: int | None = None,
        ) -> tuple[str, str]:
            self.calls += 1
            if self.calls <= 2:
                raise subprocess.TimeoutExpired(
                    cmd=["plasclass"],
                    timeout=timeout,
                )
            return "stdout", "stderr"

    fake_process = FakeProcess()
    sent_signals: list[tuple[int, signal.Signals]] = []

    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *unused_args, **unused_kwargs: fake_process,
    )
    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda process_id, sent_signal: sent_signals.append((process_id, sent_signal)),
    )

    result = runner.run_process(["plasclass"])

    assert result["timed_out"] is True
    assert result["termination_signal"] == "SIGKILL"
    assert sent_signals == [
        (4321, signal.SIGTERM),
        (4321, signal.SIGKILL),
    ]


def test_direct_help_execution_is_manuscript_only() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_SOURCE),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "manuscript-only" in completed.stdout
    assert "--cohort-role" in completed.stdout
    assert "--processes" not in completed.stdout
    assert "--timeout-seconds" not in completed.stdout
    assert "--threshold" not in completed.stdout


def test_postflight_failure_invalidates_successful_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "environment"
    make_fake_environment(environment)
    input_fasta = tmp_path / "input.fasta"
    write_fasta(input_fasta)
    output = tmp_path / "postflight_failure"

    verification_results = iter(
        [
            valid_environment_result(),
            {
                "valid": False,
                "mismatches": [
                    "m1000: SHA-256 mismatch",
                ],
            },
        ]
    )

    monkeypatch.setattr(
        runner,
        "verify_environment",
        lambda unused_prefix, unused_contract: next(verification_results),
    )

    def fake_run_process(
        command: list[str],
    ) -> dict[str, Any]:
        raw_path = Path(command[command.index("-o") + 1])
        raw_path.write_text("p1\t0.90\n" "c1\t0.10\n")
        return {
            "returncode": 0,
            "timed_out": False,
            "termination_signal": "",
            "wallclock_seconds": 1.0,
            "stdout": "",
            "stderr": "",
            "resource_usage": {},
        }

    monkeypatch.setattr(
        runner,
        "run_process",
        fake_run_process,
    )

    return_code = runner.run_benchmark(
        input_fasta=input_fasta,
        output_directory=output,
        environment_prefix=environment,
        scope_contract=TRACKED_CONTRACT,
        cohort_role="development",
    )

    assert return_code == 1

    standardized = read_tsv(output / "standardized_predictions.tsv")
    assert len(standardized) == 2
    assert {row["predicted_label"] for row in standardized} == {"unclassified"}
    assert {row["prediction_status"] for row in standardized} == {"missing_output"}

    metadata = json.loads((output / "adapter_metadata.json").read_text())
    provenance = json.loads((output / "runner_provenance.json").read_text())

    assert metadata["predictions_invalidated"] is True
    assert metadata["failure_abstentions_written"] is True
    assert metadata["discarded_adapter_metadata"]["complete_output"] is True

    assert provenance["success"] is False
    assert provenance["cohort_role"] == "development"
    assert provenance["confirmatory_data_accessed"] is False
    assert any(
        "Post-run environment verification failed" in message for message in provenance["errors"]
    )
