"""Tests for the frozen manuscript-only PlasFlow2 confirmatory runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.benchmark.runners import plasflow2 as runner


def _contract_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmark/contracts/plasflow2_confirmatory_runner_contract_v1.json"
    )


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(f">{header}\n{sequence}\n" for header, sequence in records),
        encoding="utf-8",
    )


def _input_records() -> list[runner.InputRecord]:
    return [
        runner.InputRecord(
            contig_id="blind_0001",
            header="blind_0001 description",
            length=1_000,
            sequence_sha256=hashlib.sha256(b"A" * 1_000).hexdigest(),
        ),
        runner.InputRecord(
            contig_id="blind_0002",
            header="blind_0002",
            length=1_200,
            sequence_sha256=hashlib.sha256(b"C" * 1_200).hexdigest(),
        ),
    ]


def _write_raw_predictions(
    path: Path,
    rows: list[list[str]] | None = None,
    columns: list[str] | None = None,
) -> None:
    fieldnames = columns or [
        "contig_id",
        "label",
        "confidence",
        "plasmid_score",
        "chromosome_score",
        "phage_score",
    ]
    values = rows or [
        ["blind_0001", "plasmid", "0.80", "0.80", "0.10", "0.10"],
        ["blind_0002", "unclassified", "0.50", "0.25", "0.50", "0.25"],
    ]
    with path.open("wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(fieldnames)
        writer.writerows(values)


def test_contract_identity_and_frozen_parameters() -> None:
    contract = runner.load_and_validate_contract()
    assert contract["contract_sha256"] == runner.RUNNER_CONTRACT_SHA256
    assert contract["input_contract"]["sha256"] == runner.FINAL_COHORT_SHA256
    assert contract["publication_parameters"]["profile"] == "balanced"
    assert contract["publication_parameters"]["threads"] == 4
    assert contract["publication_parameters"]["timeout_seconds"] == 86_400
    assert contract["publication_parameters"]["marker_fusion"] is False
    assert contract["authorization"]["confirmatory_predictions_authorized"] is False


def test_contract_rejects_file_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    changed = tmp_path / "contract.json"
    changed.write_text(_contract_path().read_text() + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "CONTRACT_PATH", changed)
    with pytest.raises(runner.RunnerContractError, match="file identity changed"):
        runner.load_and_validate_contract()


def test_contract_rejects_canonical_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = json.loads(_contract_path().read_text())
    value["publication_parameters"]["threads"] = 5
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(runner, "CONTRACT_PATH", changed)
    monkeypatch.setattr(runner, "RUNNER_CONTRACT_FILE_SHA256", runner.sha256_file(changed))
    with pytest.raises(runner.RunnerContractError, match="canonical identity"):
        runner.load_and_validate_contract()


def test_contract_cannot_self_authorize_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = json.loads(_contract_path().read_text())
    value["authorization"]["confirmatory_predictions_authorized"] = True
    value["contract_sha256"] = runner.canonical_hash(value)
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(runner, "CONTRACT_PATH", changed)
    monkeypatch.setattr(runner, "RUNNER_CONTRACT_FILE_SHA256", runner.sha256_file(changed))
    monkeypatch.setattr(runner, "RUNNER_CONTRACT_SHA256", value["contract_sha256"])
    with pytest.raises(runner.RunnerContractError, match="must not self-authorize"):
        runner.load_and_validate_contract()


def test_parse_fasta_preserves_opaque_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "input.fasta"
    records = [("blind_0002 description", "A" * 1_000), ("blind_0001", "C" * 1_200)]
    _write_fasta(fasta, records)
    monkeypatch.setattr(runner, "EXPECTED_RECORDS", 2)
    monkeypatch.setattr(runner, "EXPECTED_BASES", 2_200)
    parsed = runner.parse_fasta(fasta)
    assert [item.contig_id for item in parsed] == ["blind_0002", "blind_0001"]
    assert [item.length for item in parsed] == [1_000, 1_200]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([("same first", "A" * 1_000), ("same second", "C" * 1_000)], "Duplicate"),
        ([("plasmid_001", "A" * 1_000)], "class term"),
        ([("blind", "A" * 999)], "outside the frozen domain"),
        ([("blind", "A" * 500_001)], "outside the frozen domain"),
        ([("blind", "A" * 999 + "!")], "invalid nucleotide"),
    ],
)
def test_parse_fasta_rejects_invalid_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: list[tuple[str, str]],
    message: str,
) -> None:
    fasta = tmp_path / "input.fasta"
    _write_fasta(fasta, records)
    monkeypatch.setattr(runner, "EXPECTED_RECORDS", len(records))
    monkeypatch.setattr(runner, "EXPECTED_BASES", sum(len(sequence) for _, sequence in records))
    with pytest.raises(runner.RunnerContractError, match=message):
        runner.parse_fasta(fasta)


def test_official_command_preserves_frozen_behavior(tmp_path: Path) -> None:
    contract = runner.load_and_validate_contract()
    command = runner.build_official_command(
        contract, tmp_path / "input.fasta", tmp_path / "predictions.tsv"
    )
    assert command[:4] == ["/usr/bin/caffeinate", "-i", "/usr/bin/time", "-l"]
    assert command.count("classify") == 1
    assert command[command.index("--profile") + 1] == "balanced"
    assert command[command.index("--threads") + 1] == "4"
    assert "--no-marker-model" in command
    assert not set(contract["command_contract"]["forbidden_arguments"]) & set(command)


def test_official_command_rejects_forbidden_override(tmp_path: Path) -> None:
    contract = runner.load_and_validate_contract()
    contract["command_contract"]["arguments"].extend(["--threshold", "0.1"])
    with pytest.raises(runner.RunnerContractError, match="forbidden token"):
        runner.build_official_command(
            contract, tmp_path / "input.fasta", tmp_path / "predictions.tsv"
        )


def test_frozen_environment_is_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/untrusted")
    contract = runner.load_and_validate_contract()
    environment, overrides = runner.frozen_environment(contract)
    assert overrides["PLASFLOW_USE_MPS"] == "0"
    assert overrides["CUDA_VISIBLE_DEVICES"] == ""
    assert overrides["PYTHONHASHSEED"] == "0"
    assert environment["PATH"] == f"{runner.EXPECTED_ENVIRONMENT}/bin:/usr/bin:/bin"


def test_frozen_environment_rejects_changed_threads() -> None:
    contract = runner.load_and_validate_contract()
    contract["command_contract"]["environment"]["OMP_NUM_THREADS"] = "2"
    with pytest.raises(runner.RunnerContractError, match="environment contract changed"):
        runner.frozen_environment(contract)


def test_valid_native_output_is_standardized_without_reclassification(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.tsv"
    standardized = tmp_path / "standardized.tsv"
    _write_raw_predictions(raw)
    result = runner.validate_and_standardize_output(raw, standardized, _input_records())
    assert result["rows"] == 2
    assert result["label_counts"]["plasmid"] == 1
    assert result["label_counts"]["unclassified"] == 1
    with standardized.open("rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        assert reader.fieldnames is not None
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
        rows = list(reader)
    assert [row["predicted_label"] for row in rows] == ["plasmid", "unclassified"]
    assert rows[1]["prediction_status"] == "unclassified_abstention"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [["blind_0002", "unclassified", "0.50", "0.25", "0.50", "0.25"]],
            "order differs",
        ),
        (
            [
                ["blind_0001", "plasmid", "0.80", "0.80", "0.10", "0.10"],
                ["blind_0001", "plasmid", "0.80", "0.80", "0.10", "0.10"],
            ],
            "Duplicate raw",
        ),
        (
            [
                ["blind_0001", "virus", "0.80", "0.80", "0.10", "0.10"],
                ["blind_0002", "unclassified", "0.50", "0.25", "0.50", "0.25"],
            ],
            "Invalid native label",
        ),
        (
            [
                ["blind_0001", "plasmid", "nan", "0.80", "0.10", "0.10"],
                ["blind_0002", "unclassified", "0.50", "0.25", "0.50", "0.25"],
            ],
            "outside \\[0,1\\]",
        ),
        (
            [
                ["blind_0001", "plasmid", "0.80", "0.80", "0.10", "0.05"],
                ["blind_0002", "unclassified", "0.50", "0.25", "0.50", "0.25"],
            ],
            "do not sum to one",
        ),
        (
            [
                ["blind_0001", "plasmid", "0.70", "0.80", "0.10", "0.10"],
                ["blind_0002", "unclassified", "0.50", "0.25", "0.50", "0.25"],
            ],
            "confidence does not equal",
        ),
        (
            [
                ["blind_0001", "chromosome", "0.80", "0.80", "0.10", "0.10"],
                ["blind_0002", "unclassified", "0.50", "0.25", "0.50", "0.25"],
            ],
            "inconsistent with class scores",
        ),
    ],
)
def test_invalid_native_outputs_are_rejected(
    tmp_path: Path, rows: list[list[str]], message: str
) -> None:
    raw = tmp_path / "raw.tsv"
    _write_raw_predictions(raw, rows)
    with pytest.raises(runner.RunnerContractError, match=message):
        runner.validate_and_standardize_output(raw, tmp_path / "standardized.tsv", _input_records())


def test_raw_output_requires_exact_columns(tmp_path: Path) -> None:
    raw = tmp_path / "raw.tsv"
    columns = ["contig_id", "label"]
    _write_raw_predictions(raw, [["blind_0001", "plasmid"]], columns)
    with pytest.raises(runner.RunnerContractError, match="columns changed"):
        runner.validate_and_standardize_output(raw, tmp_path / "standardized.tsv", _input_records())


def test_artifact_checksums_cover_retained_files(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two\n", encoding="utf-8")
    checksum_path = runner.write_artifact_checksums(tmp_path)
    lines = checksum_path.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("  one.txt")
    assert lines[1].endswith("  two.txt")


def _configure_mocked_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    return_code: int = 0,
    timed_out: bool = False,
    changed_after: bool = False,
) -> tuple[argparse.Namespace, Path]:
    input_fasta = tmp_path / "input.fasta"
    output_dir = tmp_path / "output"
    _write_fasta(input_fasta, [("blind_0001", "A" * 1_000)])
    input_hash = runner.sha256_file(input_fasta)
    monkeypatch.setattr(runner, "FINAL_COHORT_SHA256", input_hash)
    monkeypatch.setattr(runner, "EXPECTED_RECORDS", 1)
    monkeypatch.setattr(runner, "EXPECTED_BASES", 1_000)
    monkeypatch.setattr(runner, "load_and_validate_contract", lambda: {})
    identities: list[dict[str, Any]] = [{"identity": "before"}]
    identities.append({"identity": "after" if changed_after else "before"})
    monkeypatch.setattr(
        runner,
        "verify_runtime_identities",
        lambda contract, environment: identities.pop(0),
    )
    monkeypatch.setattr(runner, "build_official_command", lambda *args: ["fake-tool"])
    monkeypatch.setattr(runner, "frozen_environment", lambda contract: ({}, {}))

    def fake_process(
        command: list[str],
        environment: dict[str, str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> runner.ProcessResult:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if return_code == 0 and not timed_out:
            _write_raw_predictions(
                output_dir / "all_predictions.tsv",
                [["blind_0001", "plasmid", "0.80", "0.80", "0.10", "0.10"]],
            )
        return runner.ProcessResult(return_code, 0.1, timed_out, None, None)

    monkeypatch.setattr(runner, "run_process", fake_process)
    args = argparse.Namespace(
        input_fasta=input_fasta,
        output_dir=output_dir,
        environment_prefix=runner.EXPECTED_ENVIRONMENT,
        cohort_role="confirmatory",
    )
    return args, output_dir


def test_successful_mocked_execution_records_complete_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, output_dir = _configure_mocked_execution(tmp_path, monkeypatch)
    assert runner.execute(args) == 0
    provenance = json.loads((output_dir / "runner_provenance.json").read_text())
    assert provenance["status"] == "PASS"
    assert provenance["cohort_role"] == "confirmatory"
    assert provenance["marker_fusion"] is False
    assert provenance["performance_metrics_calculated"] is False
    assert provenance["ground_truth_performance_labels_accessed"] is False
    assert (output_dir / "standardized_predictions.tsv").is_file()
    assert (output_dir / "artifact_checksums.sha256").is_file()


@pytest.mark.parametrize(
    ("return_code", "timed_out", "message"),
    [(2, False, "returned status 2"), (124, True, "exceeded the frozen timeout")],
)
def test_process_failures_retain_partial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
    timed_out: bool,
    message: str,
) -> None:
    args, output_dir = _configure_mocked_execution(
        tmp_path, monkeypatch, return_code=return_code, timed_out=timed_out
    )
    with pytest.raises(runner.RunnerContractError, match=message):
        runner.execute(args)
    assert (output_dir / "command.json").is_file()
    assert (output_dir / "resource_usage.json").is_file()
    assert (output_dir / "stdout.log").is_file()
    assert (output_dir / "stderr.log").is_file()


def test_post_run_identity_change_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _configure_mocked_execution(tmp_path, monkeypatch, changed_after=True)
    with pytest.raises(runner.RunnerContractError, match="identities changed"):
        runner.execute(args)


def test_existing_output_directory_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, output_dir = _configure_mocked_execution(tmp_path, monkeypatch)
    output_dir.mkdir()
    with pytest.raises(runner.RunnerContractError, match="already exists"):
        runner.execute(args)


def test_changed_input_identity_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _configure_mocked_execution(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "FINAL_COHORT_SHA256", "0" * 64)
    with pytest.raises(runner.RunnerContractError, match="frozen confirmatory cohort"):
        runner.execute(args)


def test_runner_help_works_from_external_directory(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    runner_path = project_root / "scripts/benchmark/runners/plasflow2.py"
    completed = subprocess.run(
        [sys.executable, str(runner_path), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    normalized = " ".join(completed.stdout.split())
    assert completed.returncode == 0
    assert "frozen PlasFlow2 2.1.2 target method" in normalized
    assert "not part of the production PlasFlow2 prediction workflow" in normalized
    assert completed.stderr == ""


def test_runner_is_isolated_from_production_imports_and_label_map() -> None:
    project_root = Path(__file__).resolve().parents[2]
    runner_path = project_root / "scripts/benchmark/runners/plasflow2.py"
    source = runner_path.read_text(encoding="utf-8")
    assert "from plasflow2" not in source
    assert "import plasflow2" not in source
    assert "sealed_final_confirmatory_label_map" not in source
    production = project_root / "src/plasflow2"
    violations = [
        str(path.relative_to(project_root))
        for path in production.rglob("*.py")
        if "benchmark.runners.plasflow2" in path.read_text(encoding="utf-8")
    ]
    assert violations == []
