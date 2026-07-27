#!/usr/bin/env python3
"""Run the frozen manuscript-only PlasClass 0.1 comparator.

This runner is exclusively for comparative manuscript benchmarking.
It is not part of the PlasFlow2 prediction workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.benchmark.adapters.plasclass import (  # noqa: E402
    CONTRACT_SHA256 as ADAPTER_CONTRACT_SHA256,
)
from scripts.benchmark.adapters.plasclass import (  # noqa: E402
    OUTPUT_FIELDS,
    adapt_plasclass,
    load_fasta_records,
)

TOOL_NAME = "PlasClass"
TOOL_VERSION = "0.1"
RUNNER_SCHEMA = "nar-plasclass-runner-v1"
SCOPE = "manuscript-only comparative benchmarking"

EXPECTED_RUNNER_CONTRACT_SHA256 = "434dd2d3414797d95c00c54f1a0dfa8de3bbaa8750ad15a02b694e64cc3b08e9"
EXPECTED_ADAPTER_SHA256 = "916bff9fb47300e0db3361feadb1dfe5" "e1e486d8caadb0b3448a0aad1186c300"
EXPECTED_ADAPTER_CONTRACT_SHA256 = (
    "70e984280cffacea651945f1dddedee1dc" "948b4f2f4787191caaafd074a79adf"
)

PUBLICATION_PROCESSES = 10
PUBLICATION_TIMEOUT_SECONDS = 14_400
TIMEOUT_GRACE_SECONDS = 10


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def sha256_file(file_name: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with file_name.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def canonical_contract_hash(
    contract: dict[str, Any],
) -> str:
    """Calculate the canonical hash excluding contract_sha256."""

    payload = dict(contract)
    payload.pop("contract_sha256", None)

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return hashlib.sha256(canonical).hexdigest()


def adapter_source_path() -> Path:
    """Return the frozen adapter source path."""

    return Path(__file__).resolve().parents[1] / "adapters" / "plasclass.py"


def load_runner_contract(
    contract_file: Path,
) -> dict[str, Any]:
    """Load and cryptographically verify the runner contract."""

    if not contract_file.is_file():
        raise FileNotFoundError(f"Runner contract does not exist: {contract_file}")

    contract = json.loads(contract_file.read_text())
    recorded_hash = contract.get("contract_sha256")

    if not isinstance(recorded_hash, str):
        raise ValueError("Runner contract has no contract_sha256")

    calculated_hash = canonical_contract_hash(contract)

    if calculated_hash != recorded_hash:
        raise ValueError("Runner contract canonical SHA-256 mismatch")

    if recorded_hash != EXPECTED_RUNNER_CONTRACT_SHA256:
        raise ValueError("Runner contract does not match the frozen " "publication contract")

    if contract.get("schema_version") != "nar-plasclass-runner-contract-v1.1":
        raise ValueError("Unsupported PlasClass runner contract schema")

    if contract.get("scope") != SCOPE:
        raise ValueError("Runner contract scope is not manuscript-only")

    if contract.get("production_workflow_component") is not False:
        raise ValueError("Runner contract incorrectly declares a production role")

    parameters = contract.get("publication_parameters", {})

    if parameters.get("processes") != PUBLICATION_PROCESSES:
        raise ValueError("Runner contract publication process count changed")

    if parameters.get("process_override_allowed") is not False:
        raise ValueError("Runner contract permits a process-count override")

    if parameters.get("timeout_seconds") != PUBLICATION_TIMEOUT_SECONDS:
        raise ValueError("Runner contract publication timeout changed")

    if parameters.get("timeout_override_allowed") is not False:
        raise ValueError("Runner contract permits a timeout override")

    if parameters.get("decision_threshold") != 0.5:
        raise ValueError("Runner contract decision threshold changed")

    if parameters.get("threshold_override_allowed") is not False:
        raise ValueError("Runner contract permits threshold tuning")

    adapter_record = contract.get("adapter_contract", {})

    if adapter_record.get("contract_sha256") != EXPECTED_ADAPTER_CONTRACT_SHA256:
        raise ValueError("Adapter contract identity changed")

    current_adapter = adapter_source_path()

    if not current_adapter.is_file():
        raise FileNotFoundError(f"PlasClass adapter is missing: {current_adapter}")

    current_adapter_sha256 = sha256_file(current_adapter)

    if current_adapter_sha256 != EXPECTED_ADAPTER_SHA256:
        raise ValueError("PlasClass adapter source SHA-256 does not match " "the frozen runner")

    if adapter_record.get("adapter_sha256") != current_adapter_sha256:
        raise ValueError("Runner contract adapter SHA-256 mismatch")

    if ADAPTER_CONTRACT_SHA256 != EXPECTED_ADAPTER_CONTRACT_SHA256:
        raise ValueError("Imported PlasClass adapter contract identity changed")

    return contract


def runtime_probe(
    environment_prefix: Path,
) -> dict[str, str]:
    """Read exact dependency versions without loading models."""

    environment_python = environment_prefix / "bin" / "python"

    if not environment_python.exists():
        raise FileNotFoundError(f"Environment Python is missing: {environment_python}")

    probe_source = """
import json
import platform
import joblib
import numpy
import pkg_resources
import scipy
import sklearn

print(json.dumps({
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__,
    "joblib": joblib.__version__,
    "plasclass": pkg_resources.get_distribution(
        "plasclass"
    ).version,
}, sort_keys=True))
"""

    completed = subprocess.run(
        [
            str(environment_python),
            "-c",
            probe_source,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if completed.returncode != 0:
        raise RuntimeError("Unable to probe PlasClass environment: " + completed.stderr.strip())

    try:
        observed = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("PlasClass runtime probe returned invalid JSON") from error

    return {str(name): str(value) for name, value in observed.items()}


def safe_relative_path(
    relative_name: str,
) -> Path:
    """Validate a contract-controlled relative path."""

    relative_path = Path(relative_name)

    if not relative_name or relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe contract path: {relative_name!r}")

    return relative_path


def verify_environment(
    environment_prefix: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Verify versions, executable, adapter, and all model assets."""

    if not environment_prefix.is_dir():
        raise FileNotFoundError(f"PlasClass environment is missing: " f"{environment_prefix}")

    observed_versions = runtime_probe(environment_prefix)
    required_versions = contract["environment_contract"]["required_versions"]
    version_mismatches: list[str] = []

    for name, expected in sorted(required_versions.items()):
        observed = observed_versions.get(name)

        if observed != expected:
            version_mismatches.append(f"{name}: observed {observed!r}, " f"expected {expected!r}")

    official_relative = safe_relative_path(contract["tool"]["official_script_relative_path"])
    official_script = environment_prefix / official_relative

    if not official_script.is_file():
        raise FileNotFoundError(f"Official PlasClass script is missing: " f"{official_script}")

    official_sha256 = sha256_file(official_script)
    expected_official_sha256 = contract["tool"]["official_script_sha256"]

    model_records: list[dict[str, Any]] = []
    model_mismatches: list[str] = []

    for model_name, expected in sorted(contract["model_assets"].items()):
        relative_path = safe_relative_path(expected["relative_path"])
        model_file = environment_prefix / relative_path

        if not model_file.is_file():
            model_mismatches.append(f"{model_name}: missing")
            model_records.append(
                {
                    "name": model_name,
                    "present": False,
                    "relative_path": str(relative_path),
                }
            )
            continue

        observed_size = model_file.stat().st_size
        observed_sha256 = sha256_file(model_file)

        if observed_size != expected["bytes"]:
            model_mismatches.append(f"{model_name}: size mismatch")

        if observed_sha256 != expected["sha256"]:
            model_mismatches.append(f"{model_name}: SHA-256 mismatch")

        model_records.append(
            {
                "name": model_name,
                "present": True,
                "relative_path": str(relative_path),
                "expected_bytes": expected["bytes"],
                "observed_bytes": observed_size,
                "expected_sha256": expected["sha256"],
                "observed_sha256": observed_sha256,
            }
        )

    adapter_sha256 = sha256_file(adapter_source_path())
    mismatches = list(version_mismatches)

    if official_sha256 != expected_official_sha256:
        mismatches.append("official classify_fasta.py SHA-256 mismatch")

    if adapter_sha256 != EXPECTED_ADAPTER_SHA256:
        mismatches.append("PlasClass adapter SHA-256 mismatch")

    mismatches.extend(model_mismatches)

    return {
        "valid": not mismatches,
        "environment_prefix": str(environment_prefix),
        "versions": observed_versions,
        "required_versions": required_versions,
        "official_script": str(official_script),
        "official_script_sha256": official_sha256,
        "expected_official_script_sha256": (expected_official_sha256),
        "adapter_sha256": adapter_sha256,
        "expected_adapter_sha256": (EXPECTED_ADAPTER_SHA256),
        "verified_model_assets": len(model_records),
        "model_assets": model_records,
        "mismatches": mismatches,
    }


def fasta_statistics(
    input_fasta: Path,
) -> tuple[list[dict[str, str | int]], dict[str, Any]]:
    """Validate FASTA and return records plus provenance statistics."""

    records = load_fasta_records(input_fasta)
    bases = sum(int(record["length"]) for record in records)

    return records, {
        "path": str(input_fasta),
        "sha256": sha256_file(input_fasta),
        "sequences": len(records),
        "bases": bases,
    }


def prepare_output_directory(
    output_directory: Path,
) -> None:
    """Create a new output directory without overwriting evidence."""

    if output_directory.exists():
        raise FileExistsError(
            "Refusing to overwrite existing manuscript benchmark " f"output: {output_directory}"
        )

    output_directory.mkdir(parents=True)


def build_command(
    environment_prefix: Path,
    input_fasta: Path,
    raw_scores: Path,
) -> list[str]:
    """Build the exact frozen official PlasClass command."""

    environment_python = environment_prefix / "bin" / "python"
    official_script = environment_prefix / "bin" / "classify_fasta.py"

    if not environment_python.exists():
        raise FileNotFoundError(environment_python)

    if not official_script.is_file():
        raise FileNotFoundError(official_script)

    return [
        "/usr/bin/caffeinate",
        "-i",
        "/usr/bin/time",
        "-l",
        str(environment_python),
        str(official_script),
        "-f",
        str(input_fasta),
        "-o",
        str(raw_scores),
        "-p",
        str(PUBLICATION_PROCESSES),
    ]


def parse_resource_usage(
    stderr_text: str,
) -> dict[str, float | int]:
    """Parse macOS or GNU time resource measurements."""

    metrics: dict[str, float | int] = {}

    mac_time = re.search(
        r"^\s*([0-9.]+)\s+real\s+([0-9.]+)\s+user\s+" r"([0-9.]+)\s+sys\s*$",
        stderr_text,
        flags=re.MULTILINE,
    )

    if mac_time:
        metrics["time_reported_real_seconds"] = float(mac_time.group(1))
        metrics["user_seconds"] = float(mac_time.group(2))
        metrics["system_seconds"] = float(mac_time.group(3))

    mac_rss = re.search(
        r"^\s*(\d+)\s+maximum resident set size\s*$",
        stderr_text,
        flags=re.MULTILINE,
    )

    if mac_rss:
        metrics["peak_rss_bytes"] = int(mac_rss.group(1))

    footprint = re.search(
        r"^\s*(\d+)\s+peak memory footprint\s*$",
        stderr_text,
        flags=re.MULTILINE,
    )

    if footprint:
        metrics["peak_memory_footprint_bytes"] = int(footprint.group(1))

    gnu_rss = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        stderr_text,
    )

    if gnu_rss and "peak_rss_bytes" not in metrics:
        metrics["peak_rss_bytes"] = int(gnu_rss.group(1)) * 1024

    for name, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite resource measurement: {name}")

        if value < 0:
            raise ValueError(f"Negative resource measurement: {name}")

    return metrics


def run_process(
    command: list[str],
) -> dict[str, Any]:
    """Execute in a new process group with frozen timeout handling."""

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    timed_out = False
    termination_signal = ""

    try:
        stdout_text, stderr_text = process.communicate(timeout=PUBLICATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        termination_signal = "SIGTERM"
        os.killpg(process.pid, signal.SIGTERM)

        try:
            stdout_text, stderr_text = process.communicate(timeout=TIMEOUT_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            termination_signal = "SIGKILL"
            os.killpg(process.pid, signal.SIGKILL)
            stdout_text, stderr_text = process.communicate()

    elapsed = time.monotonic() - started

    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "termination_signal": termination_signal,
        "wallclock_seconds": elapsed,
        "stdout": stdout_text or "",
        "stderr": stderr_text or "",
        "resource_usage": parse_resource_usage(stderr_text or ""),
    }


def write_json(
    output_file: Path,
    payload: dict[str, Any],
) -> None:
    """Write deterministic readable JSON."""

    output_file.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_failure_abstentions(
    records: list[dict[str, str | int]],
    output_file: Path,
) -> None:
    """Write one explicit abstention for every input sequence."""

    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "contig_id": str(record["contig_id"]),
                    "input_header": str(record["input_header"]),
                    "length": str(record["length"]),
                    "model_scale": str(record["model_scale"]),
                    "raw_tool_contig_id": "",
                    "predicted_label": "unclassified",
                    "prediction_status": "missing_output",
                    "plasmid_score": "",
                    "decision_threshold": "0.5",
                    "source_tool": TOOL_NAME,
                    "source_version": TOOL_VERSION,
                }
            )


def write_artifact_checksums(
    output_directory: Path,
) -> None:
    """Write checksums for regular evidence files."""

    checksum_file = output_directory / "artifact_checksums.sha256"
    lines: list[str] = []

    for artifact in sorted(output_directory.iterdir()):
        if not artifact.is_file() or artifact == checksum_file:
            continue

        lines.append(f"{sha256_file(artifact)}  {artifact.name}")

    checksum_file.write_text("\n".join(lines) + "\n")


def run_benchmark(
    input_fasta: Path,
    output_directory: Path,
    environment_prefix: Path,
    scope_contract: Path,
    cohort_role: str,
) -> int:
    """Run PlasClass and retain complete provenance."""

    if cohort_role not in {"development", "confirmatory"}:
        raise ValueError("cohort_role must be 'development' or 'confirmatory'")

    contract = load_runner_contract(scope_contract)
    records, input_metadata = fasta_statistics(input_fasta)
    preflight = verify_environment(
        environment_prefix,
        contract,
    )

    if not preflight["valid"]:
        raise RuntimeError(
            "PlasClass environment preflight failed: " + "; ".join(preflight["mismatches"])
        )

    prepare_output_directory(output_directory)

    raw_scores = output_directory / "raw_scores.tsv"
    standardized = output_directory / "standardized_predictions.tsv"
    adapter_metadata_file = output_directory / "adapter_metadata.json"
    provenance_file = output_directory / "runner_provenance.json"
    command_file = output_directory / "command.json"
    stdout_file = output_directory / "stdout.log"
    stderr_file = output_directory / "stderr.log"

    command = build_command(
        environment_prefix=environment_prefix,
        input_fasta=input_fasta.resolve(),
        raw_scores=raw_scores.resolve(),
    )

    write_json(
        command_file,
        {
            "schema_version": RUNNER_SCHEMA,
            "scope": SCOPE,
            "command": command,
            "shell": False,
            "processes": PUBLICATION_PROCESSES,
            "timeout_seconds": PUBLICATION_TIMEOUT_SECONDS,
            "cohort_role": cohort_role,
            "confirmatory_tuning": False,
        },
    )

    started_at = utc_now()
    execution = run_process(command)
    finished_at = utc_now()

    stdout_file.write_text(execution.pop("stdout"))
    stderr_file.write_text(execution.pop("stderr"))

    errors: list[str] = []
    adapter_metadata: dict[str, Any] = {}
    adapter_succeeded = False

    if execution["timed_out"]:
        errors.append("PlasClass timed out")

    if execution["returncode"] != 0:
        errors.append("PlasClass return code: " f"{execution['returncode']}")

    if not raw_scores.is_file():
        raw_scores.write_text("")
        errors.append("Official PlasClass raw score file was not produced")

    if not errors:
        try:
            adapter_metadata = adapt_plasclass(
                input_fasta=input_fasta,
                raw_scores=raw_scores,
                output_path=standardized,
                metadata_output=adapter_metadata_file,
            )
            adapter_succeeded = bool(adapter_metadata.get("runner_success_allowed"))

            if not adapter_succeeded:
                errors.append("PlasClass output was incomplete")
        except Exception as error:
            errors.append("Adapter failure: " f"{type(error).__name__}: {error}")

    try:
        postflight = verify_environment(
            environment_prefix,
            contract,
        )
    except Exception as error:
        postflight = {
            "valid": False,
            "mismatches": [f"{type(error).__name__}: {error}"],
        }

    postflight_valid = bool(postflight.get("valid", False))

    if not postflight_valid:
        errors.append(
            "Post-run environment verification failed: "
            + "; ".join(postflight.get("mismatches", []))
        )

    predictions_invalidated = adapter_succeeded and not postflight_valid

    if not adapter_succeeded or not postflight_valid:
        discarded_adapter_metadata = dict(adapter_metadata) if predictions_invalidated else None

        write_failure_abstentions(
            records,
            standardized,
        )

        adapter_metadata = {
            "schema_version": ("nar-comparator-adapter-failure-v1"),
            "contract_sha256": (EXPECTED_ADAPTER_CONTRACT_SHA256),
            "source_tool": TOOL_NAME,
            "source_version": TOOL_VERSION,
            "input_sequences": len(records),
            "standardized_rows": len(records),
            "complete_output": False,
            "runner_success_allowed": False,
            "failure_abstentions_written": True,
            "predictions_invalidated": (predictions_invalidated),
            "errors": list(errors),
            "production_workflow_component": False,
        }

        if discarded_adapter_metadata is not None:
            adapter_metadata["discarded_adapter_metadata"] = discarded_adapter_metadata

        write_json(
            adapter_metadata_file,
            adapter_metadata,
        )
        adapter_succeeded = False

    success = not errors and adapter_succeeded and postflight_valid

    provenance = {
        "schema_version": RUNNER_SCHEMA,
        "scope": SCOPE,
        "production_workflow_component": False,
        "cohort_role": cohort_role,
        "confirmatory_data_accessed": (cohort_role == "confirmatory"),
        "confirmatory_tuning": False,
        "tool": {
            "name": TOOL_NAME,
            "version": TOOL_VERSION,
        },
        "contract": {
            "path": str(scope_contract),
            "sha256": (EXPECTED_RUNNER_CONTRACT_SHA256),
        },
        "runner_source_sha256": sha256_file(Path(__file__)),
        "adapter_source_sha256": sha256_file(adapter_source_path()),
        "input": input_metadata,
        "started_at": started_at,
        "finished_at": finished_at,
        "execution": execution,
        "preflight": preflight,
        "postflight": postflight,
        "adapter": adapter_metadata,
        "success": success,
        "errors": errors,
    }
    write_json(provenance_file, provenance)
    write_artifact_checksums(output_directory)

    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    """Parse the frozen runner command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen manuscript-only PlasClass 0.1 "
            "comparator with full provenance capture. "
            "This command is not part of the PlasFlow2 "
            "prediction workflow."
        )
    )
    parser.add_argument(
        "--input-fasta",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--environment-prefix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--scope-contract",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--cohort-role",
        required=True,
        choices=("development", "confirmatory"),
        help=("Truthfully record whether the input is a " "development or confirmatory cohort."),
    )
    return parser.parse_args()


def main() -> int:
    """Run the manuscript comparator CLI."""

    args = parse_args()

    try:
        return run_benchmark(
            input_fasta=args.input_fasta,
            output_directory=args.output_dir,
            environment_prefix=args.environment_prefix,
            scope_contract=args.scope_contract,
            cohort_role=args.cohort_role,
        )
    except Exception as error:
        print(
            "PlasClass benchmark runner failed: " f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
