#!/usr/bin/env python3
"""Foundation for the frozen manuscript-only PlasFlow v1.1 runner.

This runner is not part of the PlasFlow2 prediction workflow. It executes
the immutable official PlasFlow v1.1 Linux container only for frozen
comparative benchmarking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.adapters.plasflow_v1 import (  # noqa: E402
    adapt_plasflow_v1,
)

RUNNER_CONTRACT_SHA256 = "1a1666397b5a8aca209e34feabb2bb89cb2eaca6238dc2a281164195c6171f43"
ADAPTER_CONTRACT_SHA256 = "7e24ac29aedb6f0b106e421302cb1ee147f076c96c24d299f9b9d55d1b42c3bf"
ADAPTER_SHA256 = "431b92f4d7150220cf42a14832ed5529ad930ad16fe4f095ef6a973906fb4236"
IMAGE_REFERENCE = (
    "quay.io/biocontainers/plasflow@"
    "sha256:e69acee3233010dbf5a5245620252bf5"
    "b9bde930ad5546473ec496992995a7da"
)
IMAGE_DIGEST = "sha256:e69acee3233010dbf5a5245620252bf5" "b9bde930ad5546473ec496992995a7da"
DOCKER_CONTEXT = "colima-plasflow1-nar"
THRESHOLD = 0.7
TIMEOUT_SECONDS = 172_800
ALLOWED_COHORT_ROLES = {
    "development",
    "confirmatory",
}

CONTRACT_RELATIVE_PATH = Path("scripts/benchmark/contracts/" "plasflow_v1_runner_contract_v1.json")
ADAPTER_RELATIVE_PATH = Path("scripts/benchmark/adapters/plasflow_v1.py")


class RunnerContractError(RuntimeError):
    """Raised when the frozen runner contract is violated."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    """Hash one dictionary using the frozen canonical JSON form."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    """Load a JSON document."""

    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise RunnerContractError(f"Required JSON file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise RunnerContractError(f"Invalid JSON document: {path}: {error}") from error


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def validate_runner_contract(path: Path) -> dict[str, Any]:
    """Validate the complete frozen runner contract."""

    contract = load_json(path)

    if not isinstance(contract, dict):
        raise RunnerContractError("Runner contract must contain one JSON object")

    declared = contract.get("contract_sha256")

    if declared != RUNNER_CONTRACT_SHA256:
        raise RunnerContractError("Unexpected runner contract identity: " f"{declared!r}")

    canonical = dict(contract)
    canonical.pop("contract_sha256", None)
    calculated = canonical_json_sha256(canonical)

    if calculated != RUNNER_CONTRACT_SHA256:
        raise RunnerContractError("Runner contract content hash mismatch: " f"{calculated}")

    if contract.get("production_workflow_component") is not False:
        raise RunnerContractError("Runner contract does not preserve manuscript-only scope")

    parameters = contract.get(
        "publication_parameters",
        {},
    )

    expected_parameters = {
        "threshold": THRESHOLD,
        "native_batch_size_argument_supported": False,
        "native_cache_files_produced": False,
        "tool_invocations_per_cohort": 1,
        "cohort_sharding_allowed": False,
        "parallel_sequence_partitions_allowed": False,
        "timeout_seconds": TIMEOUT_SECONDS,
    }

    for field, expected in expected_parameters.items():
        if parameters.get(field) != expected:
            raise RunnerContractError(
                "Runner contract publication parameter mismatch: "
                f"{field}={parameters.get(field)!r}; "
                f"expected {expected!r}"
            )

    cohort = contract.get("cohort_dependence", {})

    if cohort.get("full_frozen_cohort_single_invocation_required") is not True:
        raise RunnerContractError("Runner contract does not require a single cohort invocation")

    if cohort.get("external_sequence_sharding_prohibited") is not True:
        raise RunnerContractError("Runner contract does not prohibit external sharding")

    if cohort.get("native_batching_supported") is not False:
        raise RunnerContractError("Runner contract incorrectly declares native batching support")

    if cohort.get("native_cache_files_produced") is not False:
        raise RunnerContractError("Runner contract incorrectly declares native cache files")

    return contract


def validate_adapter_identity(
    adapter_path: Path,
) -> str:
    """Verify that the exact frozen adapter will be executed."""

    if not adapter_path.is_file():
        raise RunnerContractError(f"Frozen adapter does not exist: {adapter_path}")

    digest = sha256_file(adapter_path)

    if digest != ADAPTER_SHA256:
        raise RunnerContractError(
            "Frozen adapter SHA-256 mismatch: " f"{digest}; expected {ADAPTER_SHA256}"
        )

    source = adapter_path.read_text(errors="replace")

    if ADAPTER_CONTRACT_SHA256 not in source:
        raise RunnerContractError("Frozen adapter does not embed the required contract")

    return digest


def find_docker_binary() -> Path:
    """Locate the Docker client without changing its context."""

    candidates = [
        Path("/opt/homebrew/bin/docker"),
        Path("/usr/local/bin/docker"),
    ]

    discovered = shutil.which("docker")

    if discovered:
        candidates.append(Path(discovered))

    for candidate in candidates:
        if candidate.is_file() and candidate.exists():
            return candidate.resolve()

    raise RunnerContractError("Docker client executable was not found")


def run_capture(
    command: list[str],
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    """Run one short diagnostic command without a shell."""

    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RunnerContractError("Diagnostic command timed out: " + " ".join(command)) from error


def docker_command(
    docker_binary: Path,
    *arguments: str,
) -> list[str]:
    """Build an explicit-context Docker client command."""

    return [
        str(docker_binary),
        "--context",
        DOCKER_CONTEXT,
        *arguments,
    ]


def current_host_docker_context(
    docker_binary: Path,
) -> str:
    """Return the caller's current Docker context."""

    completed = run_capture(
        [
            str(docker_binary),
            "context",
            "show",
        ]
    )

    if completed.returncode != 0:
        raise RunnerContractError(
            "Unable to query the host Docker context: " + completed.stderr.strip()
        )

    return completed.stdout.strip()


def container_inventory(
    docker_binary: Path,
) -> list[str]:
    """Return the sorted container IDs in the frozen runtime."""

    completed = run_capture(
        docker_command(
            docker_binary,
            "ps",
            "-aq",
        )
    )

    if completed.returncode != 0:
        raise RunnerContractError(
            "Unable to query container inventory: " + completed.stderr.strip()
        )

    return sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())


def inspect_frozen_image(
    docker_binary: Path,
) -> dict[str, Any]:
    """Verify the immutable PlasFlow v1 image identity."""

    completed = run_capture(
        docker_command(
            docker_binary,
            "image",
            "inspect",
            IMAGE_REFERENCE,
        )
    )

    if completed.returncode != 0:
        raise RunnerContractError(
            "Frozen PlasFlow image is unavailable: " + completed.stderr.strip()
        )

    try:
        documents = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RunnerContractError("Docker image inspection returned invalid JSON") from error

    if len(documents) != 1:
        raise RunnerContractError("Expected exactly one frozen image inspection result")

    image = documents[0]
    repo_digests = image.get("RepoDigests", [])

    if not any(digest.endswith("@" + IMAGE_DIGEST) for digest in repo_digests):
        raise RunnerContractError("Frozen image manifest digest is absent from RepoDigests")

    if image.get("Os") != "linux":
        raise RunnerContractError(f"Frozen image OS is not Linux: {image.get('Os')!r}")

    if image.get("Architecture") != "amd64":
        raise RunnerContractError(
            "Frozen image architecture is not amd64: " f"{image.get('Architecture')!r}"
        )

    return {
        "id": image.get("Id"),
        "repo_digests": repo_digests,
        "os": image.get("Os"),
        "architecture": image.get("Architecture"),
    }


def validate_cohort_role(cohort_role: str) -> None:
    """Reject undeclared cohort roles."""

    if cohort_role not in ALLOWED_COHORT_ROLES:
        raise RunnerContractError(f"Invalid cohort role: {cohort_role!r}")


def inventory_fasta(path: Path) -> dict[str, Any]:
    """Inventory FASTA records without changing or copying them."""

    if not path.is_file():
        raise RunnerContractError(f"Input FASTA does not exist: {path}")

    identifiers: list[str] = []
    observed: set[str] = set()
    sequence_count = 0
    base_count = 0
    current_identifier: str | None = None
    current_length = 0

    def finalize() -> None:
        nonlocal sequence_count
        nonlocal base_count
        nonlocal current_identifier
        nonlocal current_length

        if current_identifier is None:
            return

        if current_length <= 0:
            raise RunnerContractError(f"FASTA record has no sequence: {current_identifier}")

        sequence_count += 1
        base_count += current_length
        current_identifier = None
        current_length = 0

    with path.open() as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                finalize()
                header = line[1:].strip()

                if not header:
                    raise RunnerContractError(f"Empty FASTA header at line {line_number}")

                identifier = header.split()[0]

                if identifier in observed:
                    raise RunnerContractError(
                        "Canonical FASTA identifier collision: " f"{identifier!r}"
                    )

                observed.add(identifier)
                identifiers.append(identifier)
                current_identifier = identifier
                continue

            if current_identifier is None:
                raise RunnerContractError(
                    "Sequence data appears before the first " f"FASTA header at line {line_number}"
                )

            current_length += len(line)

    finalize()

    if sequence_count == 0:
        raise RunnerContractError(f"No FASTA records found: {path}")

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sequence_count": sequence_count,
        "base_count": base_count,
        "identifiers": identifiers,
    }


def prepare_output_directory(path: Path) -> Path:
    """Require a new or completely empty output directory."""

    resolved = path.resolve()

    if resolved.exists():
        if not resolved.is_dir():
            raise RunnerContractError(f"Output path is not a directory: {resolved}")

        if any(resolved.iterdir()):
            raise RunnerContractError("Output directory must be new or empty: " f"{resolved}")
    else:
        resolved.mkdir(parents=True)

    return resolved


def stage_input(
    input_fasta: Path,
    output_directory: Path,
    expected_sha256: str,
) -> Path:
    """Create and verify the required byte-identical input copy."""

    staged = output_directory / "input.fasta"

    if staged.exists():
        raise RunnerContractError(f"Staged input already exists: {staged}")

    shutil.copyfile(input_fasta, staged)
    staged_sha256 = sha256_file(staged)

    if staged_sha256 != expected_sha256:
        raise RunnerContractError(
            "Staged FASTA SHA-256 differs from the original: "
            f"{staged_sha256} versus {expected_sha256}"
        )

    return staged


FOUNDATION_COMPLETE = True


def required_output_paths(
    output_directory: Path,
) -> list[Path]:
    """Return the frozen successful-run artifact set."""

    return [
        output_directory / "input.fasta",
        output_directory / "raw_predictions.tsv",
        output_directory / "raw_predictions.tsv_plasmids.fasta",
        output_directory / "raw_predictions.tsv_chromosomes.fasta",
        output_directory / "raw_predictions.tsv_unclassified.fasta",
        output_directory / "standardized_predictions.tsv",
        output_directory / "adapter_metadata.json",
        output_directory / "runner_provenance.json",
        output_directory / "command.json",
        output_directory / "stdout.log",
        output_directory / "stderr.log",
        output_directory / "resource_usage.txt",
    ]


def build_tool_command(
    docker_binary: Path,
    output_directory: Path,
    container_name: str,
) -> list[str]:
    """Build the exact frozen PlasFlow invocation."""

    caffeinate = Path("/usr/bin/caffeinate")
    time_binary = Path("/usr/bin/time")

    for launcher in (caffeinate, time_binary):
        if not launcher.is_file():
            raise RunnerContractError(f"Required launcher is missing: {launcher}")

    return [
        str(caffeinate),
        "-i",
        str(time_binary),
        "-l",
        str(docker_binary),
        "--context",
        DOCKER_CONTEXT,
        "run",
        "--name",
        container_name,
        "--rm",
        "--network",
        "none",
        "--platform",
        "linux/amd64",
        "--volume",
        f"{output_directory}:/work",
        "--workdir",
        "/work",
        IMAGE_REFERENCE,
        "PlasFlow.py",
        "--input",
        "/work/input.fasta",
        "--output",
        "/work/raw_predictions.tsv",
        "--threshold",
        str(THRESHOLD),
    ]


def cleanup_container(
    docker_binary: Path,
    container_name: str,
) -> dict[str, Any]:
    """Force-remove the named container when it exists."""

    completed = run_capture(
        docker_command(
            docker_binary,
            "rm",
            "-f",
            container_name,
        )
    )

    already_absent = completed.returncode != 0 and (
        "No such container" in completed.stderr or "No such object" in completed.stderr
    )

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "already_absent": already_absent,
        "successful": (completed.returncode == 0 or already_absent),
    }


def terminate_process_group(
    process: subprocess.Popen[str],
    grace_seconds: int = 30,
) -> dict[str, Any]:
    """Terminate one complete process group."""

    result: dict[str, Any] = {
        "sigterm_sent": False,
        "sigkill_sent": False,
        "process_already_exited": False,
    }

    if process.poll() is not None:
        result["process_already_exited"] = True
        return result

    try:
        os.killpg(process.pid, signal.SIGTERM)
        result["sigterm_sent"] = True
    except ProcessLookupError:
        result["process_already_exited"] = True
        return result

    try:
        process.wait(timeout=grace_seconds)
        return result
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
        result["sigkill_sent"] = True
    except ProcessLookupError:
        result["process_already_exited"] = True

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        result["failed_to_exit"] = True

    return result


def execute_tool_process(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    """Execute the tool with timeout and signal handling."""

    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    process: subprocess.Popen[str] | None = None
    timed_out = False
    received_signal: int | None = None
    termination: dict[str, Any] = {}
    previous_handlers: dict[int, Any] = {}

    def signal_handler(
        signum: int,
        _frame: Any,
    ) -> None:
        nonlocal received_signal
        nonlocal termination

        received_signal = signum

        if process is not None:
            termination = terminate_process_group(process)

    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )

        for signum in (
            signal.SIGINT,
            signal.SIGTERM,
        ):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, signal_handler)

        try:
            process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            termination = terminate_process_group(process)
        finally:
            for restore_signum, handler in previous_handlers.items():
                signal.signal(restore_signum, handler)

    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)

    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "received_signal": received_signal,
        "termination": termination,
        "wallclock_seconds": (time.monotonic() - started),
        "user_seconds_delta": (usage_after.ru_utime - usage_before.ru_utime),
        "system_seconds_delta": (usage_after.ru_stime - usage_before.ru_stime),
        "maximum_resident_set_size_raw": (usage_after.ru_maxrss),
    }


def verify_successful_native_outputs(
    output_directory: Path,
) -> None:
    """Require the complete official native output set."""

    required = [
        output_directory / "raw_predictions.tsv",
        output_directory / "raw_predictions.tsv_plasmids.fasta",
        output_directory / "raw_predictions.tsv_chromosomes.fasta",
        output_directory / "raw_predictions.tsv_unclassified.fasta",
    ]
    missing = [str(path) for path in required if not path.is_file()]

    if missing:
        raise RunnerContractError(
            "Successful execution is missing native outputs: " + ", ".join(missing)
        )

    if (output_directory / "raw_predictions.tsv").stat().st_size <= 0:
        raise RunnerContractError("Native prediction table is empty")


def write_resource_usage(
    path: Path,
    execution: dict[str, Any],
) -> None:
    """Write a stable resource summary."""

    lines = [
        ("wallclock_seconds\t" f"{execution.get('wallclock_seconds', 0.0):.6f}"),
        ("user_seconds_delta\t" f"{execution.get('user_seconds_delta', 0.0):.6f}"),
        ("system_seconds_delta\t" f"{execution.get('system_seconds_delta', 0.0):.6f}"),
        ("maximum_resident_set_size_raw\t" f"{execution.get('maximum_resident_set_size_raw', '')}"),
        f"returncode\t{execution.get('returncode', '')}",
        ("timed_out\t" f"{str(execution.get('timed_out', False)).lower()}"),
    ]
    path.write_text("\n".join(lines) + "\n")


def write_artifact_checksums(
    output_directory: Path,
) -> Path:
    """Write checksums for all retained artifacts."""

    checksum_path = output_directory / "artifact_checksums.sha256"
    paths = sorted(
        path for path in output_directory.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "\n".join(
            (f"{sha256_file(path)}  " f"{path.relative_to(output_directory)}") for path in paths
        )
        + "\n"
    )
    return checksum_path


def verify_standardized_completion(
    adapter_metadata: dict[str, Any],
    input_inventory: dict[str, Any],
) -> None:
    """Require one standardized row per input sequence."""

    if adapter_metadata.get("complete_output") is not True:
        raise RunnerContractError("Adapter reported incomplete output")

    if adapter_metadata.get("runner_success_allowed") is not True:
        raise RunnerContractError("Adapter did not authorize runner success")

    expected = input_inventory["sequence_count"]

    for field in (
        "standardized_rows",
        "raw_prediction_rows",
    ):
        if adapter_metadata.get(field) != expected:
            raise RunnerContractError(
                f"{field} mismatch: " f"{adapter_metadata.get(field)} " f"versus {expected}"
            )


def execute_frozen_run(
    *,
    input_fasta: Path,
    output_directory: Path,
    scope_contract: Path,
    docker_context: str,
    cohort_role: str,
) -> dict[str, Any]:
    """Execute one complete frozen cohort without sharding."""

    if docker_context != DOCKER_CONTEXT:
        raise RunnerContractError(
            "Docker context must equal the frozen value " f"{DOCKER_CONTEXT!r}"
        )

    validate_cohort_role(cohort_role)
    contract = validate_runner_contract(scope_contract)

    adapter_path = PROJECT_ROOT / ADAPTER_RELATIVE_PATH
    adapter_sha256 = validate_adapter_identity(adapter_path)
    docker_binary = find_docker_binary()
    host_context_before = current_host_docker_context(docker_binary)
    containers_before = container_inventory(docker_binary)
    image_before = inspect_frozen_image(docker_binary)

    input_fasta = input_fasta.resolve()
    input_inventory = inventory_fasta(input_fasta)
    original_sha256 = input_inventory["sha256"]

    output_directory = prepare_output_directory(output_directory)

    staged_input = stage_input(
        input_fasta,
        output_directory,
        original_sha256,
    )

    container_name = "plasflow-v1-nar-" + uuid.uuid4().hex[:12]
    command = build_tool_command(
        docker_binary,
        output_directory,
        container_name,
    )

    write_json(
        output_directory / "command.json",
        {
            "command": command,
            "shell": False,
            "container_name": container_name,
            "docker_context": DOCKER_CONTEXT,
            "cohort_role": cohort_role,
            "threshold": THRESHOLD,
            "native_batch_size_argument_supported": False,
            "native_cache_files_produced": False,
            "timeout_seconds": TIMEOUT_SECONDS,
            "single_cohort_invocation": True,
            "external_sharding": False,
        },
    )

    execution: dict[str, Any] = {}
    cleanup: dict[str, Any] = {}
    adapter_metadata: dict[str, Any] = {}
    errors: list[str] = []

    try:
        try:
            execution = execute_tool_process(
                command,
                output_directory / "stdout.log",
                output_directory / "stderr.log",
            )
        except Exception as error:
            errors.append("Tool launch failure: " f"{type(error).__name__}: {error}")
            execution = {
                "returncode": None,
                "timed_out": False,
                "launch_failed": True,
            }
    finally:
        try:
            cleanup = cleanup_container(
                docker_binary,
                container_name,
            )
        except Exception as error:
            cleanup = {
                "successful": False,
                "error": (f"{type(error).__name__}: {error}"),
            }

    write_resource_usage(
        output_directory / "resource_usage.txt",
        execution,
    )

    if execution.get("timed_out"):
        errors.append("PlasFlow v1 execution timed out")

    if execution.get("received_signal") is not None:
        errors.append("Runner received signal " f"{execution['received_signal']}")

    if execution.get("returncode") != 0:
        errors.append("PlasFlow v1 returned nonzero status " f"{execution.get('returncode')}")

    if cleanup.get("successful") is not True:
        errors.append("Container cleanup failed")

    if not errors:
        try:
            verify_successful_native_outputs(output_directory)
            adapter_metadata = adapt_plasflow_v1(
                input_fasta=staged_input,
                raw_predictions=(output_directory / "raw_predictions.tsv"),
                output_path=(output_directory / "standardized_predictions.tsv"),
                metadata_output=(output_directory / "adapter_metadata.json"),
            )
            verify_standardized_completion(
                adapter_metadata,
                input_inventory,
            )
        except Exception as error:
            errors.append("Output or adapter failure: " f"{type(error).__name__}: {error}")

    try:
        image_after = inspect_frozen_image(docker_binary)
    except Exception as error:
        image_after = {}
        errors.append("Post-run image verification failure: " f"{type(error).__name__}: {error}")

    try:
        containers_after = container_inventory(docker_binary)
    except Exception as error:
        containers_after = []
        errors.append("Post-run container inventory failure: " f"{type(error).__name__}: {error}")

    try:
        host_context_after = current_host_docker_context(docker_binary)
    except Exception as error:
        host_context_after = ""
        errors.append("Post-run context verification failure: " f"{type(error).__name__}: {error}")

    if image_after != image_before:
        errors.append("Frozen image identity changed during execution")

    if containers_after != containers_before:
        errors.append("Container inventory was not restored")

    if host_context_after != host_context_before:
        errors.append("Host Docker context changed during execution")

    staged_sha256 = sha256_file(staged_input)

    if staged_sha256 != original_sha256:
        errors.append("Staged input identity changed during execution")

    provenance = {
        "schema_version": ("nar-plasflow-v1-runner-provenance-v1"),
        "status": "PASS" if not errors else "FAIL",
        "overall_status": 0 if not errors else 1,
        "scope": ("manuscript-only comparative benchmarking"),
        "production_workflow_component": False,
        "cohort_role": cohort_role,
        "confirmatory_data_accessed": (cohort_role == "confirmatory"),
        "contract_sha256": contract.get("contract_sha256"),
        "adapter_contract_sha256": (ADAPTER_CONTRACT_SHA256),
        "adapter_sha256": adapter_sha256,
        "image_reference": IMAGE_REFERENCE,
        "image_before": image_before,
        "image_after": image_after,
        "input": input_inventory,
        "staged_input": {
            "path": str(staged_input),
            "sha256": staged_sha256,
            "matches_original": (staged_sha256 == original_sha256),
        },
        "single_cohort_invocation": True,
        "external_sharding": False,
        "threshold": THRESHOLD,
        "native_batch_size_argument_supported": False,
        "native_cache_files_produced": False,
        "execution": execution,
        "container_cleanup": cleanup,
        "container_inventory_restored": (containers_after == containers_before),
        "host_docker_context_before": (host_context_before),
        "host_docker_context_after": (host_context_after),
        "host_docker_context_unchanged": (host_context_after == host_context_before),
        "adapter_metadata": adapter_metadata,
        "errors": errors,
    }

    write_json(
        output_directory / "runner_provenance.json",
        provenance,
    )
    write_artifact_checksums(output_directory)

    if errors:
        raise RunnerContractError("; ".join(errors))

    return provenance


def parse_args() -> argparse.Namespace:
    """Parse the frozen runner interface."""

    parser = argparse.ArgumentParser(
        description=(
            "This command is not part of the PlasFlow2 prediction workflow. "
            "Run the frozen manuscript-only PlasFlow v1.1 comparator on one "
            "complete cohort. The cohort is "
            "never externally sharded because native TF-IDF "
            "preprocessing is cohort dependent."
        )
    )
    parser.add_argument(
        "--input-fasta",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--docker-context",
        required=True,
    )
    parser.add_argument(
        "--scope-contract",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cohort-role",
        choices=sorted(ALLOWED_COHORT_ROLES),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    """Run one frozen comparator invocation."""

    args = parse_args()

    try:
        provenance = execute_frozen_run(
            input_fasta=args.input_fasta,
            output_directory=args.output_dir,
            scope_contract=args.scope_contract,
            docker_context=args.docker_context,
            cohort_role=args.cohort_role,
        )
    except (
        RunnerContractError,
        OSError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "overall_status": 1,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "production_workflow_component": False,
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
