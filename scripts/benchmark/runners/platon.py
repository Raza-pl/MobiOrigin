#!/usr/bin/env python3
"""Foundation for the frozen manuscript-only Platon 1.7 runner.

This runner is restricted to comparative analyses for the manuscript.
It is not part of the PlasFlow2 prediction workflow.
"""

from __future__ import annotations

import argparse
import csv
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNNER_CONTRACT_SHA256 = "b7a2283ab8ba36d3bb2ee12317b879af5db19ed34f5aae46becf358ffe71271e"
ADAPTER_CONTRACT_SHA256 = "b8add8c173bdd049f750133cdfd6bef2d8be42b0315c2bcf66e4182079aa6e96"
ADAPTER_SHA256 = "09221c780bd243d326d5597a3207e2b0e56b96bc968fd8d3f1b866372732bc0c"
DATABASE_MANIFEST_SHA256 = "ba247e74f80caeff348af80e92c375921d1eac1d8627447e74fa7136994e4906"
IMAGE_REFERENCE = (
    "quay.io/biocontainers/platon@"
    "sha256:74d96300053a9ce3d4f10bbb935b20631e1d8547c1df632d5f05b178eb2cbbf6"
)
IMAGE_ID = "sha256:74d96300053a9ce3d4f10bbb935b20631e1d8547c1df632d5f05b178eb2cbbf6"
DOCKER_CONTEXT = "colima-plasflow1-nar"
MODE = "accuracy"
METAGENOME_MODE = True
CHARACTERIZE_MODE = False
THREADS = 4
TIMEOUT_SECONDS = 172_800
ALLOWED_COHORT_ROLES = {"development", "confirmatory"}

CONTRACT_RELATIVE_PATH = Path("scripts/benchmark/contracts/platon_runner_contract_v1.json")
ADAPTER_RELATIVE_PATH = Path("scripts/benchmark/adapters/platon.py")

EXECUTION_BLOCK_INSTALLED = True


class RunnerContractError(RuntimeError):
    """Raised when a frozen Platon runner requirement is violated."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    """Hash a dictionary using the contract's canonical representation."""

    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    """Load a required JSON document."""

    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise RunnerContractError(f"Required JSON file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise RunnerContractError(f"Invalid JSON document: {path}: {error}") from error


def validate_runner_contract(path: Path) -> dict[str, Any]:
    """Validate the complete frozen runner contract."""

    contract = load_json(path)
    if not isinstance(contract, dict):
        raise RunnerContractError("Runner contract must contain one JSON object")

    declared = contract.get("contract_sha256")
    if declared != RUNNER_CONTRACT_SHA256:
        raise RunnerContractError(f"Unexpected runner contract identity: {declared!r}")

    canonical = dict(contract)
    canonical.pop("contract_sha256", None)
    calculated = canonical_json_sha256(canonical)
    if calculated != RUNNER_CONTRACT_SHA256:
        raise RunnerContractError(f"Runner contract content hash mismatch: {calculated}")

    if contract.get("production_workflow_component") is not False:
        raise RunnerContractError("Runner contract violates manuscript-only scope")

    parameters = contract.get("publication_parameters", {})
    expected_parameters = {
        "mode": MODE,
        "mode_override_allowed": False,
        "metagenome_mode": METAGENOME_MODE,
        "metagenome_mode_override_allowed": False,
        "characterize_mode": CHARACTERIZE_MODE,
        "characterize_mode_override_allowed": False,
        "threads": THREADS,
        "threads_override_allowed": False,
        "timeout_seconds": TIMEOUT_SECONDS,
        "timeout_override_allowed": False,
        "external_sequence_sharding_allowed": False,
        "threshold_tuning_allowed": False,
    }
    for field, expected in expected_parameters.items():
        if parameters.get(field) != expected:
            raise RunnerContractError(
                f"Publication parameter mismatch: {field}="
                f"{parameters.get(field)!r}; expected {expected!r}"
            )

    container = contract.get("container_contract", {})
    expected_container = {
        "image_reference": IMAGE_REFERENCE,
        "image_id": IMAGE_ID,
        "docker_context": DOCKER_CONTEXT,
        "container_platform": "linux/amd64",
        "network_mode": "none",
        "image_read_only": True,
        "input_mount_read_only": True,
        "database_mount_read_only": True,
        "output_mount_writable": True,
    }
    for field, expected in expected_container.items():
        if container.get(field) != expected:
            raise RunnerContractError(
                f"Container contract mismatch: {field}="
                f"{container.get(field)!r}; expected {expected!r}"
            )

    database = contract.get("database_contract", {})
    if database.get("manifest_sha256") != DATABASE_MANIFEST_SHA256:
        raise RunnerContractError("Database manifest identity does not match")
    if database.get("database_mutation_during_prediction_allowed") is not False:
        raise RunnerContractError("Runner contract permits database mutation")

    protocol = contract.get("protocol_contract", {})
    if protocol.get("confirmatory_tuning_allowed") is not False:
        raise RunnerContractError("Runner contract permits confirmatory tuning")
    if set(protocol.get("allowed_cohort_roles", [])) != ALLOWED_COHORT_ROLES:
        raise RunnerContractError("Runner contract cohort roles do not match")

    decision = contract.get("decision_contract", {})
    if decision.get("positive_label") != "plasmid":
        raise RunnerContractError("Runner contract positive label does not match")
    if decision.get("negative_label") != "non-plasmid":
        raise RunnerContractError("Runner contract negative label does not match")
    if decision.get("adapter_reimplements_native_decision") is not False:
        raise RunnerContractError("Runner contract permits adapter reclassification")
    if decision.get("rds_treated_as_probability") is not False:
        raise RunnerContractError("Runner contract treats RDS as a probability")

    command = contract.get("command_contract", {})
    expected_command = [
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
    if command.get("official_command") != expected_command:
        raise RunnerContractError("Runner contract official command does not match")
    if command.get("one_tool_invocation_per_complete_cohort") is not True:
        raise RunnerContractError("Runner contract does not require one cohort invocation")
    if command.get("shell_execution_allowed") is not False:
        raise RunnerContractError("Runner contract permits shell execution")

    return contract


def validate_adapter_identity(adapter_path: Path) -> str:
    """Verify the exact frozen manuscript adapter."""

    if not adapter_path.is_file():
        raise RunnerContractError(f"Frozen adapter does not exist: {adapter_path}")

    digest = sha256_file(adapter_path)
    if digest != ADAPTER_SHA256:
        raise RunnerContractError(
            f"Frozen adapter SHA-256 mismatch: {digest}; expected {ADAPTER_SHA256}"
        )

    namespace: dict[str, Any] = {}
    exec(compile(adapter_path.read_text(), str(adapter_path), "exec"), namespace)
    if namespace.get("CONTRACT_SHA256") != ADAPTER_CONTRACT_SHA256:
        raise RunnerContractError("Frozen adapter contract identity does not match")
    return digest


def validate_scope_contract(path: Path, contract: dict[str, Any]) -> str:
    """Verify the frozen scope/protocol report."""

    if not path.is_file():
        raise RunnerContractError(f"Scope contract does not exist: {path}")
    digest = sha256_file(path)
    expected = contract.get("protocol_contract", {}).get("sha256")
    if digest != expected:
        raise RunnerContractError(f"Scope contract SHA-256 mismatch: {digest}; expected {expected}")
    return digest


def validate_cohort_role(cohort_role: str) -> None:
    """Reject undeclared cohort roles."""

    if cohort_role not in ALLOWED_COHORT_ROLES:
        raise RunnerContractError(f"Invalid cohort role: {cohort_role!r}")


def inventory_fasta(path: Path) -> dict[str, Any]:
    """Inventory input FASTA while rejecting ambiguous identifiers."""

    if not path.is_file():
        raise RunnerContractError(f"Input FASTA does not exist: {path}")

    identifiers: set[str] = set()
    sequence_count = 0
    base_count = 0
    unsupported_count = 0
    current_id: str | None = None
    current_length = 0

    def finalize() -> None:
        nonlocal sequence_count, base_count, unsupported_count
        nonlocal current_id, current_length
        if current_id is None:
            return
        if current_length <= 0:
            raise RunnerContractError(f"FASTA record has no sequence: {current_id}")
        sequence_count += 1
        base_count += current_length
        if not 1_000 <= current_length <= 500_000:
            unsupported_count += 1
        current_id = None
        current_length = 0

    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finalize()
                header = line[1:].strip()
                if not header:
                    raise RunnerContractError(f"Empty FASTA header at line {line_number}")
                identifier = header.split()[0]
                if identifier in identifiers:
                    raise RunnerContractError(f"Duplicate canonical FASTA identifier: {identifier}")
                identifiers.add(identifier)
                current_id = identifier
                continue
            if current_id is None:
                raise RunnerContractError(
                    f"Sequence data precedes the first header at line {line_number}"
                )
            sequence = "".join(line.split()).upper()
            if not sequence:
                continue
            invalid = set(sequence) - set("ACGTURYSWKMBDHVN.-")
            if invalid:
                raise RunnerContractError(
                    f"Invalid FASTA symbols at line {line_number}: " f"{''.join(sorted(invalid))}"
                )
            current_length += len(sequence)

    finalize()
    if sequence_count == 0:
        raise RunnerContractError("Input FASTA contains no records")

    return {
        "sha256": sha256_file(path),
        "sequence_count": sequence_count,
        "base_count": base_count,
        "unsupported_length_count": unsupported_count,
    }


def verify_database_manifest(
    database_directory: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Verify the frozen database manifest against the read-only database."""

    if not database_directory.is_dir():
        raise RunnerContractError(f"Database directory does not exist: {database_directory}")

    manifest_relative = contract.get("database_contract", {}).get("manifest_path")
    manifest = PROJECT_ROOT / str(manifest_relative)
    if not manifest.is_file():
        raise RunnerContractError(f"Database manifest does not exist: {manifest}")
    if sha256_file(manifest) != DATABASE_MANIFEST_SHA256:
        raise RunnerContractError("Database manifest SHA-256 mismatch")

    verified_files = 0
    verified_bytes = 0
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["sha256", "bytes", "path"]:
            raise RunnerContractError("Unexpected database manifest schema")
        for row in reader:
            relative = Path(row["path"])
            if relative.parts[:1] != ("db",):
                raise RunnerContractError(f"Unexpected database manifest path: {relative}")
            candidate = database_directory.parent / relative
            try:
                candidate.resolve().relative_to(database_directory.resolve())
            except ValueError as error:
                raise RunnerContractError(
                    f"Database manifest path escapes the database: {relative}"
                ) from error
            if not candidate.is_file():
                raise RunnerContractError(f"Database file is missing: {candidate}")
            expected_bytes = int(row["bytes"])
            if candidate.stat().st_size != expected_bytes:
                raise RunnerContractError(f"Database file size mismatch: {candidate}")
            if sha256_file(candidate) != row["sha256"]:
                raise RunnerContractError(f"Database file SHA-256 mismatch: {candidate}")
            verified_files += 1
            verified_bytes += expected_bytes

    if verified_files != 31:
        raise RunnerContractError(
            f"Database manifest file count mismatch: {verified_files}; expected 31"
        )
    return {
        "manifest_sha256": DATABASE_MANIFEST_SHA256,
        "verified_files": verified_files,
        "verified_bytes": verified_bytes,
    }


def find_docker_binary() -> Path:
    """Locate the Docker client without changing its active context."""

    candidates = [Path("/opt/homebrew/bin/docker"), Path("/usr/local/bin/docker")]
    discovered = shutil.which("docker")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
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


def docker_command(docker_binary: Path, *arguments: str) -> list[str]:
    """Build an explicit-context Docker command."""

    return [str(docker_binary), "--context", DOCKER_CONTEXT, *arguments]


def current_host_docker_context(docker_binary: Path) -> str:
    """Return the caller's active host Docker context."""

    completed = run_capture([str(docker_binary), "context", "show"])
    if completed.returncode != 0:
        raise RunnerContractError(
            "Unable to query the host Docker context: " + completed.stderr.strip()
        )
    return completed.stdout.strip()


def inspect_frozen_image(docker_binary: Path) -> dict[str, str]:
    """Verify the frozen Linux-amd64 Platon image."""

    completed = run_capture(docker_command(docker_binary, "image", "inspect", IMAGE_REFERENCE))
    if completed.returncode != 0:
        raise RunnerContractError("Frozen Platon image is unavailable: " + completed.stderr.strip())
    try:
        documents = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RunnerContractError("Docker image inspection returned invalid JSON") from error
    if len(documents) != 1:
        raise RunnerContractError("Expected exactly one image inspection result")
    image = documents[0]
    actual_id = str(image.get("Id") or "")
    operating_system = str(image.get("Os") or "")
    architecture = str(image.get("Architecture") or "")
    if actual_id != IMAGE_ID:
        raise RunnerContractError(f"Frozen Platon image ID mismatch: {actual_id!r}")
    if operating_system != "linux" or architecture != "amd64":
        raise RunnerContractError(
            f"Frozen Platon image platform mismatch: " f"{operating_system!r}/{architecture!r}"
        )
    return {
        "id": actual_id,
        "os": operating_system,
        "architecture": architecture,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen manuscript-only runner interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen manuscript-only Platon 1.7 comparator on one "
            "complete cohort. This command is not part of the PlasFlow2 "
            "prediction workflow."
        )
    )
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--database-directory", type=Path, required=True)
    parser.add_argument("--docker-context", required=True)
    parser.add_argument("--scope-contract", type=Path, required=True)
    parser.add_argument(
        "--cohort-role",
        choices=sorted(ALLOWED_COHORT_ROLES),
        required=True,
    )
    return parser


def utc_now() -> str:
    """Return a UTC timestamp for provenance."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare_output_directory(path: Path) -> Path:
    """Create a new output directory without overwriting prior evidence."""

    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise RunnerContractError(f"Output directory already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.mkdir()
    return resolved


def validate_required_native_outputs(raw_directory: Path) -> dict[str, int]:
    """Require every official Platon output path, including empty class FASTAs."""

    required = {
        "plasmid_fasta": raw_directory / "platon.plasmid.fasta",
        "chromosome_fasta": raw_directory / "platon.chromosome.fasta",
        "json": raw_directory / "platon.json",
        "tsv": raw_directory / "platon.tsv",
        "log": raw_directory / "platon.log",
    }
    sizes: dict[str, int] = {}
    for name, path in required.items():
        if not path.is_file():
            raise RunnerContractError(f"Required Platon output is missing: {path}")
        sizes[name] = path.stat().st_size
    return sizes


def build_container_command(
    docker_binary: Path,
    input_fasta: Path,
    database_directory: Path,
    output_directory: Path,
    container_name: str,
) -> list[str]:
    """Build the exact isolated Docker invocation without a shell."""

    return docker_command(
        docker_binary,
        "run",
        "--name",
        container_name,
        "--rm",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=4g",
        "--workdir",
        "/work",
        "--mount",
        f"type=bind,src={input_fasta.resolve()},dst=/work/input.fasta,readonly",
        "--mount",
        (f"type=bind,src={database_directory.resolve()}," "dst=/database,readonly"),
        "--mount",
        f"type=bind,src={output_directory.resolve()},dst=/work",
        "--label",
        "org.plasflow2.benchmark.scope=manuscript-only",
        "--label",
        f"org.plasflow2.benchmark.contract={RUNNER_CONTRACT_SHA256}",
        IMAGE_REFERENCE,
        "/usr/local/bin/platon",
        "--db",
        "/database",
        "--output",
        "/work/raw",
        "--prefix",
        "platon",
        "--mode",
        MODE,
        "--meta",
        "--threads",
        str(THREADS),
        "/work/input.fasta",
    )


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate the Docker client process group, then force it if needed."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def execute_container(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    """Execute the frozen container with timeout and resource capture."""

    started_at = utc_now()
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    timed_out = False
    interrupted = False

    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            return_code = 124
        except KeyboardInterrupt:
            interrupted = True
            terminate_process_group(process)
            return_code = 130

    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss = int(after.ru_maxrss)
    if sys.platform != "darwin":
        peak_rss *= 1024

    return {
        "started_at": started_at,
        "finished_at": utc_now(),
        "wallclock_seconds": time.monotonic() - started,
        "return_code": return_code,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "user_seconds": after.ru_utime - before.ru_utime,
        "system_seconds": after.ru_stime - before.ru_stime,
        "peak_rss_bytes_when_available": peak_rss,
    }


def remove_container(docker_binary: Path, container_name: str) -> dict[str, Any]:
    """Remove a possible residual container after timeout or interruption."""

    completed = run_capture(docker_command(docker_binary, "rm", "--force", container_name))
    missing = "No such container" in completed.stderr
    return {
        "return_code": completed.returncode,
        "already_absent": missing,
        "stderr": completed.stderr.strip(),
    }


def run_adapter(
    adapter_path: Path,
    input_fasta: Path,
    raw_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Invoke the exact frozen adapter as a separate no-shell process."""

    command = [
        sys.executable,
        str(adapter_path),
        "--input-fasta",
        str(input_fasta),
        "--plasmid-fasta",
        str(raw_directory / "platon.plasmid.fasta"),
        "--chromosome-fasta",
        str(raw_directory / "platon.chromosome.fasta"),
        "--raw-json",
        str(raw_directory / "platon.json"),
        "--raw-tsv",
        str(raw_directory / "platon.tsv"),
        "--output",
        str(output_directory / "standardized_predictions.tsv"),
        "--metadata-output",
        str(output_directory / "adapter_metadata.json"),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=3_600,
    )
    (output_directory / "adapter_stdout.log").write_text(completed.stdout)
    (output_directory / "adapter_stderr.log").write_text(completed.stderr)
    return {
        "argv": command,
        "return_code": completed.returncode,
        "stdout_path": "adapter_stdout.log",
        "stderr_path": "adapter_stderr.log",
    }


def validate_standardized_count(path: Path, expected_records: int) -> None:
    """Require exactly one standardized row per input FASTA record."""

    if not path.is_file():
        raise RunnerContractError(f"Standardized prediction table is missing: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != expected_records:
        raise RunnerContractError(
            f"Standardized output row count mismatch: {len(rows)}; " f"expected {expected_records}"
        )


def write_artifact_checksums(output_directory: Path) -> None:
    """Write SHA-256 checksums for every retained runner artifact."""

    checksum_path = output_directory / "artifact_checksums.sha256"
    lines: list[str] = []
    for path in sorted(output_directory.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        relative = path.relative_to(output_directory).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    checksum_path.write_text("\n".join(lines) + "\n")


def run_platon(
    *,
    input_fasta: Path,
    output_directory: Path,
    database_directory: Path,
    docker_context: str,
    scope_contract: Path,
    cohort_role: str,
) -> dict[str, Any]:
    """Execute one complete frozen Platon cohort run."""

    if docker_context != DOCKER_CONTEXT:
        raise RunnerContractError(f"Docker context must equal the frozen value {DOCKER_CONTEXT!r}")

    validate_cohort_role(cohort_role)
    contract_path = PROJECT_ROOT / CONTRACT_RELATIVE_PATH
    adapter_path = PROJECT_ROOT / ADAPTER_RELATIVE_PATH
    contract = validate_runner_contract(contract_path)
    adapter_sha256 = validate_adapter_identity(adapter_path)
    scope_sha256 = validate_scope_contract(scope_contract, contract)
    input_inventory = inventory_fasta(input_fasta)
    output_directory = prepare_output_directory(output_directory)
    raw_directory = output_directory / "raw"
    raw_directory.mkdir()

    errors: list[str] = []
    docker_binary: Path | None = None
    container_name = "platon-nar-" + uuid.uuid4().hex[:12]
    command: list[str] | None = None
    tool_result: dict[str, Any] | None = None
    adapter_result: dict[str, Any] | None = None
    native_outputs: dict[str, int] | None = None
    image_before: dict[str, str] | None = None
    image_after: dict[str, str] | None = None
    database_before: dict[str, Any] | None = None
    database_after: dict[str, Any] | None = None
    host_context_before: str | None = None
    host_context_after: str | None = None
    cleanup: dict[str, Any] | None = None

    try:
        docker_binary = find_docker_binary()
        host_context_before = current_host_docker_context(docker_binary)
        image_before = inspect_frozen_image(docker_binary)
        database_before = verify_database_manifest(database_directory, contract)

        command = build_container_command(
            docker_binary,
            input_fasta,
            database_directory,
            output_directory,
            container_name,
        )
        write_json(
            output_directory / "command.json",
            {
                "argv": command,
                "shell_execution": False,
                "container_name": container_name,
            },
        )
        tool_result = execute_container(
            command,
            stdout_path=output_directory / "stdout.log",
            stderr_path=output_directory / "stderr.log",
        )
        write_json(output_directory / "resource_usage.json", tool_result)

        if tool_result["return_code"] != 0:
            errors.append(f"Platon returned nonzero status {tool_result['return_code']}")
        else:
            native_outputs = validate_required_native_outputs(raw_directory)
            adapter_result = run_adapter(
                adapter_path,
                input_fasta,
                raw_directory,
                output_directory,
            )
            if adapter_result["return_code"] != 0:
                errors.append(
                    f"Frozen Platon adapter returned status " f"{adapter_result['return_code']}"
                )
            else:
                validate_standardized_count(
                    output_directory / "standardized_predictions.tsv",
                    int(input_inventory["sequence_count"]),
                )
    except (RunnerContractError, OSError, subprocess.SubprocessError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        if docker_binary is not None:
            cleanup = remove_container(docker_binary, container_name)
            try:
                image_after = inspect_frozen_image(docker_binary)
            except RunnerContractError as error:
                errors.append(f"Post-run image verification failed: {error}")
            try:
                database_after = verify_database_manifest(
                    database_directory,
                    contract,
                )
            except RunnerContractError as error:
                errors.append(f"Post-run database verification failed: {error}")
            try:
                host_context_after = current_host_docker_context(docker_binary)
            except RunnerContractError as error:
                errors.append(f"Post-run Docker-context check failed: {error}")

    if image_before is not None and image_after != image_before:
        errors.append("Container image identity changed during execution")
    if database_before is not None and database_after != database_before:
        errors.append("Database identity changed during execution")
    if host_context_before is not None and host_context_after != host_context_before:
        errors.append("Host Docker context changed during execution")

    provenance = {
        "status": "PASS" if not errors else "FAIL",
        "scope": "manuscript-only comparative benchmarking",
        "production_workflow_component": False,
        "tool": "Platon",
        "tool_version": "1.7",
        "cohort_role": cohort_role,
        "confirmatory_tuning_performed": False,
        "confirmatory_data_accessed_by_runner_logic": (cohort_role == "confirmatory"),
        "runner_contract_sha256": RUNNER_CONTRACT_SHA256,
        "adapter_contract_sha256": ADAPTER_CONTRACT_SHA256,
        "adapter_sha256": adapter_sha256,
        "scope_contract_sha256": scope_sha256,
        "input": input_inventory,
        "publication_parameters": {
            "mode": MODE,
            "metagenome_mode": METAGENOME_MODE,
            "characterize_mode": CHARACTERIZE_MODE,
            "threads": THREADS,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
        "docker_context": docker_context,
        "image_before": image_before,
        "image_after": image_after,
        "database_before": database_before,
        "database_after": database_after,
        "container_name": container_name,
        "command": command,
        "tool_result": tool_result,
        "native_outputs": native_outputs,
        "adapter_result": adapter_result,
        "container_cleanup": cleanup,
        "host_context_before": host_context_before,
        "host_context_after": host_context_after,
        "errors": errors,
        "recorded_at": utc_now(),
    }
    write_json(output_directory / "runner_provenance.json", provenance)
    write_artifact_checksums(output_directory)

    if errors:
        raise RunnerContractError("; ".join(errors))
    return provenance


def main() -> None:
    """Run one frozen manuscript-only Platon cohort."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_platon(
            input_fasta=args.input_fasta,
            output_directory=args.output_dir,
            database_directory=args.database_directory,
            docker_context=args.docker_context,
            scope_contract=args.scope_contract,
            cohort_role=args.cohort_role,
        )
    except RunnerContractError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
