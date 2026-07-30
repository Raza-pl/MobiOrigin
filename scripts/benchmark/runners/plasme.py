#!/usr/bin/env python3
"""Foundation for the frozen manuscript-only PLASMe 1.1 runner.

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
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.adapters.plasme import adapt_plasme  # noqa: E402

RUNNER_CONTRACT_SHA256 = "af79ce9f8947f4c5855ecdff74bafb61959184503f2a02abbfec273231ef0dd8"
ADAPTER_CONTRACT_SHA256 = "c852f1c16aee4cf2fb7e0f46a5f95ebe3ccd7b3c44d2c1940e4e4e014c28bbaa"
ADAPTER_SHA256 = "c496d08ab002a526431fb819bb6c3e308e2c1d0765461cda674e3cf26e44b800"
IMAGE_TAG = "plasme-nar:v1.1-env-ef0409b-execstack-cleared"
IMAGE_ID = "sha256:fbc29e53cf4b331f328241da0e7a835c" "84a50e8aa51a6baf94931aa43559f9a7"
DOCKER_CONTEXT = "colima-plasflow1-nar"
THREADS = 8
TIMEOUT_SECONDS = 172_800
IDENTITY_THRESHOLD = 0.9
COVERAGE_THRESHOLD = 0.9
PROBABILITY_THRESHOLD = 0.5
ALLOWED_COHORT_ROLES = {
    "development",
    "confirmatory",
}

CONTRACT_RELATIVE_PATH = Path("scripts/benchmark/contracts/plasme_runner_contract_v1.json")
ADAPTER_RELATIVE_PATH = Path("scripts/benchmark/adapters/plasme.py")

STATIC_MANIFEST_SHA256 = "9aff6eecfea71f199b3ff599c2b4699d9ddd34fac86beebb02b8a2135b37aa1a"
GENERATED_MANIFEST_SHA256 = "ccc001d681b6d12bb1f6e97001698420513654831079ce4e6dd0501425d6acf6"

EXECUTION_BLOCK_INSTALLED = True


class RunnerContractError(RuntimeError):
    """Raised when a frozen PLASMe runner requirement is violated."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    """Hash a dictionary using the frozen canonical JSON representation."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
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
        "threads": THREADS,
        "threads_override_allowed": False,
        "timeout_seconds": TIMEOUT_SECONDS,
        "timeout_override_allowed": False,
        "identity_threshold": IDENTITY_THRESHOLD,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "transformer_probability_threshold": (PROBABILITY_THRESHOLD),
        "threshold_override_allowed": False,
        "unified_model": False,
        "unified_model_override_allowed": False,
    }

    for field, expected in expected_parameters.items():
        if parameters.get(field) != expected:
            raise RunnerContractError(
                "Publication parameter mismatch: "
                f"{field}={parameters.get(field)!r}; "
                f"expected {expected!r}"
            )

    container = contract.get("container_contract", {})

    if container.get("image_tag") != IMAGE_TAG:
        raise RunnerContractError("Runner contract image tag does not match")

    if container.get("image_id") != IMAGE_ID:
        raise RunnerContractError("Runner contract image ID does not match")

    if container.get("docker_context") != DOCKER_CONTEXT:
        raise RunnerContractError("Runner contract Docker context does not match")

    if container.get("network_mode") != "none":
        raise RunnerContractError("Runner contract does not disable networking")

    if container.get("image_read_only") is not True:
        raise RunnerContractError("Runner contract does not require a read-only image")

    if container.get("database_mount_read_only") is not True:
        raise RunnerContractError("Runner contract does not require a read-only database")

    protocol = contract.get("protocol_contract", {})

    if protocol.get("confirmatory_tuning_allowed") is not False:
        raise RunnerContractError("Runner contract permits confirmatory tuning")

    if set(protocol.get("allowed_cohort_roles", [])) != (ALLOWED_COHORT_ROLES):
        raise RunnerContractError("Runner contract cohort roles do not match")

    decision = contract.get("decision_contract", {})

    if decision.get("positive_label") != "plasmid":
        raise RunnerContractError("Runner contract positive label does not match")

    if decision.get("negative_label") != "non-plasmid":
        raise RunnerContractError("Runner contract negative label does not match")

    if decision.get("chromosome_relabeling_allowed") is not False:
        raise RunnerContractError("Runner contract permits chromosome relabeling")

    if decision.get("phage_relabeling_allowed") is not False:
        raise RunnerContractError("Runner contract permits phage relabeling")

    command = contract.get("command_contract", {})
    official_command = command.get("official_command")

    if official_command != [
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
    ]:
        raise RunnerContractError("Runner contract official command does not match")

    if command.get("raw_output_container_path") != "/work/raw/predicted_plasmids.fasta":
        raise RunnerContractError("Runner contract output FASTA path does not match")

    if command.get("output_argument_type") != "FASTA file":
        raise RunnerContractError("Runner contract output argument is not a FASTA file")

    if command.get("output_path_must_not_exist_before_run") is not True:
        raise RunnerContractError("Runner contract does not protect output-file creation")

    if command.get("writable_container_directory") != "/work/raw":
        raise RunnerContractError("Runner contract writable directory does not match")

    return contract


def validate_adapter_identity(adapter_path: Path) -> str:
    """Verify the exact frozen manuscript adapter."""

    if not adapter_path.is_file():
        raise RunnerContractError(f"Frozen adapter does not exist: {adapter_path}")

    digest = sha256_file(adapter_path)

    if digest != ADAPTER_SHA256:
        raise RunnerContractError(
            "Frozen adapter SHA-256 mismatch: " f"{digest}; expected {ADAPTER_SHA256}"
        )

    source = adapter_path.read_text(errors="replace")

    if ADAPTER_CONTRACT_SHA256 not in source.replace('" "', ""):
        runtime_namespace: dict[str, Any] = {}
        compiled = compile(
            source,
            str(adapter_path),
            "exec",
        )
        exec(compiled, runtime_namespace)

        if runtime_namespace.get("CONTRACT_SHA256") != ADAPTER_CONTRACT_SHA256:
            raise RunnerContractError("Frozen adapter does not embed the required contract")

    return digest


def find_docker_binary() -> Path:
    """Locate the Docker client without changing its active context."""

    candidates = [
        Path("/opt/homebrew/bin/docker"),
        Path("/usr/local/bin/docker"),
    ]
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


def docker_command(
    docker_binary: Path,
    *arguments: str,
) -> list[str]:
    """Build an explicit-context Docker command."""

    return [
        str(docker_binary),
        "--context",
        DOCKER_CONTEXT,
        *arguments,
    ]


def current_host_docker_context(
    docker_binary: Path,
) -> str:
    """Return the caller's active host Docker context."""

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


def inspect_frozen_image(
    docker_binary: Path,
) -> dict[str, str]:
    """Verify the frozen Linux-amd64 PLASMe image."""

    completed = run_capture(
        docker_command(
            docker_binary,
            "image",
            "inspect",
            IMAGE_TAG,
        )
    )

    if completed.returncode != 0:
        raise RunnerContractError("Frozen PLASMe image is unavailable: " + completed.stderr.strip())

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
        raise RunnerContractError(f"Frozen PLASMe image ID mismatch: {actual_id!r}")

    if operating_system != "linux":
        raise RunnerContractError(f"Frozen image OS is not Linux: {operating_system!r}")

    if architecture != "amd64":
        raise RunnerContractError("Frozen image architecture is not amd64: " f"{architecture!r}")

    return {
        "id": actual_id,
        "os": operating_system,
        "architecture": architecture,
    }


def validate_cohort_role(cohort_role: str) -> None:
    """Reject undeclared cohort roles."""

    if cohort_role not in ALLOWED_COHORT_ROLES:
        raise RunnerContractError(f"Invalid cohort role: {cohort_role!r}")


def inventory_fasta(path: Path) -> dict[str, Any]:
    """Inventory FASTA input without changing it."""

    if not path.is_file():
        raise RunnerContractError(f"Input FASTA does not exist: {path}")

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

                if identifier in observed:
                    raise RunnerContractError(
                        "Duplicate canonical FASTA identifier: " f"{identifier}"
                    )

                observed.add(identifier)
                current_identifier = identifier
                current_length = 0
                continue

            if current_identifier is None:
                raise RunnerContractError(
                    "FASTA sequence appears before the first header " f"at line {line_number}"
                )

            current_length += len("".join(line.split()))

    finalize()

    if sequence_count == 0:
        raise RunnerContractError("Input FASTA contains no records")

    return {
        "sha256": sha256_file(path),
        "sequence_count": sequence_count,
        "base_count": base_count,
    }


def validate_manifest_file(
    path: Path,
    *,
    expected_sha256: str,
) -> None:
    """Verify one frozen database manifest file."""

    if not path.is_file():
        raise RunnerContractError(f"Database manifest does not exist: {path}")

    actual = sha256_file(path)

    if actual != expected_sha256:
        raise RunnerContractError(
            "Database manifest SHA-256 mismatch: " f"{path}: {actual}; expected {expected_sha256}"
        )


def validate_database_foundation(
    database_directory: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate the frozen runtime database path and manifests."""

    database_directory = database_directory.resolve()
    database_contract = contract.get("database_contract", {})

    expected_directory = (
        PROJECT_ROOT / str(database_contract.get("verified_runtime_directory"))
    ).resolve()

    if database_directory != expected_directory:
        raise RunnerContractError(
            "Database directory must equal the frozen runtime path: " f"{expected_directory}"
        )

    if not database_directory.is_dir():
        raise RunnerContractError(f"Database directory does not exist: {database_directory}")

    static_manifest = (PROJECT_ROOT / str(database_contract.get("static_manifest_path"))).resolve()
    generated_manifest = (
        PROJECT_ROOT / str(database_contract.get("generated_manifest_path"))
    ).resolve()

    validate_manifest_file(
        static_manifest,
        expected_sha256=STATIC_MANIFEST_SHA256,
    )
    validate_manifest_file(
        generated_manifest,
        expected_sha256=GENERATED_MANIFEST_SHA256,
    )

    required_assets = [
        database_directory / "plas_chrom_thres.csv",
        database_directory / "plas_overlap.csv",
        database_directory / "plsdb_taxon.tsv",
        database_directory / "plsdb_Mar30.clusters.p2a",
        database_directory / "plsdb_Mar30.dmnd",
        database_directory / "plsdb_Mar30.nsq",
        database_directory / "trans_model" / "unified.pt",
        database_directory / "trans_model" / "other.pt",
    ]

    missing = [str(path) for path in required_assets if not path.is_file()]

    if missing:
        raise RunnerContractError(
            "Required PLASMe database assets are missing: " + ", ".join(missing)
        )

    transformer_models = sorted((database_directory / "trans_model").glob("*.pt"))

    if len(transformer_models) != 36:
        raise RunnerContractError(
            "Expected 36 PLASMe transformer models, found " f"{len(transformer_models)}"
        )

    return {
        "directory": str(database_directory),
        "static_manifest": str(static_manifest),
        "generated_manifest": str(generated_manifest),
        "transformer_model_count": len(transformer_models),
    }


def validate_candidate_header(path: Path) -> None:
    """Validate the frozen native candidate CSV header."""

    if not path.is_file():
        raise RunnerContractError(f"PLASMe candidate CSV is missing: {path}")

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)

    expected = [
        "order",
        "query",
        "identity",
        "coverage",
        "PLASMe",
        "overlap",
    ]

    if header != expected:
        raise RunnerContractError(f"Unexpected PLASMe candidate columns: {header!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen runner command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen manuscript-only PLASMe 1.1 comparator "
            "on one complete cohort. This command is not part of "
            "the PlasFlow2 prediction workflow."
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
        "--database-directory",
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
    return parser


STATIC_RUNTIME_REMOVED_PATHS = {
    "DB/plsdb.zip",
    "DB/plsdb_Mar30.fna",
    "DB/plsdb_Mar30.fna.aa",
}


def utc_now() -> str:
    """Return a compact UTC timestamp."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_scope_contract(
    scope_contract: Path,
    contract: dict[str, Any],
) -> str:
    """Verify the exact frozen benchmark protocol document."""

    protocol = contract.get("protocol_contract", {})
    expected_path = (PROJECT_ROOT / str(protocol.get("path"))).resolve()
    supplied_path = scope_contract.resolve()

    if supplied_path != expected_path:
        raise RunnerContractError(
            "Scope contract must equal the frozen protocol path: " f"{expected_path}"
        )

    if not supplied_path.is_file():
        raise RunnerContractError(f"Frozen scope contract does not exist: {supplied_path}")

    actual = sha256_file(supplied_path)
    expected = str(protocol.get("sha256") or "")

    if actual != expected:
        raise RunnerContractError(
            "Scope-contract SHA-256 mismatch: " f"{actual}; expected {expected}"
        )

    return actual


def expected_database_files(
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load the frozen static and generated database manifests."""

    database_contract = contract.get("database_contract", {})
    specifications = [
        (
            PROJECT_ROOT / str(database_contract.get("static_manifest_path")),
            True,
        ),
        (
            PROJECT_ROOT / str(database_contract.get("generated_manifest_path")),
            False,
        ),
    ]
    expected: dict[str, dict[str, Any]] = {}

    for manifest_path, is_static in specifications:
        manifest_path = manifest_path.resolve()

        with manifest_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"relative_path", "bytes", "sha256"}

            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise RunnerContractError(f"Invalid database manifest columns: {manifest_path}")

            for row in reader:
                manifest_relative = str(row["relative_path"])

                if is_static and manifest_relative in STATIC_RUNTIME_REMOVED_PATHS:
                    continue

                runtime_relative = manifest_relative

                if is_static and runtime_relative.startswith("DB/"):
                    runtime_relative = runtime_relative[3:]

                relative_path = Path(runtime_relative)

                if (
                    relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or not relative_path.parts
                ):
                    raise RunnerContractError(
                        "Unsafe database manifest path: " f"{manifest_relative!r}"
                    )

                entry = {
                    "bytes": int(row["bytes"]),
                    "sha256": str(row["sha256"]),
                }
                key = relative_path.as_posix()
                previous = expected.get(key)

                if previous is not None and previous != entry:
                    raise RunnerContractError(f"Conflicting database manifest entry: {key}")

                expected[key] = entry

    return expected


def verify_database_snapshot(
    database_directory: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Hash the complete frozen runtime database."""

    database_directory = database_directory.resolve()
    expected = expected_database_files(contract)
    actual_paths = {
        path.relative_to(database_directory).as_posix()
        for path in database_directory.rglob("*")
        if path.is_file()
    }
    expected_paths = set(expected)
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)

    if missing:
        raise RunnerContractError("Runtime database files are missing: " + ", ".join(missing[:20]))

    if unexpected:
        raise RunnerContractError(
            "Unexpected runtime database files: " + ", ".join(unexpected[:20])
        )

    verified: dict[str, dict[str, Any]] = {}
    total_bytes = 0

    for relative_path in sorted(expected):
        path = database_directory / relative_path
        expected_entry = expected[relative_path]
        observed_bytes = path.stat().st_size

        if observed_bytes != expected_entry["bytes"]:
            raise RunnerContractError(
                "Runtime database size mismatch: "
                f"{relative_path}: {observed_bytes}; "
                f"expected {expected_entry['bytes']}"
            )

        observed_sha256 = sha256_file(path)

        if observed_sha256 != expected_entry["sha256"]:
            raise RunnerContractError(
                "Runtime database SHA-256 mismatch: " f"{relative_path}: {observed_sha256}"
            )

        verified[relative_path] = {
            "bytes": observed_bytes,
            "sha256": observed_sha256,
        }
        total_bytes += observed_bytes

    return {
        "file_count": len(verified),
        "total_bytes": total_bytes,
        "content_sha256": canonical_json_sha256({"files": verified}),
    }


def prepare_output_directory(output_directory: Path) -> Path:
    """Create a new output directory without overwriting prior evidence."""

    output_directory = output_directory.resolve()

    if output_directory.exists():
        raise RunnerContractError(
            "Output directory already exists and may not be overwritten: " f"{output_directory}"
        )

    output_directory.mkdir(parents=True)
    return output_directory


def build_container_command(
    docker_binary: Path,
    input_fasta: Path,
    database_directory: Path,
    output_directory: Path,
    container_name: str,
    contract: dict[str, Any],
) -> list[str]:
    """Build the exact no-shell Docker invocation."""

    official_command = [str(value) for value in contract["command_contract"]["official_command"]]

    if len(official_command) < 3:
        raise RunnerContractError("Official PLASMe command is incomplete")

    entrypoint = official_command[0]
    container_arguments = official_command[1:]
    raw_output_directory = output_directory.resolve() / "raw"

    return docker_command(
        docker_binary,
        "run",
        "--rm",
        "--name",
        container_name,
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--read-only",
        "--workdir",
        "/opt/plasme",
        "--mount",
        (f"type=bind,source={input_fasta.resolve()}," "target=/work/input.fasta,readonly"),
        "--mount",
        (f"type=bind,source={database_directory.resolve()}," "target=/opt/plasme/DB,readonly"),
        "--mount",
        (f"type=bind,source={raw_output_directory}," "target=/work/raw"),
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=1073741824",
        "--entrypoint",
        entrypoint,
        IMAGE_TAG,
        *container_arguments,
    )


def terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: int = 15,
) -> None:
    """Terminate the complete Docker client process group."""

    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return

    process.wait()


def execute_tool(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    """Run PLASMe with process-group timeout and resource capture."""

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


def remove_container(
    docker_binary: Path,
    container_name: str,
) -> dict[str, Any]:
    """Remove a possible residual container after interruption."""

    completed = run_capture(
        docker_command(
            docker_binary,
            "rm",
            "--force",
            container_name,
        )
    )

    missing = "No such container" in completed.stderr

    return {
        "return_code": completed.returncode,
        "already_absent": missing,
        "stderr": completed.stderr.strip(),
    }


def validate_standardized_count(
    path: Path,
    expected_records: int,
) -> None:
    """Require exactly one standardized row per input record."""

    if not path.is_file():
        raise RunnerContractError(f"Standardized prediction table is missing: {path}")

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    if len(rows) != expected_records:
        raise RunnerContractError(
            "Standardized output row count mismatch: " f"{len(rows)}; expected {expected_records}"
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


def run_plasme(
    *,
    input_fasta: Path,
    output_directory: Path,
    database_directory: Path,
    docker_context: str,
    scope_contract: Path,
    cohort_role: str,
) -> dict[str, Any]:
    """Execute one complete frozen PLASMe cohort run."""

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

    errors: list[str] = []
    docker_binary: Path | None = None
    container_name = "plasme-nar-" + uuid.uuid4().hex[:12]
    command: list[str] | None = None
    tool_result: dict[str, Any] | None = None
    adapter_result: dict[str, Any] | None = None
    image_before: dict[str, str] | None = None
    image_after: dict[str, str] | None = None
    database_before: dict[str, Any] | None = None
    database_after: dict[str, Any] | None = None
    database_foundation: dict[str, Any] | None = None
    host_context_before: str | None = None
    host_context_after: str | None = None
    cleanup: dict[str, Any] | None = None

    try:
        docker_binary = find_docker_binary()
        host_context_before = current_host_docker_context(docker_binary)
        image_before = inspect_frozen_image(docker_binary)
        database_foundation = validate_database_foundation(
            database_directory,
            contract,
        )
        database_before = verify_database_snapshot(
            database_directory,
            contract,
        )

        (output_directory / "raw").mkdir()

        command = build_container_command(
            docker_binary,
            input_fasta,
            database_directory,
            output_directory,
            container_name,
            contract,
        )
        write_json(
            output_directory / "command.json",
            {
                "argv": command,
                "shell_execution": False,
                "container_name": container_name,
            },
        )

        tool_result = execute_tool(
            command,
            output_directory / "stdout.log",
            output_directory / "stderr.log",
        )
        write_json(
            output_directory / "resource_usage.json",
            tool_result,
        )

        if tool_result["return_code"] != 0:
            errors.append("PLASMe returned nonzero status " f"{tool_result['return_code']}")

        if tool_result["timed_out"]:
            errors.append("PLASMe exceeded the frozen timeout")

        if tool_result["interrupted"]:
            errors.append("PLASMe was interrupted by a signal")

        if not errors:
            positive_fasta = (
                output_directory / contract["command_contract"]["positive_fasta_relative_path"]
            )
            candidate_csv = (
                output_directory / contract["command_contract"]["candidate_csv_relative_path"]
            )

            if not positive_fasta.is_file():
                raise RunnerContractError(f"PLASMe positive FASTA is missing: {positive_fasta}")

            validate_candidate_header(candidate_csv)

            adapter_result = adapt_plasme(
                input_fasta=input_fasta,
                positive_fasta=positive_fasta,
                candidate_csv=candidate_csv,
                output_path=(output_directory / "standardized_predictions.tsv"),
                metadata_output=(output_directory / "adapter_metadata.json"),
            )
            validate_standardized_count(
                output_directory / "standardized_predictions.tsv",
                int(input_inventory["sequence_count"]),
            )

    except KeyboardInterrupt:
        errors.append("Runner interrupted during preflight or validation")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        if docker_binary is not None:
            try:
                cleanup = remove_container(
                    docker_binary,
                    container_name,
                )
            except Exception as error:
                errors.append(
                    "Container cleanup validation failed: " f"{type(error).__name__}: {error}"
                )

            try:
                image_after = inspect_frozen_image(docker_binary)

                if image_before is not None and image_after != image_before:
                    errors.append("Frozen container image identity changed")
            except Exception as error:
                errors.append(
                    "Post-run image verification failed: " f"{type(error).__name__}: {error}"
                )

            try:
                host_context_after = current_host_docker_context(docker_binary)

                if host_context_before is not None and host_context_after != host_context_before:
                    errors.append("The caller's Docker context changed")
            except Exception as error:
                errors.append(
                    "Post-run Docker context verification failed: "
                    f"{type(error).__name__}: {error}"
                )

        if database_before is not None:
            try:
                database_after = verify_database_snapshot(
                    database_directory,
                    contract,
                )

                if database_after != database_before:
                    errors.append("Frozen runtime database identity changed")
            except Exception as error:
                errors.append(
                    "Post-run database verification failed: " f"{type(error).__name__}: {error}"
                )

    provenance = {
        "status": "PASS" if not errors else "FAIL",
        "schema_version": "nar-comparator-runner-v1",
        "tool": "PLASMe",
        "version": "1.1",
        "manuscript_only": True,
        "production_workflow_component": False,
        "cohort_role": cohort_role,
        "confirmatory_data_accessed": cohort_role == "confirmatory",
        "confirmatory_tuning_allowed": False,
        "runner_contract_sha256": RUNNER_CONTRACT_SHA256,
        "adapter_contract_sha256": ADAPTER_CONTRACT_SHA256,
        "adapter_sha256": adapter_sha256,
        "scope_contract_sha256": scope_sha256,
        "container_image_tag": IMAGE_TAG,
        "container_image_before": image_before,
        "container_image_after": image_after,
        "docker_context_argument": docker_context,
        "host_docker_context_before": host_context_before,
        "host_docker_context_after": host_context_after,
        "input_fasta": str(input_fasta.resolve()),
        "input_inventory": input_inventory,
        "database_foundation": database_foundation,
        "database_before": database_before,
        "database_after": database_after,
        "container_name": container_name,
        "command": command,
        "tool_result": tool_result,
        "container_cleanup": cleanup,
        "adapter_result": adapter_result,
        "errors": errors,
        "recorded_at": utc_now(),
    }

    write_json(
        output_directory / "runner_provenance.json",
        provenance,
    )
    write_artifact_checksums(output_directory)

    if errors:
        raise RunnerContractError("; ".join(errors))

    return provenance


def main() -> None:
    """Run one frozen manuscript-only PLASMe cohort."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        provenance = run_plasme(
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

    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
