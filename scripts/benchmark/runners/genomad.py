#!/usr/bin/env python3
"""Manuscript-only frozen geNomad publication benchmark runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Permit both module execution and direct execution from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.benchmark.adapters.genomad import (  # noqa: E402
    OUTPUT_FIELDS,
    adapt_genomad,
    load_fasta_headers,
    sha256_file,
)

TOOL_NAME = "geNomad"
TOOL_VERSION = "1.12.0"
DATABASE_VERSION = "1.9"
RUNNER_SCHEMA = "nar-genomad-runner-v1"
SCOPE = "manuscript-only comparative benchmarking"
MMSEQS_SPLITS = 8

EXPECTED_DATABASE_FINGERPRINT = "1a23156892a2ee1aa149641b39f65bfb5c7a9fe8ed6c9647dc0b9fb26633677d"
EXPECTED_DATABASE_TABLE_SHA256 = "1a23156892a2ee1aa149641b39f65bfb5c7a9fe8ed6c9647dc0b9fb26633677d"
EXPECTED_ADAPTER_SHA256 = "caf96fa26865c204ec56c9cb9075badb2d813d8ec16241704af6b9e23b07390e"
EXPECTED_SCOPE_CONTRACT_SHA256 = "be643711b689315189ea45b08d01c8978a3a391807e9bcac8d971e55d448c286"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_database_checksum_table(
    table_file: Path,
) -> dict[str, str]:
    records: dict[str, str] = {}

    with table_file.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.rstrip("\n")
            if not value:
                continue

            parts = value.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    "Invalid database checksum row "
                    f"{line_number}: expected two tab-separated fields"
                )

            checksum, relative_name = parts
            checksum = checksum.strip().lower()
            relative_name = relative_name.strip()

            if not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ValueError(f"Invalid SHA-256 at row {line_number}")

            relative_path = Path(relative_name)
            if not relative_name or relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Unsafe database path at row {line_number}: " f"{relative_name!r}"
                )

            if relative_name in records:
                raise ValueError(f"Duplicate database path: {relative_name}")

            records[relative_name] = checksum

    if len(records) != 27:
        raise ValueError(
            "Frozen geNomad database table must contain exactly " f"27 files, found {len(records)}"
        )

    return records


def database_fingerprint(checksums: dict[str, str]) -> str:
    canonical = "".join(
        f"{checksums[relative_name]}\t{relative_name}\n" for relative_name in sorted(checksums)
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_database(
    database_directory: Path,
    expected_checksums: dict[str, str],
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    actual_checksums: dict[str, str] = {}
    mismatches: list[str] = []

    for relative_name in sorted(expected_checksums):
        expected_sha = expected_checksums[relative_name]
        database_file = database_directory / relative_name

        if not database_file.is_file():
            mismatches.append(f"{relative_name}: missing")
            files.append(
                {
                    "relative_path": relative_name,
                    "present": False,
                    "expected_sha256": expected_sha,
                }
            )
            continue

        actual_sha = sha256_file(database_file)
        actual_checksums[relative_name] = actual_sha

        if actual_sha != expected_sha:
            mismatches.append(f"{relative_name}: SHA-256 mismatch")

        files.append(
            {
                "relative_path": relative_name,
                "present": True,
                "size_bytes": database_file.stat().st_size,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            }
        )

    actual_fingerprint = (
        database_fingerprint(actual_checksums)
        if len(actual_checksums) == len(expected_checksums)
        else ""
    )

    if actual_fingerprint != EXPECTED_DATABASE_FINGERPRINT:
        mismatches.append("Aggregate database fingerprint does not match " "the frozen value")

    return {
        "valid": not mismatches,
        "database_version": DATABASE_VERSION,
        "expected_fingerprint_sha256": (EXPECTED_DATABASE_FINGERPRINT),
        "actual_fingerprint_sha256": actual_fingerprint,
        "expected_files": len(expected_checksums),
        "verified_files": len(actual_checksums),
        "mismatches": mismatches,
        "files": files,
    }


def fasta_statistics(input_fasta: Path) -> dict[str, int]:
    records = load_fasta_headers(input_fasta)
    bases = 0
    seen_header = False

    with input_fasta.open() as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            if value.startswith(">"):
                seen_header = True
                continue
            if not seen_header:
                raise ValueError("FASTA sequence data appeared before a header")
            bases += len(value)

    return {
        "sequences": len(records),
        "bases": bases,
    }


def get_tool_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = "\n".join(value for value in (completed.stdout, completed.stderr) if value)

    match = re.search(
        r"geNomad,\s+version\s+([0-9.]+)",
        combined,
    )

    if completed.returncode != 0 or match is None:
        raise RuntimeError("Unable to determine geNomad version: " + combined.strip())

    version = match.group(1)
    if version != TOOL_VERSION:
        raise RuntimeError(
            f"geNomad version {version} does not match " f"the frozen version {TOOL_VERSION}"
        )

    return version


def prepare_output_directory(output_directory: Path) -> None:
    if output_directory.exists():
        raise FileExistsError(
            "Refusing to overwrite existing manuscript benchmark " f"output: {output_directory}"
        )
    output_directory.mkdir(parents=True)


def build_command(
    executable: Path,
    input_fasta: Path,
    raw_output_directory: Path,
    database_directory: Path,
    threads: int,
) -> list[str]:
    if threads < 1:
        raise ValueError("Thread count must be at least one")

    return [
        "/usr/bin/caffeinate",
        "-i",
        "/usr/bin/time",
        "-l",
        str(executable),
        "end-to-end",
        str(input_fasta),
        str(raw_output_directory),
        str(database_directory),
        "--threads",
        str(threads),
        "--splits",
        str(MMSEQS_SPLITS),
        "--enable-score-calibration",
        "--composition",
        "auto",
        "--force-auto",
    ]


def parse_resource_usage(
    stderr_text: str,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}

    timing = re.search(
        r"^\s*([0-9.]+)\s+real\s+([0-9.]+)\s+user\s+" r"([0-9.]+)\s+sys\s*$",
        stderr_text,
        flags=re.MULTILINE,
    )
    if timing:
        metrics["time_reported_real_seconds"] = float(timing.group(1))
        metrics["user_seconds"] = float(timing.group(2))
        metrics["system_seconds"] = float(timing.group(3))

    resident = re.search(
        r"^\s*(\d+)\s+maximum resident set size\s*$",
        stderr_text,
        flags=re.MULTILINE,
    )
    if resident:
        metrics["peak_rss_bytes"] = int(resident.group(1))

    footprint = re.search(
        r"^\s*(\d+)\s+peak memory footprint\s*$",
        stderr_text,
        flags=re.MULTILINE,
    )
    if footprint:
        metrics["peak_memory_footprint_bytes"] = int(footprint.group(1))

    gnu_resident = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        stderr_text,
    )
    if gnu_resident and "peak_rss_bytes" not in metrics:
        metrics["peak_rss_bytes"] = int(gnu_resident.group(1)) * 1024

    return metrics


def safe_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def validate_resource_usage(
    resource_usage: dict[str, float | int],
) -> None:
    for name, value in resource_usage.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite resource measurement: {name}")
        if value < 0:
            raise ValueError(f"Negative resource measurement: {name}")


def package_inventory(
    environment_prefix: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    conda_metadata = environment_prefix / "conda-meta"

    if not conda_metadata.is_dir():
        return records

    for metadata_file in sorted(conda_metadata.glob("*.json")):
        try:
            metadata = json.loads(metadata_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        records.append(
            {
                "name": str(metadata.get("name", "")),
                "version": str(metadata.get("version", "")),
                "build": str(metadata.get("build", "")),
                "metadata_sha256": sha256_file(metadata_file),
            }
        )

    return records


def artifact_inventory(
    directory: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not directory.is_dir():
        return records

    for artifact_file in sorted(directory.rglob("*")):
        if not artifact_file.is_file():
            continue

        records.append(
            {
                "relative_path": str(artifact_file.relative_to(directory)),
                "size_bytes": artifact_file.stat().st_size,
                "sha256": sha256_file(artifact_file),
            }
        )

    return records


def host_inventory() -> dict[str, Any]:
    memory_bytes = 0
    try:
        memory_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        memory_bytes = 0

    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": memory_bytes,
    }


def find_single_output(
    raw_output_directory: Path,
    pattern: str,
    description: str,
) -> Path:
    matches = sorted(raw_output_directory.rglob(pattern))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {description}, found "
            f"{len(matches)} using pattern {pattern!r}"
        )

    return matches[0]


def discover_genomad_outputs(
    raw_output_directory: Path,
) -> dict[str, Path]:
    return {
        "plasmid_summary": find_single_output(
            raw_output_directory,
            "*_plasmid_summary.tsv",
            "geNomad plasmid summary",
        ),
        "virus_summary": find_single_output(
            raw_output_directory,
            "*_virus_summary.tsv",
            "geNomad virus summary",
        ),
        "calibrated_scores": find_single_output(
            raw_output_directory,
            "*_calibrated_aggregated_classification.tsv",
            "geNomad calibrated classification table",
        ),
    }


def write_failed_standardized_output(
    input_fasta: Path,
    output_directory: Path,
    failure_reason: str,
) -> dict[str, Any]:
    records = load_fasta_headers(input_fasta)
    standardized_path = output_directory / "standardized_predictions.tsv"
    metadata_path = output_directory / "adapter_metadata.json"

    rows: list[dict[str, str]] = []

    for contig_id, input_header in records:
        row = {field: "" for field in OUTPUT_FIELDS}
        row.update(
            {
                "contig_id": contig_id,
                "input_header": input_header,
                "predicted_label": "unclassified",
                "prediction_status": "tool_failed",
                "source_tool": TOOL_NAME,
                "source_version": TOOL_VERSION,
            }
        )
        rows.append(row)

    with standardized_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata: dict[str, Any] = {
        "schema_version": "nar-comparator-adapter-v1",
        "runner_schema_version": RUNNER_SCHEMA,
        "scope": SCOPE,
        "source_tool": TOOL_NAME,
        "source_version": TOOL_VERSION,
        "database_version": DATABASE_VERSION,
        "database_fingerprint_sha256": (EXPECTED_DATABASE_FINGERPRINT),
        "scope_contract_sha256": (EXPECTED_SCOPE_CONTRACT_SHA256),
        "input_fasta": str(input_fasta),
        "input_fasta_sha256": sha256_file(input_fasta),
        "standardized_output": str(standardized_path),
        "standardized_output_sha256": (sha256_file(standardized_path)),
        "input_sequences": len(records),
        "standardized_rows": len(rows),
        "label_counts": {"unclassified": len(rows)},
        "status_counts": {"tool_failed": len(rows)},
        "failure_reason": failure_reason,
        "successful_tool_output": False,
        "confirmatory_tuning": False,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def standardize_genomad_output(
    input_fasta: Path,
    raw_output_directory: Path,
    output_directory: Path,
) -> tuple[bool, dict[str, Any], dict[str, str], str]:
    try:
        raw_outputs = discover_genomad_outputs(raw_output_directory)

        metadata = adapt_genomad(
            input_fasta=input_fasta,
            plasmid_summary=raw_outputs["plasmid_summary"],
            virus_summary=raw_outputs["virus_summary"],
            calibrated_scores=raw_outputs["calibrated_scores"],
            output_path=(output_directory / "standardized_predictions.tsv"),
            metadata_output=(output_directory / "adapter_metadata.json"),
        )

        raw_output_strings = {name: str(file_path) for name, file_path in raw_outputs.items()}
        return True, metadata, raw_output_strings, ""

    except Exception as error:
        failure_reason = f"{type(error).__name__}: {error}"
        metadata = write_failed_standardized_output(
            input_fasta=input_fasta,
            output_directory=output_directory,
            failure_reason=failure_reason,
        )
        return False, metadata, {}, failure_reason


def execute_command(
    command: list[str],
    environment: dict[str, str],
    working_directory: Path,
    timeout_seconds: int,
) -> tuple[int, bool, str, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        cwd=working_directory,
        start_new_session=True,
    )

    timed_out = False

    try:
        stdout_text, stderr_text = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            stdout_text, stderr_text = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout_text, stderr_text = process.communicate()

        stderr_text = safe_text(stderr_text)
        stderr_text += f"\nRunner timeout after {timeout_seconds} seconds.\n"

    return (
        124 if timed_out else int(process.returncode or 0),
        timed_out,
        safe_text(stdout_text),
        safe_text(stderr_text),
    )


def run_genomad(
    input_fasta: Path,
    output_directory: Path,
    environment_prefix: Path,
    database_directory: Path,
    checksum_table: Path,
    scope_contract: Path,
    threads: int = 4,
    timeout_seconds: int = 86400,
) -> dict[str, Any]:
    input_fasta = input_fasta.resolve()
    output_directory = output_directory.resolve()
    environment_prefix = environment_prefix.resolve()
    database_directory = database_directory.resolve()
    checksum_table = checksum_table.resolve()
    scope_contract = scope_contract.resolve()

    if platform.system() != "Darwin":
        raise RuntimeError(
            "The frozen manuscript runner requires the " "preregistered macOS execution host"
        )

    if timeout_seconds < 1:
        raise ValueError("Timeout must be at least one second")

    if threads < 1:
        raise ValueError("Thread count must be at least one")

    if not input_fasta.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    if output_directory.exists():
        raise FileExistsError(
            "Refusing to overwrite existing manuscript benchmark " f"output: {output_directory}"
        )

    executable = environment_prefix / "bin" / "genomad"
    if not executable.is_file():
        raise FileNotFoundError(f"geNomad executable not found: {executable}")

    if not database_directory.is_dir():
        raise FileNotFoundError(f"geNomad database not found: {database_directory}")

    if not checksum_table.is_file():
        raise FileNotFoundError(f"Database checksum table not found: {checksum_table}")

    if not scope_contract.is_file():
        raise FileNotFoundError(f"Manuscript scope contract not found: {scope_contract}")

    if sha256_file(checksum_table) != EXPECTED_DATABASE_TABLE_SHA256:
        raise RuntimeError(
            "Database checksum table does not match " "the frozen manuscript contract"
        )

    if sha256_file(scope_contract) != EXPECTED_SCOPE_CONTRACT_SHA256:
        raise RuntimeError(
            "Runner scope contract does not match " "the frozen manuscript-only contract"
        )

    adapter_file = Path(__file__).resolve().parents[1] / "adapters" / "genomad.py"
    if sha256_file(adapter_file) != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("geNomad adapter does not match the frozen contract")

    tool_version = get_tool_version(executable)
    expected_database = load_database_checksum_table(checksum_table)

    table_fingerprint = database_fingerprint(expected_database)
    if table_fingerprint != EXPECTED_DATABASE_FINGERPRINT:
        raise RuntimeError(
            "Checksum-table fingerprint does not match " "the frozen geNomad database"
        )

    pre_database = verify_database(
        database_directory,
        expected_database,
    )
    if not pre_database["valid"]:
        raise RuntimeError(
            "Pre-run geNomad database verification failed: " + "; ".join(pre_database["mismatches"])
        )

    prepare_output_directory(output_directory)

    raw_output_directory = output_directory / "raw_tool_output"
    stdout_file = output_directory / "stdout.log"
    stderr_file = output_directory / "stderr.log"
    command_file = output_directory / "command.json"
    environment_file = output_directory / "environment.json"

    command = build_command(
        executable=executable,
        input_fasta=input_fasta,
        raw_output_directory=raw_output_directory,
        database_directory=database_directory,
        threads=threads,
    )

    command_file.write_text(
        json.dumps(
            {
                "scope": SCOPE,
                "argv": command,
                "cwd": str(Path.cwd().resolve()),
                "shell": False,
            },
            indent=2,
        )
        + "\n"
    )

    environment = os.environ.copy()
    environment["PATH"] = str(environment_prefix / "bin") + os.pathsep + environment.get("PATH", "")
    environment["CONDA_PREFIX"] = str(environment_prefix)

    environment_file.write_text(
        json.dumps(
            {
                "scope": SCOPE,
                "environment_prefix": str(environment_prefix),
                "packages": package_inventory(environment_prefix),
                "host": host_inventory(),
            },
            indent=2,
        )
        + "\n"
    )

    fasta_info = fasta_statistics(input_fasta)
    started_at = utc_now()
    started = time.monotonic()

    (
        child_returncode,
        timed_out,
        stdout_text,
        stderr_text,
    ) = execute_command(
        command=command,
        environment=environment,
        working_directory=Path.cwd().resolve(),
        timeout_seconds=timeout_seconds,
    )

    wall_seconds = time.monotonic() - started
    finished_at = utc_now()

    stdout_file.write_text(stdout_text)
    stderr_file.write_text(stderr_text)

    post_database = verify_database(
        database_directory,
        expected_database,
    )

    (
        adapter_succeeded,
        adapter_metadata,
        discovered_outputs,
        adapter_error,
    ) = standardize_genomad_output(
        input_fasta=input_fasta,
        raw_output_directory=raw_output_directory,
        output_directory=output_directory,
    )

    resource_usage = parse_resource_usage(stderr_text)
    validate_resource_usage(resource_usage)

    throughput = fasta_info["bases"] / wall_seconds if wall_seconds > 0 else 0.0
    if not math.isfinite(throughput) or throughput < 0:
        raise ValueError("Invalid throughput measurement")

    failure_reasons: list[str] = []

    if child_returncode != 0:
        failure_reasons.append(f"geNomad return code: {child_returncode}")
    if timed_out:
        failure_reasons.append("geNomad timed out")
    if not adapter_succeeded:
        failure_reasons.append(f"adapter failure: {adapter_error}")
    if not post_database["valid"]:
        failure_reasons.append("post-run database verification failed")

    run_ok = (
        child_returncode == 0
        and not timed_out
        and adapter_succeeded
        and bool(pre_database["valid"])
        and bool(post_database["valid"])
    )

    standardized_file = output_directory / "standardized_predictions.tsv"
    adapter_metadata_file = output_directory / "adapter_metadata.json"

    manifest: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA,
        "scope": SCOPE,
        "production_workflow_component": False,
        "run_status": "ok" if run_ok else "failed",
        "failure_reasons": failure_reasons,
        "source_tool": TOOL_NAME,
        "source_version": tool_version,
        "input": {
            "fasta": str(input_fasta),
            "sha256": sha256_file(input_fasta),
            **fasta_info,
        },
        "execution": {
            "command": command,
            "shell": False,
            "environment_prefix": str(environment_prefix),
            "executable": str(executable),
            "threads": threads,
            "splits": MMSEQS_SPLITS,
            "score_calibration": True,
            "composition": "auto",
            "force_auto": True,
            "custom_thresholds": False,
            "timeout_seconds": timeout_seconds,
            "timed_out": timed_out,
            "returncode": child_returncode,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": wall_seconds,
            "throughput_bases_per_second": throughput,
            "resource_usage": resource_usage,
        },
        "database": {
            "directory": str(database_directory),
            "version": DATABASE_VERSION,
            "checksum_table": str(checksum_table),
            "checksum_table_sha256": (sha256_file(checksum_table)),
            "pre_run": pre_database,
            "post_run": post_database,
        },
        "contracts": {
            "scope_contract": str(scope_contract),
            "scope_contract_sha256": (sha256_file(scope_contract)),
            "adapter_source": str(adapter_file),
            "adapter_source_sha256": (sha256_file(adapter_file)),
        },
        "adapter": {
            "succeeded": adapter_succeeded,
            "error": adapter_error,
            "discovered_raw_outputs": discovered_outputs,
            "metadata": adapter_metadata,
        },
        "artifacts": {
            "raw_tool_output": artifact_inventory(raw_output_directory),
            "stdout_sha256": sha256_file(stdout_file),
            "stderr_sha256": sha256_file(stderr_file),
            "command_sha256": sha256_file(command_file),
            "environment_sha256": sha256_file(environment_file),
            "standardized_predictions_sha256": (sha256_file(standardized_file)),
            "adapter_metadata_sha256": (sha256_file(adapter_metadata_file)),
        },
        "host": host_inventory(),
        "confirmatory_tuning": False,
    }

    manifest_file = output_directory / "run_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen manuscript-only geNomad comparator "
            "with full provenance capture. This command is not "
            "part of the PlasFlow2 prediction workflow."
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
        "--environment-prefix",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--database-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--database-checksums",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--scope-contract",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=86400,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        manifest = run_genomad(
            input_fasta=args.input_fasta,
            output_directory=args.output_dir,
            environment_prefix=args.environment_prefix,
            database_directory=args.database_directory,
            checksum_table=args.database_checksums,
            scope_contract=args.scope_contract,
            threads=args.threads,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as error:
        print(f"ERROR: {error}")
        return 2

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["run_status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
