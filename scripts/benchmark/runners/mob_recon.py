#!/usr/bin/env python3
"""Frozen, provenance-preserving MOB-recon publication runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.benchmark.adapters.mob_recon import adapt_mob_recon

TOOL_NAME = "MOB-recon"
TOOL_VERSION = "3.1.9"
RUNNER_SCHEMA = "nar-mob-recon-runner-v1"

EXPECTED_DATABASE_FINGERPRINT = "4c168e1c05266be7509d6313525f428eabb2656f516d9740f06290d2f30711e1"
EXPECTED_DATABASE_TABLE_SHA256 = "603d4ccb2d57f98bfe7234d89007344aa6e7a2240d13a32f3e87d3075fa60c58"
EXPECTED_ADAPTER_SHA256 = "8c48815b7ad850bfad4302ffa72b5b83e45e9bd1d63e84d5a98ed63eb22d84e5"


def sha256_file(file_name: Path) -> str:
    digest = hashlib.sha256()
    with file_name.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_checksum_table(
    table_file: Path,
) -> dict[str, tuple[int, str]]:
    records: dict[str, tuple[int, str]] = {}

    with table_file.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"filename", "size_bytes", "sha256"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "Database checksum table is missing columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            filename = (row.get("filename") or "").strip()
            size_text = (row.get("size_bytes") or "").strip()
            checksum = (row.get("sha256") or "").strip().lower()

            if not filename:
                raise ValueError("Empty database filename is not allowed")

            if Path(filename).name != filename:
                raise ValueError(f"Unsafe database filename in checksum table: {filename}")

            if filename in records:
                raise ValueError(f"Duplicate database filename in checksum table: {filename}")

            try:
                size_bytes = int(size_text)
            except ValueError as exc:
                raise ValueError(f"Invalid size for database file {filename}: {size_text}") from exc

            if size_bytes < 0:
                raise ValueError(f"Negative size for database file {filename}")

            if not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ValueError(f"Invalid SHA-256 for database file {filename}")

            records[filename] = (size_bytes, checksum)

    if not records:
        raise ValueError("Database checksum table is empty")

    return records


def database_fingerprint(
    checksums: dict[str, str],
) -> str:
    source = "\n".join(f"{filename}\t{checksums[filename]}" for filename in sorted(checksums))
    return hashlib.sha256(source.encode()).hexdigest()


def verify_database(
    database_directory: Path,
    expected: dict[str, tuple[int, str]],
    expected_fingerprint: str,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    actual_checksums: dict[str, str] = {}
    mismatches: list[str] = []

    for filename in sorted(expected):
        expected_size, expected_sha = expected[filename]
        database_file = database_directory / filename

        if not database_file.is_file():
            mismatches.append(f"{filename}: missing")
            files.append(
                {
                    "filename": filename,
                    "present": False,
                    "expected_size_bytes": expected_size,
                    "expected_sha256": expected_sha,
                }
            )
            continue

        actual_size = database_file.stat().st_size
        actual_sha = sha256_file(database_file)
        actual_checksums[filename] = actual_sha

        if actual_size != expected_size:
            mismatches.append(f"{filename}: size {actual_size} != {expected_size}")

        if actual_sha != expected_sha:
            mismatches.append(f"{filename}: SHA-256 mismatch")

        files.append(
            {
                "filename": filename,
                "present": True,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            }
        )

    fingerprint = (
        database_fingerprint(actual_checksums) if len(actual_checksums) == len(expected) else ""
    )

    if fingerprint != expected_fingerprint:
        mismatches.append("aggregate database fingerprint does not match the frozen value")

    return {
        "valid": not mismatches,
        "expected_fingerprint_sha256": expected_fingerprint,
        "actual_fingerprint_sha256": fingerprint,
        "expected_files": len(expected),
        "verified_files": sum(item.get("present", False) for item in files),
        "mismatches": mismatches,
        "files": files,
    }


def parse_resource_usage(stderr_text: str) -> dict[str, float | int]:
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

    peak_footprint = re.search(
        r"^\s*(\d+)\s+peak memory footprint\s*$",
        stderr_text,
        flags=re.MULTILINE,
    )

    if peak_footprint:
        metrics["peak_memory_footprint_bytes"] = int(peak_footprint.group(1))

    gnu_rss = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        stderr_text,
    )

    if gnu_rss and "peak_rss_bytes" not in metrics:
        metrics["peak_rss_bytes"] = int(gnu_rss.group(1)) * 1024

    return metrics


def fasta_statistics(input_fasta: Path) -> dict[str, int]:
    sequences = 0
    bases = 0
    seen_header = False

    with input_fasta.open() as handle:
        for line in handle:
            value = line.strip()

            if not value:
                continue

            if value.startswith(">"):
                sequences += 1
                seen_header = True
                continue

            if not seen_header:
                raise ValueError("FASTA sequence data appeared before the first header")

            bases += len(value)

    if sequences == 0:
        raise ValueError(f"No FASTA records found in {input_fasta}")

    return {
        "sequences": sequences,
        "bases": bases,
    }


def get_tool_version(mob_executable: Path) -> str:
    completed = subprocess.run(
        [str(mob_executable), "-V"],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = "\n".join(value for value in (completed.stdout, completed.stderr) if value)

    match = re.search(r"mob_recon\s+([0-9.]+)", combined)

    if completed.returncode != 0 or match is None:
        raise RuntimeError("Unable to determine the MOB-recon version: " + combined.strip())

    version = match.group(1)

    if version != TOOL_VERSION:
        raise RuntimeError(
            f"MOB-recon version {version} does not match " f"the frozen version {TOOL_VERSION}"
        )

    return version


def prepare_output_directory(output_directory: Path) -> None:
    if output_directory.exists():
        raise FileExistsError(
            "Refusing to overwrite existing publication output: " f"{output_directory}"
        )

    output_directory.mkdir(parents=True)


def build_command(
    mob_executable: Path,
    input_fasta: Path,
    raw_output_directory: Path,
    database_directory: Path,
    threads: int,
    sample_id: str,
) -> list[str]:
    if threads < 1:
        raise ValueError("Thread count must be at least one")

    return [
        "/usr/bin/caffeinate",
        "-i",
        "/usr/bin/time",
        "-l",
        str(mob_executable),
        "--infile",
        str(input_fasta),
        "--outdir",
        str(raw_output_directory),
        "--database_directory",
        str(database_directory),
        "--num_threads",
        str(threads),
        "--sample_id",
        sample_id,
    ]


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def package_inventory(
    environment_prefix: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    conda_meta = environment_prefix / "conda-meta"

    if not conda_meta.is_dir():
        return records

    for metadata_file in sorted(conda_meta.glob("*.json")):
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


def standardize_output(
    input_fasta: Path,
    raw_output_directory: Path,
    output_directory: Path,
) -> tuple[bool, dict[str, Any]]:
    contig_report = raw_output_directory / "contig_report.txt"
    report_was_emitted = contig_report.is_file()

    if not report_was_emitted:
        raw_output_directory.mkdir(parents=True, exist_ok=True)
        contig_report = raw_output_directory / "runner_empty_contig_report.tsv"
        contig_report.write_text("contig_id\tmolecule_type\n")

    standardized = output_directory / "standardized_predictions.tsv"
    adapter_metadata = output_directory / "adapter_metadata.json"

    metadata = adapt_mob_recon(
        input_fasta=input_fasta,
        contig_report=contig_report,
        output_path=standardized,
        metadata_output=adapter_metadata,
    )

    return report_was_emitted, metadata


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


def run_mob_recon(
    input_fasta: Path,
    output_directory: Path,
    environment_prefix: Path,
    database_directory: Path,
    checksum_table: Path,
    threads: int,
    sample_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    input_fasta = input_fasta.resolve()
    output_directory = output_directory.resolve()
    environment_prefix = environment_prefix.resolve()
    database_directory = database_directory.resolve()
    checksum_table = checksum_table.resolve()

    if platform.system() != "Darwin":
        raise RuntimeError(
            "The frozen publication runner requires the preregistered " "macOS execution host"
        )

    if timeout_seconds < 1:
        raise ValueError("Timeout must be at least one second")

    if not input_fasta.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    mob_executable = environment_prefix / "bin" / "mob_recon"

    if not mob_executable.is_file():
        raise FileNotFoundError(f"MOB-recon executable not found: {mob_executable}")

    if not database_directory.is_dir():
        raise FileNotFoundError(f"MOB-recon database not found: {database_directory}")

    if not checksum_table.is_file():
        raise FileNotFoundError(f"Database checksum table not found: {checksum_table}")

    if sha256_file(checksum_table) != (EXPECTED_DATABASE_TABLE_SHA256):
        raise RuntimeError("Database checksum table does not match the frozen contract")

    adapter_file = Path(__file__).resolve().parents[1] / "adapters" / "mob_recon.py"

    if sha256_file(adapter_file) != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("MOB-recon adapter does not match the frozen contract")

    tool_version = get_tool_version(mob_executable)
    expected_database = load_checksum_table(checksum_table)

    expected_table_fingerprint = database_fingerprint(
        {filename: checksum for filename, (_, checksum) in expected_database.items()}
    )

    if expected_table_fingerprint != EXPECTED_DATABASE_FINGERPRINT:
        raise RuntimeError(
            "Checksum-table fingerprint does not match the frozen " "database fingerprint"
        )

    pre_database = verify_database(
        database_directory,
        expected_database,
        EXPECTED_DATABASE_FINGERPRINT,
    )

    if not pre_database["valid"]:
        raise RuntimeError(
            "Pre-run MOB-recon database verification failed: "
            + "; ".join(pre_database["mismatches"])
        )

    prepare_output_directory(output_directory)

    raw_output_directory = output_directory / "raw_tool_output"
    stdout_file = output_directory / "stdout.log"
    stderr_file = output_directory / "stderr.log"
    command_file = output_directory / "command.json"
    environment_file = output_directory / "environment.json"

    command = build_command(
        mob_executable=mob_executable,
        input_fasta=input_fasta,
        raw_output_directory=raw_output_directory,
        database_directory=database_directory,
        threads=threads,
        sample_id=sample_id,
    )

    command_file.write_text(
        json.dumps(
            {
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
    timed_out = False
    child_returncode = 0
    stdout_text = ""
    stderr_text = ""

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            cwd=Path.cwd(),
            timeout=timeout_seconds,
        )
        child_returncode = completed.returncode
        stdout_text = completed.stdout
        stderr_text = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        child_returncode = 124
        stdout_text = _as_text(exc.stdout)
        stderr_text = _as_text(exc.stderr)
        stderr_text += f"\nRunner timeout after {timeout_seconds} seconds.\n"

    wall_seconds = time.monotonic() - started
    finished_at = utc_now()

    stdout_file.write_text(stdout_text)
    stderr_file.write_text(stderr_text)

    post_database = verify_database(
        database_directory,
        expected_database,
        EXPECTED_DATABASE_FINGERPRINT,
    )

    report_was_emitted, adapter_metadata = standardize_output(
        input_fasta,
        raw_output_directory,
        output_directory,
    )

    resource_usage = parse_resource_usage(stderr_text)
    throughput = fasta_info["bases"] / wall_seconds if wall_seconds > 0 else 0.0

    run_ok = (
        child_returncode == 0
        and not timed_out
        and report_was_emitted
        and bool(pre_database["valid"])
        and bool(post_database["valid"])
    )

    manifest: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA,
        "run_status": "ok" if run_ok else "failed",
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
            "executable": str(mob_executable),
            "threads": threads,
            "sample_id": sample_id,
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
            "checksum_table": str(checksum_table),
            "checksum_table_sha256": sha256_file(checksum_table),
            "pre_run": pre_database,
            "post_run": post_database,
        },
        "adapter": {
            "source": str(adapter_file),
            "source_sha256": sha256_file(adapter_file),
            "report_was_emitted": report_was_emitted,
            "metadata": adapter_metadata,
        },
        "artifacts": {
            "raw_tool_output": artifact_inventory(raw_output_directory),
            "stdout_sha256": sha256_file(stdout_file),
            "stderr_sha256": sha256_file(stderr_file),
            "command_sha256": sha256_file(command_file),
            "environment_sha256": sha256_file(environment_file),
            "standardized_predictions_sha256": sha256_file(
                output_directory / "standardized_predictions.tsv"
            ),
            "adapter_metadata_sha256": sha256_file(output_directory / "adapter_metadata.json"),
        },
        "host": host_inventory(),
    }

    manifest_file = output_directory / "run_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--sample-id",
        default="mob_recon_benchmark",
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
        manifest = run_mob_recon(
            input_fasta=args.input_fasta,
            output_directory=args.output_dir,
            environment_prefix=args.environment_prefix,
            database_directory=args.database_directory,
            checksum_table=args.database_checksums,
            threads=args.threads,
            sample_id=args.sample_id,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["run_status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
