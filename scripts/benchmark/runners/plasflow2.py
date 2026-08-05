#!/usr/bin/env python3
"""Run frozen PlasFlow2 predictions for manuscript-only benchmarking.

This runner is not part of the production PlasFlow2 workflow. It executes one
complete frozen cohort without reading labels or calculating performance.
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
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = PROJECT_ROOT / (
    "scripts/benchmark/contracts/plasflow2_confirmatory_runner_contract_v1.json"
)
RUNNER_CONTRACT_SHA256 = "4317560cc67b2155ede9689b8e2d0256419b2cb1e01e3e45a225c5deeee0e8db"
RUNNER_CONTRACT_FILE_SHA256 = "19e6c4a288a406c7dce7320b8855837904d1442356cae04ecdee5b3ab5b0ee89"
FINAL_COHORT_SHA256 = "6edf2e3b55b672baebbd833e588ed2ce0a507fb300b4b76d6aa85d3209e7d556"
MODEL_SHA256 = "3564913c14eaad068a217572f9641907fa8c86b1527abbb2f855b0df4bf3cb23"
EXPECTED_ENVIRONMENT = Path("/Users/shahbazraza/miniconda3/envs/plasflow2")
EXPECTED_PROFILE = "balanced"
EXPECTED_THRESHOLD_POLICY = "rev5-balanced-p080-20260724-v1"
EXPECTED_COHORT_ROLE = "confirmatory"
EXPECTED_RECORDS = 3000
EXPECTED_BASES = 186076925
MINIMUM_LENGTH = 1000
MAXIMUM_LENGTH = 500000
TIMEOUT_SECONDS = 86400
ALLOWED_LABELS = frozenset({"plasmid", "chromosome", "phage", "unclassified"})
CLASS_LABELS = ("plasmid", "chromosome", "phage")
CLASS_TERMS = re.compile(r"plasmid|chromosome|phage", flags=re.IGNORECASE)


class RunnerContractError(RuntimeError):
    """Raised when a frozen runner requirement is violated."""


@dataclass(frozen=True)
class InputRecord:
    """Prediction-facing FASTA record metadata."""

    contig_id: str
    header: str
    length: int
    sequence_sha256: str


@dataclass(frozen=True)
class ProcessResult:
    """Official process outcome."""

    return_code: int
    wallclock_seconds: float
    timed_out: bool
    first_signal: str | None
    final_signal: str | None


def utc_now() -> str:
    """Return a stable UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    """Reproduce the embedded hash of a canonical JSON contract."""
    payload = dict(value)
    payload.pop("contract_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write deterministic human-readable JSON."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object with a contract-focused error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerContractError(f"Unable to read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise RunnerContractError(f"Expected a JSON object: {path}")
    return value


def load_and_validate_contract() -> dict[str, Any]:
    """Load the tracked contract and validate its immutable identity."""
    if not CONTRACT_PATH.is_file():
        raise RunnerContractError(f"Tracked runner contract is missing: {CONTRACT_PATH}")
    if sha256_file(CONTRACT_PATH) != RUNNER_CONTRACT_FILE_SHA256:
        raise RunnerContractError("Tracked runner contract file identity changed")

    value = load_json_object(CONTRACT_PATH)
    declared = value.get("contract_sha256")
    recalculated = canonical_hash(value)
    if declared != RUNNER_CONTRACT_SHA256 or recalculated != RUNNER_CONTRACT_SHA256:
        raise RunnerContractError(
            "Tracked runner contract canonical identity does not match the frozen value"
        )
    if value.get("status") != "FROZEN":
        raise RunnerContractError("Runner contract status must be FROZEN")

    authorization = value.get("authorization", {})
    if authorization.get("confirmatory_predictions_authorized") is not False:
        raise RunnerContractError("Runner contract must not self-authorize predictions")
    if authorization.get("ground_truth_performance_label_release_authorized") is not False:
        raise RunnerContractError("Runner contract must keep performance labels sealed")
    return value


def require_file_identity(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    """Require one frozen file identity and return provenance."""
    if not path.is_file():
        raise RunnerContractError(f"Frozen {label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RunnerContractError(
            f"Frozen {label} identity mismatch: {observed}; expected {expected_sha256}"
        )
    return {
        "path": relative(path),
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def resolve_contract_path(raw_path: str) -> Path:
    """Resolve a contract path against the project root."""
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify_runtime_identities(
    contract: dict[str, Any],
    environment_prefix: Path,
) -> dict[str, Any]:
    """Verify model, environment, source, evaluator, and upstream contracts."""
    if environment_prefix.resolve() != EXPECTED_ENVIRONMENT.resolve():
        raise RunnerContractError(
            f"Environment prefix mismatch: {environment_prefix}; "
            f"expected {EXPECTED_ENVIRONMENT}"
        )

    tool = contract.get("tool", {})
    model_contract = contract.get("model_contract", {})
    evaluation_contract = contract.get("evaluation_contract", {})
    upstream = contract.get("upstream_contracts", {})

    identities: dict[str, Any] = {}
    identities["cli_executable"] = require_file_identity(
        Path(str(tool.get("cli_executable", ""))),
        str(tool.get("cli_executable_sha256", "")),
        "CLI executable",
    )
    identities["python_executable"] = require_file_identity(
        Path(str(tool.get("python_executable", ""))),
        str(tool.get("python_executable_sha256", "")),
        "Python executable",
    )
    identities["model"] = require_file_identity(
        resolve_contract_path(str(model_contract.get("path", ""))),
        MODEL_SHA256,
        "model",
    )
    identities["model_manifest"] = require_file_identity(
        resolve_contract_path(str(model_contract.get("manifest_path", ""))),
        str(model_contract.get("manifest_file_sha256", "")),
        "model manifest",
    )
    identities["evaluator"] = require_file_identity(
        resolve_contract_path(str(evaluation_contract.get("path", ""))),
        str(evaluation_contract.get("sha256", "")),
        "evaluator",
    )

    upstream_specs = {
        "protocol": ("protocol_path", "protocol_file_sha256", False),
        "endpoint_policy": (
            "endpoint_policy_path",
            "endpoint_policy_contract_sha256",
            True,
        ),
        "cohort_contract": (
            "cohort_contract_path",
            "cohort_contract_sha256",
            True,
        ),
    }
    for name, (path_key, hash_key, canonical) in upstream_specs.items():
        path = resolve_contract_path(str(upstream.get(path_key, "")))
        expected = str(upstream.get(hash_key, ""))
        if canonical:
            value = load_json_object(path)
            observed = canonical_hash(value)
            if value.get("contract_sha256") != expected or observed != expected:
                raise RunnerContractError(f"Frozen {name} canonical identity changed")
            identities[name] = {
                "path": relative(path),
                "contract_sha256": observed,
                "file_sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        else:
            identities[name] = require_file_identity(path, expected, name)

    source_paths = {
        "pyproject": PROJECT_ROOT / "pyproject.toml",
        "package_init": PROJECT_ROOT / "src/plasflow2/__init__.py",
        "cli_source": PROJECT_ROOT / "src/plasflow2/cli.py",
        "predict_source": PROJECT_ROOT / "src/plasflow2/classify/predict.py",
        "features_source": PROJECT_ROOT / "src/plasflow2/classify/features.py",
        "model_source": PROJECT_ROOT / "src/plasflow2/classify/model.py",
        "model_contract_source": PROJECT_ROOT / "src/plasflow2/classify/model_contract.py",
        "threshold_policy_source": (PROJECT_ROOT / "src/plasflow2/classify/threshold_policy.py"),
        "device_source": PROJECT_ROOT / "src/plasflow2/utils/device.py",
        "fasta_source": PROJECT_ROOT / "src/plasflow2/utils/fasta.py",
    }
    frozen_sources = tool.get("source_identities", {})
    if set(frozen_sources) != set(source_paths):
        raise RunnerContractError("Frozen source-identity inventory is incomplete")
    identities["source_files"] = {
        name: require_file_identity(path, str(frozen_sources[name]), name)
        for name, path in source_paths.items()
    }

    manifest = load_json_object(resolve_contract_path(str(model_contract["manifest_path"])))
    if manifest.get("model_sha256") != MODEL_SHA256:
        raise RunnerContractError("Model manifest does not declare the frozen model identity")
    if manifest.get("model_id") != model_contract.get("model_id"):
        raise RunnerContractError("Model manifest identifier changed")
    profiles = manifest.get("profile_threshold_policies", {})
    if profiles.get(EXPECTED_PROFILE) != EXPECTED_THRESHOLD_POLICY:
        raise RunnerContractError("Model manifest no longer pairs the frozen profile and policy")
    return identities


def parse_fasta(path: Path) -> list[InputRecord]:
    """Parse and validate the exact prediction-facing FASTA."""
    records: list[InputRecord] = []
    seen: set[str] = set()
    header: str | None = None
    contig_id: str | None = None
    sequence_digest = hashlib.sha256()
    length = 0

    def finish() -> None:
        nonlocal header, contig_id, sequence_digest, length
        if header is None or contig_id is None:
            return
        if length == 0:
            raise RunnerContractError(f"Input FASTA contains an empty sequence: {contig_id}")
        if not MINIMUM_LENGTH <= length <= MAXIMUM_LENGTH:
            raise RunnerContractError(
                f"Input sequence length is outside the frozen domain: {contig_id}={length}"
            )
        records.append(
            InputRecord(
                contig_id=contig_id,
                header=header,
                length=length,
                sequence_sha256=sequence_digest.hexdigest(),
            )
        )

    with path.open("rt", encoding="utf-8", errors="strict", newline=None) as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                header = line[1:].strip()
                if not header:
                    raise RunnerContractError(f"Empty FASTA header at line {line_number}")
                contig_id = header.split()[0]
                if contig_id in seen:
                    raise RunnerContractError(f"Duplicate canonical identifier: {contig_id}")
                if CLASS_TERMS.search(contig_id):
                    raise RunnerContractError(
                        f"Prediction-facing identifier exposes a class term: {contig_id}"
                    )
                seen.add(contig_id)
                sequence_digest = hashlib.sha256()
                length = 0
                continue
            if header is None:
                raise RunnerContractError(
                    f"Sequence data precede the first FASTA header at line {line_number}"
                )
            sequence = "".join(line.split()).upper()
            if not sequence:
                continue
            if not re.fullmatch(r"[ACGTRYSWKMBDHVN]+", sequence):
                raise RunnerContractError(
                    f"Input FASTA contains an invalid nucleotide at line {line_number}"
                )
            sequence_digest.update(sequence.encode("ascii"))
            length += len(sequence)
    finish()

    if len(records) != EXPECTED_RECORDS:
        raise RunnerContractError(
            f"Input record count mismatch: {len(records)}; expected {EXPECTED_RECORDS}"
        )
    total_bases = sum(record.length for record in records)
    if total_bases != EXPECTED_BASES:
        raise RunnerContractError(
            f"Input base count mismatch: {total_bases}; expected {EXPECTED_BASES}"
        )
    return records


def build_official_command(
    contract: dict[str, Any],
    input_fasta: Path,
    raw_predictions: Path,
) -> list[str]:
    """Construct the exact frozen official command without a shell."""
    command_contract = contract.get("command_contract", {})
    launcher = [str(value) for value in command_contract.get("launcher_prefix", [])]
    arguments = [
        str(value)
        .replace("{input_fasta}", str(input_fasta.resolve()))
        .replace("{raw_predictions_tsv}", str(raw_predictions.resolve()))
        for value in command_contract.get("arguments", [])
    ]
    command = launcher + arguments
    if not command or command[:4] != [
        "/usr/bin/caffeinate",
        "-i",
        "/usr/bin/time",
        "-l",
    ]:
        raise RunnerContractError("Frozen launcher prefix changed")

    required = [
        str(EXPECTED_ENVIRONMENT / "bin/plasflow2"),
        "classify",
        "--profile",
        EXPECTED_PROFILE,
        "--no-marker-model",
    ]
    for token in required:
        if token not in command:
            raise RunnerContractError(f"Official command is missing frozen token: {token}")
    for forbidden in command_contract.get("forbidden_arguments", []):
        if str(forbidden) in command:
            raise RunnerContractError(f"Official command contains forbidden token: {forbidden}")
    return command


def frozen_environment(contract: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Return the isolated process environment and recorded overrides."""
    raw_overrides = contract.get("command_contract", {}).get("environment", {})
    overrides = {str(key): str(value) for key, value in raw_overrides.items()}
    required = {
        "PLASFLOW_USE_MPS": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if overrides != required:
        raise RunnerContractError("Frozen process-environment contract changed")
    environment = os.environ.copy()
    environment.update(overrides)
    environment["PATH"] = str(EXPECTED_ENVIRONMENT / "bin") + os.pathsep + "/usr/bin:/bin"
    return environment, overrides


def run_process(
    command: list[str],
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> ProcessResult:
    """Execute the official command with process-group timeout enforcement."""
    started = time.monotonic()
    timed_out = False
    first_signal: str | None = None
    final_signal: str | None = None
    with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            shell=False,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            first_signal = "SIGTERM"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                final_signal = "SIGKILL"
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()
    return ProcessResult(
        return_code=return_code,
        wallclock_seconds=time.monotonic() - started,
        timed_out=timed_out,
        first_signal=first_signal,
        final_signal=final_signal,
    )


def parse_finite_score(text: str, field: str, contig_id: str) -> float:
    """Parse one finite unit-interval score."""
    try:
        value = float(text)
    except ValueError as error:
        raise RunnerContractError(f"Invalid {field} for {contig_id}: {text!r}") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RunnerContractError(f"{field} is outside [0,1] for {contig_id}: {text!r}")
    return value


def validate_and_standardize_output(
    raw_path: Path,
    standardized_path: Path,
    records: list[InputRecord],
) -> dict[str, Any]:
    """Validate complete native output and create a non-reclassifying table."""
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        raise RunnerContractError("PlasFlow2 produced no non-empty raw prediction table")

    expected_columns = [
        "contig_id",
        "label",
        "confidence",
        "plasmid_score",
        "chromosome_score",
        "phage_score",
    ]
    input_by_id = {record.contig_id: record for record in records}
    observed_rows: list[dict[str, str]] = []
    seen: set[str] = set()

    with raw_path.open("rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != expected_columns:
            raise RunnerContractError(
                f"Raw prediction columns changed: {reader.fieldnames}; expected {expected_columns}"
            )
        for position, row in enumerate(reader):
            contig_id = (row.get("contig_id") or "").strip()
            if not contig_id:
                raise RunnerContractError("Raw prediction contains an empty identifier")
            if contig_id in seen:
                raise RunnerContractError(f"Duplicate raw prediction identifier: {contig_id}")
            if contig_id not in input_by_id:
                raise RunnerContractError(f"Unexpected raw prediction identifier: {contig_id}")
            if position >= len(records) or records[position].contig_id != contig_id:
                raise RunnerContractError("Raw prediction order differs from input FASTA order")
            seen.add(contig_id)

            label = (row.get("label") or "").strip().lower()
            if label not in ALLOWED_LABELS:
                raise RunnerContractError(f"Invalid native label for {contig_id}: {label!r}")
            confidence = parse_finite_score(
                (row.get("confidence") or "").strip(), "confidence", contig_id
            )
            scores = {
                class_name: parse_finite_score(
                    (row.get(f"{class_name}_score") or "").strip(),
                    f"{class_name}_score",
                    contig_id,
                )
                for class_name in CLASS_LABELS
            }
            if abs(sum(scores.values()) - 1.0) > 0.001:
                raise RunnerContractError(f"Class scores do not sum to one for {contig_id}")
            maximum_score = max(scores.values())
            if abs(confidence - maximum_score) > 0.0001:
                raise RunnerContractError(
                    f"Native confidence does not equal maximum class score for {contig_id}"
                )
            if label != "unclassified" and maximum_score - scores[label] > 0.0001:
                raise RunnerContractError(
                    f"Native called label is inconsistent with class scores for {contig_id}"
                )
            observed_rows.append(row)

    expected_ids = [record.contig_id for record in records]
    if len(observed_rows) != len(records) or seen != set(expected_ids):
        missing = sorted(set(expected_ids) - seen)
        raise RunnerContractError(
            f"Incomplete raw predictions: rows={len(observed_rows)}, "
            f"expected={len(records)}, missing={len(missing)}"
        )

    status_by_label = {
        "plasmid": "called_plasmid",
        "chromosome": "called_chromosome",
        "phage": "called_phage",
        "unclassified": "unclassified_abstention",
    }
    standardized_columns = [
        "contig_id",
        "input_header",
        "length",
        "predicted_label",
        "prediction_status",
        "confidence",
        "plasmid_score",
        "chromosome_score",
        "phage_score",
        "source_tool",
        "source_version",
        "profile",
        "threshold_policy",
        "runner_contract_sha256",
    ]
    counts = {label: 0 for label in sorted(ALLOWED_LABELS)}
    with standardized_path.open("wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=standardized_columns, delimiter="\t")
        writer.writeheader()
        for input_record, raw_row in zip(records, observed_rows, strict=True):
            label = str(raw_row["label"]).strip().lower()
            counts[label] += 1
            writer.writerow(
                {
                    "contig_id": input_record.contig_id,
                    "input_header": input_record.header,
                    "length": input_record.length,
                    "predicted_label": label,
                    "prediction_status": status_by_label[label],
                    "confidence": raw_row["confidence"],
                    "plasmid_score": raw_row["plasmid_score"],
                    "chromosome_score": raw_row["chromosome_score"],
                    "phage_score": raw_row["phage_score"],
                    "source_tool": "PlasFlow2",
                    "source_version": "2.1.2",
                    "profile": EXPECTED_PROFILE,
                    "threshold_policy": EXPECTED_THRESHOLD_POLICY,
                    "runner_contract_sha256": RUNNER_CONTRACT_SHA256,
                }
            )
    return {
        "rows": len(observed_rows),
        "label_counts": counts,
        "raw_sha256": sha256_file(raw_path),
        "standardized_sha256": sha256_file(standardized_path),
    }


def write_artifact_checksums(output_dir: Path) -> Path:
    """Write checksums for every retained runner artifact except the checksum file."""
    checksum_path = output_dir / "artifact_checksums.sha256"
    paths = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != checksum_path.name
    )
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths),
        encoding="utf-8",
    )
    return checksum_path


def build_parser() -> argparse.ArgumentParser:
    """Build the manuscript-only runner interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen PlasFlow2 2.1.2 target method on one complete "
            "confirmatory cohort. This command is not part of the production "
            "PlasFlow2 prediction workflow."
        )
    )
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment-prefix", type=Path, required=True)
    parser.add_argument(
        "--cohort-role",
        choices=[EXPECTED_COHORT_ROLE],
        required=True,
    )
    return parser


def execute(args: argparse.Namespace) -> int:
    """Run the complete frozen PlasFlow2 prediction transaction."""
    input_fasta = args.input_fasta.resolve()
    output_dir = args.output_dir.resolve()
    environment_prefix = args.environment_prefix.resolve()

    if output_dir.exists():
        raise RunnerContractError(f"Output directory already exists: {output_dir}")
    if not input_fasta.is_file():
        raise RunnerContractError(f"Input FASTA is missing: {input_fasta}")
    if sha256_file(input_fasta) != FINAL_COHORT_SHA256:
        raise RunnerContractError("Input FASTA does not match the frozen confirmatory cohort")

    contract = load_and_validate_contract()
    before_identities = verify_runtime_identities(contract, environment_prefix)
    records = parse_fasta(input_fasta)

    output_dir.mkdir(parents=True, exist_ok=False)
    raw_predictions = output_dir / "all_predictions.tsv"
    standardized_predictions = output_dir / "standardized_predictions.tsv"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    command_path = output_dir / "command.json"
    resource_path = output_dir / "resource_usage.json"
    provenance_path = output_dir / "runner_provenance.json"

    command = build_official_command(contract, input_fasta, raw_predictions)
    environment, environment_overrides = frozen_environment(contract)
    started_at = utc_now()
    write_json(
        command_path,
        {
            "argv": command,
            "cwd": str(PROJECT_ROOT),
            "environment_overrides": environment_overrides,
            "shell": False,
            "cohort_role": args.cohort_role,
            "started_at_utc": started_at,
        },
    )

    result = run_process(command, environment, stdout_path, stderr_path)
    write_json(resource_path, asdict(result))
    if result.timed_out:
        raise RunnerContractError(
            f"PlasFlow2 exceeded the frozen timeout of {TIMEOUT_SECONDS} seconds"
        )
    if result.return_code != 0:
        raise RunnerContractError(f"PlasFlow2 returned status {result.return_code}")

    output_summary = validate_and_standardize_output(
        raw_predictions,
        standardized_predictions,
        records,
    )
    if sha256_file(input_fasta) != FINAL_COHORT_SHA256:
        raise RunnerContractError("Input FASTA identity changed during execution")
    after_identities = verify_runtime_identities(contract, environment_prefix)
    if before_identities != after_identities:
        raise RunnerContractError("Frozen runtime identities changed during execution")

    provenance = {
        "status": "PASS",
        "scope": "manuscript-only comparative benchmarking",
        "production_workflow_component": False,
        "tool": "PlasFlow2",
        "version": "2.1.2",
        "cohort_role": args.cohort_role,
        "input_fasta": relative(input_fasta),
        "input_sha256": FINAL_COHORT_SHA256,
        "input_records": len(records),
        "input_bases": sum(record.length for record in records),
        "profile": EXPECTED_PROFILE,
        "threshold_policy": EXPECTED_THRESHOLD_POLICY,
        "marker_fusion": False,
        "hallmark_gate": False,
        "compass_filter": False,
        "argmax_fallback": False,
        "external_cohort_sharding": False,
        "runner_contract_sha256": RUNNER_CONTRACT_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "runtime_identities_before": before_identities,
        "runtime_identities_after": after_identities,
        "process": asdict(result),
        "output": output_summary,
        "ground_truth_performance_labels_accessed": False,
        "performance_metrics_calculated": False,
        "completed_at_utc": utc_now(),
    }
    write_json(provenance_path, provenance)
    write_artifact_checksums(output_dir)
    return 0


def main() -> int:
    """Parse arguments, execute the frozen runner, and report failures."""
    args = build_parser().parse_args()
    try:
        return execute(args)
    except (OSError, RunnerContractError) as error:
        print(f"ERROR: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
