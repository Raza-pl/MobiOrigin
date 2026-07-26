"""Cryptographic compatibility contract for PlasFlow MLP models."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

MODEL_CONTRACT_SCHEMA_VERSION = 1
MODEL_ARTIFACT_TYPE = "plasflow-mlp"

SUPPORTED_CLASSIFICATION_PROFILES = frozenset(
    {
        "sequence-only",
        "balanced",
        "evidence-assisted",
        "conservative",
    }
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelContractError(ValueError):
    """Raised when a model artifact violates its compatibility contract."""


@dataclass(frozen=True)
class ModelContract:
    """Validated identity and compatibility metadata for one MLP artifact."""

    schema_version: int
    artifact_type: str
    model_id: str
    model_sha256: str
    compatible_profiles: tuple[str, ...]
    threshold_policy_id: str
    calibration_id: str
    input_dim: int
    num_classes: int
    model_path: Path
    manifest_path: Path


def model_manifest_path(model_path: Path | str) -> Path:
    """Return the canonical sidecar path for a model artifact."""
    path = Path(model_path)
    return Path(f"{path}.manifest.json")


def calculate_sha256(
    path: Path | str,
    chunk_size: int = 1 << 20,
) -> str:
    """Calculate a file SHA-256 without loading the full artifact into RAM."""
    artifact = Path(path)
    digest = hashlib.sha256()

    with artifact.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def _required_text(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    value = payload.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ModelContractError(f"Model manifest field {field!r} must be a non-empty string.")

    return value.strip()


def _required_positive_int(
    payload: Mapping[str, Any],
    field: str,
) -> int:
    value = payload.get(field)

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelContractError(f"Model manifest field {field!r} must be a positive integer.")

    return value


def _parse_manifest(
    payload: Mapping[str, Any],
    *,
    model_path: Path,
    manifest_path: Path,
) -> ModelContract:
    schema_version = payload.get("schema_version")

    if schema_version != MODEL_CONTRACT_SCHEMA_VERSION:
        raise ModelContractError(
            "Unsupported model manifest schema_version: "
            f"{schema_version!r}; expected "
            f"{MODEL_CONTRACT_SCHEMA_VERSION}."
        )

    artifact_type = _required_text(
        payload,
        "artifact_type",
    )

    if artifact_type != MODEL_ARTIFACT_TYPE:
        raise ModelContractError(
            "Unsupported model artifact_type: "
            f"{artifact_type!r}; expected "
            f"{MODEL_ARTIFACT_TYPE!r}."
        )

    model_sha256 = _required_text(
        payload,
        "model_sha256",
    ).lower()

    if not _SHA256_PATTERN.fullmatch(model_sha256):
        raise ModelContractError(
            "Model manifest field 'model_sha256' must contain "
            "exactly 64 lowercase hexadecimal characters."
        )

    raw_profiles = payload.get("compatible_profiles")

    if (
        not isinstance(raw_profiles, list)
        or not raw_profiles
        or not all(isinstance(profile, str) and profile for profile in raw_profiles)
    ):
        raise ModelContractError(
            "Model manifest field 'compatible_profiles' must be "
            "a non-empty list of profile names."
        )

    compatible_profiles = tuple(dict.fromkeys(raw_profiles))

    unsupported = set(compatible_profiles) - SUPPORTED_CLASSIFICATION_PROFILES

    if unsupported:
        raise ModelContractError(
            "Model manifest declares unsupported profiles: " + ", ".join(sorted(unsupported))
        )

    num_classes = _required_positive_int(
        payload,
        "num_classes",
    )

    if num_classes != 3:
        raise ModelContractError("PlasFlow production MLP contracts must declare " "num_classes=3.")

    return ModelContract(
        schema_version=MODEL_CONTRACT_SCHEMA_VERSION,
        artifact_type=artifact_type,
        model_id=_required_text(payload, "model_id"),
        model_sha256=model_sha256,
        compatible_profiles=compatible_profiles,
        threshold_policy_id=_required_text(
            payload,
            "threshold_policy_id",
        ),
        calibration_id=_required_text(
            payload,
            "calibration_id",
        ),
        input_dim=_required_positive_int(
            payload,
            "input_dim",
        ),
        num_classes=num_classes,
        model_path=model_path,
        manifest_path=manifest_path,
    )


def load_model_contract(
    model_path: Path | str,
    manifest_path: Path | str | None = None,
    *,
    verify_sha256: bool = True,
) -> ModelContract:
    """Load and validate a model sidecar contract."""
    artifact = Path(model_path)

    if not artifact.is_file():
        raise ModelContractError(f"Model artifact does not exist: {artifact}")

    sidecar = Path(manifest_path) if manifest_path is not None else model_manifest_path(artifact)

    if not sidecar.is_file():
        raise ModelContractError(
            "Model manifest not found: "
            f"{sidecar}. A verified model/profile contract is required."
        )

    try:
        raw_payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelContractError(f"Unable to read model manifest {sidecar}: {error}") from error

    if not isinstance(raw_payload, dict):
        raise ModelContractError("Model manifest root must be a JSON object.")

    contract = _parse_manifest(
        raw_payload,
        model_path=artifact,
        manifest_path=sidecar,
    )

    if verify_sha256:
        actual_sha256 = calculate_sha256(artifact)

        if actual_sha256 != contract.model_sha256:
            raise ModelContractError(
                "Model SHA-256 does not match its manifest: "
                f"expected {contract.model_sha256}, "
                f"observed {actual_sha256}."
            )

    return contract


def validate_model_profile_pair(
    model_path: Path | str,
    profile: str,
    *,
    manifest_path: Path | str | None = None,
    allow_unverified_custom_model: bool = False,
) -> ModelContract | None:
    """Validate that a model artifact is approved for a classifier profile.

    An explicit custom-model escape hatch permits only a missing manifest.
    It never bypasses a malformed manifest, checksum mismatch, or declared
    profile incompatibility.
    """
    if profile not in SUPPORTED_CLASSIFICATION_PROFILES:
        raise ModelContractError(f"Unsupported classification profile: {profile!r}.")

    artifact = Path(model_path)
    sidecar = Path(manifest_path) if manifest_path is not None else model_manifest_path(artifact)

    if not sidecar.is_file() and allow_unverified_custom_model:
        logger.warning(
            "Running unverified custom MLP model %s with profile %s. "
            "No compatibility or checksum guarantees are active.",
            artifact,
            profile,
        )
        return None

    contract = load_model_contract(
        artifact,
        sidecar,
        verify_sha256=True,
    )

    if profile not in contract.compatible_profiles:
        raise ModelContractError(
            f"Model {contract.model_id!r} is not compatible with "
            f"profile {profile!r}. Approved profiles: " + ", ".join(contract.compatible_profiles)
        )

    return contract
