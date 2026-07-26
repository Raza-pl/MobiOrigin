import hashlib
import json
from pathlib import Path

import pytest
from plasflow2.classify.model_contract import (
    MODEL_ARTIFACT_TYPE,
    MODEL_CONTRACT_SCHEMA_VERSION,
    ModelContractError,
    load_model_contract,
    model_manifest_path,
    validate_model_profile_pair,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_model_and_manifest(
    tmp_path: Path,
    *,
    compatible_profiles: list[str] | None = None,
) -> tuple[Path, Path]:
    model = tmp_path / "model.pt"
    model.write_bytes(b"safe-test-model")

    manifest = model_manifest_path(model)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": MODEL_CONTRACT_SCHEMA_VERSION,
                "artifact_type": MODEL_ARTIFACT_TYPE,
                "model_id": "test-model",
                "model_sha256": _sha256(model),
                "compatible_profiles": compatible_profiles or ["sequence-only", "balanced"],
                "threshold_policy_id": "test-thresholds-v1",
                "calibration_id": "test-calibration-v1",
                "input_dim": 9557,
                "num_classes": 3,
            }
        )
    )

    return model, manifest


def test_model_manifest_path_is_unambiguous(
    tmp_path: Path,
) -> None:
    model = tmp_path / "mlp_v2.pt"

    assert model_manifest_path(model) == Path(f"{model}.manifest.json")


def test_load_model_contract_verifies_identity(
    tmp_path: Path,
) -> None:
    model, manifest = _write_model_and_manifest(tmp_path)

    contract = load_model_contract(model)

    assert contract.model_id == "test-model"
    assert contract.model_sha256 == _sha256(model)
    assert contract.model_path == model
    assert contract.manifest_path == manifest
    assert contract.compatible_profiles == (
        "sequence-only",
        "balanced",
    )


def test_tampered_model_is_rejected(
    tmp_path: Path,
) -> None:
    model, _ = _write_model_and_manifest(tmp_path)
    model.write_bytes(b"tampered-model")

    with pytest.raises(
        ModelContractError,
        match="does not match",
    ):
        load_model_contract(model)


def test_incompatible_profile_is_rejected(
    tmp_path: Path,
) -> None:
    model, _ = _write_model_and_manifest(
        tmp_path,
        compatible_profiles=["conservative"],
    )

    with pytest.raises(
        ModelContractError,
        match="not compatible",
    ):
        validate_model_profile_pair(
            model,
            "balanced",
        )


def test_missing_manifest_fails_closed(
    tmp_path: Path,
) -> None:
    model = tmp_path / "custom.pt"
    model.write_bytes(b"custom")

    with pytest.raises(
        ModelContractError,
        match="manifest not found",
    ):
        validate_model_profile_pair(
            model,
            "sequence-only",
        )


def test_explicit_custom_escape_hatch_allows_only_missing_manifest(
    tmp_path: Path,
) -> None:
    model = tmp_path / "custom.pt"
    model.write_bytes(b"custom")

    assert (
        validate_model_profile_pair(
            model,
            "sequence-only",
            allow_unverified_custom_model=True,
        )
        is None
    )


def test_escape_hatch_does_not_bypass_checksum_mismatch(
    tmp_path: Path,
) -> None:
    model, _ = _write_model_and_manifest(tmp_path)
    model.write_bytes(b"tampered")

    with pytest.raises(
        ModelContractError,
        match="does not match",
    ):
        validate_model_profile_pair(
            model,
            "sequence-only",
            allow_unverified_custom_model=True,
        )


def test_unsupported_schema_is_rejected(
    tmp_path: Path,
) -> None:
    model, manifest = _write_model_and_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["schema_version"] = 999
    manifest.write_text(json.dumps(payload))

    with pytest.raises(
        ModelContractError,
        match="schema_version",
    ):
        load_model_contract(model)
