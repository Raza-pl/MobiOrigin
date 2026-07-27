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
    profile_threshold_policies: dict[str, str] | None = None,
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
                "profile_threshold_policies": profile_threshold_policies
                or {
                    "sequence-only": "test-tiered-v1",
                    "balanced": "test-balanced-v1",
                },
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


def test_load_model_contract_verifies_identity_and_policies(
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
    assert contract.threshold_policy_for("sequence-only") == "test-tiered-v1"
    assert contract.threshold_policy_for("balanced") == "test-balanced-v1"


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
        profile_threshold_policies={
            "conservative": "candidate-conservative-v1",
        },
    )

    with pytest.raises(
        ModelContractError,
        match="not compatible",
    ):
        validate_model_profile_pair(
            model,
            "balanced",
        )


def test_threshold_policy_lookup_rejects_undeclared_profile(
    tmp_path: Path,
) -> None:
    model, _ = _write_model_and_manifest(tmp_path)
    contract = load_model_contract(model)

    with pytest.raises(
        ModelContractError,
        match="no threshold policy",
    ):
        contract.threshold_policy_for("evidence-assisted")


def test_missing_profile_policy_map_is_rejected(
    tmp_path: Path,
) -> None:
    model, manifest = _write_model_and_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload.pop("profile_threshold_policies")
    manifest.write_text(json.dumps(payload))

    with pytest.raises(
        ModelContractError,
        match="profile_threshold_policies",
    ):
        load_model_contract(model)


def test_blank_threshold_policy_is_rejected(
    tmp_path: Path,
) -> None:
    model, manifest = _write_model_and_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["profile_threshold_policies"] = {"balanced": "   "}
    manifest.write_text(json.dumps(payload))

    with pytest.raises(
        ModelContractError,
        match="threshold-policy ID",
    ):
        load_model_contract(model)


def test_unsupported_declared_profile_is_rejected(
    tmp_path: Path,
) -> None:
    model, manifest = _write_model_and_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["profile_threshold_policies"] = {"unknown-profile": "policy-v1"}
    manifest.write_text(json.dumps(payload))

    with pytest.raises(
        ModelContractError,
        match="unsupported profile",
    ):
        load_model_contract(model)


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
