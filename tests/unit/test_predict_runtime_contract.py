import hashlib
import json
from pathlib import Path

import pytest
from plasflow2.classify.model_contract import ModelContractError
from plasflow2.classify.predict import (
    _assign_label,
    _resolve_prediction_policy,
    _validate_prediction_policy_options,
)
from plasflow2.classify.threshold_policy import (
    BALANCED_POLICY,
    CONSERVATIVE_POLICY,
    EVIDENCE_ASSISTED_POLICY,
    ThresholdPolicyError,
)


def _write_model_contract(
    tmp_path: Path,
    *,
    profile: str,
    policy_id: str,
) -> Path:
    model = tmp_path / "model.pt"
    model.write_bytes(b"contract-test-model")

    manifest = {
        "schema_version": 1,
        "artifact_type": "plasflow-mlp",
        "model_id": "unit-test-model",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "profile_threshold_policies": {profile: policy_id},
        "calibration_id": "unit-test-calibration",
        "input_dim": 9557,
        "num_classes": 3,
    }
    Path(f"{model}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return model


def _validate_options(policy, **overrides) -> bool:
    options = {
        "threshold": None,
        "plasmid_threshold": None,
        "argmax_fallback": False,
        "marker_model_path": None,
        "compass_sketch_path": None,
        "apply_prior": None,
    }
    options.update(overrides)
    return _validate_prediction_policy_options(policy, **options)


def test_verified_model_resolves_declared_profile_policy(
    tmp_path: Path,
) -> None:
    model = _write_model_contract(
        tmp_path,
        profile="balanced",
        policy_id=BALANCED_POLICY.policy_id,
    )

    contract, policy = _resolve_prediction_policy(
        model,
        "balanced",
        allow_unverified_custom_model=False,
    )

    assert contract is not None
    assert contract.model_id == "unit-test-model"
    assert policy is BALANCED_POLICY


def test_declared_policy_profile_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    model = _write_model_contract(
        tmp_path,
        profile="balanced",
        policy_id=CONSERVATIVE_POLICY.policy_id,
    )

    with pytest.raises(
        ThresholdPolicyError,
        match="belongs to profile",
    ):
        _resolve_prediction_policy(
            model,
            "balanced",
            allow_unverified_custom_model=False,
        )


def test_unverified_custom_model_requires_explicit_escape(
    tmp_path: Path,
) -> None:
    model = tmp_path / "custom.pt"
    model.write_bytes(b"unverified-custom-model")

    with pytest.raises(
        ModelContractError,
        match="manifest not found",
    ):
        _resolve_prediction_policy(
            model,
            "balanced",
            allow_unverified_custom_model=False,
        )

    contract, policy = _resolve_prediction_policy(
        model,
        "balanced",
        allow_unverified_custom_model=True,
    )

    assert contract is None
    assert policy is BALANCED_POLICY


def test_evidence_assisted_requires_compass() -> None:
    with pytest.raises(
        ThresholdPolicyError,
        match="requires a COMPASS sketch",
    ):
        _validate_options(EVIDENCE_ASSISTED_POLICY)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"threshold": 0.50}, "forbids threshold overrides"),
        ({"plasmid_threshold": 0.80}, "forbids threshold overrides"),
        ({"argmax_fallback": True}, "forbids argmax fallback"),
        ({"marker_model_path": "marker.json"}, "forbids marker-model fusion"),
        ({"compass_sketch_path": "compass.npy"}, "forbids COMPASS"),
        ({"apply_prior": True}, "requires apply_prior=False"),
    ],
)
def test_conservative_policy_rejects_semantic_mutations(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(
        ThresholdPolicyError,
        match=message,
    ):
        _validate_options(CONSERVATIVE_POLICY, **overrides)


def test_conservative_policy_disables_prior_by_default() -> None:
    assert _validate_options(CONSERVATIVE_POLICY) is False


@pytest.mark.parametrize(
    ("length_bp", "class_name"),
    [
        (2_000, "plasmid"),
        (2_001, "phage"),
        (5_000, "chromosome"),
        (10_000, "phage"),
        (20_000, "plasmid"),
    ],
)
def test_assign_label_uses_resolved_conservative_policy(
    length_bp: int,
    class_name: str,
) -> None:
    tier = CONSERVATIVE_POLICY.thresholds_for_length(length_bp)
    threshold = float(getattr(tier, class_name))
    remainder = (1.0 - threshold) / 2.0
    scores = {
        "plasmid": remainder,
        "chromosome": remainder,
        "phage": remainder,
    }
    scores[class_name] = threshold

    accepted, _ = _assign_label(
        scores,
        length_bp,
        None,
        None,
        False,
        threshold_policy=CONSERVATIVE_POLICY,
    )
    assert accepted == class_name

    scores[class_name] = threshold - 0.000001
    rejected, _ = _assign_label(
        scores,
        length_bp,
        None,
        None,
        False,
        threshold_policy=CONSERVATIVE_POLICY,
    )
    assert rejected == "unclassified"
