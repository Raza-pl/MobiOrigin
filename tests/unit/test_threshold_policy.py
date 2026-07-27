import pytest
from plasflow2.classify.predict import _assign_label
from plasflow2.classify.threshold_policy import (
    BALANCED_POLICY,
    CONSERVATIVE_POLICY,
    CONSERVATIVE_POLICY_ID,
    EVIDENCE_ASSISTED_POLICY,
    SEQUENCE_ONLY_POLICY,
    ThresholdPolicyError,
    get_threshold_policy,
    validate_profile_threshold_policy,
)


@pytest.mark.parametrize(
    ("length_bp", "expected"),
    [
        (2_000, (0.950, 0.700, 0.855)),
        (2_001, (0.950, 0.680, 0.850)),
        (4_999, (0.950, 0.680, 0.850)),
        (5_000, (0.859, 0.650, 0.845)),
        (10_000, (0.857, 0.630, 0.835)),
        (20_000, (0.809, 0.620, 0.750)),
    ],
)
def test_sequence_policy_records_effective_shipped_thresholds(
    length_bp: int,
    expected: tuple[float, float, float],
) -> None:
    tier = SEQUENCE_ONLY_POLICY.thresholds_for_length(length_bp)

    assert (
        tier.plasmid,
        tier.chromosome,
        tier.phage,
    ) == expected


@pytest.mark.parametrize(
    "length_bp",
    [2_000, 2_001, 4_999, 5_000, 10_000, 20_000],
)
@pytest.mark.parametrize(
    "class_name",
    ["plasmid", "chromosome", "phage"],
)
def test_sequence_policy_matches_legacy_label_assignment(
    length_bp: int,
    class_name: str,
) -> None:
    tier = SEQUENCE_ONLY_POLICY.thresholds_for_length(length_bp)
    threshold = float(getattr(tier, class_name))
    remainder = (1.0 - threshold) / 2.0
    scores = {name: remainder for name in ("plasmid", "chromosome", "phage")}
    scores[class_name] = threshold

    accepted, _ = _assign_label(
        scores,
        length_bp,
        None,
        None,
        False,
    )
    assert accepted == class_name

    scores[class_name] = threshold - 0.000001
    rejected, _ = _assign_label(
        scores,
        length_bp,
        None,
        None,
        False,
    )
    assert rejected == "unclassified"


def test_balanced_policy_changes_only_plasmid_thresholds() -> None:
    for balanced, sequence_only in zip(
        BALANCED_POLICY.tiers,
        SEQUENCE_ONLY_POLICY.tiers,
        strict=True,
    ):
        assert balanced.plasmid == 0.80
        assert balanced.chromosome == sequence_only.chromosome
        assert balanced.phage == sequence_only.phage


def test_evidence_assisted_policy_requires_compass() -> None:
    assert EVIDENCE_ASSISTED_POLICY.requires_compass is True
    assert EVIDENCE_ASSISTED_POLICY.tiers == BALANCED_POLICY.tiers


@pytest.mark.parametrize(
    ("length_bp", "expected"),
    [
        (1, (0.950, 0.340, 0.935)),
        (2_000, (0.950, 0.340, 0.935)),
        (2_001, (0.950, 0.340, 0.980)),
        (4_999, (0.950, 0.340, 0.980)),
        (5_000, (0.950, 0.350, 0.935)),
        (9_999, (0.950, 0.350, 0.935)),
        (10_000, (0.945, 0.350, 0.905)),
        (19_999, (0.945, 0.350, 0.905)),
        (20_000, (0.945, 0.365, 0.555)),
        (1_000_000, (0.945, 0.365, 0.555)),
    ],
)
def test_conservative_policy_exact_boundaries(
    length_bp: int,
    expected: tuple[float, float, float],
) -> None:
    tier = CONSERVATIVE_POLICY.thresholds_for_length(length_bp)

    assert (
        tier.plasmid,
        tier.chromosome,
        tier.phage,
    ) == expected


def test_conservative_policy_locks_confirmatory_semantics() -> None:
    assert CONSERVATIVE_POLICY.requires_compass is False
    assert CONSERVATIVE_POLICY.allow_compass is False
    assert CONSERVATIVE_POLICY.allow_threshold_overrides is False
    assert CONSERVATIVE_POLICY.apply_prior_correction is False
    assert CONSERVATIVE_POLICY.allow_marker_fusion is False
    assert CONSERVATIVE_POLICY.allow_argmax_fallback is False
    assert CONSERVATIVE_POLICY.allow_postclassification_label_changes is False


def test_get_threshold_policy_returns_frozen_candidate_policy() -> None:
    assert get_threshold_policy(CONSERVATIVE_POLICY_ID) is CONSERVATIVE_POLICY


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(
        ThresholdPolicyError,
        match="Unknown threshold policy",
    ):
        get_threshold_policy("not-a-real-policy")


def test_profile_policy_mismatch_is_rejected() -> None:
    with pytest.raises(
        ThresholdPolicyError,
        match="belongs to profile",
    ):
        validate_profile_threshold_policy(
            "balanced",
            CONSERVATIVE_POLICY_ID,
        )


def test_nonpositive_sequence_length_is_rejected() -> None:
    with pytest.raises(
        ThresholdPolicyError,
        match="positive integer",
    ):
        CONSERVATIVE_POLICY.thresholds_for_length(0)
