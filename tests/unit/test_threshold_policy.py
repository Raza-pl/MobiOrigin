import math

import pytest
from plasflow2.classify.predict import LENGTH_THRESHOLD_TIERS
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


def test_sequence_policy_matches_existing_runtime_thresholds() -> None:
    expected = tuple(
        (
            None if math.isinf(maximum) else int(maximum),
            plasmid,
            chromosome,
            phage,
        )
        for maximum, plasmid, phage, chromosome in LENGTH_THRESHOLD_TIERS
    )
    observed = tuple(
        (
            tier.max_length_bp,
            tier.plasmid,
            tier.chromosome,
            tier.phage,
        )
        for tier in SEQUENCE_ONLY_POLICY.tiers
    )

    assert observed == expected


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


def test_conservative_policy_is_locked_against_overrides() -> None:
    assert CONSERVATIVE_POLICY.allow_threshold_overrides is False
    assert CONSERVATIVE_POLICY.requires_compass is False


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
