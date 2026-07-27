"""Immutable classification-threshold policies paired with MLP contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

SEQUENCE_ONLY_POLICY_ID = "rev5-tiered-default-20260724-v1"
BALANCED_POLICY_ID = "rev5-balanced-p080-20260724-v1"
EVIDENCE_ASSISTED_POLICY_ID = "rev5-balanced-p080-compass-k21-s5m-20260724-v1"
CONSERVATIVE_POLICY_ID = "three-class-conservative-tiered-20260726-v1"


class ThresholdPolicyError(ValueError):
    """Raised when a threshold policy or profile pairing is invalid."""


@dataclass(frozen=True)
class ThresholdTier:
    """Per-class thresholds up to one inclusive sequence-length boundary."""

    max_length_bp: int | None
    plasmid: float
    chromosome: float
    phage: float

    def __post_init__(self) -> None:
        if self.max_length_bp is not None and self.max_length_bp <= 0:
            raise ThresholdPolicyError("Threshold tier boundaries must be positive.")

        for class_name, value in (
            ("plasmid", self.plasmid),
            ("chromosome", self.chromosome),
            ("phage", self.phage),
        ):
            if not 0.0 <= value <= 1.0:
                raise ThresholdPolicyError(f"{class_name} threshold must be between 0 and 1.")


@dataclass(frozen=True)
class ThresholdPolicy:
    """Frozen threshold behavior approved for one classifier profile."""

    policy_id: str
    profile: str
    tiers: tuple[ThresholdTier, ...]
    requires_compass: bool = False
    allow_compass: bool = True
    allow_threshold_overrides: bool = True
    apply_prior_correction: bool = True
    allow_marker_fusion: bool = True
    allow_argmax_fallback: bool = True
    allow_postclassification_label_changes: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ThresholdPolicyError("Threshold policy ID cannot be blank.")
        if not self.profile:
            raise ThresholdPolicyError("Threshold policy profile cannot be blank.")
        if not self.tiers:
            raise ThresholdPolicyError("Threshold policy requires at least one tier.")

        previous_boundary = 0
        for index, tier in enumerate(self.tiers):
            boundary = tier.max_length_bp

            if boundary is None:
                if index != len(self.tiers) - 1:
                    raise ThresholdPolicyError("Only the final threshold tier may be unbounded.")
                continue

            if boundary <= previous_boundary:
                raise ThresholdPolicyError("Threshold tier boundaries must increase strictly.")
            previous_boundary = boundary

        if self.tiers[-1].max_length_bp is not None:
            raise ThresholdPolicyError("The final threshold tier must be unbounded.")

    def thresholds_for_length(self, length_bp: int) -> ThresholdTier:
        """Return the approved thresholds for one positive sequence length."""
        if length_bp <= 0:
            raise ThresholdPolicyError("Sequence length must be a positive integer.")

        for tier in self.tiers:
            if tier.max_length_bp is None or length_bp <= tier.max_length_bp:
                return tier

        raise ThresholdPolicyError(f"No threshold tier covers sequence length {length_bp}.")


def _tier(
    maximum: int | None,
    plasmid: float,
    chromosome: float,
    phage: float,
) -> ThresholdTier:
    return ThresholdTier(
        max_length_bp=maximum,
        plasmid=plasmid,
        chromosome=chromosome,
        phage=phage,
    )


SEQUENCE_ONLY_POLICY = ThresholdPolicy(
    policy_id=SEQUENCE_ONLY_POLICY_ID,
    profile="sequence-only",
    tiers=(
        # Effective shipped behavior includes the historical <5 kb
        # floors applied by _assign_label(), not merely the raw tier table.
        _tier(2_000, 0.950, 0.70, 0.855),
        _tier(4_999, 0.950, 0.68, 0.850),
        _tier(9_999, 0.859, 0.65, 0.845),
        _tier(19_999, 0.857, 0.63, 0.835),
        _tier(None, 0.809, 0.62, 0.750),
    ),
)

BALANCED_POLICY = ThresholdPolicy(
    policy_id=BALANCED_POLICY_ID,
    profile="balanced",
    tiers=(
        _tier(2_000, 0.800, 0.70, 0.855),
        _tier(4_999, 0.800, 0.68, 0.850),
        _tier(9_999, 0.800, 0.65, 0.845),
        _tier(19_999, 0.800, 0.63, 0.835),
        _tier(None, 0.800, 0.62, 0.750),
    ),
)

EVIDENCE_ASSISTED_POLICY = ThresholdPolicy(
    policy_id=EVIDENCE_ASSISTED_POLICY_ID,
    profile="evidence-assisted",
    tiers=BALANCED_POLICY.tiers,
    requires_compass=True,
)

CONSERVATIVE_POLICY = ThresholdPolicy(
    policy_id=CONSERVATIVE_POLICY_ID,
    profile="conservative",
    tiers=(
        _tier(2_000, 0.950, 0.340, 0.935),
        _tier(4_999, 0.950, 0.340, 0.980),
        _tier(9_999, 0.950, 0.350, 0.935),
        _tier(19_999, 0.945, 0.350, 0.905),
        _tier(None, 0.945, 0.365, 0.555),
    ),
    allow_compass=False,
    allow_threshold_overrides=False,
    apply_prior_correction=False,
    allow_marker_fusion=False,
    allow_argmax_fallback=False,
    allow_postclassification_label_changes=False,
)

THRESHOLD_POLICIES: Mapping[str, ThresholdPolicy] = MappingProxyType(
    {
        policy.policy_id: policy
        for policy in (
            SEQUENCE_ONLY_POLICY,
            BALANCED_POLICY,
            EVIDENCE_ASSISTED_POLICY,
            CONSERVATIVE_POLICY,
        )
    }
)


DEFAULT_PROFILE_POLICY_IDS: Mapping[str, str] = MappingProxyType(
    {
        "sequence-only": SEQUENCE_ONLY_POLICY_ID,
        "balanced": BALANCED_POLICY_ID,
        "evidence-assisted": EVIDENCE_ASSISTED_POLICY_ID,
        "conservative": CONSERVATIVE_POLICY_ID,
    }
)


def get_threshold_policy(policy_id: str) -> ThresholdPolicy:
    """Return a known immutable threshold policy."""
    try:
        return THRESHOLD_POLICIES[policy_id]
    except KeyError as error:
        raise ThresholdPolicyError(f"Unknown threshold policy ID: {policy_id!r}.") from error


def default_threshold_policy_for_profile(
    profile: str,
) -> ThresholdPolicy:
    """Return the canonical policy for an explicitly allowed custom model."""
    try:
        policy_id = DEFAULT_PROFILE_POLICY_IDS[profile]
    except KeyError as error:
        raise ThresholdPolicyError(f"Unsupported classification profile: {profile!r}.") from error

    return get_threshold_policy(policy_id)


def validate_profile_threshold_policy(
    profile: str,
    policy_id: str,
) -> ThresholdPolicy:
    """Require a threshold policy to belong to the requested profile."""
    policy = get_threshold_policy(policy_id)

    if policy.profile != profile:
        raise ThresholdPolicyError(
            f"Threshold policy {policy_id!r} belongs to profile "
            f"{policy.profile!r}, not {profile!r}."
        )

    return policy
