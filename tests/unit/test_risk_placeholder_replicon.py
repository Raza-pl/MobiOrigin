from types import SimpleNamespace

from plasflow2.risk.scorer import score_plasmid


def test_placeholder_replicon_does_not_increase_risk():
    mobility = SimpleNamespace(
        mobility_class="non-mobilizable",
        replicon_type="-",
        relaxase_type="none",
        mpf_type="none",
    )
    result = score_plasmid(
        "contig",
        mobility,
        [],
        "unspecified",
        None,
    )
    assert result.replicon_score == 0
    assert not any("Known replicon -" in item for item in result.evidence)
