from plasflow2.pipeline import _resolve_pipeline_plasmid_threshold


def test_explicit_threshold_wins_in_lenient_mode():
    assert _resolve_pipeline_plasmid_threshold(True, 0.80, None) == 0.80


def test_lenient_supplies_threshold_without_explicit_override():
    assert _resolve_pipeline_plasmid_threshold(True, None, 0.65) == 0.65


def test_standard_mode_preserves_tiered_default():
    assert _resolve_pipeline_plasmid_threshold(False, None, None) is None
