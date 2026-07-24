import pytest

from plasflow2.classify.predict import _get_length_thresholds


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (1_000, (0.862, 0.855, 0.75)),
        (2_000, (0.862, 0.855, 0.75)),
        (2_001, (0.864, 0.850, 0.68)),
        (4_999, (0.864, 0.850, 0.68)),
        (5_000, (0.859, 0.845, 0.65)),
        (9_999, (0.859, 0.845, 0.65)),
        (10_000, (0.857, 0.835, 0.63)),
        (19_999, (0.857, 0.835, 0.63)),
        (20_000, (0.809, 0.750, 0.62)),
    ],
)
def test_calibrated_length_threshold_boundaries(length, expected):
    assert _get_length_thresholds(length) == pytest.approx(expected)
