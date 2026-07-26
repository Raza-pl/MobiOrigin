import click
import plasflow2.cli as cli
import pytest


def test_sequence_only_profile_preserves_settings():
    assert cli._resolve_classification_profile("sequence-only", None, None) == (None, None)


def test_balanced_profile_sets_threshold_without_compass():
    assert cli._resolve_classification_profile("balanced", None, None) == (
        None,
        0.80,
    )


def test_balanced_profile_respects_explicit_settings():
    assert cli._resolve_classification_profile("balanced", "custom.npy", 0.85) == (
        "custom.npy",
        0.85,
    )


def test_evidence_assisted_profile_autodetects_sketch(tmp_path, monkeypatch):
    sketch = tmp_path / "sketch.npy"
    sketch.touch()
    monkeypatch.setattr(cli, "_DEFAULT_COMPASS_SKETCH", sketch)
    monkeypatch.setattr(cli, "_DOCKER_COMPASS_SKETCH", tmp_path / "missing.npy")

    resolved, threshold = cli._resolve_classification_profile("evidence-assisted", None, None)

    assert resolved == str(sketch)
    assert threshold == 0.80


def test_evidence_assisted_profile_respects_explicit_threshold(tmp_path):
    sketch = tmp_path / "custom.npy"
    sketch.touch()

    resolved, threshold = cli._resolve_classification_profile(
        "evidence-assisted", str(sketch), 0.85
    )

    assert resolved == str(sketch)
    assert threshold == 0.85


def test_evidence_assisted_profile_requires_sketch(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_DEFAULT_COMPASS_SKETCH", tmp_path / "missing-default.npy")
    monkeypatch.setattr(cli, "_DOCKER_COMPASS_SKETCH", tmp_path / "missing-docker.npy")

    with pytest.raises(click.ClickException):
        cli._resolve_classification_profile("evidence-assisted", None, None)


def test_marker_fusion_requires_explicit_model(tmp_path):
    marker = tmp_path / "marker_xgb.json"
    marker.write_text("{}")

    assert cli._resolve_explicit_marker_model(None, False) is None
    assert cli._resolve_explicit_marker_model(str(marker), True) is None
    assert cli._resolve_explicit_marker_model(str(marker), False) == str(marker)


def test_marker_fusion_rejects_missing_native_model(tmp_path):
    with pytest.raises(click.BadParameter):
        cli._resolve_explicit_marker_model(
            str(tmp_path / "missing_marker.pkl"),
            False,
        )
