"""Regression tests for verified model distribution and installation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_database_setup_requires_and_verifies_model_manifest() -> None:
    script = (ROOT / "scripts/setup_databases.sh").read_text()

    assert 'MLP_MANIFEST="$MODEL_DIR/mlp_v2.pt.manifest.json"' in script
    assert (
        "for fname in mlp_v2.pt mlp_v2.pt.manifest.json "
        "marker_xgb.json marker_xgb.json.meta.json"
    ) in script
    assert 'verify_mlp_contract "$MLP_PT" "$MLP_MANIFEST"' in script
    assert "MLP contract verification failed" in script


def test_installation_check_fails_closed_without_verified_manifest() -> None:
    script = (ROOT / "scripts/test_installation.sh").read_text()

    assert 'MODEL_MANIFEST="$MODEL.manifest.json"' in script
    assert "load_model_contract" in script
    assert "MLP manifest not found" in script
    assert "MLP contract verification failed" in script


def test_readme_documents_cryptographic_model_verification() -> None:
    readme = (ROOT / "README.md").read_text()

    assert "cryptographic manifest" in readme
    assert "SHA-256 verified before use" in readme
