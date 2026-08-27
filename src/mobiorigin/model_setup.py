"""Atomic retrieval and identity verification of frozen MobiOrigin models."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Final

from mobiorigin.provenance import sha256_file

MODEL_RELEASE_TAG: Final = "v0.1.3"
MODEL_ARCHIVE_NAME: Final = "mobiorigin-models-dev1.tar"
MODEL_ARCHIVE_URL: Final = (
    f"https://github.com/Raza-pl/MobiOrigin/releases/download/"
    f"{MODEL_RELEASE_TAG}/{MODEL_ARCHIVE_NAME}"
)
MODEL_ARCHIVE_SHA256: Final = "10a3e599eae31a72a4d09a4a58685666058f88d6995fbb9ed450e965a6a513cf"
MODEL_ARCHIVE_ROOT: Final = "mobiorigin-models-dev1"
MODEL_INSTALLATION_MANIFEST: Final = "mobiorigin_model_installation_manifest.json"
MODEL_ARTIFACTS: Final = {
    "seed_20260810.pt": {
        "bytes": 40_361_658,
        "sha256": "2ed9a2ae4cbe00213504c27ef705b6af965aae97a8e33259661cb2c630a495c3",
    },
    "seed_20260811.pt": {
        "bytes": 40_361_658,
        "sha256": "9270b5d2213ac95cae2821d26d6840974105905eb080ee39a178fe945140037d",
    },
    "seed_20260812.pt": {
        "bytes": 40_361_658,
        "sha256": "085608214f4aac424e841cfd57b39c7b968deedebed943a626695fe815fe1c0f",
    },
    "marker_normalization.npy": {
        "bytes": 264,
        "sha256": "cb93c881032356f970bc0963969f852f88ab3a9a3a4a3d6c391437e11a4cd8bc",
    },
    "model_manifest.json": {
        "bytes": 1_753,
        "sha256": "d50e25bd3578cbd30f845e2b09b20a5a3a7abb880e248febf5de848006787c63",
    },
}


def default_model_dir() -> Path:
    """Return the stable user model directory without creating it."""
    override = os.environ.get("MOBIORIGIN_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "mobiorigin" / "models" / "dev1"


def default_model_cache_dir() -> Path:
    """Return the resumable cache directory for the frozen model archive."""
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "mobiorigin" / "model_downloads"


def packaged_model_dir() -> Path:
    """Return the legacy package-local directory used by model-inclusive releases."""
    return Path(__file__).parent / "data" / "models" / "dev1"


def _all_artifacts_present(path: Path) -> bool:
    return all((path / name).is_file() for name in MODEL_ARTIFACTS)


def resolve_model_dir(model_dir: Path | None = None) -> Path:
    """Resolve explicit, environment, packaged, or user-installed model artifacts."""
    if model_dir is not None:
        return model_dir.expanduser()
    override = os.environ.get("MOBIORIGIN_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    packaged = packaged_model_dir()
    if _all_artifacts_present(packaged):
        return packaged
    return default_model_dir()


def check_models(model_dir: Path) -> dict[str, object]:
    """Fail closed unless every frozen model artifact has its exact identity."""
    observed: dict[str, dict[str, object]] = {}
    for filename, contract in MODEL_ARTIFACTS.items():
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Frozen model artifact is missing: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != contract["bytes"] or digest != contract["sha256"]:
            raise ValueError(f"Frozen model artifact identity changed: {path}")
        observed[filename] = {"bytes": size, "sha256": digest}
    return {
        "status": "PASS",
        "model_dir": str(model_dir.resolve()),
        "artifacts_verified": len(observed),
        "artifacts": observed,
    }


def _download_archive(url: str, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == MODEL_ARCHIVE_SHA256:
        return {"url": url, "bytes": destination.stat().st_size, "reused": True}
    if destination.exists():
        destination.unlink()
    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "MobiOrigin-model-setup/1"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        append = start > 0 and getattr(response, "status", None) == 206
        with partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    os.replace(partial, destination)
    if sha256_file(destination) != MODEL_ARCHIVE_SHA256:
        destination.unlink(missing_ok=True)
        raise ValueError("Downloaded frozen model archive identity changed")
    return {"url": url, "bytes": destination.stat().st_size, "reused": False}


def _extract_verified(archive_path: Path, destination: Path) -> None:
    expected_members = {
        f"{MODEL_ARCHIVE_ROOT}/{filename}": filename for filename in MODEL_ARTIFACTS
    }
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        observed_names = {member.name for member in members}
        if observed_names != set(expected_members) or any(
            not member.isfile() for member in members
        ):
            raise ValueError("Frozen model archive member inventory changed")
        for member in members:
            filename = expected_members[member.name]
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read frozen model artifact: {filename}")
            with source, (destination / filename).open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())


def setup_models(
    output_dir: Path,
    *,
    archive: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, object]:
    """Retrieve or consume the frozen archive and publish verified models atomically."""
    if output_dir.exists():
        raise FileExistsError(f"Model output directory already exists: {output_dir}")
    network_accessed = archive is None
    download_result: dict[str, object] | None = None
    if archive is None:
        cache = cache_dir or default_model_cache_dir()
        archive_path = cache / MODEL_ARCHIVE_NAME
        download_result = _download_archive(MODEL_ARCHIVE_URL, archive_path)
    else:
        archive_path = archive.expanduser()
    if not archive_path.is_file() or sha256_file(archive_path) != MODEL_ARCHIVE_SHA256:
        raise ValueError(f"Frozen model archive identity changed: {archive_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _extract_verified(archive_path, temporary)
        result = check_models(temporary)
        installation = {
            "schema_version": "mobiorigin-model-installation-v1",
            "model_release_tag": MODEL_RELEASE_TAG,
            "archive_url": MODEL_ARCHIVE_URL,
            "archive_sha256": MODEL_ARCHIVE_SHA256,
            "archive_bytes": archive_path.stat().st_size,
            "network_accessed": network_accessed,
            "download": download_result,
            "artifacts": result["artifacts"],
        }
        (temporary / MODEL_INSTALLATION_MANIFEST).write_text(
            json.dumps(installation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checksums = "".join(
            f"{MODEL_ARTIFACTS[name]['sha256']}  {name}\n" for name in sorted(MODEL_ARTIFACTS)
        )
        (temporary / "SHA256SUMS.txt").write_text(checksums, encoding="ascii")
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        **check_models(output_dir),
        "archive_sha256": MODEL_ARCHIVE_SHA256,
        "network_accessed": network_accessed,
        "download": download_result,
    }
