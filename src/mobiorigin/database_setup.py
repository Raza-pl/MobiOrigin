"""Atomic retrieval and verification of MobiOrigin marker databases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO

from mobiorigin.marker_features import DATABASE_SHA256

DATABASE_FILENAMES = {
    "rep": "rep_proteins.dmnd",
    "mob": "mob_proteins.dmnd",
    "mpf": "mpf_proteins.dmnd",
}
DEFAULT_BASE_URL = "https://github.com/Raza-pl/MobiOrigin/releases/download/v2.0.0"
ARCHIVE_DOI = "https://doi.org/10.5281/zenodo.10304948"
MANIFEST_NAME = "mobiorigin_mob_suite_database_manifest.json"
NOTICE = """MobiOrigin marker-database notice

These MOB-suite-derived biological database files are retrieved for local use and
are not part of the MobiOrigin Python distribution. The MOB-suite source-code
repository is Apache-2.0 licensed. The audited database archive did not expose an
explicit license covering redistribution of every biological sequence record.
Users are responsible for confirming that their use complies with the upstream
terms and applicable law.

Official MOB-suite source: https://github.com/phac-nml/mob-suite
Audited database archive: https://doi.org/10.5281/zenodo.10304948
"""


def _copy_and_hash(source: BinaryIO, destination: Path) -> str:
    digest = hashlib.sha256()
    with destination.open("xb") as handle:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest()


def _open_url(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": "MobiOrigin/0.1"})
    return urllib.request.urlopen(request, timeout=120)  # noqa: S310


def setup_databases(
    output_dir: Path,
    *,
    source_dir: Path | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> None:
    """Retrieve or copy the three frozen databases and publish them atomically."""
    if output_dir.exists():
        raise FileExistsError("Database output directory already exists")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        manifest_databases: dict[str, dict[str, str]] = {}
        for family, filename in DATABASE_FILENAMES.items():
            destination = temporary / filename
            expected = DATABASE_SHA256[family]
            if source_dir is None:
                source_identity = f"{base_url.rstrip('/')}/{filename}"
                with _open_url(source_identity) as source:
                    observed = _copy_and_hash(source, destination)
            else:
                source_path = source_dir / filename
                if not source_path.is_file():
                    raise FileNotFoundError(f"Missing source database: {source_path}")
                source_identity = str(source_path.resolve())
                with source_path.open("rb") as source:
                    observed = _copy_and_hash(source, destination)
            if observed != expected:
                raise ValueError(f"MOB-suite {family} database SHA-256 mismatch")
            manifest_databases[family] = {
                "path": filename,
                "sha256": expected,
                "source": source_identity,
            }

        manifest = {
            "schema_version": "mobiorigin-mob-suite-database-manifest-v1",
            "databases": manifest_databases,
            "upstream_archive_doi": ARCHIVE_DOI,
            "network_accessed": source_dir is None,
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "THIRD_PARTY_DATABASE_NOTICE.txt").write_text(NOTICE, encoding="utf-8")
        checksums = "".join(
            f"{DATABASE_SHA256[family]}  {DATABASE_FILENAMES[family]}\n"
            for family in sorted(DATABASE_FILENAMES)
        )
        (temporary / "SHA256SUMS.txt").write_text(checksums, encoding="ascii")
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
