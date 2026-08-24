"""Atomic retrieval and verification of MobiOrigin marker databases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO

from mobiorigin.marker_features import DATABASE_SHA256, load_database_manifest

DATABASE_FILENAMES = {
    "rep": "rep_proteins.dmnd",
    "mob": "mob_proteins.dmnd",
    "mpf": "mpf_proteins.dmnd",
}
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


def setup_databases(
    output_dir: Path,
    *,
    source_dir: Path,
) -> None:
    """Verify and copy three official-source databases, then publish atomically."""
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
            "network_accessed": False,
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


def check_databases(database_dir: Path, *, diamond: Path = Path("diamond")) -> dict[str, object]:
    """Fail closed unless DIAMOND and all frozen marker databases are usable."""
    executable = shutil.which(str(diamond))
    if executable is None:
        candidate = diamond.expanduser()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FileNotFoundError(f"DIAMOND executable not found: {diamond}")
        executable = str(candidate.resolve())
    completed = subprocess.run([executable, "version"], text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"DIAMOND version check failed: {completed.stderr.strip()}")
    databases = load_database_manifest(database_dir)
    return {
        "status": "PASS",
        "diamond": executable,
        "diamond_version": (completed.stdout or completed.stderr).strip(),
        "database_dir": str(database_dir.resolve()),
        "databases_verified": len(databases),
        "database_sha256": {family: DATABASE_SHA256[family] for family in sorted(DATABASE_SHA256)},
    }
