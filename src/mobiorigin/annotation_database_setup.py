"""Atomic staging and verification of annotation database resources."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Final

from mobiorigin.provenance import sha256_file

ANNOTATION_MANIFEST_NAME: Final = "mobiorigin_annotation_database_manifest.json"
ANNOTATION_NOTICE_NAME: Final = "THIRD_PARTY_ANNOTATION_DATABASE_NOTICE.txt"

ARG_DATABASE_FILES: Final = (
    "card/card.dmnd",
    "card/aro_index.tsv",
    "sarg/sarg.dmnd",
)
COMPREHENSIVE_DATABASE_FILES: Final = (
    *ARG_DATABASE_FILES,
    "vfdb/vfdb_setA.dmnd",
    "vfdb/vfdb_indx.txt",
    "mge/isfinder.dmnd",
    "mge/mge_database.tsv",
    "bacmet/bacmet.dmnd",
    "bacmet/Bacmet_list.tsv",
    "mob_suite/rep_proteins.dmnd",
    "mob_suite/mob_proteins.dmnd",
    "mob_suite/mpf_proteins.dmnd",
)

UPSTREAM_TERMS: Final = {
    "CARD": "https://card.mcmaster.ca/about",
    "SARG": "https://smile.hku.hk/SARGs",
    "VFDB": "https://www.mgc.ac.cn/VFs/main.htm",
    "ISfinder": "https://isfinder.biotoul.fr/about.php",
    "BacMet": "http://bacmet.biomedicine.gu.se/",
    "MOB-suite": "https://github.com/phac-nml/mob-suite",
}

NOTICE: Final = """MobiOrigin annotation-database notice

These third-party biological database files are installed for local use and are
not part of the MobiOrigin Python distribution. Possession of a database file
does not grant permission to redistribute it. In particular, ISfinder states
that downloading its database requires written authorization and that its
database may not be distributed to third parties. VFDB describes its data as
CC BY-NC 4.0 for non-commercial research use. Other upstream sources may impose
additional academic-use, registration, attribution, or commercial-use terms.

The installing user confirms that they obtained every source lawfully and that
their intended use complies with the applicable upstream terms. MobiOrigin
records file identities for reproducibility but does not grant rights in any
third-party database content.

Upstream terms:
  CARD: https://card.mcmaster.ca/about
  SARG: https://smile.hku.hk/SARGs
  VFDB: https://www.mgc.ac.cn/VFs/main.htm
  ISfinder: https://isfinder.biotoul.fr/about.php
  BacMet: http://bacmet.biomedicine.gu.se/
  MOB-suite: https://github.com/phac-nml/mob-suite
"""


def annotation_files(profile: str) -> tuple[str, ...]:
    """Return the exact files required for an annotation profile."""
    if profile == "arg":
        return ARG_DATABASE_FILES
    if profile == "comprehensive":
        return COMPREHENSIVE_DATABASE_FILES
    raise ValueError("Annotation database profile must be 'arg' or 'comprehensive'")


def default_annotation_database_dir() -> Path:
    """Return the stable annotation database location without creating it."""
    override = os.environ.get("MOBIORIGIN_ANNOTATION_DATABASE_DIR")
    if override:
        return Path(override).expanduser()
    root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "mobiorigin" / "annotation_databases"


def _copy_file(source: Path, destination: Path) -> dict[str, object]:
    if source.is_symlink():
        raise ValueError(f"Annotation database source must not be a symlink: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"Missing annotation database file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"Annotation database copy identity mismatch: {source}")
    return {
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": destination_hash,
        "source": str(source.resolve()),
    }


def _source_files(directory: Path) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError(f"Official AMRFinderPlus database directory not found: {directory}")
    if not (directory / "version.txt").is_file():
        raise FileNotFoundError(
            f"Official AMRFinderPlus database version file not found: {directory / 'version.txt'}"
        )
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Official AMRFinderPlus database directory is empty")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"AMRFinderPlus database source must not be a symlink: {path}")
    return files


def setup_annotation_databases(
    output_dir: Path,
    *,
    source_dir: Path,
    amrfinder_database: Path,
    profile: str = "comprehensive",
    accept_third_party_terms: bool = False,
) -> dict[str, object]:
    """Copy an authorized annotation resource set and publish it atomically."""
    if not accept_third_party_terms:
        raise PermissionError(
            "Annotation database setup requires --accept-third-party-terms. "
            "Review docs/MOBIORIGIN_ANNOTATION.md and each upstream source first."
        )
    if output_dir.exists():
        raise FileExistsError("Annotation database output directory already exists")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Annotation database source directory not found: {source_dir}")
    required = annotation_files(profile)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        resources: dict[str, dict[str, object]] = {}
        total_bytes = 0
        for relative in required:
            item = _copy_file(source_dir / relative, temporary / relative)
            item["path"] = relative
            resources[relative] = item
            copied_bytes = item["bytes"]
            if not isinstance(copied_bytes, int):
                raise TypeError(f"Annotation database byte count is invalid: {relative}")
            total_bytes += copied_bytes
        amrfinder_files = _source_files(amrfinder_database)
        for source in amrfinder_files:
            relative = f"amrfinderplus/{source.relative_to(amrfinder_database).as_posix()}"
            item = _copy_file(source, temporary / relative)
            item["path"] = relative
            resources[relative] = item
            copied_bytes = item["bytes"]
            if not isinstance(copied_bytes, int):
                raise TypeError(f"AMRFinderPlus database byte count is invalid: {relative}")
            total_bytes += copied_bytes
        manifest = {
            "schema_version": "mobiorigin-annotation-database-manifest-v1",
            "profile": profile,
            "resources": resources,
            "resource_count": len(resources),
            "total_bytes": total_bytes,
            "network_accessed": False,
            "third_party_terms_accepted": True,
            "official_amrfinderplus_included": True,
            "upstream_terms": UPSTREAM_TERMS,
        }
        (temporary / ANNOTATION_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / ANNOTATION_NOTICE_NAME).write_text(NOTICE, encoding="utf-8")
        checksums = "".join(
            f"{resources[relative]['sha256']}  {relative}\n" for relative in sorted(resources)
        )
        (temporary / "SHA256SUMS.txt").write_text(checksums, encoding="ascii")
        os.replace(temporary, output_dir)
        return check_annotation_databases(output_dir, profile=profile)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def check_annotation_databases(
    database_dir: Path, *, profile: str = "comprehensive"
) -> dict[str, object]:
    """Verify the manifest and every byte required by an annotation profile."""
    manifest_path = database_dir / ANNOTATION_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Annotation database manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Annotation database manifest is invalid") from error
    if manifest.get("schema_version") != "mobiorigin-annotation-database-manifest-v1":
        raise ValueError("Annotation database manifest schema is unsupported")
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("Annotation database manifest resources are invalid")
    verified: dict[str, str] = {}
    total_bytes = 0
    for relative in annotation_files(profile):
        entry = resources.get(relative)
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"Annotation database manifest entry is missing: {relative}")
        path = database_dir / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Annotation database file is missing: {path}")
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise ValueError(f"Annotation database SHA-256 mismatch: {relative}")
        expected_bytes = entry.get("bytes")
        if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
            raise ValueError(f"Annotation database size mismatch: {relative}")
        verified[relative] = observed
        total_bytes += expected_bytes
    amrfinder_entries = {
        relative: entry
        for relative, entry in resources.items()
        if relative.startswith("amrfinderplus/")
    }
    if "amrfinderplus/version.txt" not in amrfinder_entries:
        raise ValueError("Official AMRFinderPlus database is missing from the manifest")
    for relative, entry in sorted(amrfinder_entries.items()):
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"AMRFinderPlus manifest entry is invalid: {relative}")
        path = database_dir / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"AMRFinderPlus database file is missing: {path}")
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise ValueError(f"AMRFinderPlus database SHA-256 mismatch: {relative}")
        expected_bytes = entry.get("bytes")
        if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
            raise ValueError(f"AMRFinderPlus database size mismatch: {relative}")
        verified[relative] = observed
        total_bytes += expected_bytes
    return {
        "status": "PASS",
        "profile": profile,
        "database_dir": str(database_dir.resolve()),
        "resources_verified": len(verified),
        "total_bytes": total_bytes,
        "database_sha256": verified,
        "third_party_terms": UPSTREAM_TERMS,
    }
