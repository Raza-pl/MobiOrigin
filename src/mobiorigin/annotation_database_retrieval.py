"""Official-source retrieval and local construction of annotation databases."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Final

from mobiorigin.provenance import sha256_file

CARD_URL: Final = "https://card.mcmaster.ca/latest/data"
SARG_URL: Final = (
    "https://files.pythonhosted.org/packages/f4/20/"
    "99c43f0b19993e1af5c375b51fd3841bcc5c1fc5dfd82ce65d8e0eebb33c/"
    "ARGs_OAP-2.3.2.tar.gz"
)
SARG_SHA256: Final = "b40d0528c1a9025ab0c1b97928fd453f230d1b6cf0e618ab8b50cb9cc2283303"
VFDB_URL: Final = "http://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz"
MOBILEOG_URL: Final = (
    "https://zenodo.org/api/records/17506721/files/" "mobileOG2-dino.90_sequences.faa/content"
)
MOBILEOG_MD5: Final = "42da36e410bd7e91c01f1de67c0a5d48"
MOBILEOG_RECORD: Final = "https://doi.org/10.5281/zenodo.17506721"
BACMET_COMMIT: Final = "2e76ce454c9e5b7353af5b7ed63a196a06960788"
BACMET_BASE_URL: Final = (
    f"https://raw.githubusercontent.com/ZhihaoXie/BacMet/{BACMET_COMMIT}/BacMet2_EXP"
)
BACMET_FASTA_SHA256: Final = "18c1ca2244dbe49c3add39ee28a4a68f89713d46c82f7f673dc6fccc1c4d526b"
BACMET_METADATA_SHA256: Final = "f0d63e9f5dd305503fa85dcf211aea15b3a6df5ee21dc904cb0304fbd8c44380"


def default_annotation_cache_dir() -> Path:
    """Return the stable cache used to resume large official-source downloads."""
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "mobiorigin" / "annotation_downloads"


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(
    path: Path, *, expected_sha256: str | None = None, expected_md5: str | None = None
) -> bool:
    if not path.is_file():
        return False
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        return False
    if expected_md5 is not None and _digest(path, "md5") != expected_md5:
        return False
    return True


def download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
) -> dict[str, object]:
    """Download a file with retained partial progress and optional frozen identity."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _verified(destination, expected_sha256=expected_sha256, expected_md5=expected_md5):
        return {
            "url": url,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "reused": True,
        }
    if destination.exists():
        destination.unlink()
    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "MobiOrigin-database-setup/1"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        append = start > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(partial, destination)
    if not _verified(destination, expected_sha256=expected_sha256, expected_md5=expected_md5):
        destination.unlink(missing_ok=True)
        raise ValueError(f"Downloaded database identity mismatch: {url}")
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "reused": False,
    }


def _archive_member(archive: Path, suffix: str, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as handle:
        matches = [item for item in handle.getmembers() if item.name.endswith(suffix)]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError(f"Archive does not contain exactly one {suffix}")
        source = handle.extractfile(matches[0])
        if source is None:
            raise ValueError(f"Could not read {suffix} from archive")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def _diamond_executable(diamond: Path) -> Path:
    resolved = shutil.which(str(diamond))
    if resolved is None:
        candidate = diamond.expanduser()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FileNotFoundError(f"DIAMOND executable not found: {diamond}")
        resolved = str(candidate.resolve())
    return Path(resolved)


def _build_diamond(diamond: Path, fasta: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(destination).removesuffix(".dmnd")
    completed = subprocess.run(
        [str(diamond), "makedb", "--in", str(fasta), "--db", prefix, "--quiet"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode or not destination.is_file():
        raise RuntimeError(
            f"DIAMOND database construction failed for {fasta.name}: " f"{completed.stderr.strip()}"
        )


def _vfdb_index(fasta: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with (
        fasta.open(encoding="utf-8", errors="replace") as source,
        destination.open("x", encoding="utf-8") as output,
    ):
        for raw in source:
            if not raw.startswith(">"):
                continue
            header = raw[1:].strip()
            subject = header.split(None, 1)[0]
            groups = [part for part in header.split("[")[1:] if "]" in part]
            category = groups[0].split("]", 1)[0] if groups else "unknown"
            output.write(f"{subject}\tVFDB_CORE\t{category}\n")
            rows += 1
    if not rows:
        raise ValueError("VFDB core FASTA contains no supported records")


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def _find_amrfinder_version(root: Path) -> Path:
    candidates = sorted({path.parent.resolve() for path in root.rglob("version.txt")})
    if not candidates:
        raise RuntimeError("AMRFinderPlus update completed without a versioned database")
    return candidates[-1]


def prepare_official_annotation_sources(
    destination: Path,
    *,
    profile: str,
    cache_dir: Path,
    diamond: Path,
    marker_database_dir: Path | None,
    amrfinder_database: Path | None,
    amrfinder_update: Path,
) -> tuple[Path, dict[str, object]]:
    """Retrieve permitted official resources and build the local search indexes."""
    if profile not in {"arg", "comprehensive"}:
        raise ValueError("Annotation database profile must be 'arg' or 'comprehensive'")
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    executable = _diamond_executable(diamond)
    downloads: dict[str, object] = {}

    card_archive = cache_dir / "card-latest.tar.bz2"
    downloads["CARD"] = download(CARD_URL, card_archive)
    card_fasta = destination / ".build" / "card.fasta"
    _archive_member(card_archive, "protein_fasta_protein_homolog_model.fasta", card_fasta)
    _archive_member(card_archive, "aro_index.tsv", destination / "card" / "aro_index.tsv")
    _build_diamond(executable, card_fasta, destination / "card" / "card.dmnd")

    sarg_archive = cache_dir / "ARGs_OAP-2.3.2.tar.gz"
    downloads["SARG"] = download(SARG_URL, sarg_archive, expected_sha256=SARG_SHA256)
    sarg_fasta = destination / ".build" / "sarg.fasta"
    _archive_member(sarg_archive, "/SARG.2.2.fasta", sarg_fasta)
    _build_diamond(executable, sarg_fasta, destination / "sarg" / "sarg.dmnd")

    if profile == "comprehensive":
        vfdb_archive = cache_dir / "VFDB_setA_pro.fas.gz"
        downloads["VFDB"] = download(VFDB_URL, vfdb_archive)
        vfdb_fasta = destination / ".build" / "vfdb.fasta"
        vfdb_fasta.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(vfdb_archive, "rb") as gzip_source, vfdb_fasta.open("xb") as output:
            shutil.copyfileobj(gzip_source, output, length=1024 * 1024)
        _vfdb_index(vfdb_fasta, destination / "vfdb" / "vfdb_indx.txt")
        _build_diamond(executable, vfdb_fasta, destination / "vfdb" / "vfdb_setA.dmnd")

        mobileog_fasta = cache_dir / "mobileOG2-dino.90_sequences.faa"
        downloads["mobileOG-db"] = download(MOBILEOG_URL, mobileog_fasta, expected_md5=MOBILEOG_MD5)
        _build_diamond(executable, mobileog_fasta, destination / "mge" / "mobileog.dmnd")

        bacmet_fasta = cache_dir / "BacMet_EXP_database.fasta"
        bacmet_metadata = cache_dir / "BacMet_EXP.753.mapping.txt"
        downloads["BacMet-fasta"] = download(
            f"{BACMET_BASE_URL}/BacMet_EXP_database.fasta",
            bacmet_fasta,
            expected_sha256=BACMET_FASTA_SHA256,
        )
        downloads["BacMet-metadata"] = download(
            f"{BACMET_BASE_URL}/BacMet_EXP.753.mapping.txt",
            bacmet_metadata,
            expected_sha256=BACMET_METADATA_SHA256,
        )
        _build_diamond(executable, bacmet_fasta, destination / "bacmet" / "bacmet.dmnd")
        _copy(bacmet_metadata, destination / "bacmet" / "Bacmet_list.tsv")

        if marker_database_dir is None:
            raise ValueError(
                "Comprehensive annotation setup requires the MobiOrigin marker database"
            )
        for filename in ("rep_proteins.dmnd", "mob_proteins.dmnd", "mpf_proteins.dmnd"):
            marker_source = marker_database_dir / filename
            if not marker_source.is_file():
                raise FileNotFoundError(f"Required marker database is missing: {marker_source}")
            _copy(marker_source, destination / "mob_suite" / filename)

    if amrfinder_database is None:
        updater = shutil.which(str(amrfinder_update))
        if updater is None:
            raise FileNotFoundError(f"AMRFinderPlus updater is not available: {amrfinder_update}")
        update_root = destination / ".amrfinder-update"
        completed = subprocess.run(
            [updater, "--database", str(update_root), "--quiet"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"AMRFinderPlus database update failed: {completed.stderr.strip()}")
        amrfinder_database = _find_amrfinder_version(update_root)

    shutil.rmtree(destination / ".build", ignore_errors=True)
    return amrfinder_database, {
        "downloads": downloads,
        "diamond": str(executable),
        "mobileog_record": MOBILEOG_RECORD,
    }
