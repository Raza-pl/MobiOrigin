"""Shared runtime limits and validation."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MAX_THREADS = 128
TEMPORARY_ENVIRONMENT_VARIABLES = ("TMPDIR", "TEMP", "TMP")


def validate_threads(threads: int) -> int:
    """Return a valid external-tool worker count or fail before execution."""
    if not 1 <= threads <= MAX_THREADS:
        raise ValueError(f"Threads must be between 1 and {MAX_THREADS}")
    return threads


def is_wsl() -> bool:
    """Return whether the current Linux process is running under WSL."""
    if sys.platform != "linux":
        return False
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="ascii").lower()
    except OSError:
        release = ""
    return "microsoft" in release or "wsl" in release


def is_wsl_windows_mount(path: Path) -> bool:
    """Return whether a path is on a conventional Windows drive mount in WSL."""
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser().absolute()
    parts = resolved.parts
    return is_wsl() and len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt"


def _writable_temporary_root(path: Path) -> Path:
    """Create and verify a candidate directory for external-tool temporary files."""
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise OSError(f"Temporary path is not a directory: {path}")
    with tempfile.NamedTemporaryFile(prefix=".mobiorigin-write-test.", dir=path):
        pass
    return path.resolve()


def resolve_temporary_root() -> tuple[Path, str, list[str]]:
    """Choose writable temporary storage, avoiding Windows mounts under WSL."""
    warnings: list[str] = []
    explicit = os.environ.get("MOBIORIGIN_TMPDIR")
    if explicit:
        requested = Path(explicit)
        if is_wsl_windows_mount(requested):
            raise RuntimeError(
                "MOBIORIGIN_TMPDIR points to a Windows-mounted WSL path. "
                "Use a Linux-native path such as $HOME/.cache/mobiorigin/tmp."
            )
        try:
            return _writable_temporary_root(requested), "MOBIORIGIN_TMPDIR", warnings
        except OSError as error:
            raise RuntimeError(
                f"MOBIORIGIN_TMPDIR is not usable: {requested}. "
                "Choose an existing writable Linux-native directory."
            ) from error

    candidates: list[tuple[Path, str]] = []
    if is_wsl():
        candidates.append((Path.home() / ".cache" / "mobiorigin" / "tmp", "WSL user cache"))
    for variable in TEMPORARY_ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable)
        if not value:
            continue
        candidate = Path(value)
        if is_wsl_windows_mount(candidate):
            warnings.append(
                f"Ignored {variable}={candidate} because Windows-mounted WSL temporary "
                "storage is unreliable for AMRFinderPlus and NCBI BLAST."
            )
            continue
        candidates.append((candidate, variable))
    candidates.extend(
        [
            (Path.home() / ".cache" / "mobiorigin" / "tmp", "user cache"),
            (Path(tempfile.gettempdir()) / "mobiorigin", "system temporary directory"),
        ]
    )

    attempted: list[str] = []
    seen: set[str] = set()
    for candidate, source in candidates:
        key = str(candidate.expanduser())
        if key in seen:
            continue
        seen.add(key)
        try:
            return _writable_temporary_root(candidate), source, warnings
        except OSError as error:
            attempted.append(f"{candidate}: {error}")
    detail = "; ".join(attempted) or "no candidate directories were available"
    raise RuntimeError(
        "No writable temporary directory is available for external tools. "
        "Set MOBIORIGIN_TMPDIR to a writable Linux-native directory. "
        f"Checked: {detail}"
    )


def temporary_storage_report() -> dict[str, Any]:
    """Return a concise user-facing report for external-tool temporary storage."""
    try:
        root, source, warnings = resolve_temporary_root()
        free_bytes = shutil.disk_usage(root).free
        return {
            "status": "PASS",
            "directory": str(root),
            "source": source,
            "free_bytes": free_bytes,
            "wsl": is_wsl(),
            "warnings": warnings,
        }
    except RuntimeError as error:
        return {
            "status": "FAIL",
            "directory": None,
            "source": None,
            "free_bytes": None,
            "wsl": is_wsl(),
            "warnings": [],
            "error": str(error),
        }


@contextmanager
def external_tool_environment(tool: str) -> Iterator[tuple[dict[str, str], Path]]:
    """Provide one isolated, writable temporary directory to an external tool."""
    root, _, _ = resolve_temporary_root()
    with tempfile.TemporaryDirectory(prefix=f"mobiorigin-{tool}.", dir=root) as temporary:
        path = Path(temporary)
        environment = os.environ.copy()
        for variable in TEMPORARY_ENVIRONMENT_VARIABLES:
            environment[variable] = str(path)
        yield environment, path


def resolve_executable(
    value: str | Path,
    *,
    label: str = "executable",
    required: bool = True,
) -> Path | None:
    """Resolve a tool while preferring the environment running MobiOrigin.

    Conda can preserve a user PATH in which ``~/bin`` precedes the selected
    environment. Looking beside the active Python first prevents an unrelated
    host DIAMOND or AMRFinderPlus executable from shadowing the solved version.
    An explicit path is always honored.
    """
    raw = os.fspath(value)
    requested = Path(raw).expanduser()
    explicit = requested.is_absolute() or requested.parent != Path(".")
    if explicit:
        if requested.is_file() and os.access(requested, os.X_OK):
            return requested.resolve()
        if required:
            raise FileNotFoundError(f"Required {label} is not executable: {requested}")
        return None

    names = [requested.name]
    if os.name == "nt" and requested.suffix.lower() != ".exe":
        names.insert(0, f"{requested.name}.exe")
    for directory in (Path(sys.prefix) / "bin", Path(sys.prefix) / "Scripts"):
        for name in names:
            candidate = directory / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()

    resolved = shutil.which(raw)
    if resolved is not None:
        return Path(resolved).resolve()
    if required:
        raise FileNotFoundError(f"Required {label} is not available: {value}")
    return None
