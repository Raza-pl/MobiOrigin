"""Shared runtime limits and validation."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MAX_THREADS = 128


def validate_threads(threads: int) -> int:
    """Return a valid external-tool worker count or fail before execution."""
    if not 1 <= threads <= MAX_THREADS:
        raise ValueError(f"Threads must be between 1 and {MAX_THREADS}")
    return threads


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
