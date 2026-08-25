"""Shared runtime limits and validation."""

from __future__ import annotations

MAX_THREADS = 128


def validate_threads(threads: int) -> int:
    """Return a valid external-tool worker count or fail before execution."""
    if not 1 <= threads <= MAX_THREADS:
        raise ValueError(f"Threads must be between 1 and {MAX_THREADS}")
    return threads
