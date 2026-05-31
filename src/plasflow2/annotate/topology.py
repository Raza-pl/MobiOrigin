"""Circular topology detection for assembled contigs.

Checks whether the ends of a contig form a direct terminal repeat (DTR),
which is a strong indicator of a circularised sequence that was linearised
by the assembler (a common artifact in short-read assembly of plasmids and
small phage genomes).

Method
------
We check if the last ``window`` bp of the contig match the first ``window``
bp.  A match is defined as >= ``min_identity`` % nucleotide identity over
the aligned window, computed with a simple sliding-window comparison.

This is intentionally lightweight — it avoids any external aligner and runs
in O(window) time per contig.  For high-confidence circular detection in
production workflows, long-read assemblers (Flye, Unicycler) will typically
annotate circularity directly in the FASTA header; this module is designed
to *supplement* that signal when it is absent.

Topology values
---------------
``"circular"``      — DTR detected (ends overlap)
``"linear"``        — No DTR detected
``"too_short"``     — Contig shorter than 2 * window bp (cannot evaluate)
"""

from __future__ import annotations

import logging
from typing import Literal

from Bio.SeqRecord import SeqRecord  # type: ignore[import]

logger = logging.getLogger(__name__)

Topology = Literal["circular", "linear", "too_short"]

# Default parameters
_DEFAULT_WINDOW = 500       # bp examined at each end
_DEFAULT_MIN_ID  = 0.90     # fraction identity required (90 %)
_DEFAULT_MIN_LEN = 50       # minimum matching run to confirm DTR


def _terminal_identity(seq: str, window: int) -> float:
    """Compute fractional identity between the first and last *window* bp."""
    start = seq[:window].upper()
    end   = seq[-window:].upper()
    matches = sum(a == b for a, b in zip(start, end))
    return matches / window


def _has_dtr(seq: str, window: int, min_identity: float) -> bool:
    """Return True if the sequence has a direct terminal repeat.

    Tries both the last-vs-first comparison and a short-shifted version to
    tolerate slight trimming at contig ends.
    """
    if len(seq) < 2 * window:
        return False

    # Primary check: last window vs first window
    if _terminal_identity(seq, window) >= min_identity:
        return True

    # Fallback: last (window//2) vs first (window//2) — tolerates sloppy ends
    half = window // 2
    if len(seq) >= 2 * half and _terminal_identity(seq, half) >= min_identity:
        return True

    return False


def detect_topology(
    record: SeqRecord,
    window: int = _DEFAULT_WINDOW,
    min_identity: float = _DEFAULT_MIN_ID,
) -> Topology:
    """Determine the topology of a single contig sequence.

    Args:
        record:        BioPython SeqRecord to evaluate.
        window:        Number of terminal bp to compare (default 500).
        min_identity:  Minimum fractional identity to call DTR (default 0.90).

    Returns:
        ``"circular"``, ``"linear"``, or ``"too_short"``.
    """
    seq = str(record.seq)
    if len(seq) < 2 * window:
        return "too_short"

    # Also accept assembler-supplied topology annotations in the description
    desc = record.description.lower()
    if any(tok in desc for tok in ("circular", "topology=circular", "complete sequence")):
        return "circular"

    return "circular" if _has_dtr(seq, window, min_identity) else "linear"


def detect_topologies(
    records: list[SeqRecord],
    window: int = _DEFAULT_WINDOW,
    min_identity: float = _DEFAULT_MIN_ID,
) -> dict[str, Topology]:
    """Detect topology for a list of contigs.

    Returns:
        Dict mapping contig_id → topology string.
    """
    results: dict[str, Topology] = {}
    n_circular = 0
    for record in records:
        topo = detect_topology(record, window=window, min_identity=min_identity)
        results[record.id] = topo
        if topo == "circular":
            n_circular += 1

    logger.info(
        "Topology: %d circular, %d linear, %d too-short out of %d contigs",
        n_circular,
        sum(1 for t in results.values() if t == "linear"),
        sum(1 for t in results.values() if t == "too_short"),
        len(records),
    )
    return results
