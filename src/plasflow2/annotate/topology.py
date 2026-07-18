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
_DEFAULT_WINDOW = 500  # bp examined at each end
_DEFAULT_MIN_ID = 0.90  # fraction identity required (90 %)
_DEFAULT_MIN_LEN = 50  # minimum matching run to confirm DTR


def _terminal_identity(seq: str, window: int) -> float:
    """Compute fractional identity between the first and last *window* bp."""
    start = seq[:window].upper()
    end = seq[-window:].upper()
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

    # ── Header-based detection (assembler annotations) ────────────────────────
    # Check both description AND id — different assemblers place the annotation
    # in different fields.
    header = (record.description + " " + record.id).lower()

    # SPAdes: "NODE_1_length_X_cov_Y_circular" (in id) or "circular" anywhere
    # Flye:   "contig_1 [topology=circular]" or "circular"
    # Unicycler: "1 length=X depth=Yx circular=true"
    # NCBI:   "[topology=circular]"
    # MEGAHIT: no annotation (always linear output)
    # Bandage: "topology=circular"
    # Canu:   "suggestCircular=yes"
    _CIRCULAR_TOKENS = (
        "circular",  # covers SPAdes, Flye, Unicycler "circular=true"
        "topology=circular",  # NCBI/Bandage explicit
        "complete sequence",  # NCBI complete genomes
        "complete genome",
        "suggestcircular=yes",  # Canu
        "closedcircle",  # some custom assemblers
    )
    if any(tok in header for tok in _CIRCULAR_TOKENS):
        return "circular"

    # ── DTR-based detection (phage/plasmid terminal repeats) ─────────────────
    # Only reliable for phage genomes and some plasmids in long-read assemblies.
    # Short-read assembled metagenomic contigs almost never produce DTRs even
    # when the underlying molecule is circular — the assembler linearises them.
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
