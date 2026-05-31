"""FASTA parsing, filtering, and writing utilities.

Week 1 — Day 5 implementation target.
"""

from __future__ import annotations

import bz2
import gzip
import logging
from collections.abc import Generator
from pathlib import Path

from Bio import SeqIO  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]

logger = logging.getLogger(__name__)


def _open_fasta(path: Path):
    """Open a FASTA file transparently, supporting .gz and .bz2 compression."""
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt")
    if suffix in (".bz2", ".bzip2"):
        return bz2.open(path, "rt")
    return open(path)


def load_fasta(path: Path | str, min_length: int = 1000) -> list[SeqRecord]:
    """Load sequences from a FASTA file, filtering by minimum length.

    Supports uncompressed, gzip (.gz), and bzip2 (.bz2) input files.

    Args:
        path: Path to FASTA file (plain, .gz, or .bz2).
        min_length: Minimum sequence length to keep (default 1000 bp).

    Returns:
        List of SeqRecord objects passing the length filter.
    """
    path = Path(path)
    records: list[SeqRecord] = []
    total = 0
    with _open_fasta(path) as fh:
        for record in SeqIO.parse(fh, "fasta"):
            total += 1
            if len(record.seq) >= min_length:
                records.append(record)
    logger.info(
        "Loaded %d/%d sequences from %s (min_length=%d)",
        len(records),
        total,
        path,
        min_length,
    )
    return records


def gc_content(seq: str) -> float:
    """Compute GC content (0–1) for a DNA string."""
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq) if seq else 0.0


def write_fasta(records: list[SeqRecord], path: Path | str) -> None:
    """Write SeqRecord list to a FASTA file.

    Args:
        records: Sequences to write.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(path), "fasta")
    logger.info("Wrote %d sequences to %s", len(records), path)


def iter_fasta(path: Path | str) -> Generator[SeqRecord, None, None]:
    """Lazily iterate over sequences in a FASTA file (memory-efficient).

    Supports uncompressed, .gz, and .bz2 files.
    """
    path = Path(path)
    with _open_fasta(path) as fh:
        yield from SeqIO.parse(fh, "fasta")


def split_by_label(
    records: list[SeqRecord],
    labels: list[str],
) -> dict[str, list[SeqRecord]]:
    """Bin sequences into groups by their predicted label.

    Args:
        records: Sequences (same order as labels).
        labels: Predicted class per sequence.

    Returns:
        Dict mapping class name → list of SeqRecord.

    TODO (Week 4 — Day 20): integrate with full output writer.
    """
    bins: dict[str, list[SeqRecord]] = {}
    for record, label in zip(records, labels, strict=True):
        bins.setdefault(label, []).append(record)
    return bins
