"""Strict, order-preserving FASTA input handling."""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from pathlib import Path

IUPAC_DNA = frozenset("ACGTRYSWKMBDHVN")
UNAMBIGUOUS_DNA = frozenset("ACGT")
STRONG_AMBIGUITY_WARNING_FRACTION = 0.001
MINIMUM_SUPPORTED_BP = 1_000
MAXIMUM_SUPPORTED_BP = 500_000
FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".fas")


@dataclass(frozen=True)
class FastaRecord:
    """One normalized FASTA record."""

    identifier: str
    sequence: str

    @property
    def supported(self) -> bool:
        return MINIMUM_SUPPORTED_BP <= len(self.sequence) <= MAXIMUM_SUPPORTED_BP


@dataclass(frozen=True)
class SequenceQuality:
    """Transparent input-quality measurements reported with each prediction."""

    non_acgt_bases: int
    non_acgt_fraction: float
    n_bases: int
    n_fraction: float
    normalized_4mer_entropy: float
    warnings: tuple[str, ...]


def normalized_kmer_entropy(sequence: str, k: int = 4) -> float:
    """Return normalized Shannon entropy for unambiguous DNA k-mers.

    Windows containing an IUPAC ambiguity code are excluded. The result ranges
    from 0 to 1 and is descriptive; MobiOrigin does not use it to change or
    reject a prediction.
    """
    if len(sequence) < k:
        return 0.0
    counts = [0] * (4**k)
    total = 0
    code = 0
    valid_run = 0
    mask = (4**k) - 1
    values = {"A": 0, "C": 1, "G": 2, "T": 3}
    for base in sequence:
        value = values.get(base)
        if value is None:
            code = 0
            valid_run = 0
            continue
        code = ((code << 2) | value) & mask
        valid_run += 1
        if valid_run >= k:
            counts[code] += 1
            total += 1
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count:
            probability = count / total
            entropy -= probability * math.log(probability)
    return entropy / math.log(4**k)


def sequence_quality(sequence: str) -> SequenceQuality:
    """Measure ambiguity and complexity without modifying the input record."""
    length = len(sequence)
    non_acgt = sum(base not in UNAMBIGUOUS_DNA for base in sequence)
    n_bases = sequence.count("N")
    non_acgt_fraction = non_acgt / length
    warnings: list[str] = []
    if non_acgt:
        warnings.append("non_acgt_present")
    if non_acgt_fraction >= STRONG_AMBIGUITY_WARNING_FRACTION:
        warnings.append("non_acgt_fraction_ge_0.001")
    return SequenceQuality(
        non_acgt_bases=non_acgt,
        non_acgt_fraction=non_acgt_fraction,
        n_bases=n_bases,
        n_fraction=n_bases / length,
        normalized_4mer_entropy=normalized_kmer_entropy(sequence),
        warnings=tuple(warnings),
    )


def resolve_fasta_input(path: Path) -> Path:
    """Resolve one FASTA path or raise an actionable path error."""
    expanded = path.expanduser()
    if expanded.is_file():
        return expanded
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    parent = absolute.parent
    nearby: list[str] = []
    if parent.is_dir():
        nearby = sorted(
            item.name
            for item in parent.iterdir()
            if item.is_file()
            and (
                item.suffix.lower() in FASTA_SUFFIXES
                or any(item.name.lower().endswith(f"{suffix}.gz") for suffix in FASTA_SUFFIXES)
            )
        )[:10]
    message = f"Input FASTA was not found: {absolute}. " f"Current directory: {Path.cwd()}."
    if nearby:
        message += f" FASTA files in {parent}: {', '.join(nearby)}."
    message += " Use the exact filename or an absolute path."
    raise FileNotFoundError(message)


def read_fasta(path: Path) -> list[FastaRecord]:
    """Read a non-empty FASTA with unique first-token identifiers."""
    path = resolve_fasta_input(path)
    records: list[FastaRecord] = []
    identifier: str | None = None
    sequence: list[str] = []

    def append_record() -> None:
        if identifier is None:
            return
        value = "".join(sequence).upper()
        if not value:
            raise ValueError(f"FASTA record is empty: {identifier}")
        unexpected = sorted(set(value) - IUPAC_DNA)
        if unexpected:
            raise ValueError(
                f"FASTA record {identifier} contains unsupported symbols: {''.join(unexpected)}"
            )
        records.append(FastaRecord(identifier, value))

    if path.name.lower().endswith(".gz"):
        handle_context = gzip.open(path, "rt", encoding="ascii")
    else:
        handle_context = path.open("r", encoding="ascii")
    with handle_context as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                append_record()
                identifier = line[1:].split(None, 1)[0]
                if not identifier:
                    raise ValueError("FASTA header has no identifier")
                sequence = []
            else:
                if identifier is None:
                    raise ValueError("FASTA sequence occurs before its first header")
                sequence.append(line)
    append_record()
    if not records:
        raise ValueError("Input FASTA contains no records")
    identifiers = [record.identifier for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("FASTA identifiers must be unique")
    return records
