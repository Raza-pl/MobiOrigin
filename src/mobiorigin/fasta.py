"""Strict, order-preserving FASTA input handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IUPAC_DNA = frozenset("ACGTRYSWKMBDHVN")
MINIMUM_SUPPORTED_BP = 1_000
MAXIMUM_SUPPORTED_BP = 500_000


@dataclass(frozen=True)
class FastaRecord:
    """One normalized FASTA record."""

    identifier: str
    sequence: str

    @property
    def supported(self) -> bool:
        return MINIMUM_SUPPORTED_BP <= len(self.sequence) <= MAXIMUM_SUPPORTED_BP


def read_fasta(path: Path) -> list[FastaRecord]:
    """Read a non-empty FASTA with unique first-token identifiers."""
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

    with path.open("r", encoding="ascii") as handle:
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
