"""Frozen 9,557-dimensional MobiOrigin sequence features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

KMER_SIZES = (1, 2, 3, 4, 5)
K7_VOCAB_SIZE = 4**7
K7_CANON_SIZE = K7_VOCAB_SIZE // 2
FEATURE_DIM = sum(4**k for k in KMER_SIZES) + K7_CANON_SIZE + 1

_ASCII_TO_BASE: NDArray[np.uint8] = np.zeros(256, dtype=np.uint8)
_ASCII_IS_ACGT: NDArray[np.bool_] = np.zeros(256, dtype=np.bool_)
for _character, _value in (
    ("A", 0),
    ("C", 1),
    ("G", 2),
    ("T", 3),
    ("a", 0),
    ("c", 1),
    ("g", 2),
    ("t", 3),
):
    _ASCII_TO_BASE[ord(_character)] = _value
    _ASCII_IS_ACGT[ord(_character)] = True

_POWERS: dict[int, NDArray[np.int64]] = {
    k: 4 ** np.arange(k - 1, -1, -1, dtype=np.int64) for k in (*KMER_SIZES, 7)
}


def _build_k7_canonical_map() -> NDArray[np.int16]:
    powers = _POWERS[7]
    complement = np.array([3, 2, 1, 0], dtype=np.int64)
    identifiers = np.arange(K7_VOCAB_SIZE, dtype=np.int64)
    bases = np.zeros((K7_VOCAB_SIZE, 7), dtype=np.int64)
    for index in range(7):
        bases[:, index] = (identifiers // powers[index]) % 4
    reverse_complement = (complement[bases[:, ::-1]] * powers).sum(axis=1)
    result = np.full(K7_VOCAB_SIZE, -1, dtype=np.int16)
    canonical_index = 0
    for identifier in range(K7_VOCAB_SIZE):
        if result[identifier] == -1:
            partner = int(reverse_complement[identifier])
            result[identifier] = canonical_index
            result[partner] = canonical_index
            canonical_index += 1
    if canonical_index != K7_CANON_SIZE:
        raise RuntimeError("Canonical k=7 feature map has the wrong size")
    return result


_K7_CANONICAL_MAP = _build_k7_canonical_map()


def _encode(sequence: str) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return _ASCII_TO_BASE[raw], _ASCII_IS_ACGT[raw]


def kmer_vector(sequence: str, k: int) -> NDArray[np.float32]:
    """Return the exact frozen strand-invariant k-mer vector."""
    if k not in KMER_SIZES:
        raise ValueError(f"Unsupported k-mer size: {k}")
    sequence = sequence.upper()
    if len(sequence) < k:
        return np.zeros(4**k, dtype=np.float32)
    encoded, valid = _encode(sequence)
    reverse_complement = (3 - encoded.astype(np.int16)).astype(np.uint8)[::-1]
    reverse_valid = valid[::-1]
    counts = np.zeros(4**k, dtype=np.float32)
    for strand, strand_valid in ((encoded, valid), (reverse_complement, reverse_valid)):
        windows = np.lib.stride_tricks.sliding_window_view(strand, k).astype(np.int64)
        valid_windows = np.lib.stride_tricks.sliding_window_view(strand_valid, k).all(axis=1)
        if not np.any(valid_windows):
            continue
        identifiers = windows @ _POWERS[k]
        counts += np.bincount(identifiers[valid_windows], minlength=4**k).astype(np.float32)
    norm = float(np.linalg.norm(counts))
    if norm:
        counts /= norm
    return counts


def k7_canonical_vector(sequence: str) -> NDArray[np.float32]:
    """Return the exact frozen canonical k=7 vector."""
    sequence = sequence.upper()
    if len(sequence) < 7:
        return np.zeros(K7_CANON_SIZE, dtype=np.float32)
    encoded, valid = _encode(sequence)
    windows = np.lib.stride_tricks.sliding_window_view(encoded, 7).astype(np.int64)
    valid_windows = np.lib.stride_tricks.sliding_window_view(valid, 7).all(axis=1)
    if not np.any(valid_windows):
        return np.zeros(K7_CANON_SIZE, dtype=np.float32)
    raw_identifiers = windows @ _POWERS[7]
    canonical = _K7_CANONICAL_MAP[raw_identifiers[valid_windows]]
    counts = np.bincount(canonical.astype(np.int64), minlength=K7_CANON_SIZE).astype(np.float32)
    norm = float(np.linalg.norm(counts))
    if norm:
        counts /= norm
    return counts


def extract_sequence_features(sequences: list[str]) -> NDArray[np.float32]:
    """Extract frozen features while excluding ambiguity-spanning windows.

    The behavior is byte-identical to the frozen representation for ACGT-only
    sequences. Windows containing an IUPAC ambiguity symbol contribute no
    k-mer count instead of being silently interpreted as adenine.
    """
    matrix = np.zeros((len(sequences), FEATURE_DIM), dtype=np.float32)
    offset = 0
    for k in KMER_SIZES:
        width = 4**k
        for index, sequence in enumerate(sequences):
            matrix[index, offset : offset + width] = kmer_vector(sequence, k)
        offset += width
    for index, sequence in enumerate(sequences):
        matrix[index, offset : offset + K7_CANON_SIZE] = k7_canonical_vector(sequence)
    offset += K7_CANON_SIZE
    log_min, log_max = np.log10(1_000), np.log10(1_000_000)
    for index, sequence in enumerate(sequences):
        log_length = np.log10(max(1, len(sequence)))
        matrix[index, offset] = float(
            np.clip((log_length - log_min) / (log_max - log_min), 0.0, 1.0)
        )
    return matrix
