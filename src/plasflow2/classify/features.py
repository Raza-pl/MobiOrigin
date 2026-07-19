"""k-mer frequency feature extraction.

Vectorised numpy implementation (rev 4 — k=7 canonical).

Feature vector layout
---------------------
  k=1  :    4 dims  — GC content signal
  k=2  :   16 dims  — dinucleotide composition
  k=3  :   64 dims  — trinucleotide patterns
  k=4  :  256 dims  — tetranucleotide (classic plasmid signal)
  k=5  : 1024 dims  — pentanucleotide
  k=7  : 8192 dims  — heptanucleotide canonical (PlasFlow v1 approach)
  len  :    1 dim   — log10(length) scaled to [0,1]

k=7 canonical features (8192 dims)
-----------------------------------
For odd k there are no palindromes, so each k-mer pairs uniquely with its
reverse complement.  We use the CANONICAL form (lexicographically smaller of
the pair) to halve the feature dimension with no information loss.

At k=7, organism-specific codon-usage patterns manifest as distinctive 7-mer
frequencies.  These are sufficiently fine-grained to separate secondary
chromosomes (chromids) from plasmids of the same organism — the key failure
mode of the k=5-only model.  This mirrors PlasFlow v1's k=7 approach which
achieves F1=0.939 vs our k=5 ceiling of ~0.376 (PR curve max).

Why canonical instead of raw (16384 dims):
  kmer_vector() already sums forward + reverse-complement strands, so
  kmer_vector[i] == kmer_vector[rc(i)] always.  Canonical folding removes
  this exact redundancy, halving the feature dimension and training time
  with zero information loss.

Total: 1364 + 8192 + 1 = 9557 dims.
"""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# k-mer sizes to use for the k=1–5 block.
KMER_SIZES = (1, 2, 3, 4, 5)

# Complement mapping — kept for _reverse_complement (used by tests and CLI)
_COMPLEMENT: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _all_kmers(k: int) -> list[str]:
    """Return sorted list of all k-mers over {A, C, G, T}."""
    return ["".join(p) for p in itertools.product("ACGT", repeat=k)]


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string."""
    return "".join(_COMPLEMENT.get(b, "") for b in reversed(seq.upper()))


# Pre-build k-mer vocabularies and index maps
_VOCAB: dict[int, list[str]] = {k: _all_kmers(k) for k in KMER_SIZES}
_KMER_TO_IDX: dict[int, dict[str, int]] = {
    k: {km: i for i, km in enumerate(vocab)} for k, vocab in _VOCAB.items()
}

_KMER_DIM = sum(len(_VOCAB[k]) for k in KMER_SIZES)  # 4+16+64+256+1024 = 1364

# k=7 canonical dimensions.
# For odd k there are no palindromes, so each k-mer pairs with a different
# reverse complement.  The canonical set has exactly 4^7 / 2 = 8192 members.
K7_VOCAB_SIZE = 4**7  # 16384 total k=7 words
K7_CANON_SIZE = K7_VOCAB_SIZE // 2  # 8192 canonical k=7 words

# Feature dimension with k=7 canonical block
FEATURE_DIM_BASE = _KMER_DIM + 1  # k=1–5 + length (no k=7)  = 1365
FEATURE_DIM = _KMER_DIM + K7_CANON_SIZE + 1  # k=1–5 + k=7 + length = 9557

# Gene content feature block (appended when gene_data is provided)
GENE_DIM = 6  # gc_content, coding_density, n_orfs_per_kb, strand_switch_rate,
# canonical_sd_freq, no_rbs_freq
FEATURE_DIM_FULL = FEATURE_DIM + GENE_DIM  # 9563

# Comparative-genomics containment features (appended for Rev6+ models)
#   col 9557 (or 9563): compass_containment  — fraction of query k-mers in COMPASS plasmid sketch
#   col 9558 (or 9564): chr_containment      — fraction of query k-mers in chromosome sketch
CONTAINMENT_DIM = 2
FEATURE_DIM_CONTAINMENT = FEATURE_DIM + CONTAINMENT_DIM  # 9559
FEATURE_DIM_FULL_CONTAINMENT = FEATURE_DIM_FULL + CONTAINMENT_DIM  # 9565

# Canonical Shine-Dalgarno motifs as recognised by pyrodigal
_CANONICAL_SD: frozenset[str] = frozenset({"AGGAG", "GGAG", "AGGA", "AGG", "GGA", "GAGG"})

# Backward-compat aliases for legacy code that references these names
K6_PCA_COMPONENTS = 128  # still referenced by older scripts
K6_RAW_DIM = 4**6  # = 4096; referenced by fit_k6_pca.py
FEATURE_DIM_WITH_K6 = _KMER_DIM + 1 + K6_PCA_COMPONENTS  # = 1493

# ---------------------------------------------------------------------------
# Vectorised internals
# ---------------------------------------------------------------------------

# ASCII lookup table: byte value → base index (A=0, C=1, G=2, T=3)
_ASCII_TO_BASE: NDArray[np.uint8] = np.zeros(256, dtype=np.uint8)
for _ch, _val in [("A", 0), ("C", 1), ("G", 2), ("T", 3), ("a", 0), ("c", 1), ("g", 2), ("t", 3)]:
    _ASCII_TO_BASE[ord(_ch)] = _val

# Pre-computed base-4 power vectors for k-mer ID computation.
_POWERS: dict[int, NDArray[np.int64]] = {
    k: (4 ** np.arange(k - 1, -1, -1, dtype=np.int64)) for k in (*KMER_SIZES, 6, 7)
}


# ---------------------------------------------------------------------------
# k=7 canonical lookup table (built once at import time, ~0.05 s)
# ---------------------------------------------------------------------------
# Maps each of the 16384 k=7 raw IDs to its canonical index in [0, 8191].
# Two raw IDs (kmer and its rc) map to the same canonical index.
# The canonical representative is the one with the smaller raw ID.
#
# Build algorithm:
#   1. For each raw ID i (0..16383), compute rc_id.
#   2. Assign the pair (min(i, rc_id)) a monotonically increasing index.
#   3. Both i and rc_id receive that index in _K7_CANON_MAP.
#
def _build_k7_canon_map() -> NDArray[np.int16]:
    k = 7
    vocab = 4**k  # 16384
    powers = np.array([4 ** (k - 1 - j) for j in range(k)], dtype=np.int64)
    comp = np.array([3, 2, 1, 0], dtype=np.int64)  # A↔T, C↔G complement indices

    # Decode each raw ID into its base sequence, compute rc ID
    ids = np.arange(vocab, dtype=np.int64)
    # Extract 7 bases for each id: base[j] = (id // 4^(6-j)) % 4
    bases = np.zeros((vocab, k), dtype=np.int64)
    for j in range(k):
        bases[:, j] = (ids // powers[j]) % 4
    # Reverse-complement: complement each base, reverse the sequence
    rc_bases = comp[bases[:, ::-1]]  # shape (16384, 7)
    rc_ids = (rc_bases * powers).sum(axis=1)  # shape (16384,)

    # Assign canonical indices: iterate in raw ID order; each pair gets the
    # same index when first encountered via the smaller member.
    canon_map = np.full(vocab, -1, dtype=np.int16)
    idx = 0
    for i in range(vocab):
        if canon_map[i] == -1:
            j = int(rc_ids[i])
            canon_map[i] = idx
            canon_map[j] = idx
            idx += 1
    assert idx == K7_CANON_SIZE, f"Expected {K7_CANON_SIZE} canonical k=7 mers, got {idx}"
    return canon_map


_K7_CANON_MAP: NDArray[np.int16] = _build_k7_canon_map()


def _encode_seq(seq: str) -> NDArray[np.uint8]:
    """Encode an ASCII DNA string to a uint8 base-index array in one numpy call.

    The string must already be uppercase (caller's responsibility).
    Non-ACGT bytes (including 'N') map to 0 (treated as A); this introduces
    negligible noise for well-assembled sequences with sparse ambiguous bases.
    """
    raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return _ASCII_TO_BASE[raw]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def kmer_vector(seq: str, k: int) -> NDArray[np.float32]:
    """Compute normalised strand-invariant k-mer frequency vector.

    Uses vectorised numpy rather than Python string loops for ~100x speedup.
    For each strand (forward + reverse complement) the algorithm:
      - Encodes the sequence once via an ASCII lookup table.
      - Creates a zero-copy sliding-window view (no data is copied).
      - Converts windows to integer k-mer IDs via matrix multiply.
      - Counts IDs with np.bincount (single C pass over the data).

    The resulting counts are identical to what the original pure-Python loop
    produced for pure-ACGT sequences (sequences containing only A, C, G, T).
    Ambiguous bases (N) are treated as A, which is equivalent to the original
    behaviour of skipping them (negligible effect for ≥1 kb assembled contigs).

    Args:
        seq: DNA string (any case; non-ACGT treated as A).
        k: k-mer size; must be in KMER_SIZES (4 or 5).

    Returns:
        Float32 array of shape (4**k,), L2-normalised.
        Returns a zero vector if the sequence is shorter than k.
    """
    vocab_size = 4**k
    seq = seq.upper()

    if len(seq) < k:
        return np.zeros(vocab_size, dtype=np.float32)

    encoded: NDArray[np.uint8] = _encode_seq(seq)

    # Reverse complement: complement each base (A↔T, C↔G → 0↔3, 1↔2 → 3-base)
    # then reverse the array.  Cast through int16 to avoid uint8 wrap-around
    # before converting back to uint8.
    rc_encoded: NDArray[np.uint8] = (3 - encoded.astype(np.int16)).astype(np.uint8)[::-1]

    counts = np.zeros(vocab_size, dtype=np.float32)
    powers = _POWERS[k]

    for strand in (encoded, rc_encoded):
        # sliding_window_view returns a (L-k+1, k) view — zero-copy
        windows = np.lib.stride_tricks.sliding_window_view(strand, k).astype(np.int64)
        # matrix multiply: each row (one k-mer) → scalar ID
        kmer_ids: NDArray[np.int64] = windows @ powers
        counts += np.bincount(kmer_ids, minlength=vocab_size).astype(np.float32)

    norm = float(np.linalg.norm(counts))
    if norm > 0:
        counts /= norm
    return counts


# ---------------------------------------------------------------------------
# k=6 PCA stubs (deprecated — kept so existing scripts don't break on import)
# ---------------------------------------------------------------------------


def fit_k6_pca(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Deprecated.  The k=7 canonical architecture does not use PCA."""
    raise NotImplementedError(
        "fit_k6_pca is no longer used.  "
        "PlasFlow v2 rev-4 uses k=7 canonical features (8192 dims) instead of k=6 PCA."
    )


def load_k6_pca(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Deprecated.  Returns None so callers that guard on None still work."""
    return None


def k6_pca_vector(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Deprecated."""
    raise NotImplementedError("k=6 PCA features are no longer used.")


# ---------------------------------------------------------------------------
# k=7 canonical feature vector
# ---------------------------------------------------------------------------


def kmer_vector_k7_canonical(seq: str) -> NDArray[np.float32]:
    """Compute normalised k=7 canonical k-mer frequency vector (8192 dims).

    Each of the 16384 raw k=7 IDs is folded to its canonical representative
    (the smaller of the kmer/rc pair) using the precomputed _K7_CANON_MAP.
    Because forward and reverse-complement windows are summed, this is
    equivalent to counting canonical occurrences with multiplicity 1.

    Args:
        seq: DNA string (any case; non-ACGT treated as A).

    Returns:
        Float32 array of shape (8192,), L2-normalised.
        Returns a zero vector if the sequence is shorter than 7.
    """
    k = 7
    seq = seq.upper()

    if len(seq) < k:
        return np.zeros(K7_CANON_SIZE, dtype=np.float32)

    encoded: NDArray[np.uint8] = _encode_seq(seq)
    powers = _POWERS[k]

    # Count raw k=7 k-mer occurrences on the forward strand only. Canonical
    # folding via _K7_CANON_MAP already accounts for both strands: every raw
    # k-mer ID and its reverse complement map to the SAME canonical index
    # (canon_map[raw] == canon_map[rc(raw)] for all raw IDs, by construction
    # of _build_k7_canon_map). Consequently the reverse-complement strand's
    # canonical-ID multiset is *exactly* the same as the forward strand's —
    # not just similar, but identical, since {canon_map[rc(w)] : w in fwd
    # windows} == {canon_map[w] : w in fwd windows}. A previous version of
    # this function counted the RC strand separately and added it in, which
    # exactly doubled every count. That doubling has zero effect after L2
    # normalisation (2C / ||2C|| == C / ||C||), so it was pure wasted compute
    # — a second sliding-window pass, matmul, and bincount for a vector that
    # was always going to normalise away. Removed.
    windows = np.lib.stride_tricks.sliding_window_view(encoded, k).astype(np.int64)
    raw_ids: NDArray[np.int64] = windows @ powers  # shape (L-6,)

    # Map raw IDs → canonical IDs
    canon_ids = _K7_CANON_MAP[raw_ids]  # shape (L-6,), dtype int16

    counts = np.bincount(canon_ids.astype(np.int64), minlength=K7_CANON_SIZE).astype(np.float32)

    norm = float(np.linalg.norm(counts))
    if norm > 0:
        counts /= norm
    return counts


# ---------------------------------------------------------------------------
# Gene content feature vector
# ---------------------------------------------------------------------------


def gene_content_vector(seq: str, genes: list) -> NDArray[np.float32]:
    """Compute 6-dim gene content feature vector from a sequence + pyrodigal genes.

    Feature layout:
      0: gc_content         — fraction of G/C bases in sequence
      1: coding_density     — fraction of bp covered by ORFs, clipped to [0, 1]
      2: n_orfs_per_kb_norm — ORF density normalised: (n_orfs/kb) / 3, clipped to [0, 1]
      3: strand_switch_rate — fraction of consecutive ORF pairs that change strand
      4: canonical_sd_freq  — fraction of ORFs with a canonical Shine-Dalgarno motif
      5: no_rbs_freq        — fraction of ORFs with no detected RBS motif

    Args:
        seq:   DNA string (any case).
        genes: List of pyrodigal.Gene objects for this sequence (may be empty).

    Returns:
        Float32 array of shape (6,).
    """
    seq_upper = seq.upper()
    length_bp = max(len(seq_upper), 1)

    # GC content — computed from raw sequence, no ORF prediction required.
    gc = (seq_upper.count("G") + seq_upper.count("C")) / length_bp

    n_orfs = len(genes)
    if n_orfs == 0:
        # Safe defaults for sequences with no detectable ORFs.
        return np.array([gc, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)

    # Coding density
    covered_bp = sum(abs(g.end - g.begin) for g in genes)
    coding_density = float(min(covered_bp / length_bp, 1.0))

    # ORF density, normalised: typical assemblies have 0-3 ORFs/kb; divide by 3.
    n_orfs_per_kb_norm = float(min((n_orfs / (length_bp / 1000.0)) / 3.0, 1.0))

    # Strand switch rate
    if n_orfs > 1:
        strands = [g.strand for g in genes]
        switches = sum(1 for a, b in zip(strands, strands[1:]) if a != b)
        strand_switch_rate = switches / (n_orfs - 1)
    else:
        strand_switch_rate = 0.0

    # RBS features
    canonical_sd = sum(1 for g in genes if g.rbs_motif in _CANONICAL_SD)
    no_rbs = sum(1 for g in genes if not g.rbs_motif or g.rbs_motif == "None")
    canonical_sd_freq = canonical_sd / n_orfs
    no_rbs_freq = no_rbs / n_orfs

    return np.array(
        [
            gc,
            coding_density,
            n_orfs_per_kb_norm,
            strand_switch_rate,
            canonical_sd_freq,
            no_rbs_freq,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    sequences: list[str],
    k6_pca_path: Path | str | None = None,
    gene_data: dict[str, list] | None = None,
    seq_ids: list[str] | None = None,
    compass_sketch: NDArray[np.uint64] | None = None,
    chr_sketch: NDArray[np.uint64] | None = None,
) -> NDArray[np.float32]:
    """Extract k=1–5 + k=7-canonical k-mer features + length feature.

    When *gene_data* and *seq_ids* are provided, 6 gene content features are
    appended, extending the output from 9557 to 9563 dims.

    When *compass_sketch* and *chr_sketch* are provided (Rev6+ models), 2
    comparative-genomics containment features are appended last:
      compass_containment  — fraction of query MinHash k-mers in COMPASS plasmid sketch
      chr_containment      — fraction of query MinHash k-mers in chromosome sketch

    Feature layout:
      cols    0–3      : k=1 (4 dims)
      cols    4–19     : k=2 (16 dims)
      cols   20–83     : k=3 (64 dims)
      cols   84–339    : k=4 (256 dims)
      cols  340–1363   : k=5 (1024 dims)
      cols 1364–9555   : k=7 canonical (8192 dims)
      col  9556        : log10(length) scaled to [0, 1]
      cols 9557–9562   : gene content (6 dims, only when gene_data provided)
      cols 9557/9563+  : compass_containment, chr_containment (2 dims, Rev6)

    The `k6_pca_path` argument is accepted for API compatibility but ignored.

    Args:
        sequences:      List of DNA strings.
        k6_pca_path:    Ignored (kept for backward compat).
        gene_data:      Optional dict mapping sequence ID → list of pyrodigal.Gene
                        objects.  Must be paired with *seq_ids*.
        seq_ids:        Sequence identifiers aligned to *sequences*.  Required when
                        *gene_data* is provided.
        compass_sketch: Sorted uint64 numpy array — bottom-S MinHash hashes from
                        the COMPASS plasmid database.  When provided, compass
                        containment is appended as a feature.  Requires chr_sketch.
        chr_sketch:     Sorted uint64 numpy array — bottom-S MinHash hashes from
                        the chromosome database.  Required when compass_sketch is set.

    Returns:
        Float32 array of shape (N, 9557), (N, 9559), (N, 9563), or (N, 9565).
    """
    if k6_pca_path is not None:
        logger.debug("k6_pca_path ignored — k=7 canonical architecture does not use PCA")

    use_gene = gene_data is not None and seq_ids is not None
    use_containment = compass_sketch is not None and chr_sketch is not None
    n = len(sequences)
    if use_gene and use_containment:
        dim = FEATURE_DIM_FULL_CONTAINMENT  # 9565
    elif use_gene:
        dim = FEATURE_DIM_FULL  # 9563
    elif use_containment:
        dim = FEATURE_DIM_CONTAINMENT  # 9559
    else:
        dim = FEATURE_DIM  # 9557
    X = np.zeros((n, dim), dtype=np.float32)

    # k=1–5 block (1364 dims)
    offset = 0
    for k in KMER_SIZES:
        k_dim = 4**k
        for i, seq in enumerate(sequences):
            X[i, offset : offset + k_dim] = kmer_vector(seq, k)
        if n >= 10_000:
            logger.info("  k=%d done (%d sequences)", k, n)
        offset += k_dim

    # k=7 canonical block (8192 dims)
    logger.info("  Computing k=7 canonical features (%d dims) …", K7_CANON_SIZE)
    for i, seq in enumerate(sequences):
        X[i, offset : offset + K7_CANON_SIZE] = kmer_vector_k7_canonical(seq)
        if (i + 1) % 10_000 == 0:
            logger.info("  k=7: %d / %d sequences", i + 1, n)
    offset += K7_CANON_SIZE

    # Length feature (1 dim)
    log_min, log_max = np.log10(1_000), np.log10(1_000_000)
    for i, seq in enumerate(sequences):
        log_len = np.log10(max(1, len(seq)))
        X[i, offset] = float(np.clip((log_len - log_min) / (log_max - log_min), 0.0, 1.0))
    offset += 1

    # Gene content block (6 dims, optional)
    if use_gene:
        logger.info("  Appending gene content features (%d dims) …", GENE_DIM)
        for i, (sid, seq) in enumerate(zip(seq_ids, sequences)):
            genes = gene_data.get(sid, [])
            X[i, offset : offset + GENE_DIM] = gene_content_vector(seq, genes)
        offset += GENE_DIM

    # Comparative-genomics containment block (2 dims, Rev6+ models)
    if use_containment:
        logger.info("  Computing COMPASS + chromosome containment features (%d seqs) …", n)
        from plasflow2.classify.containment import _containment, _minhash

        compass_arr = compass_sketch.astype(np.uint64)
        chr_arr = chr_sketch.astype(np.uint64)
        for i, seq in enumerate(sequences):
            q = _minhash(seq)
            X[i, offset] = float(_containment(q, compass_arr))
            X[i, offset + 1] = float(_containment(q, chr_arr))
            if n >= 10_000 and (i + 1) % 10_000 == 0:
                logger.info("  containment: %d / %d sequences", i + 1, n)

    logger.info("Extracted features: shape %s", X.shape)
    return X


def save_features(X: NDArray[np.float32], path: Path | str) -> None:
    """Save feature matrix to an .npy file."""
    np.save(str(path), X)


def extract_features_to_npy(
    sequences: list[str],
    path: Path | str,
    *,
    chunk_size: int = 1000,
) -> tuple[int, int]:
    """Extract features in bounded-memory chunks into an atomic ``.npy`` file.

    The output is written to ``<path>.incomplete`` and renamed only after every
    chunk has been flushed. Interrupted builds therefore cannot masquerade as a
    complete training matrix.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    incomplete_path = path.with_name(f"{path.name}.incomplete")
    shape = (len(sequences), FEATURE_DIM)
    matrix = np.lib.format.open_memmap(
        incomplete_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    try:
        for start in range(0, len(sequences), chunk_size):
            end = min(start + chunk_size, len(sequences))
            matrix[start:end] = extract_features(sequences[start:end])
            matrix.flush()
            logger.info("Feature rows written: %d / %d", end, len(sequences))
    finally:
        del matrix
    os.replace(incomplete_path, path)
    return shape


def load_features(path: Path | str) -> NDArray[np.float32]:
    """Load feature matrix from an .npy file."""
    return np.load(str(path)).astype(np.float32)
