"""Unit tests for k-mer feature extraction.

FEATURE_DIM = 9557: k=1–5 (1364) + k=7 canonical (8192) + log10 length (1).
"""

import numpy as np
from plasflow2.classify.features import (
    _KMER_DIM,
    FEATURE_DIM,
    K7_CANON_SIZE,
    extract_features,
    extract_features_to_npy,
    kmer_vector,
    kmer_vector_k7_canonical,
)


def test_kmer_vector_shape() -> None:
    seq = "ACGTACGTACGT"
    v = kmer_vector(seq, k=4)
    assert v.shape == (256,), f"Expected (256,) got {v.shape}"


def test_kmer_vector_normalised() -> None:
    seq = "ACGTACGTACGT" * 10
    v = kmer_vector(seq, k=4)
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-5, f"Expected L2-norm ≈ 1.0, got {norm}"


def test_kmer_vector_short_sequence() -> None:
    """Sequences shorter than k should produce a zero vector."""
    v = kmer_vector("ACG", k=4)
    assert np.all(v == 0)


def test_extract_features_shape() -> None:
    seqs = ["ACGT" * 100, "GGCC" * 100, "TTAA" * 100]
    X = extract_features(seqs)
    assert X.shape == (3, FEATURE_DIM), f"Expected (3, {FEATURE_DIM}), got {X.shape}"


def test_kmer_vector_default_still_encodes_n_as_a() -> None:
    """Default (skip_ambiguous=False) must stay bit-identical to the
    currently-deployed model's training-time behaviour -- an N-containing
    sequence must match one with every N literally replaced by A."""
    seq_with_n = "ACGTACGTNNNACGTACGT"
    seq_n_as_a = seq_with_n.replace("N", "A")
    v_n = kmer_vector(seq_with_n, k=4)
    v_a = kmer_vector(seq_n_as_a, k=4)
    np.testing.assert_allclose(v_n, v_a)

    v7_n = kmer_vector_k7_canonical(seq_with_n)
    v7_a = kmer_vector_k7_canonical(seq_n_as_a)
    np.testing.assert_allclose(v7_n, v7_a)


def test_kmer_vector_skip_ambiguous_excludes_windows_touching_n() -> None:
    """skip_ambiguous=True must never count a k-mer whose window touches a
    non-ACGT position, on either strand, and must not fabricate spurious
    k-mers by merely encoding N as A."""
    k = 4
    # A single N in the middle: every 4-mer window overlapping position 8
    # (0-indexed) must be excluded -- i.e. windows starting at indices 5-8.
    seq = "ACGTACGTNACGTACGT"  # len 17, N at index 8
    v = kmer_vector(seq, k=k, skip_ambiguous=True)

    # Brute-force expected count: only count clean ACGT windows (both strands).
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    rc = "".join(comp.get(b, "N") for b in reversed(seq))
    expected = np.zeros(4**k, dtype=np.float64)
    bases = "ACGT"
    for strand in (seq, rc):
        for i in range(len(strand) - k + 1):
            window = strand[i : i + k]
            if all(c in bases for c in window):
                idx = 0
                for c in window:
                    idx = idx * 4 + bases.index(c)
                expected[idx] += 1
    norm = np.linalg.norm(expected)
    if norm > 0:
        expected /= norm

    np.testing.assert_allclose(v, expected.astype(np.float32), atol=1e-6)

    # And it must differ from the encode-as-A default -- the whole point of
    # the fix is that these two policies produce different feature vectors
    # for any sequence containing an ambiguous base.
    v_default = kmer_vector(seq, k=k, skip_ambiguous=False)
    assert not np.allclose(v, v_default)


def test_kmer_vector_skip_ambiguous_matches_default_when_no_ambiguous_bases() -> None:
    """With a pure-ACGT sequence, skip_ambiguous shouldn't change anything --
    there's nothing to skip."""
    seq = "ACGTACGTACGTACGTACGTACGT"
    v_default = kmer_vector(seq, k=4, skip_ambiguous=False)
    v_skip = kmer_vector(seq, k=4, skip_ambiguous=True)
    np.testing.assert_allclose(v_default, v_skip)

    v7_default = kmer_vector_k7_canonical(seq, skip_ambiguous=False)
    v7_skip = kmer_vector_k7_canonical(seq, skip_ambiguous=True)
    np.testing.assert_allclose(v7_default, v7_skip)


def test_extract_features_dtype() -> None:
    seqs = ["ACGT" * 50]
    X = extract_features(seqs)
    assert X.dtype == np.float32


def test_extract_features_to_npy_is_chunked_and_atomic(tmp_path) -> None:
    seqs = ["ACGT" * 50, "GGCC" * 50, "TTAA" * 50]
    path = tmp_path / "features.npy"

    shape = extract_features_to_npy(seqs, path, chunk_size=2)

    assert shape == (3, FEATURE_DIM)
    assert path.exists()
    assert not (tmp_path / "features.npy.incomplete").exists()
    np.testing.assert_allclose(np.load(path), extract_features(seqs))


def test_extract_features_different_seqs() -> None:
    """Different sequences should produce different feature vectors."""
    X = extract_features(["ACGT" * 100, "GGCC" * 100])
    assert not np.allclose(X[0], X[1]), "Expected distinct vectors for distinct sequences"


def test_kmer_vector_rc_invariant() -> None:
    """Reverse complement of a sequence should produce the same feature vector."""
    from plasflow2.classify.features import _reverse_complement

    seq = "ACGTTAGCCA" * 20
    rc = _reverse_complement(seq)
    v_fwd = kmer_vector(seq, k=4)
    v_rc = kmer_vector(rc, k=4)
    np.testing.assert_allclose(
        v_fwd, v_rc, atol=1e-5, err_msg="RC of sequence should yield identical k-mer vector"
    )


def test_reverse_complement_correctness() -> None:
    """Spot-check the RC helper."""
    from plasflow2.classify.features import _reverse_complement

    assert _reverse_complement("ACGT") == "ACGT"  # palindrome
    assert _reverse_complement("AAAA") == "TTTT"
    assert _reverse_complement("GCGC") == "GCGC"  # palindrome
    assert _reverse_complement("ATCG") == "CGAT"


def test_feature_dim_is_9557() -> None:
    """FEATURE_DIM must equal _KMER_DIM + K7_CANON_SIZE + 1 (length feature)."""
    assert FEATURE_DIM == _KMER_DIM + K7_CANON_SIZE + 1
    assert FEATURE_DIM == 9557


def test_length_feature_increases_with_sequence_length() -> None:
    """Longer sequences should have a larger length feature (last column)."""
    short = "ACGT" * 250  # 1000 bp
    long = "ACGT" * 2500  # 10000 bp
    X = extract_features([short, long])
    assert X[1, -1] > X[0, -1], "Length feature should be larger for the longer sequence"


def test_length_feature_in_zero_one_range() -> None:
    """Length feature (last column) must be in [0, 1] for typical contig lengths."""
    seqs = ["ACGT" * 250, "ACGT" * 2500, "ACGT" * 25000]  # 1 kb, 10 kb, 100 kb
    X = extract_features(seqs)
    assert np.all(X[:, -1] >= 0.0)
    assert np.all(X[:, -1] <= 1.0)
