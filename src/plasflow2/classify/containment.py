"""COMPASS MinHash containment filter for post-hoc plasmid validation.

After the MLP predicts a sequence as plasmid, this module checks whether the
sequence's canonical k-mers appear in the COMPASS plasmid database sketch.
Sequences with no COMPASS k-mer overlap are likely chromosomal or novel mobile
elements that resemble plasmids by nucleotide composition alone.

Sketch format
-------------
A single .npy file containing a **sorted uint64** numpy array of the bottom-S
MinHash values across all canonical 21-mers in the COMPASS plasmid database
(k=21, splitmix64 hash, bottom-S sketch).  The default sketch is
``data/databases/sketch_compass_k21_s5m.npy`` (5 million hashes, ~40 MB).

Usage
-----
The module is intentionally side-effect-free on import.  Load the sketch once
and call ``plasmid_containment()`` on each candidate sequence::

    from plasflow2.classify.containment import CompassFilter
    filt = CompassFilter.load(sketch_path)
    keep = filt.check(sequence)          # True = retain as plasmid
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------------------------------------------------------
# MinHash constants — must match build_comparative_sketches.py
# ---------------------------------------------------------------------------
_K = 21
_BASE = np.zeros(256, dtype=np.uint64)
for _b, _v in zip(b"ACGTacgt", [0, 1, 2, 3, 0, 1, 2, 3]):
    _BASE[_b] = _v
_RC = np.array([3, 2, 1, 0], dtype=np.uint64)
_POW = np.array([4 ** (_K - 1 - j) for j in range(_K)], dtype=np.uint64)


def _mix64(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _minhash(seq: str, S: int = 5_000) -> np.ndarray:
    """Compute bottom-S canonical k-mer MinHash sketch for a sequence."""
    arr = _BASE[np.frombuffer(seq.upper().encode("ascii", errors="replace"), dtype=np.uint8)]
    if len(arr) < _K:
        return np.array([], dtype=np.uint64)
    w = sliding_window_view(arr, _K)
    fwd = (w * _POW).sum(axis=1)
    rc = (_RC[w[:, ::-1]] * _POW).sum(axis=1)
    h = _mix64(np.where(fwd < rc, fwd, rc))
    if len(h) <= S:
        return np.sort(h)
    return np.sort(h[np.argpartition(h, S)[:S]])


def _containment(query: np.ndarray, db_sketch: np.ndarray) -> float:
    """Fraction of query MinHash k-mers present in db_sketch (both sorted).

    Uses binary search (O(N log M)) instead of np.isin (O(M log M)) for a
    ~50x speedup when db_sketch is large (e.g. 5 M hashes).
    """
    if len(query) == 0 or len(db_sketch) == 0:
        return 0.0
    # Binary-search each query hash into the sorted db_sketch.
    idx = np.searchsorted(db_sketch, query)
    # Clamp to valid range before indexing.
    mask = idx < len(db_sketch)
    found = np.zeros(len(query), dtype=bool)
    found[mask] = db_sketch[idx[mask]] == query[mask]
    return float(found.mean())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CompassFilter:
    """Loaded COMPASS sketch, ready to test sequences.

    Parameters
    ----------
    sketch:
        Sorted uint64 numpy array (bottom-S MinHash hashes).
    threshold:
        Minimum containment score to retain a sequence as plasmid.
        At threshold=0.001, ~97.5% of known plasmids are retained while
        ~64% of false-positive chromosomes are rejected (Tier 1 benchmark).
    """

    #: Default sketch path relative to the Plasflow project root.
    DEFAULT_SKETCH = Path("data/databases/sketch_compass_k21_s5m.npy")

    def __init__(self, sketch: np.ndarray, threshold: float = 0.001) -> None:
        if sketch.dtype != np.uint64:
            sketch = sketch.astype(np.uint64)
        self._sketch = sketch
        self.threshold = threshold

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        sketch_path: Path | str | None = None,
        *,
        threshold: float = 0.001,
        project_root: Path | str | None = None,
    ) -> CompassFilter:
        """Load sketch from *sketch_path* (or the default location).

        Parameters
        ----------
        sketch_path:
            Explicit path to a ``.npy`` sketch file.  If *None*, the method
            searches ``project_root / DEFAULT_SKETCH`` then falls back to
            ``data/databases/sketch_compass_k21_s5m.npy`` relative to cwd.
        project_root:
            Optional root directory for the default sketch location.
        threshold:
            Containment threshold (default 0.001).

        Raises
        ------
        FileNotFoundError
            If no sketch file can be located.
        """
        path: Path | None
        if sketch_path is not None:
            path = Path(sketch_path)
        else:
            candidates: list[Path] = []
            if project_root is not None:
                candidates.append(Path(project_root) / cls.DEFAULT_SKETCH)
            candidates.append(Path(__file__).parent.parent.parent.parent / cls.DEFAULT_SKETCH)
            candidates.append(Path.cwd() / cls.DEFAULT_SKETCH)
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                raise FileNotFoundError(
                    "COMPASS sketch not found. Expected at one of:\n"
                    + "\n".join(f"  {p}" for p in candidates)
                    + "\nRun scripts/build_comparative_sketches.py to build it."
                )

        sketch = np.load(path)
        if sketch.dtype != np.uint64:
            sketch = sketch.astype(np.uint64)
        return cls(sketch, threshold=threshold)

    # ------------------------------------------------------------------
    def score(self, sequence: str) -> float:
        """Return containment score of *sequence* against the COMPASS sketch."""
        return _containment(_minhash(sequence), self._sketch)

    def check(self, sequence: str) -> bool:
        """Return True if the sequence has sufficient COMPASS containment."""
        return self.score(sequence) >= self.threshold

    def filter_label(
        self,
        predicted_label: str,
        sequence: str,
        *,
        reclassify_as: str = "chromosome",
    ) -> tuple[str, float]:
        """Apply containment check and potentially override label.

        Parameters
        ----------
        predicted_label:
            The label assigned by the MLP (e.g. ``"plasmid"``).
        sequence:
            The nucleotide sequence string.
        reclassify_as:
            Label to assign if the plasmid prediction is rejected.

        Returns
        -------
        (final_label, containment_score)
        """
        if predicted_label != "plasmid":
            return predicted_label, 0.0
        c = self.score(sequence)
        if c < self.threshold:
            return reclassify_as, c
        return "plasmid", c

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"CompassFilter(sketch_size={len(self._sketch):,}, " f"threshold={self.threshold})"
