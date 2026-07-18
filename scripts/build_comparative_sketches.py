#!/usr/bin/env python3
"""Build MinHash sketches for comparative k-mer containment (Plasmer-style).

Operates in three modes:
  --mode partial    Process seqs [--offset, --offset+--count) from FASTA, save partial hashes.
  --mode merge      Merge all partial_*.npy files in --out-dir into a single sketch.
  --mode sketch     Build full sketch in one pass (feasible for small DBs like chromosomes.fna).

Usage:
  # Build COMPASS sketch in batches (for 45-second bash environments):
  python3 build_comparative_sketches.py --mode partial \\
      --fasta data/databases/plasmids/COMPASS.fna \\
      --out-dir data/databases/sketches/compass \\
      --offset 0 --count 300 --sketch-size 500000

  # Repeat with --offset 300, 600, 900 … until done.

  python3 build_comparative_sketches.py --mode merge \\
      --out-dir data/databases/sketches/compass \\
      --sketch-size 500000 \\
      --out data/databases/sketch_compass_k21_s500k.npy

  # Build chromosome sketch in one pass (225 seqs, uses chunked k-mer extraction):
  python3 build_comparative_sketches.py --mode sketch \\
      --fasta data/databases/chromosomes.fna \\
      --out data/databases/sketch_chromosomes_k21_s500k.npy \\
      --sketch-size 500000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------------------------------------------------------
# MinHash core (k=21, canonical, splitmix64 hash)
# ---------------------------------------------------------------------------
_BASE = np.zeros(256, dtype=np.uint64)
for _b, _v in zip(b"ACGTacgt", [0, 1, 2, 3, 0, 1, 2, 3]):
    _BASE[_b] = _v
_RC = np.array([3, 2, 1, 0], dtype=np.uint64)

K = 21
_POW = np.array([4 ** (K - 1 - j) for j in range(K)], dtype=np.uint64)


def _mix64(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _canonical_hashes_chunked(arr: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Compute all canonical k-mer hashes for a sequence encoded as uint64 array.

    Processes in chunks to keep peak memory bounded (chunk * K * 8 bytes).
    """
    n = len(arr)
    if n < K:
        return np.array([], dtype=np.uint64)
    windows = sliding_window_view(arr, K)  # view – no copy
    out = []
    for start in range(0, len(windows), chunk):
        w = windows[start : start + chunk]  # materialises chunk only
        fwd = (w * _POW).sum(axis=1)
        rc = (_RC[w[:, ::-1]] * _POW).sum(axis=1)
        out.append(_mix64(np.where(fwd < rc, fwd, rc)))
    return np.concatenate(out) if out else np.array([], dtype=np.uint64)


def _seq_to_arr(seq: str) -> np.ndarray:
    return _BASE[
        np.frombuffer(seq.upper().encode("ascii", errors="replace"), dtype=np.uint8)
    ]


# ---------------------------------------------------------------------------
# Sketch accumulator: streaming bottom-S
# ---------------------------------------------------------------------------

class BottomSketch:
    """Maintain the S smallest hashes seen so far across all sequences."""

    def __init__(self, S: int) -> None:
        self.S = S
        self._pool: list[np.ndarray] = []
        self._pool_size = 0
        self._sketch: np.ndarray | None = None  # sorted, len==S once full
        self._threshold = np.iinfo(np.uint64).max

    def add(self, hashes: np.ndarray) -> None:
        if len(hashes) == 0:
            return
        # Quick filter: drop hashes above current threshold
        if self._sketch is not None:
            hashes = hashes[hashes < self._threshold]
        if len(hashes) == 0:
            return
        self._pool.append(hashes)
        self._pool_size += len(hashes)
        # Compact every time pool_size > 4×S to avoid unbounded memory
        if self._pool_size > 4 * self.S:
            self._compact()

    def _compact(self) -> None:
        all_h = np.concatenate(self._pool)
        if len(all_h) > self.S:
            idx = np.argpartition(all_h, self.S)
            all_h = np.sort(all_h[idx[: self.S]])
            self._threshold = np.uint64(all_h[-1])
        else:
            all_h = np.sort(all_h)
        self._pool = [all_h]
        self._pool_size = len(all_h)
        self._sketch = all_h

    def result(self) -> np.ndarray:
        self._compact()
        return self._sketch if self._sketch is not None else np.array([], dtype=np.uint64)


# ---------------------------------------------------------------------------
# FASTA iteration
# ---------------------------------------------------------------------------

def _iter_fasta(path: Path):
    """Yield (seq_id, seq_str) for each record."""
    cur_id: str | None = None
    parts: list[bytes] = []
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                if cur_id is not None:
                    yield cur_id, b"".join(parts).decode("ascii", errors="replace")
                cur_id = line.decode("ascii", errors="replace").strip()[1:].split()[0]
                parts = []
            else:
                parts.append(line.rstrip(b"\r\n"))
    if cur_id is not None:
        yield cur_id, b"".join(parts).decode("ascii", errors="replace")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_partial(fasta: Path, out_dir: Path, offset: int, count: int, S: int) -> int:
    """Process seqs [offset, offset+count), save partial hashes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"partial_{offset:07d}.npy"
    if out_file.exists():
        arr = np.load(out_file)
        print(f"[partial {offset}] already done ({len(arr):,} hashes), skipping")
        return int(len(arr))

    sketch = BottomSketch(S)
    processed = 0
    t0 = time.time()
    for i, (seq_id, seq) in enumerate(_iter_fasta(fasta)):
        if i < offset:
            continue
        if i >= offset + count:
            break
        arr = _seq_to_arr(seq)
        sketch.add(_canonical_hashes_chunked(arr))
        processed += 1
        if processed % 100 == 0:
            print(f"  [{processed}/{count}] {seq_id}  ({time.time()-t0:.0f}s)", flush=True)

    hashes = sketch.result()
    np.save(out_file, hashes)
    print(f"[partial {offset}] {processed} seqs → {len(hashes):,} hashes → {out_file.name}  ({time.time()-t0:.1f}s)")
    return processed


def run_merge(out_dir: Path, S: int, out_path: Path) -> np.ndarray:
    """Merge all partial_*.npy files and keep the bottom S hashes."""
    partial_files = sorted(out_dir.glob("partial_*.npy"))
    if not partial_files:
        raise FileNotFoundError(f"No partial_*.npy files in {out_dir}")
    print(f"Merging {len(partial_files)} partial files…")
    sketch = BottomSketch(S)
    for f in partial_files:
        arr = np.load(f)
        sketch.add(arr)
        print(f"  {f.name}: {len(arr):,} hashes", flush=True)
    result = sketch.result()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, result)
    print(f"Saved {len(result):,} hashes → {out_path}")
    return result


def run_sketch(fasta: Path, out_path: Path, S: int) -> np.ndarray:
    """Build sketch in one pass (for small DBs that fit in a single bash call)."""
    sketch = BottomSketch(S)
    t0 = time.time()
    count = 0
    for seq_id, seq in _iter_fasta(fasta):
        arr = _seq_to_arr(seq)
        sketch.add(_canonical_hashes_chunked(arr))
        count += 1
        if count % 50 == 0:
            print(f"  [{count}] {seq_id}  ({time.time()-t0:.0f}s)", flush=True)
    result = sketch.result()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, result)
    elapsed = time.time() - t0
    print(f"Done: {count} seqs → {len(result):,} hashes → {out_path}  ({elapsed:.1f}s)")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["partial", "merge", "sketch"], required=True)
    p.add_argument("--fasta", type=Path, help="Input FASTA (partial/sketch modes)")
    p.add_argument("--out-dir", type=Path, help="Directory for partial files (partial/merge modes)")
    p.add_argument("--out", type=Path, help="Output .npy sketch file (merge/sketch modes)")
    p.add_argument("--offset", type=int, default=0, help="First sequence index (partial mode)")
    p.add_argument("--count", type=int, default=300, help="Number of sequences per partial (partial mode)")
    p.add_argument("--sketch-size", type=int, default=500_000, help="Bottom-S sketch size")
    args = p.parse_args()

    if args.mode == "partial":
        if not args.fasta or not args.out_dir:
            p.error("--fasta and --out-dir required for partial mode")
        n = run_partial(args.fasta, args.out_dir, args.offset, args.count, args.sketch_size)
        print(f"partial done: processed {n} sequences")

    elif args.mode == "merge":
        if not args.out_dir or not args.out:
            p.error("--out-dir and --out required for merge mode")
        run_merge(args.out_dir, args.sketch_size, args.out)

    elif args.mode == "sketch":
        if not args.fasta or not args.out:
            p.error("--fasta and --out required for sketch mode")
        run_sketch(args.fasta, args.out, args.sketch_size)


if __name__ == "__main__":
    main()
