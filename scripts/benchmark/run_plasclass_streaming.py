#!/usr/bin/env python3
"""Run PlasClass on a FASTA in bounded-memory batches."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import sklearn.linear_model._logistic as _logistic
import sklearn.preprocessing._data as _preprocessing_data

# PlasClass model files were pickled with scikit-learn 0.19 module paths.
sys.modules.setdefault("sklearn.linear_model.logistic", _logistic)
sys.modules.setdefault("sklearn.preprocessing.data", _preprocessing_data)

try:
    from plasclass.plasclass import plasclass as PlasClass  # noqa: E402
except ImportError:  # pragma: no cover - PlasClass is an external benchmark tool
    PlasClass = None  # type: ignore[assignment,misc]

from plasflow2.utils.fasta import iter_fasta  # noqa: E402


def run_plasclass_streaming(
    fasta_path: Path,
    output_path: Path,
    *,
    processes: int = 1,
    batch_size: int = 2_000,
) -> int:
    """Classify every FASTA record while bounding retained sequence data."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if PlasClass is None:
        raise RuntimeError(
            "PlasClass is not installed. Install the external PlasClass tool "
            "(https://github.com/Shamir-Lab/PlasClass) to run this benchmark."
        )
    classifier = PlasClass(n_procs=processes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", newline="") as output_fh:
        writer = csv.writer(output_fh)
        writer.writerow(["name", "score"])
        sequence_ids: list[str] = []
        sequences: list[str] = []

        def flush() -> None:
            nonlocal total
            if not sequences:
                return
            scores = classifier.classify(sequences)
            for sequence_id, score in zip(sequence_ids, scores):
                writer.writerow([sequence_id, f"{float(score):.6f}"])
            total += len(sequences)
            output_fh.flush()
            print(f"PlasClass: {total:,} sequences", flush=True)
            sequence_ids.clear()
            sequences.clear()

        for record in iter_fasta(fasta_path):
            sequence_ids.append(record.id)
            sequences.append(str(record.seq))
            if len(sequences) >= batch_size:
                flush()
        flush()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2_000)
    args = parser.parse_args()
    total = run_plasclass_streaming(
        args.input,
        args.output,
        processes=args.processes,
        batch_size=args.batch_size,
    )
    print(f"PlasClass complete: {total:,} predictions → {args.output}")


if __name__ == "__main__":
    main()
