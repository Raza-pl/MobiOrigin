"""Tests for bounded-memory PlasClass benchmark execution."""

from __future__ import annotations

import csv

from scripts.benchmark import run_plasclass_streaming as runner


def test_run_plasclass_streaming_batches_and_writes_all_rows(tmp_path, monkeypatch) -> None:
    fasta = tmp_path / "input.fasta"
    output = tmp_path / "scores.csv"
    fasta.write_text(">a\nACGT\n>b\nTGCA\n>c\nAAAA\n")
    batch_sizes: list[int] = []

    class FakePlasClass:
        def __init__(self, n_procs: int) -> None:
            assert n_procs == 3

        def classify(self, sequences):
            batch_sizes.append(len(sequences))
            return [0.25] * len(sequences)

    monkeypatch.setattr(runner, "PlasClass", FakePlasClass)
    total = runner.run_plasclass_streaming(
        fasta,
        output,
        processes=3,
        batch_size=2,
    )

    with output.open() as fh:
        rows = list(csv.DictReader(fh))
    assert total == 3
    assert batch_sizes == [2, 1]
    assert [row["name"] for row in rows] == ["a", "b", "c"]
    assert {row["score"] for row in rows} == {"0.250000"}
