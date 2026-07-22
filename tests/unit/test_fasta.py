"""Unit tests for FASTA utilities.

Day 5 target: all tests pass.

Also covers the duplicate-contig-ID guard added in response to the second
external code review (docs/CODE_REVIEW_FINDINGS_2026-07.md, Round 8): every
downstream consumer keys data by contig ID in plain dicts, so a duplicate
silently overwrote an earlier record with no error. load_fasta() and
iter_fasta() now raise DuplicateContigIDError instead.
"""

from pathlib import Path

import pytest
from plasflow2.utils.fasta import (
    DuplicateContigIDError,
    gc_content,
    iter_fasta,
    load_fasta,
    split_by_label,
    write_fasta,
)


def _write(tmp_path: Path, name: str, entries: list[tuple[str, str]]) -> Path:
    path = tmp_path / name
    path.write_text("".join(f">{cid}\n{seq}\n" for cid, seq in entries))
    return path


def test_gc_content_pure_gc() -> None:
    assert gc_content("GGCC") == 1.0


def test_gc_content_no_gc() -> None:
    assert gc_content("AATT") == 0.0


def test_gc_content_mixed() -> None:
    result = gc_content("ACGT")
    assert abs(result - 0.5) < 1e-9


def test_gc_content_empty() -> None:
    assert gc_content("") == 0.0


def test_split_by_label_basic() -> None:
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    records = [SeqRecord(Seq("ACGT"), id=f"seq{i}") for i in range(4)]
    labels = ["plasmid", "chromosome", "plasmid", "phage"]
    bins = split_by_label(records, labels)

    assert len(bins["plasmid"]) == 2
    assert len(bins["chromosome"]) == 1
    assert len(bins["phage"]) == 1
    assert "archaea" not in bins


# ---------------------------------------------------------------------------
# load_fasta
# ---------------------------------------------------------------------------


class TestLoadFasta:
    def test_loads_all_sequences_above_min_length(self, tmp_path):
        fasta = _write(tmp_path, "in.fasta", [("c1", "A" * 2000), ("c2", "C" * 2000)])
        records = load_fasta(fasta, min_length=1000)
        assert {r.id for r in records} == {"c1", "c2"}

    def test_filters_by_min_length(self, tmp_path):
        fasta = _write(tmp_path, "in.fasta", [("short", "A" * 500), ("long", "C" * 2000)])
        records = load_fasta(fasta, min_length=1000)
        assert [r.id for r in records] == ["long"]

    def test_duplicate_id_raises(self, tmp_path):
        fasta = _write(tmp_path, "dup.fasta", [("c1", "A" * 2000), ("c1", "C" * 2000)])
        with pytest.raises(DuplicateContigIDError, match="c1"):
            load_fasta(fasta, min_length=1000)

    def test_duplicate_id_raises_even_if_below_min_length(self, tmp_path):
        # A duplicate that would be filtered out by min_length still
        # indicates a malformed input file -- must still raise.
        fasta = _write(tmp_path, "dup_short.fasta", [("c1", "A" * 10), ("c1", "C" * 10)])
        with pytest.raises(DuplicateContigIDError, match="c1"):
            load_fasta(fasta, min_length=1000)

    def test_no_false_positive_on_distinct_ids(self, tmp_path):
        fasta = _write(
            tmp_path,
            "ok.fasta",
            [("c1", "A" * 2000), ("c2", "A" * 2000), ("c3", "A" * 2000)],
        )
        records = load_fasta(fasta, min_length=1000)
        assert len(records) == 3

    def test_write_fasta_roundtrip(self, tmp_path):
        fasta = _write(tmp_path, "in.fasta", [("c1", "ACGT" * 300)])
        records = load_fasta(fasta, min_length=1000)
        out = tmp_path / "out.fasta"
        write_fasta(records, out)
        reloaded = load_fasta(out, min_length=1000)
        assert [r.id for r in reloaded] == ["c1"]


# ---------------------------------------------------------------------------
# iter_fasta
# ---------------------------------------------------------------------------


class TestIterFasta:
    def test_yields_all_records(self, tmp_path):
        fasta = _write(tmp_path, "in.fasta", [("c1", "ACGT"), ("c2", "TTTT")])
        records = list(iter_fasta(fasta))
        assert [r.id for r in records] == ["c1", "c2"]

    def test_duplicate_id_raises(self, tmp_path):
        fasta = _write(tmp_path, "dup.fasta", [("c1", "ACGT"), ("c1", "TTTT")])
        with pytest.raises(DuplicateContigIDError, match="c1"):
            list(iter_fasta(fasta))

    def test_duplicate_id_raised_lazily_without_buffering_whole_file(self, tmp_path):
        # The first two (unique) records should be yielded successfully
        # before the generator raises on the third, duplicate one --
        # confirms the check doesn't require buffering the whole file.
        fasta = _write(
            tmp_path,
            "dup.fasta",
            [("c1", "ACGT"), ("c2", "TTTT"), ("c2", "GGGG")],
        )
        gen = iter_fasta(fasta)
        assert next(gen).id == "c1"
        assert next(gen).id == "c2"
        with pytest.raises(DuplicateContigIDError, match="c2"):
            next(gen)
