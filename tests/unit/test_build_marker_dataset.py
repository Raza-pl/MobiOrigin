"""Tests for leakage-resistant marker-dataset construction."""

from __future__ import annotations

from scripts import build_marker_dataset as builder


def test_sample_sequences_excludes_locked_normalized_groups(tmp_path) -> None:
    fasta = tmp_path / "sources.fna"
    fasta.write_text(">COMPASS_LOCKED.1\n" + "ACGT" * 750 + "\n" ">SAFE.1\n" + "TGCA" * 750 + "\n")

    samples, observed_groups = builder.sample_sequences(
        [fasta],
        max_total=20,
        window_sizes=(2_000,),
        excluded_groups={"LOCKED.1"},
    )

    assert samples
    assert {sample[2] for sample in samples} == {"SAFE.1"}
    assert {sample[3] for sample in samples} == {str(fasta)}
    assert observed_groups == {"SAFE.1"}


def test_clean_samples_removes_conflicts_duplicates_and_balances() -> None:
    samples = {
        "plasmid": [
            ("p1_w2000_s0", "A" * 2_000, "P1", "p.fna"),
            ("shared_w2000_s0", "C" * 2_000, "SHARED", "p.fna"),
            ("p_duplicate_w2000_s0", "A" * 2_000, "P2", "p.fna"),
        ],
        "chromosome": [
            ("c1_w2000_s0", "G" * 2_000, "C1", "c.fna"),
            ("shared_w2000_s0", "T" * 2_000, "SHARED", "c.fna"),
        ],
        "phage": [
            ("v1_w2000_s0", "T" * 1_999 + "A", "V1", "v.fna"),
            ("v2_w2000_s0", "T" * 1_999 + "C", "V2", "v.fna"),
        ],
    }

    cleaned, summary = builder.clean_samples(samples, max_per_class=10)

    assert summary["conflicting_source_groups_removed"] == 1
    assert summary["within_class_duplicate_sequences_removed"] == 1
    assert summary["balanced_rows_per_class"] == 1
    assert {label: len(rows) for label, rows in cleaned.items()} == {
        "plasmid": 1,
        "chromosome": 1,
        "phage": 1,
    }
    assert not {sample[2] for rows in cleaned.values() for sample in rows} & {"SHARED"}
