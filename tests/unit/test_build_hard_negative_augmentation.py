"""Tests for source-balanced hard-negative augmentation construction."""

from __future__ import annotations

import json

import numpy as np

from scripts import build_hard_negative_augmentation as builder


def test_build_hard_negative_augmentation_filters_and_balances(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "usable.fna").write_text(
        ">KEEP chromosome\n"
        + "ACGT" * 2_000
        + "\n>LOCKED chromosome\n"
        + "TGCA" * 2_000
        + "\n>PLASMID plasmid replicon\n"
        + "ATGC" * 2_000
        + "\n"
    )
    (input_dir / "duplicate.fna").write_text(">KEEP\n" + "ACGT" * 2_000 + "\n")
    (input_dir / "too_large.fna").write_text(">LARGE\n" + "A" * 30_000 + "\n")
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("LOCKED\n")
    reference_ids = tmp_path / "reference_ids.txt"
    reference_labels = tmp_path / "reference_labels.npy"
    reference_ids.write_text("KEEP_w1000_s0\nCONFLICT_w1000_s0\n")
    np.save(reference_labels, np.array([1, 0], dtype=np.int64))

    def fake_extract(sequences, path, *, chunk_size):
        assert chunk_size == 2
        np.save(path, np.zeros((len(sequences), 3), dtype=np.float32))
        return len(sequences), 3

    monkeypatch.setattr(builder, "extract_features_to_npy", fake_extract)
    summary = builder.build_hard_negative_augmentation(
        input_dir,
        output_dir,
        exclude_groups_path=excluded,
        reference_ids_path=reference_ids,
        reference_labels_path=reference_labels,
        window_sizes=(1_000, 2_000),
        rows_per_size=2,
        max_windows_per_source_size=1_000,
        max_file_bytes=28_000,
        feature_chunk_size=2,
    )

    assert summary["rows_by_size"] == {"1000": 2, "2000": 2}
    assert summary["total_rows"] == 4
    assert summary["skipped_locked_groups"] == 1
    assert summary["skipped_plasmid_header_groups"] == 1
    assert summary["exact_reference_id_overlaps"] >= 1
    assert summary["duplicate_source_groups"] == 1
    assert summary["skipped_large_files"] == [str(input_dir / "too_large.fna")]
    assert set(np.load(output_dir / "labels.npy")) == {1}
    assert all(
        line.startswith("KEEP_") for line in (output_dir / "seq_ids.txt").read_text().splitlines()
    )
    assert json.loads((output_dir / "augmentation_summary.json").read_text())["total_rows"] == 4
