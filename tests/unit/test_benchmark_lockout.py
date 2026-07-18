"""Tests for reproducible source-level benchmark lockout construction."""

from __future__ import annotations

import json

import numpy as np

from scripts.build_benchmark_lockout import build_lockout


def test_build_lockout_reserves_external_and_phage_sources(tmp_path) -> None:
    ids = tmp_path / "ids.txt"
    labels = tmp_path / "labels.npy"
    benchmark = tmp_path / "benchmark.tsv"
    groups_out = tmp_path / "locked.txt"
    phage_manifest = tmp_path / "phage.tsv"
    phage_dev_manifest = tmp_path / "phage_dev.tsv"
    phage_final_manifest = tmp_path / "phage_final.tsv"
    summary_path = tmp_path / "summary.json"
    ids.write_text("EXT_w1000_s0\nKEEP_w1000_s0\nPHAGE_A_w1000_s0\nPHAGE_A_w2000_s0\n")
    np.save(labels, np.array([0, 1, 2, 2], dtype=np.int64))
    benchmark.write_text("contig_id\ttrue_label\tsource_accession\n" "ext_1\tplasmid\tEXT\n")

    summary = build_lockout(
        ids,
        labels,
        [benchmark],
        groups_out,
        phage_manifest,
        summary_path,
        phage_dev_manifest_path=phage_dev_manifest,
        phage_final_manifest_path=phage_final_manifest,
        phage_fraction=1.0,
        phage_dev_fraction=1.0,
        seed=42,
    )

    assert groups_out.read_text().splitlines() == ["EXT", "PHAGE_A"]
    assert summary["locked_training_rows_total"] == 3
    assert summary["remaining_class_counts"] == {1: 1}
    assert len(phage_manifest.read_text().splitlines()) == 3
    assert len(phage_dev_manifest.read_text().splitlines()) == 3
    assert len(phage_final_manifest.read_text().splitlines()) == 1
    assert json.loads(summary_path.read_text())["locked_phage_groups"] == 1
