"""Tests for classifier error analysis artifacts."""

from __future__ import annotations

import json

from scripts.analyze_classifier_errors import analyze_errors


def test_analyze_errors_groups_and_ranks_mistakes(tmp_path) -> None:
    predictions = tmp_path / "predictions.tsv"
    out = tmp_path / "errors"
    predictions.write_text(
        "contig_id\ttrue_label\targmax_prediction\tlength_tier\ttaxon\t"
        "plasmid_score\tchromosome_score\tphage_score\n"
        "a\tplasmid\tchromosome\t1-2 kb\tTaxon A\t0.1\t0.9\t0.0\n"
        "b\tplasmid\tplasmid\t2-5 kb\tTaxon B\t0.8\t0.2\t0.0\n"
        "c\tchromosome\tplasmid\t1-2 kb\tTaxon A\t0.7\t0.3\t0.0\n"
    )

    summary = analyze_errors(predictions, out)

    assert summary["n_errors"] == 2
    assert len((out / "error_summary.tsv").read_text().splitlines()) == 3
    assert json.loads((out / "summary.json").read_text())["error_rate"] == 2 / 3
    high_confidence = (out / "high_confidence_errors.tsv").read_text().splitlines()
    assert high_confidence[1].startswith("a\t")
