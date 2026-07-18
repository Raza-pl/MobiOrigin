"""Tests for streaming labeled-FASTA model evaluation."""

from __future__ import annotations

import json

from plasflow2.classify.features import FEATURE_DIM
from plasflow2.classify.model import PlasFlowMLP, save_model

from scripts.evaluate_fasta_model import evaluate_fasta_model, evaluate_fasta_models


def test_evaluate_fasta_model_streams_and_writes_metrics(tmp_path) -> None:
    fasta = tmp_path / "input.fasta"
    labels = tmp_path / "labels.tsv"
    model_path = tmp_path / "model.pt"
    out = tmp_path / "evaluation"
    fasta.write_text(">a\n" + "ACGT" * 250 + "\n>b\n" + "TGCA" * 250 + "\n")
    labels.write_text(
        "contig_id\ttrue_label\tlength\tlength_tier\tsource_accession\ttaxon\n"
        "a\tplasmid\t1000\t1-2 kb\tA\tTaxon A\n"
        "b\tchromosome\t1000\t1-2 kb\tB\tTaxon B\n"
    )
    model = PlasFlowMLP(
        input_dim=FEATURE_DIM,
        num_classes=3,
        hidden_dims=(8, 6, 4),
    )
    save_model(model, model_path)

    metrics = evaluate_fasta_model(
        fasta,
        labels,
        model_path,
        out,
        chunk_size=1,
    )

    assert metrics["argmax"]["n_rows"] == 2
    assert len((out / "predictions.tsv").read_text().splitlines()) == 3
    assert json.loads((out / "metrics.json").read_text())["argmax"]["n_rows"] == 2


def test_evaluate_fasta_model_supports_binary_production_checkpoint(tmp_path) -> None:
    fasta = tmp_path / "input.fasta"
    labels = tmp_path / "labels.tsv"
    model_path = tmp_path / "model.pt"
    out = tmp_path / "evaluation"
    fasta.write_text(">a\n" + "ACGT" * 250 + "\n>b\n" + "TGCA" * 250 + "\n")
    labels.write_text(
        "contig_id\ttrue_label\tlength\tlength_tier\tsource_accession\ttaxon\n"
        "a\tplasmid\t1000\t1-2 kb\tA\tTaxon A\n"
        "b\tchromosome\t1000\t1-2 kb\tB\tTaxon B\n"
    )
    model = PlasFlowMLP(
        input_dim=FEATURE_DIM,
        num_classes=2,
        hidden_dims=(8, 6, 4),
    )
    save_model(model, model_path)

    metrics = evaluate_fasta_model(
        fasta,
        labels,
        model_path,
        out,
        chunk_size=1,
    )

    assert metrics["argmax"]["n_rows"] == 2
    rows = (out / "predictions.tsv").read_text().splitlines()
    assert rows[0].endswith("plasmid_score\tchromosome_score\tphage_score")
    assert all(row.endswith("\t0.0") for row in rows[1:])


def test_evaluate_fasta_models_shares_one_feature_pass(tmp_path, monkeypatch) -> None:
    fasta = tmp_path / "input.fasta"
    labels = tmp_path / "labels.tsv"
    candidate_path = tmp_path / "candidate.pt"
    baseline_path = tmp_path / "baseline.pt"
    fasta.write_text(">a\n" + "ACGT" * 250 + "\n>b\n" + "TGCA" * 250 + "\n")
    labels.write_text(
        "contig_id\ttrue_label\tlength\tlength_tier\tsource_accession\ttaxon\n"
        "a\tplasmid\t1000\t1-2 kb\tA\tTaxon A\n"
        "b\tchromosome\t1000\t1-2 kb\tB\tTaxon B\n"
    )
    save_model(
        PlasFlowMLP(input_dim=FEATURE_DIM, num_classes=3, hidden_dims=(8, 6, 4)),
        candidate_path,
    )
    save_model(
        PlasFlowMLP(input_dim=FEATURE_DIM, num_classes=2, hidden_dims=(8, 6, 4)),
        baseline_path,
    )
    from scripts import evaluate_fasta_model as evaluator

    real_extract_features = evaluator.extract_features
    calls = 0

    def counted_extract_features(sequences):
        nonlocal calls
        calls += 1
        return real_extract_features(sequences)

    monkeypatch.setattr(evaluator, "extract_features", counted_extract_features)
    results = evaluate_fasta_models(
        fasta,
        labels,
        {"candidate": candidate_path, "baseline": baseline_path},
        {
            "candidate": tmp_path / "candidate_eval",
            "baseline": tmp_path / "baseline_eval",
        },
        chunk_size=1,
    )

    assert calls == 2
    assert results["candidate"]["argmax"]["n_rows"] == 2
    assert results["baseline"]["argmax"]["n_rows"] == 2
