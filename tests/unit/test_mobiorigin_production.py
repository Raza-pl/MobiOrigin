"""Focused synthetic tests for the standalone MobiOrigin production package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mobiorigin import cli
from mobiorigin.database_setup import DATABASE_FILENAMES, MANIFEST_NAME, setup_databases
from mobiorigin.fasta import FastaRecord, read_fasta
from mobiorigin.marker_features import (
    DATABASE_SHA256,
    OrfSummary,
    interval_union_length,
    load_database_manifest,
    marker_family_values,
    orf_values,
    parse_hits,
    predict_orfs,
    run_diamond,
)
from mobiorigin.model import MobiOriginMLP, ModelLoadError, load_model
from mobiorigin.predict import (
    _write_predictions,
    configure_runtime,
    ensemble_probabilities,
    fuse_features,
    predict,
    selective_labels,
)
from mobiorigin.provenance import atomic_json, atomic_text, sha256_file
from mobiorigin.sequence_features import (
    FEATURE_DIM,
    extract_sequence_features,
    k7_canonical_vector,
    kmer_vector,
)


def write(path: Path, value: str) -> Path:
    path.write_text(value, encoding="ascii")
    return path


def test_fasta_preserves_order_and_iupac(tmp_path: Path) -> None:
    path = write(tmp_path / "x.fasta", ">first description\nACGTRYSWKMBDHVN\n>second\nacgt\n")
    records = read_fasta(path)
    assert [record.identifier for record in records] == ["first", "second"]
    assert records[1].sequence == "ACGT"


@pytest.mark.parametrize(
    "payload, message",
    [
        ("", "no records"),
        ("ACGT\n", "before its first header"),
        (">x\n", "empty"),
        (">x\nACGT-\n", "unsupported symbols"),
        (">x\nACGT\n>x\nACGT\n", "unique"),
    ],
)
def test_fasta_rejects_invalid_input(tmp_path: Path, payload: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_fasta(write(tmp_path / "bad.fasta", payload))


def test_supported_length_boundaries() -> None:
    assert FastaRecord("a", "A" * 1_000).supported
    assert FastaRecord("b", "A" * 500_000).supported
    assert not FastaRecord("c", "A" * 999).supported
    assert not FastaRecord("d", "A" * 500_001).supported


def test_sequence_features_are_deterministic_and_normalized() -> None:
    sequences = ["ACGT" * 260, "NRYKMSWBDHVACGT" * 70]
    first = extract_sequence_features(sequences)
    second = extract_sequence_features(sequences)
    assert first.shape == (2, FEATURE_DIM)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.isclose(np.linalg.norm(kmer_vector(sequences[0], 5)), 1.0)
    assert np.isclose(np.linalg.norm(k7_canonical_vector(sequences[0])), 1.0)


def test_short_and_invalid_kmer_behaviour() -> None:
    assert not kmer_vector("AC", 5).any()
    assert not k7_canonical_vector("ACGT").any()
    with pytest.raises(ValueError, match="Unsupported"):
        kmer_vector("ACGT", 6)


def test_marker_helper_semantics(tmp_path: Path) -> None:
    assert interval_union_length([]) == 0
    assert interval_union_length([(1, 10), (8, 20), (25, 30)]) == 26
    summary = OrfSummary(3, 270, (30, 50, 70), (1, 1, -1))
    assert np.allclose(orf_values(summary, 1_000), [0.27, 3, np.log1p(50), 0.5, 2 / 3])
    hits_path = write(
        tmp_path / "hits.tsv",
        "q\ts2\t70\t80\t1e-10\t90\t50\nq\ts1\t70\t80\t1e-10\t100\t50\n",
    )
    hit = parse_hits(hits_path)["q"]
    assert hit.subject_id == "s1"
    assert marker_family_values({"q": hit}, {"q": "x"}, "x", 2, 1.0) == [1, 0.5, 1, 2]


def test_malformed_marker_hit_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_hits(write(tmp_path / "hits.tsv", "too\tfew\n"))


def test_database_manifest_rejects_missing_or_changed_payload(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "mobiorigin-mob-suite-database-manifest-v1",
        "databases": {
            key: {"path": f"{key}.dmnd", "sha256": value} for key, value in DATABASE_SHA256.items()
        },
    }
    (tmp_path / "mobiorigin_mob_suite_database_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest identity"):
        load_database_manifest(tmp_path)


def test_database_setup_is_atomic_and_identity_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    hashes: dict[str, str] = {}
    for family, filename in DATABASE_FILENAMES.items():
        path = write(source / filename, f"{family}-database\n")
        hashes[family] = sha256_file(path)
    monkeypatch.setattr("mobiorigin.database_setup.DATABASE_SHA256", hashes)
    monkeypatch.setattr("mobiorigin.marker_features.DATABASE_SHA256", hashes)
    output = tmp_path / "installed"
    setup_databases(output, source_dir=source)
    assert load_database_manifest(output) == {
        family: output / filename for family, filename in DATABASE_FILENAMES.items()
    }
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["network_accessed"] is False
    assert (output / "THIRD_PARTY_DATABASE_NOTICE.txt").is_file()
    with pytest.raises(FileExistsError):
        setup_databases(output, source_dir=source)
    (source / DATABASE_FILENAMES["rep"]).write_text("changed\n", encoding="ascii")
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        setup_databases(failed, source_dir=source)
    assert not failed.exists()


def test_model_round_trip_and_rejections(tmp_path: Path) -> None:
    configure_runtime()
    model = MobiOriginMLP(input_dim=8)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    loaded = load_model(path, input_dim=8)
    assert tuple(loaded.state_dict()) == tuple(model.state_dict())
    wrong = tmp_path / "wrong.pt"
    torch.save(model.state_dict(), wrong)
    with pytest.raises(ModelLoadError, match="architecture"):
        load_model(wrong, input_dim=9)
    unsafe = tmp_path / "unsafe.pt"
    torch.save({"bad": "not a tensor"}, unsafe)
    with pytest.raises(ModelLoadError):
        load_model(unsafe, input_dim=8)
    other_architecture = tmp_path / "other_architecture.pt"
    torch.save(
        MobiOriginMLP(input_dim=8, hidden_dims=(512, 128, 32)).state_dict(),
        other_architecture,
    )
    with pytest.raises(ModelLoadError, match="architecture"):
        load_model(other_architecture, input_dim=8)


def test_fusion_and_selective_policy() -> None:
    sequence = np.zeros((2, 9_557), dtype=np.float32)
    marker = np.ones((2, 17), dtype=np.float32)
    normalization = np.vstack([np.zeros(17, dtype=np.float32), np.ones(17, dtype=np.float32)])
    fused = fuse_features(sequence, marker, normalization)
    assert fused.shape == (2, 9_574)
    probabilities = np.asarray(
        [[0.1, 0.54, 0.36], [0.1, 0.8, 0.1], [0.6, 0.3, 0.1]], dtype=np.float32
    )
    labels, scores = selective_labels(probabilities)
    assert labels == ["unclassified", "plasmid", "chromosome"]
    assert scores.shape == (3,)
    with pytest.raises(ValueError):
        fuse_features(sequence, np.ones((3, 17), dtype=np.float32), normalization)
    with pytest.raises(ValueError):
        selective_labels(np.zeros((2, 4), dtype=np.float32))


class ConstantModel(torch.nn.Module):
    def __init__(self, logits: tuple[float, float, float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(logits, dtype=torch.float32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.values.repeat(len(values), 1)


def test_ensemble_is_equal_weight_and_normalized() -> None:
    values = np.zeros((2, 9_574), dtype=np.float32)
    models = [ConstantModel((1, 2, 3)), ConstantModel((3, 2, 1)), ConstantModel((2, 3, 1))]
    first = ensemble_probabilities(models, values)  # type: ignore[arg-type]
    second = ensemble_probabilities(models, values)  # type: ignore[arg-type]
    assert np.array_equal(first, second)
    assert np.allclose(first.sum(axis=1), 1.0)


def test_synthetic_orf_prediction(tmp_path: Path) -> None:
    proteins = tmp_path / "proteins.faa"
    summaries, query_map = predict_orfs([FastaRecord("x", "ATG" + "GCC" * 400)], proteins)
    assert summaries["x"].count >= 0
    assert all(identifier.startswith("x__orf_") for identifier in query_map)
    assert proteins.is_file()


def test_diamond_transport_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "hits.tsv"
    monkeypatch.setattr(
        "mobiorigin.marker_features.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
    )
    run_diamond(Path("diamond"), tmp_path / "proteins.faa", tmp_path / "rep.dmnd", output, 1)
    assert output.read_text() == ""
    monkeypatch.setattr(
        "mobiorigin.marker_features.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="failed"),
    )
    with pytest.raises(RuntimeError, match="failed"):
        run_diamond(
            Path("diamond"),
            tmp_path / "proteins.faa",
            tmp_path / "rep.dmnd",
            output,
            1,
        )


def test_prediction_table_schema_and_abstention(tmp_path: Path) -> None:
    path = tmp_path / "predictions.tsv"
    records = [FastaRecord("a", "A" * 1_000), FastaRecord("b", "A" * 999)]
    probabilities = np.asarray([[0.1, 0.8, 0.1], [1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
    _write_predictions(
        path,
        records,
        probabilities,
        ["plasmid", "unclassified"],
        np.asarray([0.7, 0], dtype=np.float32),
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["sequence_id"] for row in rows] == ["a", "b"]
    assert rows[1]["abstention_reason"] == "unsupported_length"


def test_atomic_helpers_and_hash(tmp_path: Path) -> None:
    text = tmp_path / "x.txt"
    atomic_text(text, "hello\n")
    assert len(sha256_file(text)) == 64
    payload = tmp_path / "x.json"
    atomic_json(payload, {"value": 1})
    assert json.loads(payload.read_text()) == {"value": 1}


def test_predict_synthetic_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fasta = write(
        tmp_path / "input.fasta",
        f">supported\n{'ACGT' * 250}\n>short\n{'ACGT' * 249}\n",
    )
    normalization = np.vstack([np.zeros(17, dtype=np.float32), np.ones(17, dtype=np.float32)])
    models = [ConstantModel((0.1, 3.0, 0.1))] * 3
    monkeypatch.setattr("mobiorigin.predict.configure_runtime", lambda: None)
    monkeypatch.setattr("mobiorigin.predict.load_artifacts", lambda _: (models, normalization))
    monkeypatch.setattr("mobiorigin.predict.load_database_manifest", lambda _: {})
    monkeypatch.setattr(
        "mobiorigin.predict.extract_marker_features",
        lambda records, **kwargs: np.zeros((len(records), 17), dtype=np.float32),
    )
    output = tmp_path / "output"
    predict(
        input_fasta=fasta,
        output_dir=output,
        database_dir=tmp_path,
        threads=1,
        model_dir=tmp_path,
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "SHA256SUMS.txt",
        "predictions.tsv",
        "provenance.json",
    ]
    with (output / "predictions.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["prediction"] == "plasmid"
    assert rows[1]["prediction"] == "unclassified"
    assert rows[1]["abstention_reason"] == "unsupported_length"
    assert json.loads((output / "provenance.json").read_text())["unsupported_length_records"] == 1
    with pytest.raises(FileExistsError):
        predict(
            input_fasta=fasta,
            output_dir=output,
            database_dir=tmp_path,
            threads=1,
            model_dir=tmp_path,
        )


def test_cli_dispatches_predict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "predict", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "predict",
            "--input-fasta",
            str(tmp_path / "input.fasta"),
            "--output-dir",
            str(tmp_path / "out"),
            "--database-dir",
            str(tmp_path / "db"),
            "--threads",
            "4",
        ]
    )
    assert observed["threads"] == 4


def test_cli_dispatches_database_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "setup_databases", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "setup-databases",
            "--source-dir",
            str(tmp_path / "official_source"),
            "--output-dir",
            str(tmp_path / "db"),
        ]
    )
    assert observed == {
        "output_dir": tmp_path / "db",
        "source_dir": tmp_path / "official_source",
    }
