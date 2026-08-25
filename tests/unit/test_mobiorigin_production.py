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
from mobiorigin.annotate import (
    ArgHit,
    Orf,
    annotate,
    consensus_hits,
    load_amrfinder_hierarchy,
    parse_amrfinderplus_hits,
    parse_amrprot_hits,
    parse_card_hits,
    parse_sarg_hits,
    predict_annotation_orfs,
)
from mobiorigin.biological_evidence import (
    EvidenceHit,
    arg_evidence,
    load_predictions,
    parse_amrfinderplus_non_amr,
    write_integrated_results,
    write_publication_summary,
)
from mobiorigin.database_setup import (
    DATABASE_FILENAMES,
    MANIFEST_NAME,
    check_databases,
    setup_databases,
)
from mobiorigin.fasta import FastaRecord, read_fasta
from mobiorigin.marker_database_builder import (
    DIAMOND_VERSION,
    build_rep_proteins,
    translate,
)
from mobiorigin.marker_database_builder import (
    read_fasta as read_marker_fasta,
)
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
from mobiorigin.visualize import visualize
from mobiorigin.workflow import (
    default_database_dir,
    demo,
    doctor,
    resolve_database_dir,
    run_analysis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def test_installation_environments_keep_incompatible_stacks_separate() -> None:
    runtime = (PROJECT_ROOT / "environment.yml").read_text(encoding="utf-8")
    database = (PROJECT_ROOT / "environment.mob-database.yml").read_text(encoding="utf-8")
    marker_build = (PROJECT_ROOT / "environment.marker-build.yml").read_text(encoding="utf-8")
    assert "name: mobiorigin\n" in runtime
    assert "numpy=1.26.4" in runtime
    assert "pytorch=2.5.1=cpu*" in runtime
    assert "diamond>=2.1" in runtime
    assert "ncbi-amrfinderplus=4.2.7" in runtime
    assert "-e ." not in runtime
    assert "mob_suite" not in runtime
    assert "pandas" not in runtime
    assert "name: mobiorigin-db\n" in database
    assert "mob_suite=3.1.8" in database
    assert "numpy>=1.11.1,<1.23.5" in database
    assert "pandas>=0.22,<=1.5.3" in database
    assert "blast>=2.9,<2.16" in database
    assert "diamond=2.0.15" not in database
    assert "pytorch" not in database
    assert "name: mobiorigin-marker-build\n" in marker_build
    assert "diamond=2.0.15" in marker_build
    assert "mob_suite" not in marker_build


def test_guided_installer_is_non_errexit_and_runs_demo() -> None:
    installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
    assert "set -e" not in installer
    assert "env create --file" in installer
    assert "mobiorigin doctor --software-only" in installer
    assert "scripts/setup_mobiorigin_databases.sh" in installer
    assert "mobiorigin demo" in installer
    assert "--software-only" in installer
    assert not any(line.lstrip().startswith("rm ") for line in installer.splitlines())


def test_default_database_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MOBIORIGIN_DATABASE_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert default_database_dir() == tmp_path / "data" / "mobiorigin" / "marker_databases"
    monkeypatch.setenv("MOBIORIGIN_DATABASE_DIR", str(tmp_path / "custom"))
    assert resolve_database_dir(None) == tmp_path / "custom"
    assert resolve_database_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_doctor_reports_software_and_database_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "mobiorigin.workflow._command_version",
        lambda command, arguments: {
            "status": "PASS",
            "executable": f"/bin/{command}",
            "version": "test",
        },
    )
    monkeypatch.setattr(
        "mobiorigin.workflow.check_databases", lambda path: {"status": "PASS", "path": str(path)}
    )
    result = doctor(database_dir=tmp_path / "db")
    assert result["status"] == "PASS"
    assert result["database"]["status"] == "PASS"
    assert doctor(software_only=True)["status"] == "PASS"


def test_atomic_run_and_demo_orchestration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    predictions_text = (
        "sequence_id\tlength_bp\tprediction\tp_chromosome\tp_plasmid\tp_phage\t"
        "plasmid_score\tabstention_reason\n"
        "demo\t1200\tplasmid\t0.1\t0.8\t0.1\t0.7\t\n"
    )

    def fake_predict(**kwargs: object) -> None:
        output = Path(str(kwargs["output_dir"]))
        output.mkdir()
        write(output / "predictions.tsv", predictions_text)
        write(output / "provenance.json", "{}\n")

    monkeypatch.setattr("mobiorigin.workflow.predict", fake_predict)
    output = tmp_path / "analysis"
    run_analysis(
        input_fasta=tmp_path / "input.fasta",
        output_dir=output,
        database_dir=tmp_path / "db",
    )
    assert (output / "README_RESULTS.txt").is_file()
    assert (output / "visualization" / "mobiorigin_dashboard.html").is_file()
    with pytest.raises(FileExistsError):
        run_analysis(
            input_fasta=tmp_path / "input.fasta",
            output_dir=output,
            database_dir=tmp_path / "db",
        )
    demo_output = tmp_path / "demo"
    result = demo(output_dir=demo_output, database_dir=tmp_path / "db")
    assert result["status"] == "PASS"
    assert result["records"] == 1


def test_cli_dispatches_run_doctor_and_demo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "run_analysis", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "run",
            "--input-fasta",
            str(tmp_path / "input.fasta"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert observed["database_dir"] is None
    monkeypatch.setattr(cli, "doctor", lambda **kwargs: {"status": "PASS"})
    cli.main(["doctor", "--software-only"])
    assert '"status": "PASS"' in capsys.readouterr().out
    monkeypatch.setattr(cli, "demo", lambda **kwargs: {"status": "PASS"})
    cli.main(["demo", "--output-dir", str(tmp_path / "demo")])
    assert '"status": "PASS"' in capsys.readouterr().out


def test_database_helper_is_guided_non_destructive_and_non_errexit() -> None:
    helper = PROJECT_ROOT / "scripts/setup_mobiorigin_databases.sh"
    payload = helper.read_text(encoding="utf-8")
    assert payload.startswith("#!/usr/bin/env bash\n")
    assert "set -e" not in payload
    assert "environment.mob-database.yml" in payload
    assert "environment.marker-build.yml" in payload
    assert "run -n mobiorigin-db mob_init" in payload
    assert 'root / "databases"' in payload
    assert "marker_database_builder.py" in payload
    assert "run -n mobiorigin-marker-build python" in payload
    assert "PYTHONNOUSERSITE=1" in payload
    assert "unset PYTHONPATH" in payload
    assert "--diamond diamond" in payload
    assert "run -n mobiorigin mobiorigin setup-databases" in payload
    assert "--platform osx-64" in payload
    assert "Rosetta" in payload
    assert "Existing MobiOrigin marker databases are valid" in payload
    assert not any(line.lstrip().startswith("rm ") for line in payload.splitlines())


def test_bundled_four_class_example_and_runner_are_public_and_portable() -> None:
    example = PROJECT_ROOT / "src/mobiorigin/data/examples/annotated_assembly_example.fasta"
    records = read_fasta(example)
    assert [record.identifier for record in records] == [
        "assembly_example_chromosome_01",
        "assembly_example_chromosome_02",
        "assembly_example_plasmid_01",
        "assembly_example_plasmid_02",
        "assembly_example_phage_01",
        "assembly_example_phage_02",
        "assembly_example_unclassified_01",
        "assembly_example_unclassified_02",
    ]
    assert all(record.supported for record in records)
    assert sum(len(record.sequence) for record in records) == 160_054

    runner = PROJECT_ROOT / "scripts/run_mobiorigin_assembly_example.sh"
    payload = runner.read_text(encoding="utf-8")
    assert payload.startswith("#!/usr/bin/env bash\n")
    assert "set -e" not in payload
    assert "annotated_assembly_example.fasta" in payload
    assert "chromosome=2, plasmid=2, phage=2, unclassified=2" in payload
    assert "ANNOTATION_DATABASE" in payload
    assert "accuracy or prevalence" in payload


def test_frozen_marker_translation_is_deterministic(tmp_path: Path) -> None:
    source = write(tmp_path / "rep.dna.fas", ">record description\nATG" + "GCT" * 30 + "TAA")
    destination = tmp_path / "rep_proteins.faa"
    build_rep_proteins(source, destination)
    records = list(read_marker_fasta(destination))
    assert records[0] == ("record description_s1_f0_o0", "M" + "A" * 30)
    repeated = tmp_path / "repeated.faa"
    build_rep_proteins(source, repeated)
    assert repeated.read_bytes() == destination.read_bytes()
    assert translate("ATGGCTTAA") == "MA*"
    assert translate("GCN") == "A"
    assert DIAMOND_VERSION == "2.0.15"


def test_database_setup_missing_source_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="setup_mobiorigin_databases.sh"):
        setup_databases(tmp_path / "output", source_dir=tmp_path / "missing")


def test_database_check_missing_manifest_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="setup_mobiorigin_databases.sh"):
        check_databases(tmp_path / "missing")


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


def test_database_check_verifies_diamond_and_frozen_databases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database_dir = tmp_path / "db"
    database_dir.mkdir()
    write(database_dir / MANIFEST_NAME, "{}\n")
    monkeypatch.setattr("mobiorigin.database_setup.shutil.which", lambda value: "/bin/diamond")
    monkeypatch.setattr(
        "mobiorigin.database_setup.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="diamond version 2.1.9\n", stderr=""
        ),
    )
    monkeypatch.setattr(
        "mobiorigin.database_setup.load_database_manifest",
        lambda path: {family: path / filename for family, filename in DATABASE_FILENAMES.items()},
    )
    result = check_databases(database_dir)
    assert result["status"] == "PASS"
    assert result["databases_verified"] == 3
    assert result["diamond_version"] == "diamond version 2.1.9"


def test_cli_dispatches_database_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "check_databases",
        lambda database_dir, **kwargs: observed.update({"database_dir": database_dir, **kwargs})
        or {"status": "PASS"},
    )
    cli.main(["setup-databases", "--check", "--output-dir", str(tmp_path / "db")])
    assert observed == {"database_dir": tmp_path / "db", "diamond": Path("diamond")}


def test_visualization_outputs_tables_svg_and_html(tmp_path: Path) -> None:
    predictions = write(
        tmp_path / "predictions.tsv",
        "sequence_id\tlength_bp\tprediction\tp_chromosome\tp_plasmid\tp_phage\t"
        "plasmid_score\tabstention_reason\n"
        "a\t1500\tplasmid\t0.1\t0.8\t0.1\t0.7\t\n"
        "b\t3000\tchromosome\t0.8\t0.1\t0.1\t-0.7\t\n"
        "c\t8000\tphage\t0.1\t0.1\t0.8\t-0.7\t\n"
        "d\t60000\tunclassified\t0.3\t0.4\t0.3\t0.1\tlow_plasmid_margin\n",
    )
    annotated = write(
        tmp_path / "annotated.tsv",
        "sequence_id\tprediction\tconsensus_arg_orfs\tmge_hits\tmobility_marker_hits\t"
        "evidence_priority_tier\n"
        "a\tplasmid\t1\t1\t0\tB\n"
        "b\tchromosome\t0\t0\t0\tE\n"
        "c\tphage\t0\t0\t1\tD\n"
        "d\tunclassified\t0\t0\t0\tE\n",
    )
    output = tmp_path / "visualization"
    visualize(
        predictions_tsv=predictions,
        annotated_results_tsv=annotated,
        output_dir=output,
    )
    assert {
        "prediction_summary.tsv",
        "prediction_by_length_bin.tsv",
        "visualization_summary.json",
        "mobiorigin_summary.svg",
        "mobiorigin_dashboard.html",
        "SHA256SUMS.txt",
    } == {path.name for path in output.iterdir()}
    summary = json.loads((output / "visualization_summary.json").read_text())
    assert summary["records"] == 4
    assert summary["evidence_priority_tier_counts"]["B"] == 1
    assert summary["accuracy_metrics_calculated"] is False
    assert "Interpretation boundary" in (output / "mobiorigin_dashboard.html").read_text()


def test_cli_dispatches_visualize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "visualize", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "visualize",
            "--predictions-tsv",
            str(tmp_path / "predictions.tsv"),
            "--output-dir",
            str(tmp_path / "visualization"),
        ]
    )
    assert observed == {
        "predictions_tsv": tmp_path / "predictions.tsv",
        "output_dir": tmp_path / "visualization",
        "annotated_results_tsv": None,
    }


def test_arg_parsers_preserve_independent_evidence_and_filter_non_amr(tmp_path: Path) -> None:
    orfs = {"seq__orf_1": Orf("seq__orf_1", "seq", 1, 300, 1, 100)}
    card = write(
        tmp_path / "card.tsv",
        "seq__orf_1\tgb|ABC.1|ARO:3000001|blaX\t91\t95\t1e-20\t200\t"
        "gb|ABC.1|ARO:3000001|blaX description\n",
    )
    card_hits = parse_card_hits(
        card,
        orfs,
        {
            "ARO:3000001": {
                "gene": "blaX",
                "family": "class A beta-lactamase",
                "drug_class": "beta-lactam",
                "mechanism": "antibiotic inactivation",
            }
        },
    )
    sarg = write(
        tmp_path / "sarg.tsv",
        "seq__orf_1\tSARG|beta-lactam|bla*|WP_1.1\t90\t92\t1e-15\t180\t"
        "SARG|beta-lactam|bla*|WP_1.1 protein\n",
    )
    sarg_hits = parse_sarg_hits(sarg, orfs)
    assert card_hits[0].source == "CARD"
    assert card_hits[0].resistance_mechanism == "antibiotic inactivation"
    assert sarg_hits[0].source == "SARG"
    assert sarg_hits[0].drug_class == "beta-lactam"
    assert consensus_hits([sarg_hits[0], card_hits[0]]) == [card_hits[0]]

    official = write(
        tmp_path / "official.tsv",
        "Protein id\tElement symbol\tElement name\tType\tClass\tSubclass\tMethod\t"
        "% Coverage of reference\t% Identity to reference\tClosest reference accession\t"
        "Hierarchy node\n"
        "seq__orf_1\tblaX\tbeta-lactamase\tAMR\tBETA-LACTAM\tCEPHALOSPORIN\t"
        "BLASTP\t98\t99\tWP_1.1\tblaX_fam\n"
        "seq__orf_1\tstxA\tShiga toxin\tVIRULENCE\tSTX2\tstxA\tEXACTP\t100\t100\t"
        "WP_2.1\tstxA\n",
    )
    official_hits = parse_amrfinderplus_hits(official, orfs)
    assert len(official_hits) == 1
    assert official_hits[0].source == "AMRFINDERPLUS"
    assert official_hits[0].gene_symbol == "blaX"


def test_annotation_orf_coordinates_and_amrprot_hierarchy(tmp_path: Path) -> None:
    proteins = tmp_path / "proteins.faa"
    orfs = predict_annotation_orfs([FastaRecord("seq", "ATG" + "GCC" * 400 + "TAA")], proteins)
    assert orfs
    first = next(iter(orfs.values()))
    assert first.sequence_id == "seq"
    assert first.start >= 1
    assert first.end > first.start
    assert proteins.read_text(encoding="ascii").startswith(">seq__orf_")

    hierarchy_path = write(
        tmp_path / "fam.tsv",
        "#node_id\tparent_node_id\tgene_symbol\ttype\tclass\tsubclass\tfamily_name\n"
        "ALL\t\t-\t\t\t\t\n"
        "AMR\tALL\t-\tAMR\t\t\t\n"
        "BETA\tAMR\tblaX\t\tBETA-LACTAM\tCEPHALOSPORIN\tclass A family\n"
        "VIR\tALL\tstxA\tVIRULENCE\tSTX2\tstxA\tShiga toxin\n",
    )
    hierarchy = load_amrfinder_hierarchy(hierarchy_path)
    diamond = write(
        tmp_path / "amrprot.tsv",
        f"{first.identifier}\tABC.1\t99\t100\t1e-30\t300\t"
        "ABC.1|1|1|blaX|blaX_fam||1|CEPHALOSPORIN|BETA-LACTAM|class_A_beta_lactamase\n"
        f"{first.identifier}\tVIR.1\t100\t100\t1e-40\t400\t"
        "VIR.1|1|1|stxA|stxA||1|stxA|STX2|Shiga_toxin\n",
    )
    hits = parse_amrprot_hits(diamond, orfs, hierarchy)
    assert len(hits) == 1
    assert hits[0].source == "AMRPROT_DIAMOND"
    assert hits[0].gene_symbol == "blaX"


def test_arg_annotation_integration_is_atomic_and_prediction_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = write(tmp_path / "input.fasta", ">seq\nATGGCCGCCGCC\n")
    database = tmp_path / "databases"
    for directory in (database / "card", database / "sarg"):
        directory.mkdir(parents=True)
    write(database / "card" / "card.dmnd", "card\n")
    write(
        database / "card" / "aro_index.tsv",
        "ARO Accession\tARO Name\tAMR Gene Family\tDrug Class\tResistance Mechanism\n"
        "ARO:3000001\tblaX\tclass A\tbeta-lactam\tantibiotic inactivation\n",
    )
    write(database / "sarg" / "sarg.dmnd", "sarg\n")
    official_database = tmp_path / "official_amrfinder"
    official_database.mkdir()
    write(official_database / "version.txt", "test-version\n")

    def fake_orfs(records: list[FastaRecord], output: Path) -> dict[str, Orf]:
        output.write_text(">seq__orf_1\nMAAA\n", encoding="ascii")
        return {"seq__orf_1": Orf("seq__orf_1", records[0].identifier, 1, 12, 1, 4)}

    def fake_diamond(**kwargs: object) -> None:
        output = kwargs["output"]
        database_path = kwargs["database"]
        assert isinstance(output, Path) and isinstance(database_path, Path)
        if database_path.name == "card.dmnd":
            output.write_text(
                "seq__orf_1\tgb|ABC|ARO:3000001|blaX\t99\t100\t1e-30\t300\t"
                "gb|ABC|ARO:3000001|blaX\n",
                encoding="utf-8",
            )
        else:
            output.write_text("", encoding="utf-8")

    def fake_amrfinder(**kwargs: object) -> None:
        output = kwargs["output"]
        assert isinstance(output, Path)
        output.write_text(
            "Protein id\tElement symbol\tElement name\tType\tClass\tMethod\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("mobiorigin.annotate.predict_annotation_orfs", fake_orfs)
    monkeypatch.setattr("mobiorigin.annotate.run_arg_diamond", fake_diamond)
    monkeypatch.setattr("mobiorigin.annotate.run_amrfinderplus", fake_amrfinder)
    output = tmp_path / "annotations"
    annotate(
        input_fasta=fasta,
        output_dir=output,
        database_dir=database,
        diamond=Path("true"),
        amrfinder_bin=Path("true"),
        amrfinder_database=official_database,
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "SHA256SUMS.txt",
        "annotation_provenance.json",
        "annotation_summary.tsv",
        "arg_consensus.tsv",
        "arg_hits.tsv",
        "predicted_proteins.faa",
        "raw_evidence",
    ]
    provenance = json.loads((output / "annotation_provenance.json").read_text())
    assert provenance["annotation_is_prediction_independent"] is True
    assert provenance["official_amrfinderplus_executed"] is True
    with pytest.raises(FileExistsError):
        annotate(
            input_fasta=fasta,
            output_dir=output,
            database_dir=database,
            diamond=Path("true"),
            amrfinder_bin=Path("true"),
            amrfinder_database=official_database,
        )


def test_cli_dispatches_annotate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "annotate", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "annotate",
            "--input-fasta",
            str(tmp_path / "input.fasta"),
            "--output-dir",
            str(tmp_path / "out"),
            "--database-dir",
            str(tmp_path / "db"),
            "--amrfinder-database",
            str(tmp_path / "amrfinder"),
            "--threads",
            "4",
        ]
    )
    assert observed["threads"] == 4
    assert observed["amrfinder_mode"] == "official"
    assert observed["profile"] == "arg"
    assert observed["predictions_tsv"] is None


def test_comprehensive_evidence_priority_is_transparent_and_prediction_preserving(
    tmp_path: Path,
) -> None:
    records = [FastaRecord("seq", "A" * 2000)]
    predictions_path = write(
        tmp_path / "predictions.tsv",
        "sequence_id\tlength_bp\tprediction\tp_chromosome\tp_plasmid\tp_phage\t"
        "plasmid_score\tabstention_reason\n"
        "seq\t2000\tplasmid\t0.05\t0.9\t0.05\t0.85\t\n",
    )
    predictions = load_predictions(predictions_path, records)
    arg = ArgHit(
        "seq",
        "seq__orf_1",
        1,
        300,
        1,
        "CARD",
        "blaX",
        "beta-lactamase",
        "ARO:1",
        "class A",
        "beta-lactam",
        "inactivation",
        "DIAMOND_BLASTP",
        99.0,
        100.0,
        1e-30,
        300.0,
    )
    evidence = [
        *arg_evidence([arg]),
        EvidenceHit(
            "seq",
            "seq__orf_2",
            400,
            700,
            1,
            "MOBILITY",
            "MOB_SUITE_RELAXASE",
            "relaxase",
            "MOBF",
            "MOBF",
            "mob",
            "relaxase",
            "DIAMOND_BLASTP",
            70.0,
            90.0,
            1e-20,
            200.0,
        ),
        EvidenceHit(
            "seq",
            "seq__orf_3",
            800,
            1100,
            1,
            "MOBILITY",
            "MOB_SUITE_MPF",
            "mating_pair_formation",
            "MPF_F",
            "MPF_F",
            "mpf",
            "MPF",
            "DIAMOND_BLASTP",
            65.0,
            85.0,
            1e-12,
            150.0,
        ),
    ]
    integrated = tmp_path / "integrated.tsv"
    rows = write_integrated_results(integrated, records, evidence, predictions)
    assert rows[0]["prediction"] == "plasmid"
    assert rows[0]["p_plasmid"] == "0.9"
    assert rows[0]["evidence_priority_tier"] == "A"
    assert rows[0]["mobility_class"] == "conjugative"
    summary = tmp_path / "summary.json"
    write_publication_summary(summary, rows, evidence)
    payload = json.loads(summary.read_text())
    assert payload["interpretation"]["priority_is_clinical_risk_score"] is False
    assert payload["interpretation"]["annotation_changes_origin_prediction"] is False


def test_amrfinderplus_non_amr_evidence_remains_outside_arg_consensus(tmp_path: Path) -> None:
    orfs = {"seq__orf_1": Orf("seq__orf_1", "seq", 1, 300, 1, 100)}
    output = write(
        tmp_path / "amrfinder.tsv",
        "Protein id\tElement symbol\tElement name\tType\tSubtype\tClass\tMethod\t"
        "% Coverage of reference\t% Identity to reference\tClosest reference accession\n"
        "seq__orf_1\tstxA\tShiga toxin\tVIRULENCE\tVIRULENCE\tTOXIN\tEXACTP\t"
        "100\t100\tWP_1.1\n",
    )
    hits = parse_amrfinderplus_non_amr(output, orfs)
    assert len(hits) == 1
    assert hits[0].evidence_group == "VIRULENCE"
    assert hits[0].source == "AMRFINDERPLUS"
    assert hits[0].feature_name == "stxA"


def test_comprehensive_annotation_publishes_integrated_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = write(tmp_path / "input.fasta", ">seq\n" + "ATGGCC" * 400 + "\n")
    predictions = write(
        tmp_path / "predictions.tsv",
        "sequence_id\tlength_bp\tprediction\tp_chromosome\tp_plasmid\tp_phage\t"
        "plasmid_score\tabstention_reason\n"
        "seq\t2400\tplasmid\t0.05\t0.9\t0.05\t0.85\t\n",
    )
    database = tmp_path / "databases"
    required = {
        "card/card.dmnd": "card\n",
        "card/aro_index.tsv": (
            "ARO Accession\tARO Name\tAMR Gene Family\tDrug Class\tResistance Mechanism\n"
            "ARO:3000001\tblaX\tclass A\tbeta-lactam\tantibiotic inactivation\n"
        ),
        "sarg/sarg.dmnd": "sarg\n",
        "vfdb/vfdb_setA.dmnd": "vfdb\n",
        "vfdb/vfdb_indx.txt": "VFG000001(gb|WP_1.1)\tVFC0001\tToxin\n",
        "mge/isfinder.dmnd": "mge\n",
        "mge/mge_database.tsv": (
            "ID\tSub_class\tgene_name\tClass\tLength\n"
            "1_tnpA_X\tIS3\ttnpA\tinsertion_sequence\t300\n"
        ),
        "bacmet/bacmet.dmnd": "bacmet\n",
        "bacmet/Bacmet_list.tsv": (
            "BacMet_ID\tGene_name\tClass\tAccession\tOrganism\tLength\tLocation\tCompound\n"
            "BAC0001\tabeM\tBio\tQ1\tTest\t100\tChromosome\tTriclosan [class: phenol]\n"
        ),
        "mob_suite/rep_proteins.dmnd": "rep\n",
        "mob_suite/mob_proteins.dmnd": "mob\n",
        "mob_suite/mpf_proteins.dmnd": "mpf\n",
    }
    for relative, content in required.items():
        destination = database / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        write(destination, content)
    official_database = tmp_path / "official_amrfinder"
    official_database.mkdir()
    write(official_database / "version.txt", "test\n")

    def fake_orfs(records: list[FastaRecord], output: Path) -> dict[str, Orf]:
        output.write_text(
            ">seq__orf_1\nMAAA\n>seq__orf_2\nMBBB\n>seq__orf_3\nMCCC\n",
            encoding="ascii",
        )
        return {
            "seq__orf_1": Orf("seq__orf_1", records[0].identifier, 1, 300, 1, 100),
            "seq__orf_2": Orf("seq__orf_2", records[0].identifier, 400, 700, 1, 100),
            "seq__orf_3": Orf("seq__orf_3", records[0].identifier, 800, 1100, -1, 100),
        }

    def fake_arg_search(**kwargs: object) -> None:
        output = kwargs["output"]
        database_path = kwargs["database"]
        assert isinstance(output, Path) and isinstance(database_path, Path)
        if database_path.name == "card.dmnd":
            output.write_text(
                "seq__orf_1\tgb|ABC|ARO:3000001|blaX\t99\t100\t1e-30\t300\t"
                "gb|ABC|ARO:3000001|blaX\n",
                encoding="utf-8",
            )
        else:
            output.write_text("", encoding="utf-8")

    def fake_amrfinder(**kwargs: object) -> None:
        output = kwargs["output"]
        assert isinstance(output, Path)
        output.write_text(
            "Protein id\tElement symbol\tElement name\tType\tSubtype\tClass\tSubclass\t"
            "Method\t% Coverage of reference\t% Identity to reference\t"
            "Closest reference accession\tHierarchy node\n"
            "seq__orf_1\tblaX\tbeta-lactamase\tAMR\tAMR\tBETA-LACTAM\t\tEXACTP\t"
            "100\t100\tWP_ARG\tblaX\n"
            "seq__orf_2\tstxA\tShiga toxin\tVIRULENCE\tVIRULENCE\tTOXIN\t\tEXACTP\t"
            "100\t100\tWP_VF\tstxA\n"
            "seq__orf_3\tqacX\tbiocide resistance\tSTRESS\tBIOCIDE\tQAC\t\tBLASTP\t"
            "95\t90\tWP_QAC\tqacX\n",
            encoding="utf-8",
        )

    def fake_evidence_search(**kwargs: object) -> None:
        output = kwargs["output"]
        database_path = kwargs["database"]
        assert isinstance(output, Path) and isinstance(database_path, Path)
        rows = {
            "vfdb_setA.dmnd": (
                "seq__orf_2\tVFG000001(gb|WP_1.1)\t90\t100\t1e-20\t200\t"
                "VFG000001(gb|WP_1.1) stxA [Shiga toxin (VF1)] [Escherichia coli]\n"
            ),
            "isfinder.dmnd": "seq__orf_3\t1_tnpA_X\t80\t90\t1e-10\t150\ttnpA\n",
            "bacmet.dmnd": ("seq__orf_3\tBAC0001|abeM|tr|Q1\t90\t95\t1e-15\t180\tAbeM pump\n"),
            "rep_proteins.dmnd": (
                "seq__orf_1\tNC_1|IncP_s1_f0_o0\t70\t80\t1e-12\t170\tIncP replicon\n"
            ),
            "mob_proteins.dmnd": ("seq__orf_2\tMOBF_1\t70\t90\t1e-15\t190\tMOBF relaxase\n"),
            "mpf_proteins.dmnd": (
                "seq__orf_3\tMPF_F_1\t70\t90\t1e-15\t190\tMPF_F coupling protein\n"
            ),
        }
        output.write_text(rows[database_path.name], encoding="utf-8")

    monkeypatch.setattr("mobiorigin.annotate.predict_annotation_orfs", fake_orfs)
    monkeypatch.setattr("mobiorigin.annotate.run_arg_diamond", fake_arg_search)
    monkeypatch.setattr("mobiorigin.annotate.run_amrfinderplus", fake_amrfinder)
    monkeypatch.setattr("mobiorigin.biological_evidence.run_evidence_diamond", fake_evidence_search)
    output = tmp_path / "comprehensive"
    annotate(
        input_fasta=fasta,
        output_dir=output,
        database_dir=database,
        diamond=Path("true"),
        amrfinder_bin=Path("true"),
        amrfinder_database=official_database,
        profile="comprehensive",
        predictions_tsv=predictions,
    )
    for name in (
        "biological_evidence.tsv",
        "mobiorigin_annotated_results.tsv",
        "publication_summary.json",
        "mobiorigin_report.html",
    ):
        assert (output / name).is_file()
    integrated = (output / "mobiorigin_annotated_results.tsv").read_text()
    assert "\tplasmid\t" in integrated
    assert "\tA\tARG plus relaxase and mating-pair-formation evidence\t" in integrated
    report = (output / "mobiorigin_report.html").read_text()
    assert "not clinical risk scores" in report
    provenance = json.loads((output / "annotation_provenance.json").read_text())
    assert provenance["annotation_profile"] == "comprehensive"
    assert provenance["predictions_integrated"] is True
    assert provenance["classification_labels_or_probabilities_changed"] is False
