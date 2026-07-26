"""Unit tests for the end-to-end pipeline (pipeline.py).

All external calls (predict, annotate_contigs, run_mob_typer) are mocked so
these tests run without DIAMOND, mob_typer, or trained model weights.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from plasflow2.annotate.args import ARGHit
from plasflow2.annotate.mobility import MobilityResult
from plasflow2.annotate.plasmid_db import PlasmidDBHit
from plasflow2.classify.predict import Prediction
from plasflow2.classify.threshold_policy import ThresholdPolicyError
from plasflow2.pipeline import ContigResult, PipelineResult, run_pipeline
from plasflow2.risk.scorer import RiskScore

# ---------------------------------------------------------------------------
# Helpers for synthetic data
# ---------------------------------------------------------------------------

_SEQ = "ACGT" * 12_500  # 50,000 bp — passes min_length=1000 and hallmark gate (≥50 kb)


def _record(name: str, seq: str = _SEQ) -> SeqRecord:
    return SeqRecord(Seq(seq), id=name, description="")


def _prediction(seq_id: str, label: str, confidence: float = 0.95) -> Prediction:
    scores = {"plasmid": 0.0, "chromosome": 0.0, "phage": 0.0, "archaea": 0.0}
    if label in scores:
        scores[label] = confidence
    return Prediction(sequence_id=seq_id, label=label, confidence=confidence, scores=scores)


def _arg_hit(contig_id: str) -> ARGHit:
    return ARGHit(
        contig_id=contig_id,
        gene_name="NDM-6",
        aro_accession="ARO:3002356",
        amr_family="NDM beta-lactamase",
        drug_class="carbapenem antibiotic",
        resistance_mechanism="antibiotic inactivation",
        identity=99.5,
        coverage=95.0,
        evalue=1e-120,
    )


def _plasmid_db_hit(contig_id: str) -> PlasmidDBHit:
    return PlasmidDBHit(
        contig_id=contig_id,
        match_acc="PLSDB_NZ_CP012345.1",
        source_db="PLSDB",
        ani=99.2,
        query_cov=98.5,
        organism="Escherichia coli",
    )


def _mob_result(contig_id: str, mobility_class: str = "conjugative") -> MobilityResult:
    return MobilityResult(
        contig_id=contig_id,
        mobility_class=mobility_class,
        replicon_type="IncP-1alpha",
        relaxase_type="MOBP",
        mpf_type="MPF_T",
    )


def _risk(contig_id: str, score: int = 5) -> RiskScore:
    return RiskScore(contig_id=contig_id, score=score, evidence=["mock evidence"])


# ---------------------------------------------------------------------------
# Shared mock factory
# ---------------------------------------------------------------------------


def _mock_pipeline(
    tmp_path: Path,
    *,
    fasta_records: list[SeqRecord],
    predictions: list[Prediction],
    arg_hits: list[ARGHit] | None = None,
    mob_results: list[MobilityResult] | None = None,
    skip_mobility: bool = False,
    plasmid_db_hits: dict | None = None,
    require_hallmarks: bool = False,
    enable_marker_fusion: bool = False,
) -> PipelineResult:
    """Run run_pipeline() with all external I/O mocked."""
    fasta = tmp_path / "contigs.fasta"
    # Write a minimal FASTA so FileNotFoundError checks pass
    fasta.write_text("".join(f">{r.id}\n{r.seq}\n" for r in fasta_records))

    model = tmp_path / "mlp_v2.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")

    # annotate_plasmid_db() is only invoked when plasmid_db_dir is set (see
    # pipeline.py section 4d) -- provide one whenever the test wants
    # plasmid_db_hits to actually be injected, so the mocked return value is
    # reachable rather than silently ignored.
    plasmid_db_dir = None
    if plasmid_db_hits is not None:
        plasmid_db_dir = tmp_path / "plasmid_db_src"
        plasmid_db_dir.mkdir(exist_ok=True)

    with (
        patch("plasflow2.pipeline.load_fasta", return_value=fasta_records),
        patch("plasflow2.pipeline.predict", return_value=predictions),
        patch("plasflow2.pipeline.write_fasta"),
        patch("plasflow2.pipeline.annotate_contigs_with_orfs", return_value=(arg_hits or [], [])),
        patch("plasflow2.pipeline.annotate_plasmid_db", return_value=plasmid_db_hits or {}),
        # Mock DIAMOND fast path (preferred over mob_typer) — return mob_results directly
        patch(
            "plasflow2.pipeline.find_mob_diamond_dbs",
            return_value=("mock.dmnd", "mock.dmnd", None, None),
        ),
        patch(
            "plasflow2.pipeline.annotate_mobility_diamond",
            return_value=mob_results or [],
        ),
    ):
        return run_pipeline(
            fasta_path=fasta,
            model_path=model,
            card_db=card_db,
            aro_index=aro_index,
            work_dir=tmp_path / "work",
            skip_mobility=skip_mobility,
            plasmid_db_dir=plasmid_db_dir,
            require_hallmarks=require_hallmarks,
            enable_marker_fusion=enable_marker_fusion,
        )


def test_marker_fusion_is_disabled_by_default(tmp_path: Path) -> None:
    marker_path = tmp_path / "marker_xgb.json"
    marker_path.write_text("{}")

    with (
        patch(
            "plasflow2.pipeline.resolve_marker_model_path",
            return_value=marker_path,
        ),
        patch("plasflow2.pipeline.MarkerClassifier.load") as marker_load,
    ):
        _mock_pipeline(
            tmp_path,
            fasta_records=[_record("p1")],
            predictions=[_prediction("p1", "plasmid")],
        )

    marker_load.assert_not_called()


def test_explicit_marker_fusion_requires_native_model(tmp_path: Path) -> None:
    with patch(
        "plasflow2.pipeline.resolve_marker_model_path",
        return_value=None,
    ):
        with pytest.raises(
            FileNotFoundError,
            match="Experimental marker fusion was requested",
        ):
            _mock_pipeline(
                tmp_path,
                fasta_records=[_record("p1")],
                predictions=[_prediction("p1", "plasmid")],
                enable_marker_fusion=True,
            )


def test_explicit_marker_fusion_requires_xgboost(tmp_path: Path) -> None:
    marker_path = tmp_path / "marker_xgb.json"
    marker_path.write_text("{}")

    with (
        patch(
            "plasflow2.pipeline.resolve_marker_model_path",
            return_value=marker_path,
        ),
        patch(
            "plasflow2.pipeline.marker_classifier_available",
            return_value=False,
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="XGBoost is unavailable",
        ):
            _mock_pipeline(
                tmp_path,
                fasta_records=[_record("p1")],
                predictions=[_prediction("p1", "plasmid")],
                enable_marker_fusion=True,
            )


# ---------------------------------------------------------------------------
# PipelineResult.__post_init__ (no mocking needed)
# ---------------------------------------------------------------------------


def test_pipeline_result_counts_class_labels() -> None:
    preds = [
        _prediction("c1", "plasmid"),
        _prediction("c2", "chromosome"),
        _prediction("c3", "plasmid"),
        _prediction("c4", "phage"),
    ]
    result = PipelineResult(
        input_fasta=Path("x.fasta"),
        all_predictions=preds,
        plasmid_results=[],
    )
    assert result.class_counts["plasmid"] == 2
    assert result.class_counts["chromosome"] == 1
    assert result.class_counts["phage"] == 1
    assert result.total_sequences == 4
    assert result.total_plasmids == 0


def test_pipeline_result_total_args() -> None:
    cr = ContigResult(
        record=_record("p1"),
        prediction=_prediction("p1", "plasmid"),
        arg_hits=[_arg_hit("p1"), _arg_hit("p1")],
        mobility=None,
        risk=_risk("p1"),
    )
    result = PipelineResult(
        input_fasta=Path("x.fasta"),
        all_predictions=[_prediction("p1", "plasmid")],
        plasmid_results=[cr],
    )
    assert result.total_args == 2


# ---------------------------------------------------------------------------
# run_pipeline — basic flow
# ---------------------------------------------------------------------------


def test_run_pipeline_empty_fasta_returns_empty(tmp_path: Path) -> None:
    result = _mock_pipeline(tmp_path, fasta_records=[], predictions=[])
    assert result.total_sequences == 0
    assert result.plasmid_results == []


def test_run_pipeline_no_plasmids_returns_empty_plasmid_list(tmp_path: Path) -> None:
    records = [_record("c1"), _record("c2")]
    preds = [_prediction("c1", "chromosome"), _prediction("c2", "phage")]
    result = _mock_pipeline(tmp_path, fasta_records=records, predictions=preds)
    assert result.total_plasmids == 0
    assert result.plasmid_results == []
    assert result.total_sequences == 2


def test_run_pipeline_plasmid_count(tmp_path: Path) -> None:
    records = [_record("p1"), _record("c1"), _record("p2")]
    preds = [
        _prediction("p1", "plasmid"),
        _prediction("c1", "chromosome"),
        _prediction("p2", "plasmid"),
    ]
    result = _mock_pipeline(tmp_path, fasta_records=records, predictions=preds, skip_mobility=True)
    assert result.total_plasmids == 2


def test_default_policy_keeps_short_plasmid_without_hallmarks(
    tmp_path: Path,
) -> None:
    records = [_record("novel_p1", "ACGT" * 1_000)]
    preds = [_prediction("novel_p1", "plasmid", 0.86)]

    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        skip_mobility=True,
    )

    final = next(p for p in result.all_predictions if p.sequence_id == "novel_p1")
    assert final.label == "plasmid"
    assert result.total_plasmids == 1


def test_require_hallmarks_demotes_short_unsupported_plasmid(
    tmp_path: Path,
) -> None:
    records = [_record("unsupported_p1", "ACGT" * 1_000)]
    preds = [_prediction("unsupported_p1", "plasmid", 0.86)]

    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        skip_mobility=True,
        require_hallmarks=True,
    )

    final = next(p for p in result.all_predictions if p.sequence_id == "unsupported_p1")
    assert final.label == "unclassified"
    assert result.total_plasmids == 0


def test_require_hallmarks_keeps_supported_short_plasmid(
    tmp_path: Path,
) -> None:
    records = [_record("supported_p1", "ACGT" * 1_000)]
    preds = [_prediction("supported_p1", "plasmid", 0.86)]

    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        mob_results=[_mob_result("supported_p1", "mobilizable")],
        require_hallmarks=True,
    )

    final = next(p for p in result.all_predictions if p.sequence_id == "supported_p1")
    assert final.label == "plasmid"
    assert result.total_plasmids == 1


def test_run_pipeline_arg_hits_grouped_by_contig(tmp_path: Path) -> None:
    records = [_record("p1")]
    preds = [_prediction("p1", "plasmid")]
    hits = [_arg_hit("p1"), _arg_hit("p1")]
    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        arg_hits=hits,
        skip_mobility=True,
    )
    assert len(result.plasmid_results[0].arg_hits) == 2
    assert result.total_args == 2


def test_run_pipeline_mobility_attached(tmp_path: Path) -> None:
    records = [_record("p1")]
    preds = [_prediction("p1", "plasmid")]
    mob = [_mob_result("p1", "conjugative")]
    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        mob_results=mob,
    )
    assert result.plasmid_results[0].mobility is not None
    assert result.plasmid_results[0].mobility.mobility_class == "conjugative"


def test_run_pipeline_skip_mobility_sets_none(tmp_path: Path) -> None:
    records = [_record("p1")]
    preds = [_prediction("p1", "plasmid")]
    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        skip_mobility=True,
    )
    assert result.plasmid_results[0].mobility is None


def test_run_pipeline_risk_score_present(tmp_path: Path) -> None:
    records = [_record("p1")]
    preds = [_prediction("p1", "plasmid")]
    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        skip_mobility=True,
    )
    assert isinstance(result.plasmid_results[0].risk, RiskScore)


# ---------------------------------------------------------------------------
# run_pipeline — PLSDB hits + no marker XGBoost model available
# ---------------------------------------------------------------------------
#
# A "no marker model" fallback override used to live in this code path
# (force plasmid confidence to 0.97 for any plasmid_db_hits contig not
# already labeled plasmid). It was removed as dead code: PLSDB matching
# only runs on contigs already labeled plasmid (plasmid_fasta is built from
# plasmid_records), and the hallmark gate treats any plasmid_db_hits
# membership as sufficient evidence to keep -- or, for a widened near-miss
# candidate, explicitly promote -- a contig's label as "plasmid". So by the
# time that fallback ran, its own guard condition could never be true. See
# docs/CODE_REVIEW_FINDINGS_2026-07.md, Round 8.


def test_no_marker_model_plsdb_hit_stays_plasmid_with_original_scores(
    tmp_path: Path,
) -> None:
    # No marker_xgb.pkl written alongside the mock model -- exercises the
    # "no marker model" path. Confirms a plasmid_db_hits contig simply keeps
    # its original MLP scores untouched (no override, no crash), since
    # nothing between PLSDB matching and here can move it off "plasmid".
    records = [_record("p1")]
    original_scores = {"plasmid": 0.80, "chromosome": 0.15, "phage": 0.05, "archaea": 0.0}
    preds = [
        Prediction(
            sequence_id="p1",
            label="plasmid",
            confidence=0.80,
            scores=dict(original_scores),
        )
    ]
    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        plasmid_db_hits={"p1": _plasmid_db_hit("p1")},
        skip_mobility=True,
    )
    final = next(p for p in result.all_predictions if p.sequence_id == "p1")
    assert final.label == "plasmid"
    assert final.scores == original_scores


@pytest.mark.parametrize(
    "forbidden_options",
    [
        {"confidence_threshold": 0.50},
        {"plasmid_threshold": 0.80},
        {"argmax_fallback": True},
        {"lenient": True},
        {"require_hallmarks": True},
        {"enable_marker_fusion": True},
        {"widen_candidates": True},
        {"compass_sketch_path": "compass.npy"},
    ],
)
def test_conservative_pipeline_rejects_label_mutations(
    tmp_path: Path,
    forbidden_options: dict[str, object],
) -> None:
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(f">p1\n{_SEQ}\n")
    model = tmp_path / "candidate.pt"
    model.write_text("mock")

    with pytest.raises(
        ThresholdPolicyError,
        match="forbids pipeline mutations",
    ):
        run_pipeline(
            fasta_path=fasta,
            model_path=model,
            card_db=None,
            aro_index=None,
            work_dir=tmp_path / "work",
            profile="conservative",
            **forbidden_options,
        )


# ---------------------------------------------------------------------------
# run_pipeline — FileNotFoundError checks
# ---------------------------------------------------------------------------


def test_run_pipeline_missing_fasta_raises(tmp_path: Path) -> None:
    model = tmp_path / "mlp_v2.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")
    with pytest.raises(FileNotFoundError, match="fasta_path"):
        run_pipeline(
            fasta_path=tmp_path / "nonexistent.fasta",
            model_path=model,
            card_db=card_db,
            aro_index=aro_index,
            work_dir=tmp_path / "work",
        )


def test_run_pipeline_missing_model_raises(tmp_path: Path) -> None:
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(">c1\nACGT\n")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")
    with pytest.raises(FileNotFoundError, match="model_path"):
        run_pipeline(
            fasta_path=fasta,
            model_path=tmp_path / "missing.pt",
            card_db=card_db,
            aro_index=aro_index,
            work_dir=tmp_path / "work",
        )


# ---------------------------------------------------------------------------
# run_pipeline — mob_typer failure is graceful
# ---------------------------------------------------------------------------


def test_run_pipeline_mob_typer_failure_is_graceful(tmp_path: Path) -> None:
    """If mob_typer raises RuntimeError, pipeline continues with mobility=None."""
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(f">p1\n{_SEQ}\n")
    model = tmp_path / "mlp_v2.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")

    with (
        patch("plasflow2.pipeline.load_fasta", return_value=[_record("p1")]),
        patch("plasflow2.pipeline.predict", return_value=[_prediction("p1", "plasmid")]),
        patch("plasflow2.pipeline.write_fasta"),
        patch("plasflow2.pipeline.annotate_contigs_with_orfs", return_value=([], [])),
        patch("plasflow2.pipeline.annotate_plasmid_db", return_value={}),
        patch("plasflow2.pipeline.run_mob_typer", side_effect=RuntimeError("mob_typer not found")),
    ):
        result = run_pipeline(
            fasta_path=fasta,
            model_path=model,
            card_db=card_db,
            aro_index=aro_index,
            work_dir=tmp_path / "work",
        )

    assert result.total_plasmids == 1
    assert result.plasmid_results[0].mobility is None
