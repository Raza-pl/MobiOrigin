"""User-facing diagnostics, analysis orchestration, and bundled demonstration."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

import mobiorigin
from mobiorigin.annotate import annotate
from mobiorigin.annotation_database_setup import default_annotation_database_dir
from mobiorigin.database_setup import check_databases
from mobiorigin.model_setup import check_models, resolve_model_dir
from mobiorigin.predict import predict
from mobiorigin.visualize import visualize


def default_database_dir() -> Path:
    """Return the documented marker-database directory without creating it."""
    override = os.environ.get("MOBIORIGIN_DATABASE_DIR")
    if override:
        return Path(override).expanduser()
    root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "mobiorigin" / "marker_databases"


def resolve_database_dir(database_dir: Path | None) -> Path:
    """Resolve an explicit database path or the stable user default."""
    return database_dir.expanduser() if database_dir is not None else default_database_dir()


def _command_version(command: str, arguments: list[str]) -> dict[str, Any]:
    executable = shutil.which(command)
    if executable is None:
        return {"status": "MISSING", "executable": None, "version": None}
    completed = subprocess.run(
        [executable, *arguments], text=True, capture_output=True, check=False
    )
    output = (completed.stdout or completed.stderr).strip()
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "executable": executable,
        "version": output.splitlines()[0] if output else "unknown",
    }


def doctor(
    *,
    database_dir: Path | None = None,
    model_dir: Path | None = None,
    software_only: bool = False,
) -> dict[str, Any]:
    """Inspect the installed runtime and, unless omitted, frozen marker databases."""
    tools = {
        "diamond": _command_version("diamond", ["version"]),
        "amrfinder": _command_version("amrfinder", ["--version"]),
    }
    database = resolve_database_dir(database_dir)
    database_result: dict[str, Any] | None = None
    database_error: str | None = None
    models = resolve_model_dir(model_dir)
    model_result: dict[str, Any] | None = None
    model_error: str | None = None
    if not software_only:
        try:
            database_result = check_databases(database)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            database_error = str(error)
        try:
            model_result = check_models(models)
        except (FileNotFoundError, ValueError) as error:
            model_error = str(error)
    passed = all(item["status"] == "PASS" for item in tools.values()) and (
        software_only or (database_result is not None and model_result is not None)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "mobiorigin_version": mobiorigin.__version__,
        "software": tools,
        "database_dir": str(database),
        "database": database_result,
        "database_error": database_error,
        "model_dir": str(models),
        "models": model_result,
        "model_error": model_error,
        "next_step": (
            "Run: mobiorigin demo --output-dir mobiorigin_demo"
            if passed and not software_only
            else "Run the repository installer without --software-only to prepare databases."
        ),
    }


def run_analysis(
    *,
    input_fasta: Path,
    output_dir: Path,
    database_dir: Path | None = None,
    annotation_database_dir: Path | None = None,
    annotation_profile: str = "comprehensive",
    skip_annotation: bool = False,
    threads: int = 1,
) -> None:
    """Run prediction, annotation, and visualization as one atomic analysis."""
    if output_dir.exists():
        raise FileExistsError("Analysis output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        predictions = temporary / "predictions"
        annotation = temporary / "annotation"
        figures = temporary / "visualization"
        predict(
            input_fasta=input_fasta,
            output_dir=predictions,
            database_dir=resolve_database_dir(database_dir),
            threads=threads,
        )
        annotated_results: Path | None = None
        if not skip_annotation:
            annotation_databases = (
                annotation_database_dir.expanduser()
                if annotation_database_dir is not None
                else default_annotation_database_dir()
            )
            annotate(
                input_fasta=input_fasta,
                output_dir=annotation,
                database_dir=annotation_databases,
                threads=threads,
                diamond=Path("diamond"),
                amrfinder_mode="official",
                amrfinder_bin=Path("amrfinder"),
                amrfinder_database=annotation_databases / "amrfinderplus",
                profile=annotation_profile,
                predictions_tsv=predictions / "predictions.tsv",
            )
            annotated_results = annotation / "mobiorigin_annotated_results.tsv"
        visualize(
            predictions_tsv=predictions / "predictions.tsv",
            output_dir=figures,
            annotated_results_tsv=annotated_results,
        )
        visualization_summary = json.loads(
            (figures / "visualization_summary.json").read_text(encoding="utf-8")
        )
        (temporary / "README_RESULTS.txt").write_text(
            "MobiOrigin analysis completed successfully.\n\n"
            "Start here:\n"
            "  visualization/mobiorigin_dashboard.html   interactive browser report\n"
            "  visualization/mobiorigin_summary.svg      editable figure\n"
            "  predictions/predictions.tsv               per-sequence predictions\n"
            "  predictions/provenance.json               reproducibility record\n\n"
            + (
                "Annotation outputs:\n"
                "  annotation/mobiorigin_report.html      biological-evidence report\n"
                "  annotation/mobiorigin_annotated_results.tsv  integrated contig table\n"
                "  annotation/biological_evidence.tsv     retained evidence hits\n\n"
                if annotated_results is not None
                else "Annotation was skipped explicitly.\n\n"
            )
            + f"Records analyzed: {visualization_summary['records']}\n"
            f"Bases analyzed: {visualization_summary['bases']}\n\n"
            "These are predictions, not experimental proof or clinical risk scores.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def demo(*, output_dir: Path, database_dir: Path | None = None, threads: int = 1) -> dict[str, Any]:
    """Run the bundled synthetic smoke input and summarize the files produced."""
    demo_resource = resources.files("mobiorigin").joinpath("data/examples/demo.fasta")
    with resources.as_file(demo_resource) as input_fasta:
        run_analysis(
            input_fasta=input_fasta,
            output_dir=output_dir,
            database_dir=database_dir,
            skip_annotation=True,
            threads=threads,
        )
    summary = json.loads(
        (output_dir / "visualization" / "visualization_summary.json").read_text(encoding="utf-8")
    )
    with (output_dir / "predictions" / "predictions.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        prediction = next(csv.DictReader(handle, delimiter="\t"))
    return {
        "status": "PASS",
        "records": summary["records"],
        "bases": summary["bases"],
        "expected_test_prediction": prediction["prediction"],
        "expected_test_abstention_reason": prediction["abstention_reason"],
        "output_dir": str(output_dir.resolve()),
        "prediction_table": str((output_dir / "predictions" / "predictions.tsv").resolve()),
        "html_report": str((output_dir / "visualization" / "mobiorigin_dashboard.html").resolve()),
        "svg_figure": str((output_dir / "visualization" / "mobiorigin_summary.svg").resolve()),
        "interpretation": "Synthetic installation test only; do not use as biological evidence.",
    }
