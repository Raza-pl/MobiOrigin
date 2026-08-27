"""Command-line interface for MobiOrigin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mobiorigin.annotate import annotate
from mobiorigin.annotation_database_setup import (
    check_annotation_databases,
    default_annotation_database_dir,
    setup_annotation_databases,
)
from mobiorigin.database_setup import check_databases, setup_databases
from mobiorigin.model_setup import check_models, default_model_dir, setup_models
from mobiorigin.predict import predict
from mobiorigin.visualize import visualize
from mobiorigin.workflow import demo, doctor, resolve_database_dir, run_analysis


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="mobiorigin",
        description="Classify bacterial DNA sequences as chromosome, plasmid, phage or unclassified.",
    )
    subparsers = value.add_subparsers(dest="command", required=True)
    predict_parser = subparsers.add_parser("predict", help="run MobiOrigin prediction")
    predict_parser.add_argument("--input-fasta", type=Path, required=True)
    predict_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser.add_argument(
        "--database-dir",
        type=Path,
        help="marker database directory (default: $MOBIORIGIN_DATABASE_DIR or user data directory)",
    )
    predict_parser.add_argument(
        "--threads", type=int, default=1, help="external-search workers (1-128; default: 1)"
    )
    setup_parser = subparsers.add_parser(
        "setup-databases", help="prepare and verify marker or annotation databases"
    )
    setup_parser.add_argument(
        "--component", choices=("marker", "annotation", "models"), default="marker"
    )
    setup_parser.add_argument("--output-dir", type=Path)
    setup_parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "prepared source directory: frozen MOB databases for marker setup, or the "
            "documented offline mirror for annotation setup; annotation resources are "
            "downloaded automatically when omitted"
        ),
    )
    setup_parser.add_argument(
        "--check",
        action="store_true",
        help="verify DIAMOND and an existing output directory without copying databases",
    )
    setup_parser.add_argument(
        "--profile",
        choices=("arg", "comprehensive"),
        default="comprehensive",
        help="annotation resources to stage or verify (default: comprehensive)",
    )
    setup_parser.add_argument(
        "--accept-third-party-terms",
        action="store_true",
        help="confirm authorized acquisition and acceptance of upstream database terms",
    )
    setup_parser.add_argument(
        "--amrfinder-database",
        type=Path,
        help="optional existing AMRFinderPlus version directory (otherwise downloaded)",
    )
    setup_parser.add_argument(
        "--marker-database-dir",
        type=Path,
        help="existing MobiOrigin marker directory used by comprehensive annotation",
    )
    setup_parser.add_argument(
        "--legacy-isfinder-source-dir",
        type=Path,
        help=(
            "optional authorized directory containing isfinder.dmnd and mge_database.tsv; "
            "ISfinder is never downloaded by MobiOrigin"
        ),
    )
    setup_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="download cache for resumable annotation database retrieval",
    )
    setup_parser.add_argument(
        "--amrfinder-update",
        type=Path,
        default=Path("amrfinder_update"),
        help="AMRFinderPlus database updater executable",
    )
    setup_parser.add_argument("--diamond", type=Path, default=Path("diamond"))
    setup_parser.add_argument(
        "--model-archive",
        type=Path,
        help="optional offline copy of the exact frozen model archive",
    )
    annotate_parser = subparsers.add_parser(
        "annotate", help="annotate biological evidence without changing MobiOrigin predictions"
    )
    annotate_parser.add_argument("--input-fasta", type=Path, required=True)
    annotate_parser.add_argument("--output-dir", type=Path, required=True)
    annotate_parser.add_argument(
        "--database-dir",
        type=Path,
        help=(
            "annotation database directory (default: $MOBIORIGIN_ANNOTATION_DATABASE_DIR "
            "or user data directory)"
        ),
    )
    annotate_parser.add_argument(
        "--threads", type=int, default=1, help="external-search workers (1-128; default: 1)"
    )
    annotate_parser.add_argument("--diamond", type=Path, default=Path("diamond"))
    annotate_parser.add_argument(
        "--amrfinder-mode",
        choices=("official", "amrprot"),
        default="official",
        help="official AMRFinderPlus (default) or explicitly supplemental AMRProt DIAMOND",
    )
    annotate_parser.add_argument("--amrfinder-bin", type=Path, default=Path("amrfinder"))
    annotate_parser.add_argument(
        "--amrfinder-database",
        type=Path,
        help="complete official AMRFinderPlus database directory (required in official mode)",
    )
    annotate_parser.add_argument(
        "--profile",
        choices=("arg", "comprehensive"),
        default="arg",
        help="ARG-only output or comprehensive ARG/VF/MGE/stress/mobility evidence",
    )
    annotate_parser.add_argument(
        "--predictions-tsv",
        type=Path,
        help="optional matching MobiOrigin predictions.tsv for an integrated publication table",
    )
    visualize_parser = subparsers.add_parser(
        "visualize", help="create deterministic tables, SVG, and HTML from MobiOrigin outputs"
    )
    visualize_parser.add_argument("--predictions-tsv", type=Path, required=True)
    visualize_parser.add_argument("--output-dir", type=Path, required=True)
    visualize_parser.add_argument(
        "--annotated-results-tsv",
        type=Path,
        help="optional matching mobiorigin_annotated_results.tsv for evidence-tier summaries",
    )
    run_parser = subparsers.add_parser(
        "run", help="run prediction, comprehensive annotation, and visualization"
    )
    run_parser.add_argument("--input-fasta", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--database-dir", type=Path, help="marker database directory")
    run_parser.add_argument(
        "--annotation-database-dir",
        type=Path,
        help=(
            "annotation database directory (default: "
            "$MOBIORIGIN_ANNOTATION_DATABASE_DIR or user data directory)"
        ),
    )
    run_parser.add_argument(
        "--annotation-profile",
        choices=("arg", "comprehensive"),
        default="comprehensive",
        help="annotation evidence profile (default: comprehensive)",
    )
    run_parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="run prediction and visualization only",
    )
    run_parser.add_argument(
        "--threads", type=int, default=1, help="external-search workers (1-128; default: 1)"
    )
    doctor_parser = subparsers.add_parser("doctor", help="check installation and databases")
    doctor_parser.add_argument("--database-dir", type=Path)
    doctor_parser.add_argument("--model-dir", type=Path)
    doctor_parser.add_argument(
        "--software-only", action="store_true", help="check installed commands without databases"
    )
    demo_parser = subparsers.add_parser(
        "demo", help="run a bundled synthetic installation test and create example outputs"
    )
    demo_parser.add_argument("--output-dir", type=Path, default=Path("mobiorigin_demo"))
    demo_parser.add_argument("--database-dir", type=Path)
    demo_parser.add_argument(
        "--threads", type=int, default=1, help="external-search workers (1-128; default: 1)"
    )
    return value


def main(argv: Sequence[str] | None = None) -> None:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.command == "predict":
        predict(
            input_fasta=args.input_fasta,
            output_dir=args.output_dir,
            database_dir=resolve_database_dir(args.database_dir),
            threads=args.threads,
        )
    elif args.command == "setup-databases":
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = (
                default_annotation_database_dir()
                if args.component == "annotation"
                else (
                    default_model_dir()
                    if args.component == "models"
                    else resolve_database_dir(None)
                )
            )
        if args.component == "models":
            if args.check:
                print(json.dumps(check_models(output_dir), indent=2))
            else:
                print(
                    json.dumps(
                        setup_models(
                            output_dir,
                            archive=args.model_archive,
                            cache_dir=args.cache_dir,
                        ),
                        indent=2,
                    )
                )
        elif args.component == "annotation":
            if args.check:
                result = check_annotation_databases(output_dir, profile=args.profile)
            else:
                result = setup_annotation_databases(
                    output_dir=output_dir,
                    source_dir=args.source_dir,
                    amrfinder_database=args.amrfinder_database,
                    marker_database_dir=(
                        args.marker_database_dir
                        if args.marker_database_dir is not None
                        else resolve_database_dir(None)
                    ),
                    legacy_isfinder_source_dir=args.legacy_isfinder_source_dir,
                    cache_dir=args.cache_dir,
                    diamond=args.diamond,
                    amrfinder_update=args.amrfinder_update,
                    profile=args.profile,
                    accept_third_party_terms=args.accept_third_party_terms,
                )
            print(json.dumps(result, indent=2))
        elif args.check:
            print(json.dumps(check_databases(output_dir, diamond=args.diamond), indent=2))
        else:
            if args.source_dir is None:
                argument_parser.error(
                    "setup-databases requires --source-dir unless --check is used"
                )
            setup_databases(output_dir=output_dir, source_dir=args.source_dir)
    elif args.command == "annotate":
        annotation_database_dir = (
            args.database_dir.expanduser()
            if args.database_dir is not None
            else default_annotation_database_dir()
        )
        annotate(
            input_fasta=args.input_fasta,
            output_dir=args.output_dir,
            database_dir=annotation_database_dir,
            threads=args.threads,
            diamond=args.diamond,
            amrfinder_mode=args.amrfinder_mode,
            amrfinder_bin=args.amrfinder_bin,
            amrfinder_database=(
                args.amrfinder_database
                if args.amrfinder_database is not None
                else annotation_database_dir / "amrfinderplus"
            ),
            profile=args.profile,
            predictions_tsv=args.predictions_tsv,
        )
    elif args.command == "visualize":
        visualize(
            predictions_tsv=args.predictions_tsv,
            output_dir=args.output_dir,
            annotated_results_tsv=args.annotated_results_tsv,
        )
    elif args.command == "run":
        run_analysis(
            input_fasta=args.input_fasta,
            output_dir=args.output_dir,
            database_dir=args.database_dir,
            annotation_database_dir=args.annotation_database_dir,
            annotation_profile=args.annotation_profile,
            skip_annotation=args.skip_annotation,
            threads=args.threads,
        )
    elif args.command == "doctor":
        result = doctor(
            database_dir=args.database_dir,
            model_dir=args.model_dir,
            software_only=args.software_only,
        )
        print(json.dumps(result, indent=2))
        if result["status"] != "PASS":
            raise SystemExit(1)
    elif args.command == "demo":
        print(
            json.dumps(
                demo(
                    output_dir=args.output_dir,
                    database_dir=args.database_dir,
                    threads=args.threads,
                ),
                indent=2,
            )
        )
