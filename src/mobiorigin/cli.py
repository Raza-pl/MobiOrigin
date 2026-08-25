"""Command-line interface for MobiOrigin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mobiorigin.annotate import annotate
from mobiorigin.database_setup import check_databases, setup_databases
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
        "setup-databases", help="retrieve and verify the frozen MOB marker databases"
    )
    setup_parser.add_argument("--output-dir", type=Path, required=True)
    setup_parser.add_argument(
        "--source-dir",
        type=Path,
        help="official MOB-suite data directory containing the three exact databases",
    )
    setup_parser.add_argument(
        "--check",
        action="store_true",
        help="verify DIAMOND and an existing output directory without copying databases",
    )
    setup_parser.add_argument("--diamond", type=Path, default=Path("diamond"))
    annotate_parser = subparsers.add_parser(
        "annotate", help="annotate biological evidence without changing MobiOrigin predictions"
    )
    annotate_parser.add_argument("--input-fasta", type=Path, required=True)
    annotate_parser.add_argument("--output-dir", type=Path, required=True)
    annotate_parser.add_argument(
        "--database-dir",
        type=Path,
        required=True,
        help="directory containing card/, sarg/, and amrfinder/ resources",
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
        "run", help="run prediction and create tables, SVG, and an HTML dashboard"
    )
    run_parser.add_argument("--input-fasta", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--database-dir", type=Path)
    run_parser.add_argument(
        "--threads", type=int, default=1, help="external-search workers (1-128; default: 1)"
    )
    doctor_parser = subparsers.add_parser("doctor", help="check installation and databases")
    doctor_parser.add_argument("--database-dir", type=Path)
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
        if args.check:
            print(json.dumps(check_databases(args.output_dir, diamond=args.diamond), indent=2))
        else:
            if args.source_dir is None:
                argument_parser.error(
                    "setup-databases requires --source-dir unless --check is used"
                )
            setup_databases(output_dir=args.output_dir, source_dir=args.source_dir)
    elif args.command == "annotate":
        annotate(
            input_fasta=args.input_fasta,
            output_dir=args.output_dir,
            database_dir=args.database_dir,
            threads=args.threads,
            diamond=args.diamond,
            amrfinder_mode=args.amrfinder_mode,
            amrfinder_bin=args.amrfinder_bin,
            amrfinder_database=args.amrfinder_database,
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
            threads=args.threads,
        )
    elif args.command == "doctor":
        result = doctor(database_dir=args.database_dir, software_only=args.software_only)
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
