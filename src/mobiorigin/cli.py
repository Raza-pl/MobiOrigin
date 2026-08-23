"""Command-line interface for MobiOrigin."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from mobiorigin.annotate import annotate
from mobiorigin.database_setup import setup_databases
from mobiorigin.predict import predict


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="mobiorigin",
        description="Classify bacterial DNA sequences as chromosome, plasmid, phage or unclassified.",
    )
    subparsers = value.add_subparsers(dest="command", required=True)
    predict_parser = subparsers.add_parser("predict", help="run MobiOrigin prediction")
    predict_parser.add_argument("--input-fasta", type=Path, required=True)
    predict_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser.add_argument("--database-dir", type=Path, required=True)
    predict_parser.add_argument("--threads", type=int, default=1)
    setup_parser = subparsers.add_parser(
        "setup-databases", help="retrieve and verify the frozen MOB marker databases"
    )
    setup_parser.add_argument("--output-dir", type=Path, required=True)
    setup_parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="official MOB-suite data directory containing the three exact databases",
    )
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
    annotate_parser.add_argument("--threads", type=int, default=1)
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
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "predict":
        predict(
            input_fasta=args.input_fasta,
            output_dir=args.output_dir,
            database_dir=args.database_dir,
            threads=args.threads,
        )
    elif args.command == "setup-databases":
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
