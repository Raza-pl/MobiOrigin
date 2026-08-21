"""Command-line interface for MobiOrigin."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

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
