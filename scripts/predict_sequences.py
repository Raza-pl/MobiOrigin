"""Run PlasFlow v2 prediction on a FASTA file with full marker XGBoost support.

This script mirrors what run_benchmark_evaluation.py does internally, exposing
all predict() options including --marker-model and --annotation-tsv.

Usage
-----
  python scripts/predict_sequences.py \\
      --input          data/test/W1.contigs.fa \\
      --model          data/models/mlp_v2.pt \\
      --marker-model   data/models/marker_xgb.pkl \\
      --annotation-tsv data/test/W1_annotations.tsv \\
      --out            results/W1_plasflow2/predictions.tsv

  python scripts/predict_sequences.py \\
      --input          data/test/GCA_054405655.1_ASM5440565v1_genomic.fna \\
      --model          data/models/mlp_v2.pt \\
      --marker-model   data/models/marker_xgb.pkl \\
      --annotation-tsv data/test/GCA_annotations.tsv \\
      --out            results/GCA_plasflow2/predictions.tsv

Output TSV columns
------------------
  sequence_id, label, plasmid, chromosome, phage, unclassified
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import os
import sys
import time
from pathlib import Path

# ── macOS ARM segfault fix ────────────────────────────────────────────────────
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _iter_fasta(path: Path):
    opener = gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)
    with opener as fh:
        cur_id, parts = None, []
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, "".join(parts)
                cur_id = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if cur_id is not None:
            yield cur_id, "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PlasFlow v2 prediction with marker XGBoost support"
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input FASTA (can be gzipped)")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "data/models/mlp_v2.pt",
                        help="MLP model path (default: data/models/mlp_v2.pt)")
    parser.add_argument("--marker-model", type=Path, default=None,
                        help="Marker XGBoost model path (optional)")
    parser.add_argument("--annotation-tsv", type=Path, default=None,
                        help="Pre-computed biological annotation TSV (optional)")
    parser.add_argument("--out", "-o", type=Path, required=True,
                        help="Output predictions TSV path")
    parser.add_argument("--min-length", type=int, default=1000,
                        help="Skip contigs shorter than this (default: 1000 bp)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Inference batch size (default: 512)")
    parser.add_argument("--alpha-base", type=float, default=0.3,
                        help="Base alpha for marker blending (default: 0.3)")
    parser.add_argument("--no-marker-model", action="store_true",
                        help="Skip marker XGBoost second stage even if marker_xgb.pkl exists")
    parser.add_argument("--context", type=str, default="unspecified",
                        choices=["unspecified", "wastewater", "clinical", "environmental"],
                        help="Sample context for Bayesian prior correction (default: unspecified)")
    # Cascade mode
    parser.add_argument("--stage1-model", type=Path, default=None,
                        help="Cascade Stage 1 model (.pt, plasmid vs. rest). "
                             "When given with --stage2-model, uses cascade_predict() "
                             "instead of the 3-class predict(). Ignores --model.")
    parser.add_argument("--stage2-model", type=Path, default=None,
                        help="Cascade Stage 2 model (.pt, chromosome vs. phage).")
    args = parser.parse_args()

    # ── Cascade mode auto-detect ──────────────────────────────────────────────
    use_cascade = False
    if args.stage1_model and args.stage2_model:
        use_cascade = True
        logger.info("  [cascade] Stage 1: %s", args.stage1_model)
        logger.info("  [cascade] Stage 2: %s", args.stage2_model)
    elif (args.stage1_model is None) != (args.stage2_model is None):
        # Only one stage given — check for both in data/models/
        s1_default = ROOT / "data/models/mlp_cascade_stage1.pt"
        s2_default = ROOT / "data/models/mlp_cascade_stage2.pt"
        if s1_default.exists() and s2_default.exists():
            args.stage1_model = s1_default
            args.stage2_model = s2_default
            use_cascade = True
            logger.info("  [cascade auto] Stage 1: %s", args.stage1_model)
            logger.info("  [cascade auto] Stage 2: %s", args.stage2_model)

    # ── Marker model (3-class mode only) ─────────────────────────────────────
    marker_model_path = args.marker_model
    if use_cascade:
        marker_model_path = None  # cascade doesn't use marker model yet
    elif args.no_marker_model:
        marker_model_path = None
        logger.info("  [--no-marker-model] Skipping marker XGBoost stage")
    elif marker_model_path is None:
        default_mm = ROOT / "data/models/marker_xgb.pkl"
        if default_mm.exists():
            marker_model_path = default_mm
            logger.info("  [auto] Marker XGBoost: %s", marker_model_path)

    annotation_tsv = args.annotation_tsv
    if annotation_tsv is None and not use_cascade:
        # try same directory as input, same stem
        candidate = args.input.parent / (args.input.name.split(".")[0] + "_annotations.tsv")
        if candidate.exists():
            annotation_tsv = candidate
            logger.info("  [auto] Annotation TSV: %s", annotation_tsv)

    # Load sequences
    logger.info("Loading sequences from %s …", args.input)
    seqs, ids = [], []
    skipped = 0
    for sid, seq in _iter_fasta(args.input):
        if len(seq) < args.min_length:
            skipped += 1
            continue
        seqs.append(seq.upper())
        ids.append(sid)
    logger.info("  Loaded %d sequences (skipped %d < %d bp)",
                len(seqs), skipped, args.min_length)

    if not seqs:
        logger.error("No sequences to classify.")
        sys.exit(1)

    # Run prediction
    logger.info("Running prediction …")
    t0 = time.time()

    if use_cascade:
        from plasflow2.classify.predict import cascade_predict  # noqa: E402
        results = cascade_predict(
            sequences=seqs,
            sequence_ids=ids,
            stage1_model_path=str(args.stage1_model),
            stage2_model_path=str(args.stage2_model),
            batch_size=args.batch_size,
        )
    else:
        from plasflow2.classify.predict import predict  # noqa: E402
        results = predict(
            sequences=seqs,
            sequence_ids=ids,
            model_path=str(args.model),
            marker_model_path=str(marker_model_path) if marker_model_path else None,
            annotation_tsv=str(annotation_tsv) if annotation_tsv else None,
            marker_alpha_base=args.alpha_base,
            batch_size=args.batch_size,
            source_context=args.context,
        )

    elapsed = time.time() - t0
    logger.info("  Prediction done in %.1f s", elapsed)

    # Write output TSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["sequence_id", "label", "plasmid", "chromosome", "phage", "unclassified"])
        for r in results:
            # r is a Prediction dataclass: .sequence_id, .label, .scores (dict)
            plas    = r.scores.get("plasmid", 0.0)
            chrom   = r.scores.get("chromosome", 0.0)
            phage   = r.scores.get("phage", 0.0)
            unclass = r.scores.get("unclassified", 0.0)
            writer.writerow([r.sequence_id, r.label,
                             f"{plas:.4f}", f"{chrom:.4f}",
                             f"{phage:.4f}", f"{unclass:.4f}"])
            counts[r.label] = counts.get(r.label, 0) + 1

    summary = "  ".join(f"{k}: {v:,}" for k, v in sorted(counts.items()))
    logger.info("Results: %s", summary)
    logger.info("Predictions written to %s", args.out)


if __name__ == "__main__":
    main()
