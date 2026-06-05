"""Build the marker-feature training dataset for the XGBoost second-stage classifier.

This script computes biological marker features for labelled training sequences
and saves them as a .npz file for train_marker_model.py.

What it does
------------
For each class (plasmid / chromosome / phage):
  1. Loads sequences from the same sources as build_dataset.py.
  2. Runs pyrodigal to predict ORFs.
  3. Runs DIAMOND blastp against mob_proteins.dmnd (relaxase / MPF markers).
  4. Extracts per-contig marker features (ContigMarkerFeatures).
  5. Runs the existing MLP to add MLP softmax scores as features.

The resulting feature matrix has shape (N, 15) — the 15 features defined in
marker_classifier.MARKER_FEATURE_NAMES.

Usage
-----
    python scripts/build_marker_dataset.py \\
        --plasmid-dir  data/databases/plasmids/ \\
        --chrom-dir    data/gtdb_genomes/bacteria/ \\
        --data-dir     data/databases/ \\
        --model        data/models/mlp_v2.pt \\
        --mob-db       data/databases/mob_suite/mob_proteins.dmnd \\
        --max-per-class 30000 \\
        --threads      16 \\
        --out          data/marker_features.npz

Then train:
    python scripts/train_marker_model.py \\
        --features data/marker_features.npz \\
        --out      data/models/
"""

from __future__ import annotations

# ── macOS ARM segfault fix: cap BLAS threads before numpy/torch import ───────
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import gzip
import logging
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from Bio import SeqIO  # type: ignore[import]

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plasflow2.classify.features import extract_features  # noqa: E402
from plasflow2.classify.marker_classifier import (  # noqa: E402
    ContigMarkerFeatures,
    N_MARKER_FEATURES,
    extract_marker_features,
)
from plasflow2.utils.device import CLASS_TO_IDX  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FASTA_EXTS = {".fna", ".fa", ".fasta", ".fna.gz", ".fa.gz", ".fasta.gz"}


def _open_fasta(path: Path):
    name = path.name
    if name.endswith(".gz"):
        return SeqIO.parse(gzip.open(path, "rt"), "fasta")
    return SeqIO.parse(str(path), "fasta")


def _fasta_files(directory: Path) -> list[Path]:
    files = []
    for ext in _FASTA_EXTS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(set(files))


def sample_sequences(
    files: list[Path],
    max_total: int,
    min_length: int = 1000,
    window_sizes: tuple = (2000, 5000, 10_000),
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Load files, window sequences, reservoir-sample to max_total."""
    rng = random.Random(seed)
    reservoir: list[tuple[str, str]] = []
    n_seen = 0

    for fpath in files:
        try:
            for rec in _open_fasta(fpath):
                seq = str(rec.seq).upper()
                if len(seq) < min_length:
                    continue
                for w in window_sizes:
                    if w > len(seq):
                        continue
                    step = max(1, w // 2)
                    for start in range(0, len(seq) - w + 1, step):
                        fragment = seq[start: start + w]
                        n_seen += 1
                        if len(reservoir) < max_total:
                            reservoir.append((f"{rec.id}_w{w}_s{start}", fragment))
                        else:
                            j = rng.randint(0, n_seen - 1)
                            if j < max_total:
                                reservoir[j] = (f"{rec.id}_w{w}_s{start}", fragment)
        except Exception as e:
            logger.warning("Skipping %s: %s", fpath.name, e)

    logger.info("  Sampled %d fragments (from %d windows seen)", len(reservoir), n_seen)
    return reservoir


def run_diamond_db(
    proteins_faa: Path,
    db: Path,
    out_tsv: Path,
    threads: int = 8,
    min_id: float = 40.0,
    min_cov: float = 60.0,
) -> dict[str, int]:
    """Run DIAMOND blastp against db, return contig_id → hit count dict."""
    import re
    from collections import Counter
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "diamond", "blastp",
        "--query", str(proteins_faa),
        "--db", str(db).removesuffix(".dmnd"),
        "--out", str(out_tsv),
        "--outfmt", "6", "qseqid", "sseqid", "pident", "qcovhsp", "evalue",
        "--id", str(min_id), "--query-cover", str(min_cov),
        "--threads", str(threads),
        "--max-target-seqs", "1",
        "--sensitive", "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("DIAMOND failed (%s): %s", db.name, result.stderr[:200])
        return {}
    contig_hits: Counter = Counter()
    with open(out_tsv) as fh:
        for line in fh:
            orf_id = line.split("\t")[0].strip()
            contig_id = re.sub(r"_\d+$", "", orf_id)
            contig_hits[contig_id] += 1
    return dict(contig_hits)


def predict_mlp(
    sequences: list[str],
    seq_ids: list[str],
    model_path: Path,
) -> dict[str, dict[str, float]]:
    """Run the MLP and return contig_id → {class: score} dict."""
    from plasflow2.classify.predict import predict

    preds = predict(
        sequences=sequences,
        sequence_ids=seq_ids,
        model_path=model_path,
        threshold=0.0,       # no threshold — we want raw scores
        plasmid_threshold=0.0,
        argmax_fallback=True,
    )
    return {p.sequence_id: p.scores for p in preds}


def build_class_features(
    label: str,
    samples: list[tuple[str, str]],
    model_path: Path,
    mob_db: Path | None,
    mpf_db: Path | None,
    rep_db: Path | None,
    ice_db: Path | None,
    threads: int,
    work_dir: Path,
) -> NDArray:
    """Build (N, N_MARKER_FEATURES) matrix for one class with real computed features.

    Features computed here:
      - MLP scores (real, from model inference)
      - coding_density, n_orfs_per_kb (real, from pyrodigal ORF prediction)
      - is_mobilizable (real: relaxase hit from mob_proteins DIAMOND)
      - is_conjugative (real: relaxase + MPF hit from both mob + mpf DIAMOND)
      - gc_content, log10_length (real, from sequence)
      - has_replicon, has_ice, n_arg/mge/ice_per_kb: 0 (not available here)
        — XGBoost learns these don't discriminate in training; at inference
          they carry real signal from actual annotations.
    """
    import re

    seq_ids = [s[0] for s in samples]
    sequences = [s[1] for s in samples]
    n = len(sequences)
    logger.info("  Building marker features for %d %s sequences …", n, label)

    # 1. MLP scores
    logger.info("  Running MLP …")
    mlp_by_id = predict_mlp(sequences, seq_ids, model_path)

    # 2. ORF prediction + MOB DIAMOND
    # Per-contig ORF stats (coding_density, n_orfs_per_kb)
    orfs_per_contig: dict[str, int] = {}
    covered_per_contig: dict[str, int] = {}  # total bp covered by ORFs
    relaxase_hits: dict[str, int] = {}
    mpf_hits: dict[str, int] = {}
    rep_hits: dict[str, int] = {}
    ice_hits_train: dict[str, int] = {}

    try:
        import pyrodigal  # type: ignore[import]

        proteins_faa = work_dir / f"{label}_proteins.faa"
        logger.info("  Predicting ORFs …")
        gene_pred = pyrodigal.GeneFinder(meta=True)
        with open(proteins_faa, "w") as fh:
            for sid, seq in zip(seq_ids, sequences):
                try:
                    genes = gene_pred.find_genes(seq.encode())
                    orfs_per_contig[sid] = len(genes)
                    covered_per_contig[sid] = sum(
                        abs(g.end - g.begin) for g in genes
                    )
                    for i, gene in enumerate(genes, 1):
                        fh.write(f">{sid}_{i}\n{gene.translate()}\n")
                except Exception:
                    orfs_per_contig[sid] = 0
                    covered_per_contig[sid] = 0

        if mob_db and mob_db.exists():
            logger.info("  DIAMOND vs relaxase (mob_proteins) …")
            relaxase_hits = run_diamond_db(
                proteins_faa, mob_db,
                work_dir / f"{label}_relaxase_hits.tsv", threads,
            )
            logger.info("    Relaxase hits: %d contigs", len(relaxase_hits))

        if mpf_db and mpf_db.exists():
            logger.info("  DIAMOND vs MPF (mpf_proteins) …")
            mpf_hits = run_diamond_db(
                proteins_faa, mpf_db,
                work_dir / f"{label}_mpf_hits.tsv", threads,
            )
            logger.info("    MPF hits: %d contigs", len(mpf_hits))

        if rep_db and rep_db.exists():
            logger.info("  DIAMOND vs rep proteins (rep_proteins) …")
            rep_hits = run_diamond_db(
                proteins_faa, rep_db,
                work_dir / f"{label}_rep_hits.tsv", threads,
                min_id=40.0, min_cov=60.0,
            )
            logger.info("    Rep protein hits: %d contigs", len(rep_hits))

        if ice_db and ice_db.exists():
            logger.info("  DIAMOND vs ICE proteins …")
            ice_hits_train = run_diamond_db(
                proteins_faa, ice_db,
                work_dir / f"{label}_ice_hits.tsv", threads,
                min_id=70.0, min_cov=70.0,
            )
            logger.info("    ICE hits: %d contigs", len(ice_hits_train))

    except ImportError:
        logger.warning("  pyrodigal not available — ORF features will be 0")

    # 3. Assemble feature matrix
    X = np.zeros((n, N_MARKER_FEATURES), dtype=np.float32)
    for i, (sid, seq) in enumerate(zip(seq_ids, sequences)):
        mlp_scores = mlp_by_id.get(sid, {"plasmid": 0.33, "chromosome": 0.33, "phage": 0.34})

        length_bp = max(len(seq), 1)
        length_kb = length_bp / 1000.0
        seq_upper = seq.upper()
        gc = (seq_upper.count("G") + seq_upper.count("C")) / length_bp

        # Coding density from ORFs
        n_orfs = orfs_per_contig.get(sid, 0)
        covered = covered_per_contig.get(sid, 0)
        coding_density = min(covered / length_bp, 1.0) if length_bp > 0 else 0.85
        n_orfs_kb = n_orfs / length_kb if length_kb > 0 else 1.0

        # Mobility: relaxase only → mobilizable; relaxase + MPF → conjugative
        n_relaxase = relaxase_hits.get(sid, 0)
        n_mpf      = mpf_hits.get(sid, 0)
        n_rep      = rep_hits.get(sid, 0)
        n_ice      = ice_hits_train.get(sid, 0)
        is_conj    = 1.0 if (n_relaxase > 0 and n_mpf > 0) else 0.0
        is_mob     = 1.0 if (n_relaxase > 0 and n_mpf == 0) else 0.0

        feat = ContigMarkerFeatures(
            contig_id=sid,
            mlp_plasmid_score=mlp_scores.get("plasmid", 0.0),
            mlp_chromosome_score=mlp_scores.get("chromosome", 0.0),
            mlp_phage_score=mlp_scores.get("phage", 0.0),
            is_conjugative=is_conj,
            is_mobilizable=is_mob,
            has_replicon=0.0,          # not computed at training time
            has_ice=1.0 if n_ice > 0 else 0.0,
            has_rep_protein=1.0 if n_rep > 0 else 0.0,
            n_arg_per_kb=0.0,          # not computed at training time
            n_mge_per_kb=0.0,          # not computed at training time
            n_ice_per_kb=n_ice / max(length_kb, 0.001),
            n_rep_per_kb=n_rep / max(length_kb, 0.001),
            log10_length=float(np.log10(length_bp)),
            gc_content=gc,
            coding_density=coding_density,
            n_orfs_per_kb=n_orfs_kb,
        )
        X[i] = feat.to_array()

    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="Build marker feature dataset")
    parser.add_argument("--plasmid-dir", type=Path, default=None)
    parser.add_argument("--chrom-dir",   type=Path, default=None)
    parser.add_argument("--data-dir",    type=Path, default=Path("data/databases"))
    parser.add_argument("--model",       type=Path, required=True,
                        help="Trained MLP weights (.pt)")
    parser.add_argument("--mob-db",      type=Path, default=None,
                        help="Relaxase proteins DIAMOND DB (mob_proteins.dmnd)")
    parser.add_argument("--mpf-db",      type=Path, default=None,
                        help="MPF proteins DIAMOND DB (mpf_proteins.dmnd)")
    parser.add_argument("--max-per-class", type=int, default=30_000)
    parser.add_argument("--threads",     type=int, default=8)
    parser.add_argument("--out",         type=Path, default=Path("data/marker_features.npz"))
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    work_dir = args.out.parent / "marker_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect MOB / MPF / rep / ICE DBs
    mob_db = args.mob_db
    if mob_db is None:
        mob_db = args.data_dir / "mob_suite" / "mob_proteins.dmnd"
        if not mob_db.exists():
            mob_db = None
            logger.info("MOB DB not found — relaxase features will be 0")

    mpf_db = args.mpf_db
    if mpf_db is None:
        mpf_db = args.data_dir / "mob_suite" / "mpf_proteins.dmnd"
        if not mpf_db.exists():
            mpf_db = None
            logger.info("MPF DB not found — conjugative feature will be 0")

    rep_db = args.data_dir / "mob_suite" / "rep_proteins.dmnd"
    if not rep_db.exists():
        rep_db = None
        logger.info("rep_proteins.dmnd not found — run scripts/setup_rep_diamond.sh first")

    ice_db = args.data_dir / "ice" / "ice.dmnd"
    if not ice_db.exists():
        ice_db = None
        logger.info("ice.dmnd not found — ICE feature will be 0 at training time")

    all_X: list[NDArray] = []
    all_y: list[int] = []

    # Plasmid
    logger.info("=== PLASMID ===")
    plasmid_dir = args.plasmid_dir or (args.data_dir / "plasmids")
    if plasmid_dir.is_dir():
        files = _fasta_files(plasmid_dir)
        samples = sample_sequences(files, args.max_per_class, seed=args.seed)
        X_plas = build_class_features(
            "plasmid", samples, args.model, mob_db, mpf_db, rep_db, ice_db, args.threads, work_dir
        )
        all_X.append(X_plas)
        all_y.extend([CLASS_TO_IDX["plasmid"]] * len(samples))
        logger.info("Plasmid features: %s", X_plas.shape)
    else:
        logger.error("Plasmid dir not found: %s", plasmid_dir)

    # Chromosome
    logger.info("=== CHROMOSOME ===")
    chrom_dir = args.chrom_dir or (args.data_dir.parent / "gtdb_genomes" / "bacteria")
    if chrom_dir.is_dir():
        files = _fasta_files(chrom_dir)
        samples = sample_sequences(files, args.max_per_class, seed=args.seed + 1)
        X_chrom = build_class_features(
            "chromosome", samples, args.model, mob_db, mpf_db, rep_db, ice_db, args.threads, work_dir
        )
        all_X.append(X_chrom)
        all_y.extend([CLASS_TO_IDX["chromosome"]] * len(samples))
        logger.info("Chromosome features: %s", X_chrom.shape)
    else:
        logger.warning("Chromosome dir not found: %s — skipping", chrom_dir)

    # Phage
    logger.info("=== PHAGE ===")
    inphared_candidates = [
        args.data_dir / "inphared" / "inphared_phages.fa.gz",
        args.data_dir / "14Apr2025_genomes.fa.gz",
    ]
    inphared = next((p for p in inphared_candidates if p.exists()), None)
    if inphared:
        samples = sample_sequences([inphared], args.max_per_class, seed=args.seed + 2)
        X_phage = build_class_features(
            "phage", samples, args.model, mob_db, mpf_db, rep_db, ice_db, args.threads, work_dir
        )
        all_X.append(X_phage)
        all_y.extend([CLASS_TO_IDX["phage"]] * len(samples))
        logger.info("Phage features: %s", X_phage.shape)
    else:
        logger.warning("INPHARED not found — skipping phage class")

    X = np.vstack(all_X)
    y = np.array(all_y, dtype=np.int64)
    logger.info("Total dataset: X=%s  y=%s", X.shape, y.shape)

    np.savez(args.out, X=X, y=y, feature_names=np.array(
        __import__("plasflow2.classify.marker_classifier",
                   fromlist=["MARKER_FEATURE_NAMES"]).MARKER_FEATURE_NAMES
    ))
    logger.info("Saved marker features → %s", args.out)


if __name__ == "__main__":
    main()
