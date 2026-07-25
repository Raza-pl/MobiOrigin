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

The resulting feature matrix has shape (N, 28) — the 28 features defined in
marker_classifier.MARKER_FEATURE_NAMES.

Usage
-----
    python scripts/build_marker_dataset.py \\
        --plasmid-dir  data/databases/plasmids/ \\
        --chrom-dir    data/gtdb_genomes/bacteria/ \\
        --data-dir     data/databases/ \\
        --model        data/models/mlp_v2.pt \\
        --exclude-groups data/benchmark/locked_all_training_groups.txt \\
        --mob-db       data/databases/mob_suite/mob_proteins.dmnd \\
        --max-per-class 30000 \\
        --threads      16 \\
        --out          data/marker_features.npz

Then train:
    python scripts/train_marker_model.py \\
        --features data/marker_features.npz \\
        --exclude-groups data/benchmark/locked_all_training_groups.txt \\
        --out      data/models/
"""

from __future__ import annotations

# ruff: noqa: E402
# ── macOS ARM segfault fix: cap BLAS threads before numpy/torch import ───────
import os as _os

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_v, "1")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import gzip
import hashlib
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO  # type: ignore[import]
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasflow2.classify.marker_classifier import (  # noqa: E402
    MARKER_FEATURE_NAMES,
    N_MARKER_FEATURES,
    ContigMarkerFeatures,
)
from plasflow2.classify.splits import (  # noqa: E402
    source_group_id,
    validate_group_labels,
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
    excluded_groups: set[str] | None = None,
) -> tuple[list[tuple[str, str, str, str]], set[str]]:
    """Load files, window sequences, reservoir-sample to max_total.

    Returns sampled ``(fragment_id, fragment_seq, group_id, source_file)``
    tuples plus every eligible source group observed while scanning.
    *group_id* is the normalized source accession before windowing, so every
    overlapping window
    cut from the same genome/replicon shares one group_id, so a downstream
    grouped train/val split (see MarkerClassifier.train(groups=...)) can
    keep all of a genome's windows on one side of the split. Without this,
    50%-overlapping windows of the same sequence (near-duplicates in
    feature space) can land in both train and val, inflating val_accuracy.
    """
    rng = random.Random(seed)
    excluded_groups = excluded_groups or set()
    reservoir: list[tuple[str, str, str, str]] = []
    n_seen = 0
    n_excluded_records = 0
    observed_groups: set[str] = set()

    for fpath in files:
        try:
            for rec in _open_fasta(fpath):
                seq = str(rec.seq).upper()
                if len(seq) < min_length:
                    continue
                group_id = source_group_id(rec.id)
                if group_id in excluded_groups:
                    n_excluded_records += 1
                    continue
                observed_groups.add(group_id)
                for w in window_sizes:
                    if w > len(seq):
                        continue
                    step = max(1, w // 2)
                    for start in range(0, len(seq) - w + 1, step):
                        fragment = seq[start : start + w]
                        n_seen += 1
                        sample = (
                            f"{rec.id}_w{w}_s{start}",
                            fragment,
                            group_id,
                            str(fpath),
                        )
                        if len(reservoir) < max_total:
                            reservoir.append(sample)
                        else:
                            j = rng.randint(0, n_seen - 1)
                            if j < max_total:
                                reservoir[j] = sample
        except Exception as e:
            logger.warning("Skipping %s: %s", fpath.name, e)

    logger.info(
        "  Sampled %d fragments (from %d eligible windows; excluded %d locked records)",
        len(reservoir),
        n_seen,
        n_excluded_records,
    )
    return reservoir, observed_groups


def clean_samples(
    samples_by_label: dict[str, list[tuple[str, str, str, str]]],
    max_per_class: int,
    globally_conflicting_groups: set[str] | None = None,
) -> tuple[
    dict[str, list[tuple[str, str, str, str]]],
    dict[str, int],
]:
    """Remove ambiguous sources and duplicate sequences, then balance classes.

    Source groups occurring under more than one biological label are excluded
    from every class. Exact sequence duplicates are retained once within a
    class and excluded entirely when they cross class boundaries.
    """

    group_labels: dict[str, set[str]] = {}
    hash_labels: dict[str, set[str]] = {}
    for label, samples in samples_by_label.items():
        for _, sequence, group, _ in samples:
            group_labels.setdefault(group, set()).add(label)
            digest = hashlib.sha256(sequence.encode()).hexdigest()
            hash_labels.setdefault(digest, set()).add(label)

    conflicting_groups = {group for group, labels in group_labels.items() if len(labels) > 1}
    conflicting_groups.update(globally_conflicting_groups or set())
    conflicting_hashes = {digest for digest, labels in hash_labels.items() if len(labels) > 1}

    cleaned: dict[str, list[tuple[str, str, str, str]]] = {}
    duplicate_ids = 0
    duplicate_sequences = 0
    for label, samples in samples_by_label.items():
        retained: list[tuple[str, str, str, str]] = []
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        for sample in samples:
            sequence_id, sequence, group, _ = sample
            digest = hashlib.sha256(sequence.encode()).hexdigest()
            if group in conflicting_groups or digest in conflicting_hashes:
                continue
            if sequence_id in seen_ids:
                duplicate_ids += 1
                continue
            if digest in seen_hashes:
                duplicate_sequences += 1
                continue
            seen_ids.add(sequence_id)
            seen_hashes.add(digest)
            retained.append(sample)
        cleaned[label] = retained

    balanced_count = min(
        max_per_class,
        *(len(samples) for samples in cleaned.values()),
    )
    if balanced_count <= 0:
        raise ValueError("No samples remain after lockout and deduplication")
    for label in cleaned:
        cleaned[label] = cleaned[label][:balanced_count]

    summary = {
        "conflicting_source_groups_removed": len(conflicting_groups),
        "cross_class_sequence_hashes_removed": len(conflicting_hashes),
        "duplicate_sequence_ids_removed": duplicate_ids,
        "within_class_duplicate_sequences_removed": duplicate_sequences,
        "balanced_rows_per_class": balanced_count,
    }
    return cleaned, summary


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
        "diamond",
        "blastp",
        "--query",
        str(proteins_faa),
        "--db",
        str(db).removesuffix(".dmnd"),
        "--out",
        str(out_tsv),
        "--outfmt",
        "6",
        "qseqid",
        "sseqid",
        "pident",
        "qcovhsp",
        "evalue",
        "--id",
        str(min_id),
        "--query-cover",
        str(min_cov),
        "--threads",
        str(threads),
        "--max-target-seqs",
        "1",
        "--sensitive",
        "--quiet",
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
        threshold=0.0,  # no threshold — we want raw scores
        plasmid_threshold=0.0,
        argmax_fallback=True,
    )
    return {p.sequence_id: p.scores for p in preds}


def build_class_features(
    label: str,
    samples: list[tuple[str, str, str, str]],
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

    seq_ids = [sample[0] for sample in samples]
    sequences = [sample[1] for sample in samples]
    n = len(sequences)
    logger.info("  Building marker features for %d %s sequences …", n, label)

    # Save DNA FASTA so retrain script can run geNomad annotate on it
    dna_fasta = work_dir / f"{label}_training.fna"
    logger.info("  Saving training FASTA → %s", dna_fasta)
    with open(dna_fasta, "w") as _dna_fh:
        for _sid, _seq in zip(seq_ids, sequences):
            _dna_fh.write(f">{_sid}\n{_seq}\n")

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
                    covered_per_contig[sid] = sum(abs(g.end - g.begin) for g in genes)
                    for i, gene in enumerate(genes, 1):
                        fh.write(f">{sid}_{i}\n{gene.translate()}\n")
                except Exception:
                    orfs_per_contig[sid] = 0
                    covered_per_contig[sid] = 0

        if mob_db and mob_db.exists():
            logger.info("  DIAMOND vs relaxase (mob_proteins) …")
            relaxase_hits = run_diamond_db(
                proteins_faa,
                mob_db,
                work_dir / f"{label}_relaxase_hits.tsv",
                threads,
            )
            logger.info("    Relaxase hits: %d contigs", len(relaxase_hits))

        if mpf_db and mpf_db.exists():
            logger.info("  DIAMOND vs MPF (mpf_proteins) …")
            mpf_hits = run_diamond_db(
                proteins_faa,
                mpf_db,
                work_dir / f"{label}_mpf_hits.tsv",
                threads,
            )
            logger.info("    MPF hits: %d contigs", len(mpf_hits))

        if rep_db and rep_db.exists():
            logger.info("  DIAMOND vs rep proteins (rep_proteins) …")
            rep_hits = run_diamond_db(
                proteins_faa,
                rep_db,
                work_dir / f"{label}_rep_hits.tsv",
                threads,
                min_id=40.0,
                min_cov=60.0,
            )
            logger.info("    Rep protein hits: %d contigs", len(rep_hits))

        if ice_db and ice_db.exists():
            logger.info("  DIAMOND vs ICE proteins …")
            ice_hits_train = run_diamond_db(
                proteins_faa,
                ice_db,
                work_dir / f"{label}_ice_hits.tsv",
                threads,
                min_id=70.0,
                min_cov=70.0,
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
        n_mpf = mpf_hits.get(sid, 0)
        n_rep = rep_hits.get(sid, 0)
        n_ice = ice_hits_train.get(sid, 0)
        is_conj = 1.0 if (n_relaxase > 0 and n_mpf > 0) else 0.0
        is_mob = 1.0 if (n_relaxase > 0 and n_mpf == 0) else 0.0

        feat = ContigMarkerFeatures(
            contig_id=sid,
            mlp_plasmid_score=mlp_scores.get("plasmid", 0.0),
            mlp_chromosome_score=mlp_scores.get("chromosome", 0.0),
            mlp_phage_score=mlp_scores.get("phage", 0.0),
            is_conjugative=is_conj,
            is_mobilizable=is_mob,
            has_replicon=0.0,  # not computed at training time
            has_ice=1.0 if n_ice > 0 else 0.0,
            has_rep_protein=1.0 if n_rep > 0 else 0.0,
            n_arg_per_kb=0.0,  # not computed at training time
            n_mge_per_kb=0.0,  # not computed at training time
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
    parser.add_argument("--chrom-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/databases"))
    parser.add_argument("--model", type=Path, required=True, help="Trained MLP weights (.pt)")
    parser.add_argument(
        "--mob-db",
        type=Path,
        default=None,
        help="Relaxase proteins DIAMOND DB (mob_proteins.dmnd)",
    )
    parser.add_argument(
        "--mpf-db",
        type=Path,
        default=None,
        help="MPF proteins DIAMOND DB (mpf_proteins.dmnd)",
    )
    parser.add_argument("--max-per-class", type=int, default=30_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("data/marker_features.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-groups",
        type=Path,
        required=True,
        help="One locked source accession per line; these sources are never sampled.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Candidate-specific work directory (default: <out-stem>_work beside --out).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Row-level TSV manifest (default: <out>.manifest.tsv).",
    )
    args = parser.parse_args()

    if not args.exclude_groups.is_file():
        parser.error(f"--exclude-groups file not found: {args.exclude_groups}")
    excluded_groups = {
        line.strip() for line in args.exclude_groups.read_text().splitlines() if line.strip()
    }
    if not excluded_groups:
        parser.error("--exclude-groups is empty")

    work_dir = args.work_dir or args.out.parent / f"{args.out.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.out.with_suffix(".manifest.tsv")

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

    # Sample every class before expensive feature extraction. Oversampling
    # leaves headroom for conservative conflict and duplicate removal.
    requested = max(args.max_per_class, int(args.max_per_class * 1.15))
    samples_by_label: dict[str, list[tuple[str, str, str, str]]] = {}
    observed_groups_by_label: dict[str, set[str]] = {}

    logger.info("=== SAMPLE PLASMID ===")
    plasmid_dir = args.plasmid_dir or (args.data_dir / "plasmids")
    if plasmid_dir.is_dir():
        files = _fasta_files(plasmid_dir)
        (
            samples_by_label["plasmid"],
            observed_groups_by_label["plasmid"],
        ) = sample_sequences(
            files,
            requested,
            seed=args.seed,
            excluded_groups=excluded_groups,
        )
    else:
        raise FileNotFoundError(f"Plasmid dir not found: {plasmid_dir}")

    logger.info("=== SAMPLE CHROMOSOME ===")
    chrom_dir = args.chrom_dir or (args.data_dir.parent / "gtdb_genomes" / "bacteria")
    if chrom_dir.is_dir():
        files = _fasta_files(chrom_dir)
        (
            samples_by_label["chromosome"],
            observed_groups_by_label["chromosome"],
        ) = sample_sequences(
            files,
            requested,
            seed=args.seed + 1,
            excluded_groups=excluded_groups,
        )
    else:
        raise FileNotFoundError(f"Chromosome dir not found: {chrom_dir}")

    logger.info("=== SAMPLE PHAGE ===")
    inphared_candidates = [
        args.data_dir / "inphared" / "inphared_phages.fa.gz",
        args.data_dir / "14Apr2025_genomes.fa.gz",
    ]
    inphared = next((p for p in inphared_candidates if p.exists()), None)
    if inphared:
        (
            samples_by_label["phage"],
            observed_groups_by_label["phage"],
        ) = sample_sequences(
            [inphared],
            requested,
            seed=args.seed + 2,
            excluded_groups=excluded_groups,
        )
    else:
        raise FileNotFoundError("INPHARED phage FASTA not found")

    labels_by_observed_group: dict[str, set[str]] = {}
    for label, groups_for_label in observed_groups_by_label.items():
        for group in groups_for_label:
            labels_by_observed_group.setdefault(group, set()).add(label)
    globally_conflicting_groups = {
        group for group, labels in labels_by_observed_group.items() if len(labels) > 1
    }

    samples_by_label, cleaning_summary = clean_samples(
        samples_by_label,
        args.max_per_class,
        globally_conflicting_groups,
    )
    logger.info("Cleaning summary: %s", cleaning_summary)

    all_X: list[NDArray] = []
    all_y: list[int] = []
    all_groups: list[str] = []
    all_sequence_ids: list[str] = []
    all_sequence_hashes: list[str] = []
    all_source_files: list[str] = []

    for label in ("plasmid", "chromosome", "phage"):
        samples = samples_by_label[label]
        logger.info("=== BUILD %s FEATURES ===", label.upper())
        class_X = build_class_features(
            label,
            samples,
            args.model,
            mob_db,
            mpf_db,
            rep_db,
            ice_db,
            args.threads,
            work_dir,
        )
        all_X.append(class_X)
        all_y.extend([CLASS_TO_IDX[label]] * len(samples))
        all_groups.extend(sample[2] for sample in samples)
        all_sequence_ids.extend(sample[0] for sample in samples)
        all_sequence_hashes.extend(
            hashlib.sha256(sample[1].encode()).hexdigest() for sample in samples
        )
        all_source_files.extend(sample[3] for sample in samples)
        logger.info("%s features: %s", label.capitalize(), class_X.shape)

    X = np.vstack(all_X)
    y = np.array(all_y, dtype=np.int64)
    groups = np.asarray(all_groups, dtype=str)
    sequence_ids = np.asarray(all_sequence_ids, dtype=str)
    sequence_hashes = np.asarray(all_sequence_hashes, dtype=str)
    source_files = np.asarray(all_source_files, dtype=str)

    validate_group_labels(y, all_groups)
    if X.shape[1] != len(MARKER_FEATURE_NAMES):
        raise ValueError(
            f"Feature matrix has {X.shape[1]} columns, expected " f"{len(MARKER_FEATURE_NAMES)}"
        )
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains non-finite values")
    if len(set(sequence_ids.tolist())) != len(sequence_ids):
        raise ValueError("Duplicate sequence IDs remain after cleaning")
    if len(set(sequence_hashes.tolist())) != len(sequence_hashes):
        raise ValueError("Duplicate exact sequences remain after cleaning")
    overlap = set(groups.tolist()) & excluded_groups
    if overlap:
        raise ValueError(f"Benchmark lockout failed: {len(overlap)} excluded groups remain")

    n_distinct_groups = len(set(all_groups))
    logger.info(
        "Total dataset: X=%s  y=%s  groups=%s (%d distinct source genomes)",
        X.shape,
        y.shape,
        groups.shape,
        n_distinct_groups,
    )

    lockout_sha256 = hashlib.sha256(args.exclude_groups.read_bytes()).hexdigest()
    np.savez(
        args.out,
        X=X,
        y=y,
        groups=groups,
        sequence_ids=sequence_ids,
        sequence_sha256=sequence_hashes,
        source_files=source_files,
        feature_names=np.asarray(MARKER_FEATURE_NAMES, dtype=str),
        lockout_sha256=np.asarray(lockout_sha256),
        feature_schema_version=np.asarray("marker-v2"),
        feature_profile=np.asarray("incomplete-builder-v1"),
        training_prediction_parity_verified=np.asarray(False),
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        handle.write(
            "row_index\tsequence_id\tsource_group\tlabel\tlength\t" "sequence_sha256\tsource_file\n"
        )
        row_index = 0
        for label in ("plasmid", "chromosome", "phage"):
            for sequence_id, sequence, group, source_file in samples_by_label[label]:
                digest = hashlib.sha256(sequence.encode()).hexdigest()
                handle.write(
                    f"{row_index}\t{sequence_id}\t{group}\t{label}\t"
                    f"{len(sequence)}\t{digest}\t{source_file}\n"
                )
                row_index += 1

    summary = {
        "rows": int(len(y)),
        "features": int(X.shape[1]),
        "class_counts": {
            label: int((y == CLASS_TO_IDX[label]).sum())
            for label in ("plasmid", "chromosome", "phage")
        },
        "distinct_source_groups": n_distinct_groups,
        "lockout_groups": len(excluded_groups),
        "lockout_sha256": lockout_sha256,
        "lockout_overlap": 0,
        "feature_schema_version": "marker-v2",
        "feature_profile": "incomplete-builder-v1",
        "training_prediction_parity_verified": False,
        "cleaning": cleaning_summary,
        "manifest": str(manifest_path),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Saved marker features → %s", args.out)
    logger.info("Saved row manifest → %s", manifest_path)


if __name__ == "__main__":
    main()
