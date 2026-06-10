"""Annotate a FASTA file with biological marker features.

Runs pyrodigal (ORF prediction) + DIAMOND blastp against mob_proteins,
mpf_proteins, and rep_proteins databases + optional BLASTN against rep.dna.fas
to produce a per-contig annotation TSV compatible with plasflow2's marker
XGBoost second stage.

Optionally merges per-contig geNomad SPM features from `genomad annotate`
output (--genomad-genes) to produce a combined 26-feature annotation TSV.

Requirements
------------
  pip install pyrodigal biopython
  conda install -c bioconda diamond blast  (or: brew install diamond blast)

Usage
-----
  # MOB-suite only (14 features)
  python scripts/annotate_sequences.py \\
      --fasta   data/benchmark/benchmark.fna \\
      --mob-db  data/databases/mob_suite/mob_proteins.dmnd \\
      --mpf-db  data/databases/mob_suite/mpf_proteins.dmnd \\
      --rep-db  data/databases/mob_suite/rep_proteins.dmnd \\
      --rep-dna data/databases/mob_suite/rep.dna.fas \\
      --out     data/benchmark/annotations.tsv \\
      --threads 8

  # MOB-suite + geNomad SPM (26 features — recommended)
  python scripts/annotate_sequences.py \\
      --fasta          data/benchmark/benchmark.fna \\
      --genomad-genes  data/benchmark/genomad_ann/benchmark_annotate/benchmark_genes.tsv \\
      --out            data/benchmark/annotations.tsv \\
      --threads 8

Output TSV columns
------------------
  MOB-suite (14):
    contig_id, is_conjugative, is_mobilizable, has_replicon, has_ice,
    has_rep_protein, n_arg_per_kb, n_mge_per_kb, n_ice_per_kb, n_rep_per_kb,
    coding_density, n_orfs_per_kb, gc_content, length_bp

  geNomad SPM (12, added when --genomad-genes is provided):
    p_marker_freq, c_marker_freq, v_marker_freq, pp_marker_freq,
    median_p_spm, median_c_spm, median_v_spm, p_vs_c_logistic,
    strand_switch_rate, no_rbs_freq, canonical_sd_freq, n_plasmid_markers

After running, pass --annotation-tsv to plasflow2 predict:
  plasflow2 predict --annotation-tsv data/benchmark/annotations.tsv ...
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# geNomad SPM feature extraction (optional — only imported when --genomad-genes is given)
_genomad_available = False
try:
    from extract_genomad_features import extract_all as _gn_extract_all, GENOMAD_COLS, _ZERO_FEATURES
    _genomad_available = True
except ImportError:
    pass  # will try sys.path-based import at runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FASTA streaming
# ---------------------------------------------------------------------------

def _iter_fasta(path: Path):
    """Yield (seq_id, sequence) from a FASTA file (auto-detects gzip)."""
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


# ---------------------------------------------------------------------------
# ORF prediction (pyrodigal)
# ---------------------------------------------------------------------------

def predict_orfs(
    sequences: dict[str, str],
    proteins_faa: Path,
) -> dict[str, dict]:
    """Predict ORFs with pyrodigal. Returns contig_id → {n_orfs, covered_bp}.

    Also writes all predicted proteins to proteins_faa for DIAMOND input.
    """
    try:
        import pyrodigal  # type: ignore[import]
    except ImportError:
        logger.error(
            "pyrodigal is required: pip install pyrodigal"
        )
        sys.exit(1)

    gene_pred = pyrodigal.GeneFinder(meta=True)
    orf_data: dict[str, dict] = {}

    with open(proteins_faa, "w") as fh:
        for i, (sid, seq) in enumerate(sequences.items()):
            if (i + 1) % 1000 == 0:
                logger.info("  ORF prediction: %d / %d", i + 1, len(sequences))
            try:
                genes = gene_pred.find_genes(seq.encode())
                covered = sum(abs(g.end - g.begin) for g in genes)
                orf_data[sid] = {"n_orfs": len(genes), "covered_bp": covered}
                for j, gene in enumerate(genes, 1):
                    fh.write(f">{sid}_{j}\n{gene.translate()}\n")
            except Exception as e:
                logger.debug("  ORF fail for %s: %s", sid, e)
                orf_data[sid] = {"n_orfs": 0, "covered_bp": 0}

    logger.info("ORF prediction done: %d contigs, %d total ORFs",
                len(orf_data), sum(d["n_orfs"] for d in orf_data.values()))
    return orf_data


# ---------------------------------------------------------------------------
# BLASTN vs rep.dna.fas (nucleotide-level replicon typing)
# ---------------------------------------------------------------------------

def run_blastn_replicon(
    fasta: Path,
    rep_dna_fasta: Path,
    out_tsv: Path,
    work_dir: Path,
    threads: int = 8,
    min_pident: float = 80.0,
    min_qcov: float = 80.0,
) -> dict[str, int]:
    """Run BLASTN of contigs against rep.dna.fas for nucleotide replicon typing.

    Returns contig_id → hit_count dict.
    Requires 'makeblastdb' and 'blastn' in PATH (conda install -c bioconda blast).
    """
    blastn = shutil.which("blastn")
    makeblastdb = shutil.which("makeblastdb")
    if not blastn or not makeblastdb:
        logger.info("  blastn/makeblastdb not found — skipping nucleotide replicon typing")
        return {}

    db_prefix = work_dir / "rep_dna_blastdb"
    if not (work_dir / "rep_dna_blastdb.nsi").exists():
        logger.info("  Building BLAST nucleotide DB from %s …", rep_dna_fasta.name)
        r = subprocess.run(
            ["makeblastdb", "-in", str(rep_dna_fasta), "-dbtype", "nucl",
             "-out", str(db_prefix), "-quiet"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.warning("  makeblastdb failed: %s", r.stderr[:300])
            return {}

    logger.info("  Running BLASTN vs rep.dna.fas (pident≥%.0f%% qcov≥%.0f%%) …",
                min_pident, min_qcov)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [blastn, "-query", str(fasta), "-db", str(db_prefix),
         "-out", str(out_tsv),
         "-outfmt", "6 qseqid sseqid pident qcovhsp evalue",
         "-perc_identity", str(min_pident),
         "-qcov_hsp_perc", str(min_qcov),
         "-num_threads", str(threads),
         "-max_target_seqs", "1",
         "-evalue", "1e-5",
         "-task", "blastn"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.warning("  blastn failed: %s", r.stderr[:300])
        return {}

    contig_hits: dict[str, int] = {}
    with open(out_tsv) as fh:
        for line in fh:
            cols = line.strip().split("\t")
            if cols:
                cid = cols[0]
                contig_hits[cid] = contig_hits.get(cid, 0) + 1

    logger.info("  BLASTN replicon: %d contigs with hits", len(contig_hits))
    return contig_hits


# ---------------------------------------------------------------------------
# DIAMOND blastp
# ---------------------------------------------------------------------------

def run_diamond(
    proteins_faa: Path,
    db: Path,
    out_tsv: Path,
    threads: int = 8,
    min_id: float = 40.0,
    min_cov: float = 60.0,
    label: str = "",
) -> dict[str, int]:
    """Run DIAMOND blastp. Returns contig_id → hit_count dict."""
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "diamond", "blastp",
        "--query", str(proteins_faa),
        "--db", str(db).removesuffix(".dmnd"),
        "--out", str(out_tsv),
        "--outfmt", "6", "qseqid", "sseqid", "pident", "qcovhsp", "evalue",
        "--id", str(min_id),
        "--query-cover", str(min_cov),
        "--threads", str(threads),
        "--max-target-seqs", "1",
        "--sensitive",
        "--quiet",
    ]
    logger.info("  Running DIAMOND vs %s …", db.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("  DIAMOND failed (%s):\n%s", db.name, result.stderr[:400])
        return {}

    contig_hits: dict[str, int] = {}
    with open(out_tsv) as fh:
        for line in fh:
            orf_id = line.split("\t")[0].strip()
            contig_id = re.sub(r"_\d+$", "", orf_id)
            contig_hits[contig_id] = contig_hits.get(contig_id, 0) + 1

    logger.info("  DIAMOND %s: %d contigs with hits", label, len(contig_hits))
    return contig_hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate FASTA with biological marker features for PlasFlow v2"
    )
    parser.add_argument("--fasta",    type=Path, required=True,
                        help="Input FASTA file (can be gzipped)")
    parser.add_argument("--mob-db",   type=Path, default=None,
                        help="Relaxase DIAMOND DB (mob_proteins.dmnd)")
    parser.add_argument("--mpf-db",   type=Path, default=None,
                        help="MPF (mating pair formation) DIAMOND DB")
    parser.add_argument("--rep-db",   type=Path, default=None,
                        help="Replication protein DIAMOND DB (rep_proteins.dmnd)")
    parser.add_argument("--rep-dna",  type=Path, default=None,
                        help="Replicon nucleotide FASTA for BLASTN typing (rep.dna.fas)")
    parser.add_argument("--out",      type=Path, default=Path("data/benchmark/annotations.tsv"),
                        help="Output TSV path")
    parser.add_argument("--threads",  type=int, default=8)
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Working directory for intermediate files (default: temp dir)")
    parser.add_argument("--min-id",   type=float, default=40.0,
                        help="Minimum DIAMOND percent identity (default 40)")
    parser.add_argument("--min-cov",  type=float, default=60.0,
                        help="Minimum DIAMOND query coverage (default 60)")
    parser.add_argument("--genomad-genes", type=Path, default=None,
                        help="Path to *_genes.tsv from 'genomad annotate'. "
                             "When provided, adds 12 geNomad SPM features to the output TSV "
                             "(p_marker_freq, median_p_spm, p_vs_c_logistic, etc.).")
    args = parser.parse_args()

    # Auto-detect DBs from standard locations if not specified
    root = Path(__file__).parent.parent
    mob_db_candidates = [
        args.mob_db,
        root / "data/databases/mob_suite/mob_proteins.dmnd",
    ]
    mpf_db_candidates = [
        args.mpf_db,
        root / "data/databases/mob_suite/mpf_proteins.dmnd",
    ]
    rep_db_candidates = [
        args.rep_db,
        root / "data/databases/mob_suite/rep_proteins.dmnd",
    ]
    rep_dna_candidates = [
        args.rep_dna,
        root / "data/databases/mob_suite/rep.dna.fas",
    ]

    mob_db = next((p for p in mob_db_candidates if p and p.exists()), None)
    mpf_db = next((p for p in mpf_db_candidates if p and p.exists()), None)
    rep_db = next((p for p in rep_db_candidates if p and p.exists()), None)
    rep_dna = next((p for p in rep_dna_candidates if p and p.exists()), None)

    for name, db in [("MOB", mob_db), ("MPF", mpf_db), ("REP_prot", rep_db),
                     ("REP_dna (blastn)", rep_dna)]:
        logger.info("  %s DB: %s", name, db or "NOT FOUND")

    if not any([mob_db, mpf_db, rep_db]):
        logger.warning(
            "No DIAMOND databases found — only ORF features will be computed. "
            "Install mob_suite databases to get biological marker features."
        )

    # Load sequences
    logger.info("Loading sequences from %s …", args.fasta)
    sequences: dict[str, str] = {}
    for sid, seq in _iter_fasta(args.fasta):
        if len(seq) >= 500:   # skip very short fragments
            sequences[sid] = seq.upper()
    logger.info("Loaded %d sequences", len(sequences))

    # Working directory for intermediate files
    _tmpdir = None
    if args.work_dir:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        _tmpdir = tempfile.TemporaryDirectory(prefix="plasflow2_ann_")
        work_dir = Path(_tmpdir.name)

    try:
        # ── ORF prediction ──────────────────────────────────────────────────
        proteins_faa = work_dir / "proteins.faa"
        logger.info("Predicting ORFs …")
        orf_data = predict_orfs(sequences, proteins_faa)

        # ── DIAMOND searches ─────────────────────────────────────────────────
        relaxase_hits: dict[str, int] = {}
        mpf_hits: dict[str, int] = {}
        rep_hits: dict[str, int] = {}
        rep_dna_hits: dict[str, int] = {}

        if mob_db and proteins_faa.stat().st_size > 0:
            relaxase_hits = run_diamond(
                proteins_faa, mob_db,
                work_dir / "relaxase_hits.tsv", args.threads,
                args.min_id, args.min_cov, label="relaxase",
            )

        if mpf_db and proteins_faa.stat().st_size > 0:
            mpf_hits = run_diamond(
                proteins_faa, mpf_db,
                work_dir / "mpf_hits.tsv", args.threads,
                args.min_id, args.min_cov, label="MPF",
            )

        if rep_db and proteins_faa.stat().st_size > 0:
            rep_hits = run_diamond(
                proteins_faa, rep_db,
                work_dir / "rep_hits.tsv", args.threads,
                args.min_id, args.min_cov, label="rep_protein",
            )

        # ── BLASTN vs rep.dna.fas (nucleotide replicon typing) ───────────────
        if rep_dna:
            rep_dna_hits = run_blastn_replicon(
                args.fasta, rep_dna,
                work_dir / "rep_dna_hits.tsv", work_dir, args.threads,
            )

        # ── Load geNomad SPM features (optional) ────────────────────────────
        genomad_features: dict = {}
        gn_cols: list[str] = []
        gn_zero: dict = {}

        if args.genomad_genes:
            if not args.genomad_genes.exists():
                logger.warning("--genomad-genes path not found: %s — skipping SPM features",
                               args.genomad_genes)
            else:
                # Lazy import: try package import first, then sibling-script import
                global _genomad_available, _gn_extract_all, GENOMAD_COLS, _ZERO_FEATURES
                if not _genomad_available:
                    import sys as _sys
                    _sys.path.insert(0, str(Path(__file__).parent))
                    try:
                        from extract_genomad_features import (  # type: ignore
                            extract_all as _gn_extract_all,
                            GENOMAD_COLS,
                            _ZERO_FEATURES,
                        )
                        _genomad_available = True
                    except ImportError as exc:
                        logger.error("Cannot import extract_genomad_features.py: %s", exc)

                if _genomad_available:
                    logger.info("Loading geNomad SPM features from %s …", args.genomad_genes)
                    genomad_features = _gn_extract_all(args.genomad_genes)
                    gn_cols = list(GENOMAD_COLS)
                    gn_zero = dict(_ZERO_FEATURES)
                    n_gn_matched = sum(1 for sid in sequences if sid in genomad_features)
                    logger.info("  geNomad features loaded for %d / %d contigs",
                                n_gn_matched, len(sequences))

        # ── Assemble output TSV ──────────────────────────────────────────────
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "contig_id", "is_conjugative", "is_mobilizable",
            "has_replicon", "has_ice", "has_rep_protein",
            "n_arg_per_kb", "n_mge_per_kb", "n_ice_per_kb", "n_rep_per_kb",
            "coding_density", "n_orfs_per_kb", "gc_content", "length_bp",
        ] + gn_cols

        n_conjugative = n_mobilizable = n_rep_prot = n_rep_dna = 0

        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for sid, seq in sequences.items():
                length_bp = len(seq)
                length_kb = length_bp / 1000.0
                gc = (seq.count("G") + seq.count("C")) / max(length_bp, 1)

                orf = orf_data.get(sid, {"n_orfs": 0, "covered_bp": 0})
                n_orfs = orf["n_orfs"]
                cod_density = (
                    min(orf["covered_bp"] / max(length_bp, 1), 1.0)
                    if orf["covered_bp"] > 0 else 0.0
                )
                n_orfs_kb = n_orfs / max(length_kb, 0.001)

                n_relax    = relaxase_hits.get(sid, 0)
                n_mpf      = mpf_hits.get(sid, 0)
                n_rep_prot_hits = rep_hits.get(sid, 0)
                n_rep_dna_hits  = rep_dna_hits.get(sid, 0)

                is_conj    = 1 if (n_relax > 0 and n_mpf > 0) else 0
                is_mob     = 1 if (n_relax > 0 and n_mpf == 0) else 0
                has_rep_p  = 1 if n_rep_prot_hits > 0 else 0
                has_repl   = 1 if n_rep_dna_hits > 0 else 0  # nucleotide replicon hit

                if is_conj:
                    n_conjugative += 1
                if is_mob:
                    n_mobilizable += 1
                if has_rep_p:
                    n_rep_prot += 1
                if has_repl:
                    n_rep_dna += 1

                row = {
                    "contig_id":       sid,
                    "is_conjugative":  is_conj,
                    "is_mobilizable":  is_mob,
                    "has_replicon":    has_repl,   # BLASTN vs rep.dna.fas
                    "has_ice":         0,           # not computed here
                    "has_rep_protein": has_rep_p,
                    "n_arg_per_kb":    0.0,         # not computed here
                    "n_mge_per_kb":    0.0,         # not computed here
                    "n_ice_per_kb":    0.0,         # not computed here
                    "n_rep_per_kb":    f"{n_rep_prot_hits / max(length_kb, 0.001):.4f}",
                    "coding_density":  f"{cod_density:.4f}",
                    "n_orfs_per_kb":   f"{n_orfs_kb:.4f}",
                    "gc_content":      f"{gc:.4f}",
                    "length_bp":       length_bp,
                }
                # Merge geNomad SPM features (zero-filled for contigs with no genes)
                if gn_cols:
                    gf = genomad_features.get(sid, gn_zero)
                    for col in gn_cols:
                        v = gf.get(col, gn_zero.get(col, 0.0))
                        row[col] = f"{v:.6f}" if isinstance(v, float) else str(v)
                writer.writerow(row)

        logger.info("Annotation complete:")
        logger.info("  Total contigs:         %d", len(sequences))
        logger.info("  Conjugative:           %d", n_conjugative)
        logger.info("  Mobilizable:           %d", n_mobilizable)
        logger.info("  Rep protein (DIAMOND): %d", n_rep_prot)
        logger.info("  Replicon (BLASTN):     %d", n_rep_dna)
        if gn_cols:
            n_gn = sum(1 for sid in sequences if sid in genomad_features)
            logger.info("  geNomad SPM features:  %d / %d contigs annotated", n_gn, len(sequences))
            logger.info("  Feature columns:       %d (14 MOB-suite + 12 geNomad SPM)", len(fieldnames) - 1)
        logger.info("Output TSV: %s", args.out)
        logger.info("")
        logger.info("Next: re-run prediction with marker XGBoost:")
        logger.info("  plasflow2 predict --annotation-tsv %s --marker-model data/models/marker_xgb.pkl ...", args.out)

    finally:
        if _tmpdir:
            _tmpdir.cleanup()


if __name__ == "__main__":
    main()
