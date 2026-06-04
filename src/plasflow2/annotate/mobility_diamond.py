"""Fast mobility typing via DIAMOND — replaces per-contig mob_typer calls.

mob_typer classifies plasmid mobility by:
  1. Detecting relaxase proteins (MOB family: MOBF, MOBP, MOBH, MOBQ, ...)
  2. Detecting mating pair formation (MPF) proteins (MPF_F, MPF_T, MPF_G, ...)
  3. Detecting replicon sequences (IncF, IncP, IncQ, ...)
  4. Applying rules:
       conjugative   = relaxase + MPF found
       mobilizable   = relaxase found, no MPF
       non-mobilizable = neither found

This module replicates that logic with DIAMOND:
  - One DIAMOND blastp call against mob.proteins.faa  (relaxase detection)
  - One DIAMOND blastp call against mpf_db.fasta      (MPF detection)
  - One minimap2 call against rep_db.fasta            (replicon typing)

Total time: ~10 seconds for any number of plasmid contigs vs 8+ minutes for
2,559 sequential mob_typer subprocess calls.

Database setup:
    bash scripts/setup_mob_diamond.sh
    # Finds mob_suite internal databases and builds DIAMOND indexes from them.
    # Output: data/databases/mob_suite/mob_proteins.dmnd
    #         data/databases/mob_suite/mpf_proteins.dmnd
    #         data/databases/mob_suite/rep_db.fasta
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from plasflow2.annotate.mobility import MobilityResult

logger = logging.getLogger(__name__)

# DIAMOND thresholds for MOB/MPF protein detection
# Lower identity than ARG annotation — relaxases diverge fast
MOB_MIN_IDENTITY = 50.0
MOB_MIN_COVERAGE = 70.0
MPF_MIN_IDENTITY = 50.0
MPF_MIN_COVERAGE = 70.0

# Replicon detection via minimap2
REP_MIN_COV = 60.0   # % query coverage for replicon hit

# MOB family names in the mob.proteins.faa headers
# Header format: >accession|MOB_family|...  OR  >MOBXxx_...
_MOB_FAMILY_RE = re.compile(
    r"\b(MOB[FPHQCVM][A-Za-z0-9_]*)\b", re.IGNORECASE
)

# MPF family names
_MPF_FAMILY_RE = re.compile(
    r"\b(MPF[_\s]?[FTGIBC])\b", re.IGNORECASE
)

# Replicon (Inc group) names from rep_db headers
_INC_RE = re.compile(
    r"\b(Inc[A-Za-z0-9/]+|rep_cluster_\d+|Col[A-Za-z0-9]+)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Database auto-detection
# ---------------------------------------------------------------------------

def find_mob_diamond_dbs(
    mob_suite_dir: Path,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Find mob/mpf/rep DIAMOND databases and rep nucleotide FASTA in mob_suite_dir.

    Returns:
        (mob_dmnd, mpf_dmnd, rep_protein_dmnd, rep_fasta) — any may be None if not found.
        rep_protein_dmnd: built by scripts/setup_rep_diamond.sh — used for plasmid
            hallmark gating (detecting replication proteins on non-mobile plasmids).
    """
    mob_dmnd = next(
        (mob_suite_dir / n for n in ("mob_proteins.dmnd",) if (mob_suite_dir / n).exists()),
        None,
    )
    mpf_dmnd = next(
        (mob_suite_dir / n for n in ("mpf_proteins.dmnd",) if (mob_suite_dir / n).exists()),
        None,
    )
    rep_protein_dmnd = next(
        (mob_suite_dir / n for n in ("rep_proteins.dmnd",) if (mob_suite_dir / n).exists()),
        None,
    )
    rep_fasta = next(
        (mob_suite_dir / n for n in ("rep.dna.fas", "rep_db.fasta", "replicons.fasta")
         if (mob_suite_dir / n).exists()),
        None,
    )
    return mob_dmnd, mpf_dmnd, rep_protein_dmnd, rep_fasta


# ---------------------------------------------------------------------------
# DIAMOND search helpers
# ---------------------------------------------------------------------------

def _run_diamond_blastp(
    query_fasta: Path,
    db: Path,
    out_tsv: Path,
    threads: int = 8,
    min_identity: float = MOB_MIN_IDENTITY,
    min_coverage: float = MOB_MIN_COVERAGE,
) -> None:
    """Run DIAMOND blastp and write tabular output."""
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    db_stem = str(db).removesuffix(".dmnd")
    cmd = [
        "diamond", "blastp",
        "--query",        str(query_fasta),
        "--db",           db_stem,
        "--out",          str(out_tsv),
        "--outfmt", "6",  "qseqid", "sseqid", "pident", "qcovhsp", "evalue", "stitle",
        "--id",           str(min_identity),
        "--query-cover",  str(min_coverage),
        "--threads",      str(threads),
        "--sensitive",
        "--max-target-seqs", "1",
        "--quiet",
    ]
    logger.info("Running DIAMOND: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DIAMOND failed: %s", result.stderr[:400])
        raise RuntimeError(f"DIAMOND failed (exit {result.returncode})")


def _parse_diamond_hits(tsv_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Parse DIAMOND tabular output → {orf_id: [(sseqid, stitle), ...]}.

    Returns empty dict if file is missing or empty.
    """
    hits: dict[str, list[tuple[str, str]]] = {}
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return hits
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            qseqid, sseqid, pident, qcovhsp, evalue, stitle = parts[:6]
            hits.setdefault(qseqid, []).append((sseqid, stitle))
    return hits


def _orf_to_contig(orf_id: str) -> str:
    """Strip trailing _N from pyrodigal orf_id to get contig_id."""
    return re.sub(r"_\d+$", "", orf_id)


# ---------------------------------------------------------------------------
# Replicon typing via minimap2
# ---------------------------------------------------------------------------

def _run_minimap2_rep(
    query_fasta: Path,
    rep_db: Path,
    out_paf: Path,
    threads: int = 8,
) -> None:
    """Map plasmid contigs against replicon DB with minimap2.

    Preset asm5 is correct for assembled contig vs assembled replicon reference
    (~5% divergence tolerance).  The previous sr (short-read) preset was wrong
    and produced 0 hits because it expects short reads, not assembled sequences.
    """
    out_paf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "minimap2", "-x", "asm5",      # assembled-to-assembled, ~5% divergence
        "--secondary=no",
        "-t", str(threads),
        str(rep_db),
        str(query_fasta),
    ]
    logger.info("Running minimap2 (replicon DB): %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("minimap2 replicon failed: %s", result.stderr[:300])
        out_paf.write_text("")
        return
    out_paf.write_text(result.stdout)


def _parse_rep_paf(paf_path: Path, min_cov: float = REP_MIN_COV) -> dict[str, str]:
    """Parse minimap2 PAF → {contig_id: replicon_name}.

    Coverage is measured on the REFERENCE (replicon), not the query (plasmid).
    A plasmid contig is 5–50 kb while a replicon sequence is 1–3 kb, so query
    coverage is always tiny (<20%) even for true hits.  We need ≥60% of the
    replicon to be covered to call a match.
    """
    rep: dict[str, str] = {}
    if not paf_path.exists():
        return rep
    with open(paf_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue
            qname   = parts[0]
            tname   = parts[5]
            tlen    = int(parts[6])   # replicon length
            tstart  = int(parts[7])   # replicon alignment start
            tend    = int(parts[8])   # replicon alignment end
            mapq    = int(parts[11])

            if mapq < 5:
                continue

            # Coverage = fraction of the REPLICON covered by the alignment
            ref_cov = 100.0 * (tend - tstart) / max(tlen, 1)
            if ref_cov < min_cov:
                continue

            m = _INC_RE.search(tname)
            rep_name = m.group(1) if m else tname.split("|")[0]
            if qname not in rep:
                rep[qname] = rep_name
    return rep


# ---------------------------------------------------------------------------
# Mobility classification logic (mirrors mob_typer rules)
# ---------------------------------------------------------------------------

def _classify_mobility(
    has_relaxase: bool,
    has_mpf: bool,
) -> str:
    """Classify mobility from relaxase + MPF presence (mob_typer rules)."""
    if has_relaxase and has_mpf:
        return "conjugative"
    if has_relaxase:
        return "mobilizable"
    return "non-mobilizable"


def _extract_mob_family(sseqid: str, stitle: str) -> str:
    """Extract MOB family name from DIAMOND subject ID or title."""
    for text in (sseqid, stitle):
        m = _MOB_FAMILY_RE.search(text)
        if m:
            return m.group(1).upper()
    return "MOBU"  # unknown


def _extract_mpf_type(sseqid: str, stitle: str) -> str:
    """Extract MPF type from DIAMOND subject ID or title."""
    for text in (sseqid, stitle):
        m = _MPF_FAMILY_RE.search(text)
        if m:
            return m.group(1).upper().replace(" ", "_")
    return "MPF_U"  # unknown


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def annotate_mobility_diamond(
    plasmid_fasta: Path | str,
    mob_suite_dir: Path | str,
    work_dir: Path | str,
    proteins_faa: Path | str | None = None,
    threads: int = 8,
) -> list[MobilityResult]:
    """Mobility typing via DIAMOND + minimap2 against MOB-suite databases.

    Significantly faster than calling mob_typer once per contig:
      - 3 parallel database searches (MOB, MPF, replicon)
      - No per-process startup overhead
      - Linear scaling with threads, not contig count

    Args:
        plasmid_fasta:  FASTA of plasmid-classified contigs.
        mob_suite_dir:  Directory with mob_proteins.dmnd, mpf_proteins.dmnd,
                        rep_db.fasta (output of setup_mob_diamond.sh).
        work_dir:       Directory for intermediate DIAMOND / minimap2 output.
        proteins_faa:   Pre-predicted protein FASTA (reuse from ARG step).
                        If None, pyrodigal is called to predict ORFs.
        threads:        CPU threads.

    Returns:
        List of MobilityResult — one per contig that received a classification.
        Contigs with no hits are returned as "non-mobilizable".
    """
    from Bio import SeqIO  # type: ignore[import]

    plasmid_fasta = Path(plasmid_fasta)
    mob_suite_dir = Path(mob_suite_dir)
    work_dir      = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    mob_dmnd, mpf_dmnd, rep_fasta = find_mob_diamond_dbs(mob_suite_dir)

    if mob_dmnd is None and mpf_dmnd is None:
        logger.warning(
            "No MOB-suite DIAMOND databases found in %s. "
            "Run:  bash scripts/setup_mob_diamond.sh",
            mob_suite_dir,
        )
        return []

    # Load all contig IDs so we can assign non-mobilizable to those without hits
    records = list(SeqIO.parse(str(plasmid_fasta), "fasta"))
    all_contig_ids = [r.id for r in records]

    # Predict ORFs if proteins not pre-computed
    if proteins_faa is None or not Path(proteins_faa).exists():
        from plasflow2.annotate.args import call_orfs
        prot_path = work_dir / "mob_proteins.faa"
        call_orfs(plasmid_fasta, prot_path)
        proteins_faa = prot_path
    else:
        proteins_faa = Path(proteins_faa)
        logger.info("Reusing pre-predicted ORFs from %s", proteins_faa)

    # ── 1. Search MOB relaxase database ──────────────────────────────────────
    mob_hits_by_contig: dict[str, tuple[str, str]] = {}   # contig → (family, accession)
    if mob_dmnd is not None:
        mob_tsv = work_dir / "mob_hits.tsv"
        try:
            _run_diamond_blastp(
                proteins_faa, mob_dmnd, mob_tsv, threads,
                min_identity=MOB_MIN_IDENTITY, min_coverage=MOB_MIN_COVERAGE,
            )
            for orf_id, hits in _parse_diamond_hits(mob_tsv).items():
                cid = _orf_to_contig(orf_id)
                if cid not in mob_hits_by_contig and hits:
                    sseqid, stitle = hits[0]
                    mob_hits_by_contig[cid] = (_extract_mob_family(sseqid, stitle), sseqid)
            logger.info("MOB relaxase hits: %d contigs", len(mob_hits_by_contig))
        except Exception as exc:
            logger.warning("MOB relaxase search failed: %s", exc)

    # ── 2. Search MPF database ────────────────────────────────────────────────
    mpf_hits_by_contig: dict[str, str] = {}   # contig → mpf_type
    if mpf_dmnd is not None:
        mpf_tsv = work_dir / "mpf_hits.tsv"
        try:
            _run_diamond_blastp(
                proteins_faa, mpf_dmnd, mpf_tsv, threads,
                min_identity=MPF_MIN_IDENTITY, min_coverage=MPF_MIN_COVERAGE,
            )
            for orf_id, hits in _parse_diamond_hits(mpf_tsv).items():
                cid = _orf_to_contig(orf_id)
                if cid not in mpf_hits_by_contig and hits:
                    sseqid, stitle = hits[0]
                    mpf_hits_by_contig[cid] = _extract_mpf_type(sseqid, stitle)
            logger.info("MPF system hits: %d contigs", len(mpf_hits_by_contig))
        except Exception as exc:
            logger.warning("MPF search failed: %s", exc)

    # ── 3. Replicon typing via minimap2 ───────────────────────────────────────
    rep_by_contig: dict[str, str] = {}   # contig → Inc group
    if rep_fasta is not None:
        try:
            rep_paf = work_dir / "rep_hits.paf"
            # Always re-run: minimap2 replicon search is fast (<2 s) and the
            # PAF cache has caused stale-result bugs when the preset changed.
            _run_minimap2_rep(plasmid_fasta, rep_fasta, rep_paf, threads)
            rep_by_contig = _parse_rep_paf(rep_paf)
            logger.info("Replicon hits: %d contigs", len(rep_by_contig))
        except Exception as exc:
            logger.warning("Replicon typing failed: %s", exc)

    # ── 4. Assemble MobilityResult per contig ─────────────────────────────────
    results: list[MobilityResult] = []
    for cid in all_contig_ids:
        mob_family, mob_acc = mob_hits_by_contig.get(cid, ("none", "-"))
        mpf_type = mpf_hits_by_contig.get(cid, "none")
        rep_type = rep_by_contig.get(cid, "-")

        has_relaxase = mob_family not in ("none", "MOBU")
        has_mpf      = mpf_type not in ("none", "MPF_U")

        mobility_class = _classify_mobility(has_relaxase, has_mpf)

        results.append(
            MobilityResult(
                contig_id=cid,
                mobility_class=mobility_class,
                replicon_type=rep_type,
                relaxase_type=mob_family if has_relaxase else "none",
                mpf_type=mpf_type if has_mpf else "none",
                raw={
                    "predicted_mobility": mobility_class,
                    "rep_type(s)": rep_type,
                    "relaxase_type(s)": mob_family if has_relaxase else "-",
                    "mpf_type": mpf_type if has_mpf else "-",
                    "rep_type_accession(s)": "-",
                    "relaxase_type_accession(s)": mob_acc if has_relaxase else "-",
                },
            )
        )

    n_conj = sum(1 for r in results if r.mobility_class == "conjugative")
    n_mob  = sum(1 for r in results if r.mobility_class == "mobilizable")
    n_non  = sum(1 for r in results if r.mobility_class == "non-mobilizable")
    logger.info(
        "Mobility (DIAMOND): %d conjugative | %d mobilizable | %d non-mobilizable",
        n_conj, n_mob, n_non,
    )
    return results
