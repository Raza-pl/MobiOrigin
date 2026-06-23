"""MOB-suite integration for plasmid mobility and replicon typing.

Week 3 — Day 17 implementation.

Pipeline:
    plasmid FASTA → run_mob_typer() → TSV → parse_mob_results() → [MobilityResult]

MOB-suite installation (one-time, conda recommended):
    conda install -c bioconda mob_suite

Note: MOB-suite may conflict with Apple Silicon conda envs.
Prefer running on a Linux/x86 machine or via Docker.
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# mob_typer column names (mob_suite >= 3.0)
# ---------------------------------------------------------------------------

# Canonical column names in mobtyper_results.txt (mob_suite 3.x)
_COL_SAMPLE_ID = "sample_id"
_COL_MOBILITY = "predicted_mobility"
_COL_REP_TYPE = "rep_type(s)"
_COL_RELAXASE = "relaxase_type(s)"
_COL_MPF = "mpf_type"

# Valid mobility classes returned by mob_typer
MOBILITY_CLASSES = frozenset({"conjugative", "mobilizable", "non-mobilizable"})


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class MobilityResult:
    """MOB-suite mob_typer output for one plasmid contig."""

    contig_id: str
    mobility_class: str  # conjugative | mobilizable | non-mobilizable
    replicon_type: str  # e.g., IncF, IncP, IncQ, unknown
    relaxase_type: str  # MOB family or "none"
    mpf_type: str  # Mating pair formation system or "none"
    # Raw row dict preserved for downstream callers that need other fields
    raw: dict[str, str] = field(default_factory=dict, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Running mob_typer
# ---------------------------------------------------------------------------


def _mob_typer_one(cid: str, seq: str, contig_tmp_dir: Path) -> tuple[str, list[str]]:
    """Run mob_typer on a single contig.  Returns (contig_id, result_lines).

    Caches result on disk — if the output file already exists and has content,
    returns the cached result immediately without calling mob_typer again.
    This makes interrupted runs restartable with zero re-work.

    Uses 1 thread per call — parallelism comes from the caller's thread pool.
    """
    contig_fasta = contig_tmp_dir / f"{cid}.fasta"
    contig_out = contig_tmp_dir / f"{cid}_results.txt"

    # Cache hit: result already computed (from a previous or interrupted run)
    if contig_out.exists() and contig_out.stat().st_size > 0:
        with open(contig_out) as fh:
            return cid, fh.readlines()

    with open(contig_fasta, "w") as fh:
        fh.write(f">{cid}\n{seq}\n")

    cmd = [
        "mob_typer",
        "--infile",
        str(contig_fasta),
        "--out_file",
        str(contig_out),
        "--num_threads",
        "1",  # 1 thread per call; pool provides parallelism
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    if not contig_out.exists():
        return cid, []

    with open(contig_out) as fh:
        return cid, fh.readlines()


def run_mob_typer(
    plasmid_fasta: Path | str,
    out_dir: Path | str,
    threads: int = 4,
) -> Path:
    """Run MOB-suite mob_typer on classified plasmid contigs.

    Strategy: split FASTA into one file per contig, run mob_typer in parallel
    via a ThreadPoolExecutor (threads workers, 1 mob_typer thread each), then
    merge results into a single TSV.

    Key features:
    - **Parallelism**: threads concurrent mob_typer processes → N/threads wall time
      (2559 contigs / 16 workers ≈ 8–12 min vs 4+ hours sequential)
    - **Caching**: per-contig result files are kept on disk. If the pipeline is
      interrupted and restarted, already-computed contigs are skipped instantly.
      Restarting after an interruption takes seconds for cached contigs.
    - **Progress logging**: every 200 contigs (or completion)

    Args:
        plasmid_fasta: FASTA of predicted plasmid sequences.
        out_dir:       Directory for mob_typer output files.
        threads:       Number of parallel mob_typer worker processes.

    Returns:
        Path to combined mob_typer results TSV (mobtyper_results.txt).
    """
    from Bio import SeqIO  # type: ignore[import]

    plasmid_fasta = Path(plasmid_fasta)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_tsv = out_dir / "mobtyper_results.txt"
    contig_tmp_dir = out_dir / "per_contig"
    contig_tmp_dir.mkdir(exist_ok=True)

    records = list(SeqIO.parse(str(plasmid_fasta), "fasta"))
    if not records:
        combined_tsv.write_text("")
        return combined_tsv

    # Count already-cached results so we can report how many need to run
    n_cached = sum(
        1
        for r in records
        if (contig_tmp_dir / f"{r.id}_results.txt").exists()
        and (contig_tmp_dir / f"{r.id}_results.txt").stat().st_size > 0
    )
    n_todo = len(records) - n_cached
    n_workers = min(threads, max(n_todo, 1))

    logger.info(
        "mob_typer: %d contigs total | %d cached (instant) | %d to run " "| %d parallel workers",
        len(records),
        n_cached,
        n_todo,
        n_workers,
    )

    results: dict[str, list[str]] = {}
    header_lines: list[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_mob_typer_one, r.id, str(r.seq), contig_tmp_dir): r.id for r in records
        }
        for future in as_completed(futures):
            cid, lines = future.result()
            results[cid] = lines
            done += 1
            if not header_lines and lines:
                header_lines = [lines[0]]
            if done % 200 == 0 or done == len(records):
                logger.info("mob_typer progress: %d / %d done", done, len(records))

    # Write combined TSV in original contig order
    n_success = 0
    with open(combined_tsv, "w") as fh:
        if header_lines:
            fh.write(header_lines[0])
        for record in records:
            lines = results.get(record.id, [])
            if len(lines) < 2:
                continue
            cols = lines[1].rstrip("\n").split("\t")
            if cols:
                cols[0] = record.id  # replace filename-derived sample_id
            fh.write("\t".join(cols) + "\n")
            n_success += 1

    if not header_lines:
        combined_tsv.write_text("")

    logger.info("mob_typer: %d / %d contigs typed", n_success, len(records))
    return combined_tsv


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_mob_results(tsv_path: Path | str) -> list[MobilityResult]:
    """Parse mob_typer TSV output into MobilityResult objects.

    Handles mob_suite >= 3.0 column names. Unknown / missing columns are
    filled with sensible defaults so the parser does not crash on minor
    version differences.

    Args:
        tsv_path: Path to mobtyper_results.txt produced by run_mob_typer().

    Returns:
        List of MobilityResult, one per data row.  Returns an empty list
        for header-only files (no contigs classified).
    """
    tsv_path = Path(tsv_path)
    results: list[MobilityResult] = []

    with open(tsv_path) as fh:
        raw_header = fh.readline()
        if not raw_header:
            logger.info("mob_typer results file is empty: %s", tsv_path)
            return results

        header = [h.strip() for h in raw_header.split("\t")]

        for line in fh:
            line = line.strip()
            if not line:
                continue
            values = line.split("\t")
            row = dict(zip(header, values, strict=False))

            # Normalise mobility class — default to non-mobilizable if absent
            mob_class = row.get(_COL_MOBILITY, "non-mobilizable").strip().lower()
            if mob_class not in MOBILITY_CLASSES:
                logger.debug(
                    "Unrecognised mobility class %r — defaulting to non-mobilizable",
                    mob_class,
                )
                mob_class = "non-mobilizable"

            # Normalise replicon: strip whitespace; "-" or empty -> "unknown"
            rep_type = row.get(_COL_REP_TYPE, "-").strip()
            if rep_type in ("-", ""):
                rep_type = "unknown"

            # Normalise relaxase / MPF
            relaxase = row.get(_COL_RELAXASE, "-").strip()
            if relaxase in ("-", ""):
                relaxase = "none"

            mpf = row.get(_COL_MPF, "-").strip()
            if mpf in ("-", ""):
                mpf = "none"

            results.append(
                MobilityResult(
                    contig_id=row.get(_COL_SAMPLE_ID, "unknown").strip(),
                    mobility_class=mob_class,
                    replicon_type=rep_type,
                    relaxase_type=relaxase,
                    mpf_type=mpf,
                    raw=row,
                )
            )

    logger.info("Parsed %d mobility results from %s", len(results), tsv_path)
    return results


# ---------------------------------------------------------------------------
# Convenience: full mobility annotation for one FASTA
# ---------------------------------------------------------------------------


def annotate_mobility(
    plasmid_fasta: Path | str,
    work_dir: Path | str,
    threads: int = 4,
) -> list[MobilityResult]:
    """End-to-end mobility annotation: mob_typer -> parsed results.

    Args:
        plasmid_fasta: FASTA of predicted plasmid sequences.
        work_dir: Directory for mob_typer intermediate files.
        threads: CPU threads for mob_typer.

    Returns:
        List of MobilityResult across all contigs.  Empty list if
        mob_typer produces no results (e.g., all contigs too short).
    """
    results_tsv = run_mob_typer(plasmid_fasta, work_dir, threads=threads)
    return parse_mob_results(results_tsv)


# ---------------------------------------------------------------------------
# Index helper
# ---------------------------------------------------------------------------


def index_by_contig(results: list[MobilityResult]) -> dict[str, MobilityResult]:
    """Return a dict mapping contig_id -> MobilityResult for fast lookup."""
    return {r.contig_id: r for r in results}
