"""MOB-only protein-marker features for MobiOrigin inference."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mobiorigin.fasta import IUPAC_DNA, FastaRecord
from mobiorigin.runtime import validate_threads

FEATURE_NAMES = (
    "coding_density",
    "orfs_per_kb",
    "log1p_median_orf_aa_length",
    "strand_switch_rate",
    "forward_orf_fraction",
    "rep_has_hit",
    "rep_hit_orf_fraction",
    "rep_hits_per_kb",
    "rep_max_bitscore_per_query_aa",
    "mob_has_hit",
    "mob_hit_orf_fraction",
    "mob_hits_per_kb",
    "mob_max_bitscore_per_query_aa",
    "mpf_has_hit",
    "mpf_hit_orf_fraction",
    "mpf_hits_per_kb",
    "mpf_max_bitscore_per_query_aa",
)
DATABASE_SHA256 = {
    "rep": "a70b79237026f1aece9ef70d59fbc37d6f1607d2a0ae53555a2c1dd55c54fbc0",
    "mob": "176f3e8be3aab01ddae74f73be8f19ef4f5e419e59bc0299bff54571351aad10",
    "mpf": "da7a65ac9fdb8edc80b5fdebf5b0878d97cbeebcf2ede0a7332c2af605192e37",
}
AMBIGUITY_TO_N = str.maketrans({base: "N" for base in IUPAC_DNA - set("ACGT")})


@dataclass(frozen=True)
class OrfSummary:
    count: int
    covered_bp: int
    amino_acid_lengths: tuple[int, ...]
    strands: tuple[int, ...]


@dataclass(frozen=True)
class Hit:
    query_id: str
    subject_id: str
    identity: float
    coverage: float
    evalue: float
    bitscore: float
    query_length: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval_union_length(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted((min(start, end), max(start, end)) for start, end in intervals)
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end + 1:
            end = max(end, next_end)
        else:
            total += end - start + 1
            start, end = next_start, next_end
    return total + end - start + 1


def orf_values(summary: OrfSummary, length_bp: int) -> list[float]:
    length_kb = length_bp / 1000.0
    switches = sum(left != right for left, right in zip(summary.strands, summary.strands[1:]))
    median = (
        float(np.median(np.asarray(summary.amino_acid_lengths, dtype=np.float64)))
        if summary.amino_acid_lengths
        else 0.0
    )
    forward = sum(strand > 0 for strand in summary.strands)
    return [
        min(summary.covered_bp / max(length_bp, 1), 1.0),
        summary.count / max(length_kb, 1e-9),
        math.log1p(median),
        switches / max(summary.count - 1, 1),
        forward / max(summary.count, 1),
    ]


def marker_family_values(
    hits: Mapping[str, Hit],
    query_to_contig: Mapping[str, str],
    identifier: str,
    orf_count: int,
    length_kb: float,
) -> list[float]:
    selected = [hit for query, hit in hits.items() if query_to_contig.get(query) == identifier]
    count = len(selected)
    maximum = max((hit.bitscore / max(hit.query_length, 1) for hit in selected), default=0.0)
    return [
        float(count > 0),
        count / max(orf_count, 1),
        count / max(length_kb, 1e-9),
        maximum,
    ]


def load_database_manifest(database_dir: Path) -> dict[str, Path]:
    """Resolve and identity-check the user-retrieved MOB-suite databases."""
    manifest_path = database_dir / "mobiorigin_mob_suite_database_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "mobiorigin-mob-suite-database-manifest-v1":
        raise ValueError("MOB-suite database manifest schema is unsupported")
    resolved: dict[str, Path] = {}
    for family, expected_hash in DATABASE_SHA256.items():
        item = manifest.get("databases", {}).get(family, {})
        path = database_dir / str(item.get("path", ""))
        if item.get("sha256") != expected_hash or not path.is_file():
            raise ValueError(f"MOB-suite {family} database manifest identity changed")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"MOB-suite {family} database payload identity changed")
        resolved[family] = path
    return resolved


def predict_orfs(
    records: Sequence[FastaRecord], proteins_path: Path
) -> tuple[dict[str, OrfSummary], dict[str, str]]:
    import pyrodigal  # type: ignore[import]

    bins = pyrodigal.MetagenomicBins(
        item for item in pyrodigal.METAGENOMIC_BINS if item.training_info.translation_table == 11
    )
    if not bins:
        raise RuntimeError("Pyrodigal has no metagenomic translation-table-11 bins")
    finder = pyrodigal.GeneFinder(meta=True, metagenomic_bins=bins, mask=True, min_mask=1)
    summaries: dict[str, OrfSummary] = {}
    query_to_contig: dict[str, str] = {}
    with proteins_path.open("w", encoding="ascii") as handle:
        for record in records:
            sequence = record.sequence.translate(AMBIGUITY_TO_N)
            genes = list(finder.find_genes(sequence.encode("ascii")))
            intervals: list[tuple[int, int]] = []
            lengths: list[int] = []
            strands: list[int] = []
            rank = 0
            for gene in genes:
                protein = gene.translate().rstrip("*")
                if len(protein) < 30:
                    continue
                rank += 1
                query = f"{record.identifier}__orf_{rank}"
                query_to_contig[query] = record.identifier
                handle.write(f">{query}\n{protein}\n")
                intervals.append((int(gene.begin), int(gene.end)))
                lengths.append(len(protein))
                strands.append(int(gene.strand))
            summaries[record.identifier] = OrfSummary(
                rank, interval_union_length(intervals), tuple(lengths), tuple(strands)
            )
    return summaries, query_to_contig


def run_diamond(
    diamond: Path,
    proteins: Path,
    database: Path,
    output: Path,
    threads: int,
) -> None:
    completed = subprocess.run(
        [
            str(diamond),
            "blastp",
            "--query",
            str(proteins),
            "--db",
            str(database).removesuffix(".dmnd"),
            "--out",
            str(output),
            "--outfmt",
            "6",
            "qseqid",
            "sseqid",
            "pident",
            "qcovhsp",
            "evalue",
            "bitscore",
            "qlen",
            "--id",
            "50",
            "--query-cover",
            "70",
            "--evalue",
            "1e-5",
            "--threads",
            str(threads),
            "--sensitive",
            "--max-target-seqs",
            "5",
            "--quiet",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"DIAMOND failed for {database.name}: {completed.stderr.strip()}")
    if not output.exists():
        output.write_text("", encoding="utf-8")


def parse_hits(path: Path) -> dict[str, Hit]:
    best: dict[str, Hit] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 7:
                raise ValueError(f"Malformed DIAMOND row in {path.name}")
            hit = Hit(
                query_id=parts[0],
                subject_id=parts[1],
                identity=float(parts[2]),
                coverage=float(parts[3]),
                evalue=float(parts[4]),
                bitscore=float(parts[5]),
                query_length=int(parts[6]),
            )
            previous = best.get(hit.query_id)
            ranking = (-hit.bitscore, -hit.identity, -hit.coverage, hit.subject_id)
            if previous is None or ranking < (
                -previous.bitscore,
                -previous.identity,
                -previous.coverage,
                previous.subject_id,
            ):
                best[hit.query_id] = hit
    return best


def extract_marker_features(
    records: Sequence[FastaRecord],
    *,
    databases: Mapping[str, Path],
    diamond: Path,
    threads: int,
    work_dir: Path,
) -> NDArray[np.float32]:
    """Extract the frozen 17 MOB-only marker features."""
    validate_threads(threads)
    work_dir.mkdir(parents=True, exist_ok=True)
    proteins = work_dir / "proteins.faa"
    summaries, query_to_contig = predict_orfs(records, proteins)
    hit_sets: dict[str, dict[str, Hit]] = {}
    for family in ("rep", "mob", "mpf"):
        output = work_dir / f"{family}_hits.tsv"
        if query_to_contig:
            run_diamond(diamond, proteins, databases[family], output, threads)
        else:
            output.write_text("", encoding="utf-8")
        hit_sets[family] = parse_hits(output)
    matrix = np.zeros((len(records), len(FEATURE_NAMES)), dtype=np.float32)
    for index, record in enumerate(records):
        summary = summaries[record.identifier]
        values = orf_values(summary, len(record.sequence))
        for family in ("rep", "mob", "mpf"):
            values.extend(
                marker_family_values(
                    hit_sets[family],
                    query_to_contig,
                    record.identifier,
                    summary.count,
                    len(record.sequence) / 1000.0,
                )
            )
        if len(values) != len(FEATURE_NAMES) or not np.isfinite(values).all():
            raise ValueError(f"Invalid marker vector for {record.identifier}")
        matrix[index] = np.asarray(values, dtype=np.float32)
    return matrix
