"""Reconstruct per-class training FASTAs from marker_work protein headers.

The build_marker_dataset.py script saves protein FASTAs but not the source DNA.
This script reverse-engineers the DNA sequences by:
  1. Parsing contig IDs from *_proteins.faa → {accession}_w{size}_s{start}
  2. Scanning source databases to find matching accessions
  3. Extracting the exact windows and writing per-class FASTAs

Output (per class):
  data/marker_work/plasmid_training.fna
  data/marker_work/chromosome_training.fna
  data/marker_work/phage_training.fna

Usage
-----
  python scripts/build_training_fasta.py

Then run geNomad annotate on each:
  for cls in plasmid chromosome phage; do
    genomad annotate \\
        data/marker_work/${cls}_training.fna \\
        data/marker_work/${cls}_genomad_ann/ \\
        data/databases/genomad_db/ \\
        --threads 16 --splits 4 --cleanup
  done
"""

from __future__ import annotations

import gzip
import logging
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# ── Source databases to scan (in search order) ─────────────────────────────
PLASMID_SOURCES = [
    ROOT / "data/databases/plasmids/plsdb.fasta",
    ROOT / "data/databases/plasmids/COMPASS.fna",
]
CHROMOSOME_SOURCES = [
    ROOT / "data/databases/chromosomes.fna",
    ROOT / "data/databases/extra_chromosomes",
    ROOT / "data/gtdb_genomes/bacteria",
]
PHAGE_SOURCES = [
    ROOT / "data/databases/inphared/inphared_phages.fa.gz",
]

MARKER_WORK = ROOT / "data/marker_work"


# ── ID parsing ──────────────────────────────────────────────────────────────

_ID_RE = re.compile(r"^(?:RefSeq_)?(.+?)_w(\d+)_s(\d+)$")


def parse_contig_id(cid: str) -> tuple[str, int, int] | None:
    """Parse '{accession}_w{size}_s{start}' → (accession, window_size, start).

    Returns None if the ID doesn't match the expected pattern.
    """
    m = _ID_RE.match(cid)
    if not m:
        return None
    accession = m.group(1)
    window_size = int(m.group(2))
    start = int(m.group(3))
    return accession, window_size, start


def extract_contig_ids(proteins_faa: Path) -> set[str]:
    """Extract unique contig IDs from a proteins.faa file."""
    cids = set()
    with open(proteins_faa) as fh:
        for line in fh:
            if line.startswith(">"):
                # Strip ORF index suffix: contig_id_N → contig_id
                orf_id = line[1:].strip()
                cid = re.sub(r"_\d+$", "", orf_id)
                cids.add(cid)
    return cids


# ── FASTA scanning ──────────────────────────────────────────────────────────

def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _iter_fasta(path: Path):
    """Yield (seq_id, sequence) from a FASTA file."""
    with _open(path) as fh:
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


def _fasta_files_in(path: Path) -> list[Path]:
    """Recursively find FASTA files under a directory."""
    exts = {".fna", ".fa", ".fasta", ".fna.gz", ".fa.gz", ".fasta.gz"}
    if path.is_file() and path.suffix in exts or str(path).endswith(".fa.gz"):
        return [path]
    if path.is_dir():
        found = []
        for ext in exts:
            found.extend(sorted(path.rglob(f"*{ext}")))
        return found
    return []


def _build_gtdb_index(gtdb_dir: Path, needed_accessions: set[str]) -> dict[str, Path]:
    """Fast two-pass index for GTDB: scan headers only, return {accession: file_path}."""
    log.info("Building GTDB index for %d accessions across %s …", len(needed_accessions), gtdb_dir)
    index: dict[str, Path] = {}
    fna_files = sorted(gtdb_dir.glob("*.fna.gz")) + sorted(gtdb_dir.glob("*.fna"))
    log.info("  %d GTDB files to scan (headers only)", len(fna_files))
    for i, fna in enumerate(fna_files):
        if i % 1000 == 0:
            log.info("  … indexed %d / %d files", i, len(fna_files))
        try:
            with _open(fna) as fh:
                for line in fh:
                    if not line.startswith(">"):
                        continue  # skip sequences - only read headers
                    seq_id = line[1:].split()[0].rstrip()
                    for acc_key in [seq_id, seq_id.rsplit(".", 1)[0]]:
                        if acc_key in needed_accessions and acc_key not in index:
                            index[acc_key] = fna
                            break
        except Exception as exc:
            log.warning("  Error indexing %s: %s", fna.name, exc)
    log.info("GTDB index complete: %d / %d accessions found", len(index), len(needed_accessions))
    return index


def build_windows(
    contig_ids: set[str],
    source_paths: list[Path],
    out_fasta: Path,
    label: str,
) -> int:
    """Scan source databases, extract matching windows, write to out_fasta.

    For directories with many files (e.g. GTDB), uses a two-pass approach:
      Pass 1 – read only headers to build accession→file index.
      Pass 2 – open only the relevant files to extract sequences.

    Returns number of sequences written.
    """
    # Parse all requested windows: accession → list of (cid, window_size, start)
    accession_map: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    skipped_parse = 0
    for cid in contig_ids:
        parsed = parse_contig_id(cid)
        if parsed is None:
            skipped_parse += 1
            continue
        accession, w, s = parsed
        accession_map[accession].append((cid, w, s))

    log.info("[%s] %d unique contigs → %d unique accessions  (%d unparseable IDs)",
             label, len(contig_ids), len(accession_map), skipped_parse)

    found_accessions: set[str] = set()
    written = 0
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    # Separate flat files from directories (directories get the indexed approach)
    flat_files: list[Path] = []
    dir_index: dict[str, Path] = {}  # accession → specific file path

    for src in source_paths:
        if src.is_file():
            flat_files.append(src)
        elif src.is_dir():
            # Use two-pass index for large directories
            idx = _build_gtdb_index(src, set(accession_map.keys()))
            dir_index.update(idx)
        else:
            log.warning("[%s] Source not found: %s", label, src)

    # Build file→accessions mapping for indexed directory files
    file_to_accs: dict[Path, list[str]] = defaultdict(list)
    for acc, fpath in dir_index.items():
        if acc in accession_map:
            file_to_accs[fpath].append(acc)

    log.info("[%s] Scanning %d flat files + %d indexed directory files …",
             label, len(flat_files), len(file_to_accs))

    with open(out_fasta, "w") as out_fh:
        # --- Flat file scan (sequential, reads everything) ---
        for src_path in flat_files:
            log.info("[%s]   → %s", label, src_path.name)
            try:
                for seq_id, seq in _iter_fasta(src_path):
                    for acc_key in [seq_id, seq_id.rsplit(".", 1)[0]]:
                        if acc_key not in accession_map:
                            continue
                        found_accessions.add(acc_key)
                        seq_upper = seq.upper()
                        for cid, w, start in accession_map[acc_key]:
                            fragment = seq_upper[start:start + w]
                            if len(fragment) < w // 2:
                                continue
                            out_fh.write(f">{cid}\n{fragment}\n")
                            written += 1
                        break
            except Exception as exc:
                log.warning("[%s] Error reading %s: %s", label, src_path, exc)

        # --- Indexed directory scan (opens only relevant files) ---
        # Exclude accessions already written during flat file scan to avoid duplicates
        for fpath, accs in file_to_accs.items():
            target = set(accs) - found_accessions  # skip already-found ones
            try:
                for seq_id, seq in _iter_fasta(fpath):
                    for acc_key in [seq_id, seq_id.rsplit(".", 1)[0]]:
                        if acc_key not in target:
                            continue
                        found_accessions.add(acc_key)
                        seq_upper = seq.upper()
                        for cid, w, start in accession_map[acc_key]:
                            fragment = seq_upper[start:start + w]
                            if len(fragment) < w // 2:
                                continue
                            out_fh.write(f">{cid}\n{fragment}\n")
                            written += 1
                        break
            except Exception as exc:
                log.warning("[%s] Error reading %s: %s", label, fpath.name, exc)

    missing = len(accession_map) - len(found_accessions)
    log.info("[%s] Written: %d sequences  |  Missing accessions: %d / %d",
             label, written, missing, len(accession_map))
    if missing > 0 and missing < 20:
        log.info("[%s] Missing: %s", label, sorted(set(accession_map.keys()) - found_accessions)[:10])

    return written


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    classes = {
        "plasmid":    (MARKER_WORK / "plasmid_proteins.faa",    PLASMID_SOURCES),
        "chromosome": (MARKER_WORK / "chromosome_proteins.faa", CHROMOSOME_SOURCES),
        "phage":      (MARKER_WORK / "phage_proteins.faa",      PHAGE_SOURCES),
    }

    for label, (proteins_faa, sources) in classes.items():
        if not proteins_faa.exists():
            log.warning("Missing: %s — skipping %s", proteins_faa, label)
            continue

        out_fasta = MARKER_WORK / f"{label}_training.fna"
        if out_fasta.exists():
            existing = sum(1 for l in open(out_fasta) if l.startswith(">"))
            log.info("[%s] %s already exists (%d seqs) — skipping. Delete to rebuild.",
                     label, out_fasta.name, existing)
            continue

        log.info("=== Building %s training FASTA ===", label)
        cids = extract_contig_ids(proteins_faa)
        log.info("[%s] %d unique contig IDs in proteins.faa", label, len(cids))

        n = build_windows(cids, sources, out_fasta, label)
        log.info("[%s] Done: %d sequences → %s\n", label, n, out_fasta)

    log.info("All done. Run geNomad annotate on each _training.fna:")
    log.info("  for cls in plasmid chromosome phage; do")
    log.info("    genomad annotate \\")
    log.info("        data/marker_work/${cls}_training.fna \\")
    log.info("        data/marker_work/${cls}_genomad_ann/ \\")
    log.info("        data/databases/genomad_db/ \\")
    log.info("        --threads 16 --splits 4 --cleanup")
    log.info("  done")


if __name__ == "__main__":
    main()
