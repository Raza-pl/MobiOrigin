"""Deterministically rebuild MobiOrigin's frozen MOB-suite DIAMOND databases."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from collections.abc import Iterator
from functools import lru_cache
from itertools import product
from pathlib import Path

RAW_SHA256 = {
    "rep.dna.fas": "fdc10d866d8fdc3c75db0f945c8b52797392abb7b7bf389c5de36a2429500eb0",
    "mob.proteins.faa": "e5db0e07f9e94f7252f8adceaba4355851331b1ea4e60f51ce90af97490c47a9",
    "mpf.proteins.faa": "fc6a9f78465271826120659e46c6876aa1b0f17baa0b806a525104b2e41f12fa",
}
REP_PROTEIN_SHA256 = "3157e10cdf20be3075b076e51d83cca685a1c135804aef3c3417acd26d2b1eeb"
DATABASE_SHA256 = {
    "rep_proteins.dmnd": "a70b79237026f1aece9ef70d59fbc37d6f1607d2a0ae53555a2c1dd55c54fbc0",
    "mob_proteins.dmnd": "176f3e8be3aab01ddae74f73be8f19ef4f5e419e59bc0299bff54571351aad10",
    "mpf_proteins.dmnd": "da7a65ac9fdb8edc80b5fdebf5b0878d97cbeebcf2ede0a7332c2af605192e37",
}
DIAMOND_VERSION = "2.0.15"

CODON_TABLE = {
    codon: amino_acid
    for amino_acid, codons in {
        "F": ("TTT", "TTC"),
        "L": ("TTA", "TTG", "CTT", "CTC", "CTA", "CTG"),
        "I": ("ATT", "ATC", "ATA"),
        "M": ("ATG",),
        "V": ("GTT", "GTC", "GTA", "GTG"),
        "S": ("TCT", "TCC", "TCA", "TCG", "AGT", "AGC"),
        "P": ("CCT", "CCC", "CCA", "CCG"),
        "T": ("ACT", "ACC", "ACA", "ACG"),
        "A": ("GCT", "GCC", "GCA", "GCG"),
        "Y": ("TAT", "TAC"),
        "*": ("TAA", "TAG", "TGA"),
        "H": ("CAT", "CAC"),
        "Q": ("CAA", "CAG"),
        "N": ("AAT", "AAC"),
        "K": ("AAA", "AAG"),
        "D": ("GAT", "GAC"),
        "E": ("GAA", "GAG"),
        "C": ("TGT", "TGC"),
        "W": ("TGG",),
        "R": ("CGT", "CGC", "CGA", "CGG", "AGA", "AGG"),
        "G": ("GGT", "GGC", "GGA", "GGG"),
    }.items()
    for codon in codons
}
COMPLEMENT = str.maketrans("ACGTRYKMSWBDHVN", "TGCAYRMKSWVHDBN")
IUPAC_BASES = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "K": "GT",
    "M": "AC",
    "S": "CG",
    "W": "AT",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    sequence: list[str] = []
    with path.open("r", encoding="ascii") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence).upper()
                header = line[1:]
                sequence = []
            elif header is None:
                raise ValueError(f"FASTA sequence appears before a header in {path}")
            else:
                sequence.append(line)
    if header is not None:
        yield header, "".join(sequence).upper()


@lru_cache(maxsize=3375)
def translate_codon(codon: str) -> str:
    direct = CODON_TABLE.get(codon)
    if direct is not None:
        return direct
    choices = [IUPAC_BASES.get(base, "ACGT") for base in codon]
    amino_acids = {CODON_TABLE["".join(item)] for item in product(*choices)}
    return next(iter(amino_acids)) if len(amino_acids) == 1 else "X"


def translate(sequence: str) -> str:
    usable = len(sequence) - (len(sequence) % 3)
    return "".join(translate_codon(sequence[offset : offset + 3]) for offset in range(0, usable, 3))


def build_rep_proteins(source: Path, destination: Path) -> None:
    """Apply the frozen six-frame, stop-delimited translation rule."""
    with destination.open("x", encoding="ascii", newline="\n") as handle:
        for header, sequence in read_fasta(source):
            reverse = sequence.translate(COMPLEMENT)[::-1]
            for strand, oriented in ((1, sequence), (-1, reverse)):
                for frame in range(3):
                    proteins = translate(oriented[frame:]).split("*")
                    for ordinal, protein in enumerate(proteins):
                        if len(protein) < 30:
                            continue
                        handle.write(f">{header}_s{strand}_f{frame}_o{ordinal}\n{protein}\n")


def _require_identity(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed}; expected {expected}")


def _diamond_version(diamond: Path) -> str:
    completed = subprocess.run(
        [str(diamond), "version"], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(f"DIAMOND version check failed: {completed.stderr.strip()}")
    output = (completed.stdout or completed.stderr).strip()
    if f"diamond version {DIAMOND_VERSION}" not in output:
        raise RuntimeError(
            f"Frozen database reconstruction requires DIAMOND {DIAMOND_VERSION}; got {output}"
        )
    return output


def _makedb(diamond: Path, source: Path, destination: Path) -> None:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    completed = subprocess.run(
        [
            str(diamond),
            "makedb",
            "--in",
            str(source),
            "--db",
            str(destination.with_suffix("")),
            "--threads",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(f"DIAMOND makedb failed for {source.name}: {completed.stderr.strip()}")


def build_marker_databases(raw_dir: Path, output_dir: Path, diamond: Path) -> dict[str, str]:
    """Build and fail-closed verify all three frozen database files."""
    if output_dir.exists():
        raise FileExistsError(f"Build output already exists: {output_dir}")
    executable = Path(shutil.which(str(diamond)) or str(diamond)).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"DIAMOND executable not found: {diamond}")
    version = _diamond_version(executable)
    for filename, expected in RAW_SHA256.items():
        _require_identity(raw_dir / filename, expected, f"MOB-suite {filename}")

    output_dir.mkdir(parents=True)
    try:
        rep_proteins = output_dir / "rep_proteins.faa"
        build_rep_proteins(raw_dir / "rep.dna.fas", rep_proteins)
        _require_identity(rep_proteins, REP_PROTEIN_SHA256, "translated rep protein FASTA")
        _makedb(executable, rep_proteins, output_dir / "rep_proteins.dmnd")
        _makedb(executable, raw_dir / "mob.proteins.faa", output_dir / "mob_proteins.dmnd")
        _makedb(executable, raw_dir / "mpf.proteins.faa", output_dir / "mpf_proteins.dmnd")
        for filename, expected in DATABASE_SHA256.items():
            _require_identity(output_dir / filename, expected, f"rebuilt {filename}")
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return {"diamond_version": version, **DATABASE_SHA256}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild frozen MobiOrigin marker databases from MOB-suite 3.1.8 raw data."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diamond", type=Path, default=Path("diamond"))
    args = parser.parse_args(arguments)
    result = build_marker_databases(args.raw_dir, args.output_dir, args.diamond)
    print("PASS: all three frozen marker databases were reconstructed exactly.")
    print(f"DIAMOND: {result['diamond_version']}")
    print(f"Build output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
