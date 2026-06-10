"""Compare PlasFlow v1 and PlasFlow v2 predictions on wastewater metagenome contigs.

This is the primary real-world benchmark for the paper, comparing both tools
on the same W1 wastewater metagenome assembly (no ground truth needed).

Usage
-----
    # Step 1: run PlasFlow v2 (if not already done)
    bash run_W1.sh

    # Step 2: install and run PlasFlow v1 in a conda env
    conda create -n plasflow1 python=3.7 -y
    conda activate plasflow1
    conda install -c bioconda plasflow -y
    plasflow.py --input data/test/W1.contigs.fa.gz \\
                --output results/W1_plasflow1.tsv \\
                --threshold 0.7

    # Step 3: run this comparison script
    python scripts/compare_v1_v2_wastewater.py \\
        --v2-predictions  results/W1/all_predictions.tsv \\
        --v1-predictions  results/W1_plasflow1.tsv \\
        --contigs         data/test/W1.contigs.fa.gz \\
        --out             results/W1_comparison/

Outputs
-------
    results/W1_comparison/
        comparison_summary.txt        — paper-ready comparison table
        agreement_matrix.csv          — full cross-tabulation
        v2_only_plasmids.tsv          — contigs V2 calls plasmid, V1 calls chromosome
        v1_only_plasmids.tsv          — contigs V1 calls plasmid, V2 does not
        both_plasmids.tsv             — contigs both tools agree are plasmid
        length_distribution.csv       — plasmid size bins by tool
        annotation_summary.txt        — V2 ARG/VF annotation on plasmid contigs
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LENGTH_BINS = [
    ("1-3kb",    1000,   3000),
    ("3-10kb",   3000,  10000),
    ("10-50kb", 10000,  50000),
    (">50kb",   50000, 10**9),
]


def load_contig_lengths(fasta_path: Path) -> dict[str, int]:
    """Load contig_id → length from FASTA (plain or gzipped)."""
    lengths: dict[str, int] = {}
    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    try:
        from Bio import SeqIO  # type: ignore
        with opener(str(fasta_path), "rt") as fh:
            for rec in SeqIO.parse(fh, "fasta"):
                lengths[rec.id] = len(rec.seq)
    except ImportError:
        # BioPython not available — use manual FASTA parser
        with opener(str(fasta_path), "rt") as fh:
            curr_id = None
            curr_len = 0
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if curr_id:
                        lengths[curr_id] = curr_len
                    curr_id = line[1:].split()[0]
                    curr_len = 0
                else:
                    curr_len += len(line)
            if curr_id:
                lengths[curr_id] = curr_len
    logger.info("Loaded %d contig lengths from %s", len(lengths), fasta_path.name)
    return lengths


def load_v2_predictions(tsv_path: Path) -> dict[str, dict]:
    """Load PlasFlow v2 all_predictions.tsv → {contig_id: row_dict}."""
    preds: dict[str, dict] = {}
    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cid = row.get("sequence_id") or row.get("contig_id") or row.get("id", "")
            preds[cid.strip()] = row
    logger.info("Loaded %d PlasFlow v2 predictions from %s", len(preds), tsv_path.name)
    return preds


def load_v1_predictions(tsv_path: Path) -> dict[str, dict]:
    """Load PlasFlow v1 output TSV → {contig_id: row_dict}.

    PlasFlow v1 output columns (no header in some versions):
      id  label  prob_chromosome  prob_plasmid  prob_phage  [contig_length]

    Labels: plasmid | chromosome | unclassified
    """
    preds: dict[str, dict] = {}
    with open(tsv_path) as fh:
        first = fh.readline().strip()
        has_header = first.split("\t")[0].lower() in ("id", "contig_id", "sequence_id")
        if not has_header:
            fh.seek(0)
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            cid   = parts[0].strip()
            label = parts[1].strip().lower()
            # Normalise label
            if "plasmid" in label:
                norm = "plasmid"
            elif "chromosome" in label or "chrom" in label:
                norm = "chromosome"
            else:
                norm = "unclassified"
            preds[cid] = {
                "label": norm,
                "raw_label": label,
                "prob_chromosome": parts[2] if len(parts) > 2 else "",
                "prob_plasmid":    parts[3] if len(parts) > 3 else "",
            }
    logger.info("Loaded %d PlasFlow v1 predictions from %s", len(preds), tsv_path.name)
    return preds


def length_bin(length: int) -> str:
    for name, lo, hi in LENGTH_BINS:
        if lo <= length < hi:
            return name
    return ">50kb"


def load_v2_annotations(preds_dir: Path) -> dict[str, dict]:
    """Load PlasFlow v2 annotated_predictions.tsv for extra columns (ARGs, VFs, etc.)."""
    ann_path = preds_dir / "annotated_predictions.tsv"
    if not ann_path.exists():
        return {}
    anns: dict[str, dict] = {}
    with open(ann_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cid = row.get("sequence_id") or row.get("contig_id") or row.get("id", "")
            anns[cid.strip()] = row
    logger.info("Loaded %d annotated predictions", len(anns))
    return anns


def build_comparison(
    v2: dict[str, dict],
    v1: dict[str, dict],
    lengths: dict[str, int],
) -> dict:
    """Cross-tabulate v1 and v2 predictions."""
    all_contigs = set(v2.keys()) | set(v1.keys())

    # Normalise v2 label (unclassified = chromosome for comparison purposes)
    def v2_label(cid: str) -> str:
        row = v2.get(cid, {})
        lbl = row.get("label", "unclassified")
        return lbl if lbl != "unclassified" else "chromosome"

    def v1_label(cid: str) -> str:
        row = v1.get(cid, {})
        return row.get("label", "chromosome")

    # Cross-tabulation
    xtab: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cid in all_contigs:
        xtab[v1_label(cid)][v2_label(cid)] += 1

    # Contig sets
    both_plasmid   = {c for c in all_contigs if v1_label(c) == "plasmid" and v2_label(c) == "plasmid"}
    v2_only_plasmid = {c for c in all_contigs if v1_label(c) != "plasmid" and v2_label(c) == "plasmid"}
    v1_only_plasmid = {c for c in all_contigs if v1_label(c) == "plasmid" and v2_label(c) != "plasmid"}

    # Length-bin distributions
    def length_bins_for(contig_set: set[str]) -> Counter:
        return Counter(length_bin(lengths.get(c, 0)) for c in contig_set)

    v2_all_plasmid = {c for c in all_contigs if v2_label(c) == "plasmid"}
    v1_all_plasmid = {c for c in all_contigs if v1_label(c) == "plasmid"}

    return {
        "n_total":         len(all_contigs),
        "n_v2_plasmid":    len(v2_all_plasmid),
        "n_v1_plasmid":    len(v1_all_plasmid),
        "n_both_plasmid":  len(both_plasmid),
        "n_v2_only":       len(v2_only_plasmid),
        "n_v1_only":       len(v1_only_plasmid),
        "xtab":            xtab,
        "both_plasmid":    both_plasmid,
        "v2_only_plasmid": v2_only_plasmid,
        "v1_only_plasmid": v1_only_plasmid,
        "v2_all_plasmid":  v2_all_plasmid,
        "v1_all_plasmid":  v1_all_plasmid,
        "v2_length_bins":  length_bins_for(v2_all_plasmid),
        "v1_length_bins":  length_bins_for(v1_all_plasmid),
        "both_length_bins": length_bins_for(both_plasmid),
        "lengths":         lengths,
    }


def write_summary(comp: dict, out_dir: Path, anns: dict[str, dict]) -> None:
    """Write paper-ready comparison_summary.txt."""
    n  = comp["n_total"]
    v2 = comp["n_v2_plasmid"]
    v1 = comp["n_v1_plasmid"]
    bo = comp["n_both_plasmid"]
    v2o = comp["n_v2_only"]
    v1o = comp["n_v1_only"]

    # Agreement rate (on contigs present in both)
    agree = sum(
        1 for c in set(comp["v2_all_plasmid"]) | set(comp["v1_all_plasmid"]) | set()
    )
    # Simpler: agreement on contigs both tools classified the same way
    in_both = set(comp["v2_all_plasmid"]) & set(comp["v1_all_plasmid"])
    jaccard = len(in_both) / (len(comp["v2_all_plasmid"]) | len(comp["v1_all_plasmid"]))  # type: ignore

    # ARG/VF stats for V2 plasmids
    v2_plasmid_with_arg = sum(
        1 for c in comp["v2_all_plasmid"]
        if anns.get(c, {}).get("arg_count", "0") not in ("", "0")
    )
    v2_plasmid_with_vf = sum(
        1 for c in comp["v2_all_plasmid"]
        if anns.get(c, {}).get("vf_count", "0") not in ("", "0")
    )

    # V2 phage calls
    v2_phage = sum(1 for row in comp.get("v2_raw", {}).values() if row.get("label") == "phage")

    lines = [
        "PlasFlow v1 vs PlasFlow v2 — W1 Wastewater Metagenome Comparison",
        "=" * 70,
        f"Total contigs analysed  : {n:>10,}",
        "",
        "─" * 70,
        f"{'Metric':<45} {'PlasFlow v1':>12} {'PlasFlow v2':>12}",
        "─" * 70,
        f"{'Plasmid calls':<45} {v1:>12,} {v2:>12,}",
        f"{'Plasmid rate (%)':<45} {100*v1/n:>11.2f}% {100*v2/n:>11.2f}%",
        "",
        f"{'Agreement (both call plasmid)':<45} {bo:>12,}",
        f"{'Jaccard similarity (plasmid sets)':<45} {jaccard:>11.3f}",
        f"{'V1 unique plasmids (V2 = chromosome)':<45} {v1o:>12,}",
        f"{'V2 unique plasmids (V1 = chromosome)':<45} {v2o:>12,}",
        "",
        "Plasmid length distribution:",
        f"  {'Size bin':<12}  {'PlasFlow v1':>12}  {'PlasFlow v2':>12}  {'Both agree':>12}",
    ]
    for bin_name, _, _ in LENGTH_BINS:
        v1c = comp["v1_length_bins"].get(bin_name, 0)
        v2c = comp["v2_length_bins"].get(bin_name, 0)
        boc = comp["both_length_bins"].get(bin_name, 0)
        lines.append(f"  {bin_name:<12}  {v1c:>12,}  {v2c:>12,}  {boc:>12,}")
    lines += [
        "",
        "PlasFlow v2 unique capabilities (not available in v1):",
        f"  Phage classification  : {v2_phage:,} phage contigs identified",
        f"  ARG annotation        : {v2_plasmid_with_arg:,} / {v2:,} plasmids carry resistance genes",
        f"  VF annotation         : {v2_plasmid_with_vf:,} / {v2:,} plasmids carry virulence factors",
        "",
        "Cross-tabulation (rows = V1 label, cols = V2 label):",
        f"  {'':15}  {'V2: plasmid':>12}  {'V2: chromosome':>15}  {'V2: phage':>10}  {'V2: unclassified':>17}",
    ]
    for v1_lbl in ["plasmid", "chromosome", "unclassified"]:
        row = comp["xtab"].get(v1_lbl, {})
        lines.append(
            f"  {'V1: ' + v1_lbl:<15}  "
            f"{row.get('plasmid', 0):>12,}  "
            f"{row.get('chromosome', 0):>15,}  "
            f"{row.get('phage', 0):>10,}  "
            f"{row.get('unclassified', 0):>17,}"
        )

    text = "\n".join(lines) + "\n"
    (out_dir / "comparison_summary.txt").write_text(text)
    print(text)


def write_plasmid_lists(comp: dict, v2: dict, v1: dict, out_dir: Path) -> None:
    """Write TSV files for each plasmid subset."""
    fields_v2 = ["contig_id", "v2_label", "v2_confidence", "v1_label", "length_bp"]

    def write_subset(name: str, contig_set: set[str]) -> None:
        path = out_dir / f"{name}.tsv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["contig_id", "v2_label", "v2_confidence", "v1_label",
                        "length_bp", "v2_plasmid_score"])
            for cid in sorted(contig_set,
                               key=lambda c: float(v2.get(c, {}).get("confidence", 0)),
                               reverse=True):
                v2r = v2.get(cid, {})
                v1r = v1.get(cid, {})
                w.writerow([
                    cid,
                    v2r.get("label", ""),
                    v2r.get("confidence", ""),
                    v1r.get("label", ""),
                    comp["lengths"].get(cid, ""),
                    v2r.get("plasmid", v2r.get("plasmid_score", "")),
                ])
        logger.info("Wrote %d contigs → %s", len(contig_set), path.name)

    write_subset("both_plasmids",    comp["both_plasmid"])
    write_subset("v2_only_plasmids", comp["v2_only_plasmid"])
    write_subset("v1_only_plasmids", comp["v1_only_plasmid"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PlasFlow v1 and v2 on wastewater metagenome"
    )
    parser.add_argument("--v2-predictions", type=Path, required=True,
                        help="PlasFlow v2 all_predictions.tsv (from results/W1/)")
    parser.add_argument("--v1-predictions", type=Path, default=None,
                        help="PlasFlow v1 output TSV (from plasflow.py)")
    parser.add_argument("--contigs",        type=Path, required=True,
                        help="Input contig FASTA (plain or .gz)")
    parser.add_argument("--out",            type=Path,
                        default=Path("results/W1_comparison"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load data
    lengths = load_contig_lengths(args.contigs)
    v2_preds = load_v2_predictions(args.v2_predictions)
    anns = load_v2_annotations(args.v2_predictions.parent)

    if args.v1_predictions and args.v1_predictions.exists():
        v1_preds = load_v1_predictions(args.v1_predictions)
    else:
        logger.warning(
            "--v1-predictions not provided or not found. "
            "Run PlasFlow v1 first:\n"
            "  conda create -n plasflow1 python=3.7 -y\n"
            "  conda activate plasflow1\n"
            "  conda install -c bioconda plasflow -y\n"
            "  plasflow.py --input data/test/W1.contigs.fa.gz \\\n"
            "              --output results/W1_plasflow1.tsv \\\n"
            "              --threshold 0.7\n"
        )
        logger.info("Running V2-only summary (no V1 comparison available).")
        # Show V2 summary only
        v2_plasmids = {c for c, r in v2_preds.items() if r.get("label") == "plasmid"}
        v2_phage    = {c for c, r in v2_preds.items() if r.get("label") == "phage"}
        v2_chr      = {c for c, r in v2_preds.items() if r.get("label") == "chromosome"}
        v2_unc      = {c for c, r in v2_preds.items() if r.get("label") == "unclassified"}
        print("\n=== PlasFlow v2 W1 summary (run PlasFlow v1 to get comparison) ===")
        print(f"  Total contigs   : {len(v2_preds):>8,}")
        print(f"  Plasmid         : {len(v2_plasmids):>8,}  ({100*len(v2_plasmids)/len(v2_preds):.2f}%)")
        print(f"  Chromosome      : {len(v2_chr):>8,}  ({100*len(v2_chr)/len(v2_preds):.2f}%)")
        print(f"  Phage           : {len(v2_phage):>8,}  ({100*len(v2_phage)/len(v2_preds):.2f}%)")
        print(f"  Unclassified    : {len(v2_unc):>8,}  ({100*len(v2_unc)/len(v2_preds):.2f}%)")
        print("\nPlasmid length distribution:")
        bins = Counter(length_bin(lengths.get(c, 0)) for c in v2_plasmids)
        for name, _, _ in LENGTH_BINS:
            print(f"  {name:<12}  {bins.get(name,0):>6,}")
        sys.exit(0)

    comp = build_comparison(v2_preds, v1_preds, lengths)
    comp["v2_raw"] = v2_preds

    write_summary(comp, args.out, anns)
    write_plasmid_lists(comp, v2_preds, v1_preds, args.out)

    # Write length distribution CSV
    with open(args.out / "length_distribution.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["length_bin", "v1_plasmid", "v2_plasmid", "both_agree"])
        for name, _, _ in LENGTH_BINS:
            w.writerow([
                name,
                comp["v1_length_bins"].get(name, 0),
                comp["v2_length_bins"].get(name, 0),
                comp["both_length_bins"].get(name, 0),
            ])

    logger.info("\nAll outputs written to %s", args.out)


if __name__ == "__main__":
    main()
