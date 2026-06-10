"""Compare PlasFlow v2 predictions against geNomad pseudo-labels.

Since W1 and GCA have no wet-lab ground truth, we use geNomad's high-confidence
predictions as pseudo-labels to evaluate PlasFlow v2 agreement, precision, and recall.

Usage
-----
  python scripts/compare_with_genomad.py \\
      --plasflow2  results/W1_plasflow2/predictions.tsv \\
      --genomad    results/W1/annotated_predictions.tsv \\
      --min-conf   0.9 \\
      --out        results/W1_comparison/

  python scripts/compare_with_genomad.py \\
      --plasflow2  results/GCA_plasflow2/predictions.tsv \\
      --genomad    data/test/GCA_054405655.predictions.tsv \\
      --genomad-format v4 \\
      --min-conf   0.9 \\
      --out        results/GCA_comparison/

Output
------
  comparison_summary.txt   — headline P/R/F1 + agreement rates
  confusion_matrix.tsv     — geNomad label × PlasFlow v2 label counts
  by_length.tsv            — agreement by contig length bin
  disagreements.tsv        — contigs where tools disagree (for manual inspection)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_plasflow2(path: Path) -> dict[str, dict]:
    """Load PlasFlow v2 predictions TSV.

    Expected columns: sequence_id, label, plasmid, chromosome, phage, [unclassified]
    Returns contig_id → {label, plasmid_score, chromosome_score, phage_score}
    """
    preds = {}
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sid = row.get("sequence_id") or row.get("contig_id") or row.get("id")
            if not sid:
                continue
            label = row.get("label", "unclassified").strip().lower()
            preds[sid] = {
                "label": label,
                "plasmid_score": float(row.get("plasmid", 0) or 0),
                "chromosome_score": float(row.get("chromosome", 0) or 0),
                "phage_score": float(row.get("phage", 0) or 0),
            }
    return preds


def load_genomad_v4(path: Path, min_conf: float) -> dict[str, str]:
    """Load geNomad annotated_predictions.tsv.

    Handles two column layouts:
      - Old format: prediction, <class-name score columns>
      - New format: label, confidence  (e.g. GCA_054405655.predictions.tsv)
    """
    labels = {}
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sid = (row.get("contig_id") or row.get("sequence_id") or
                   row.get("Contig") or "").strip()
            if not sid:
                continue
            # Try both column name conventions
            pred = (row.get("label") or row.get("prediction") or "").strip().lower()
            # Normalise virus → phage
            if pred == "virus":
                pred = "phage"
            if pred not in ("plasmid", "chromosome", "phage", "archaea"):
                continue
            # Confidence: dedicated column first, then per-class column, else 1.0
            conf_str = (row.get("confidence") or row.get("score") or
                        row.get(pred) or "1.0")
            conf = float(conf_str or 1.0)
            if conf >= min_conf:
                labels[sid] = pred
    return labels


def load_genomad_fasta_labels(results_dir: Path, min_conf: float) -> dict[str, str]:
    """Infer geNomad labels from which output FASTA a contig appears in.

    geNomad writes plasmid.fasta / chromosome.fasta / phage.fasta.
    This is the most reliable approach when confidence scores aren't in the TSV.
    """
    labels = {}
    for cls, fname in [("plasmid", "plasmid.fasta"),
                       ("chromosome", "chromosome.fasta"),
                       ("phage", "phage.fasta")]:
        fpath = results_dir / fname
        if fpath.exists():
            with open(fpath) as fh:
                for line in fh:
                    if line.startswith(">"):
                        sid = line[1:].split()[0].strip()
                        labels[sid] = cls
    return labels


def _contig_length_bin(length: int) -> str:
    if length < 2_000:
        return "1-2kb"
    elif length < 5_000:
        return "2-5kb"
    elif length < 10_000:
        return "5-10kb"
    elif length < 20_000:
        return "10-20kb"
    else:
        return ">20kb"


def _fasta_lengths(fasta: Path) -> dict[str, int]:
    lengths = {}
    sid, length = None, 0
    with open(fasta) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if sid:
                    lengths[sid] = length
                sid = line[1:].split()[0]
                length = 0
            else:
                length += len(line)
    if sid:
        lengths[sid] = length
    return lengths


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(
    pf2: dict[str, dict],
    gn: dict[str, str],
    lengths: dict[str, int],
    out_dir: Path,
    min_conf: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # normalise phage/virus label
    def norm(label: str) -> str:
        if label in ("virus", "phage"):
            return "phage"
        return label

    shared = set(pf2) & set(gn)
    print(f"PlasFlow v2 predictions:   {len(pf2):,}")
    print(f"geNomad labels (≥{min_conf:.0%} conf): {len(gn):,}")
    print(f"Shared contigs:            {len(shared):,}")
    if not shared:
        print("ERROR: no shared contig IDs — check that the same FASTA was used for both tools")
        sys.exit(1)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_length: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    disagreements = []

    for sid in shared:
        gn_label = norm(gn[sid])
        pf2_label = norm(pf2[sid]["label"])
        length = lengths.get(sid, 0)
        lbin = _contig_length_bin(length)

        confusion[gn_label][pf2_label] += 1
        by_length[lbin][gn_label][pf2_label] += 1

        if gn_label != pf2_label:
            disagreements.append({
                "contig_id": sid,
                "length_bp": length,
                "genomad_label": gn_label,
                "plasflow2_label": pf2_label,
                "plasflow2_plasmid_score": f"{pf2[sid]['plasmid_score']:.4f}",
                "plasflow2_chromosome_score": f"{pf2[sid]['chromosome_score']:.4f}",
                "plasflow2_phage_score": f"{pf2[sid]['phage_score']:.4f}",
            })

    # ── Confusion matrix ─────────────────────────────────────────────────────
    classes = ["plasmid", "chromosome", "phage", "unclassified"]
    with open(out_dir / "confusion_matrix.tsv", "w") as fh:
        fh.write("genomad_label\t" + "\t".join(f"pf2_{c}" for c in classes) + "\n")
        for gl in classes:
            row = confusion.get(gl, {})
            counts = "\t".join(str(row.get(pl, 0)) for pl in classes)
            fh.write(f"{gl}\t{counts}\n")

    # ── By-length breakdown ───────────────────────────────────────────────────
    len_bins = ["1-2kb", "2-5kb", "5-10kb", "10-20kb", ">20kb"]
    with open(out_dir / "by_length.tsv", "w") as fh:
        fh.write("length_bin\tgenomd_plasmid_total\tpf2_plasmid_agree\t"
                 "pf2_plasmid_recall\tpf2_chromosome_fp\n")
        for lbin in len_bins:
            d = by_length.get(lbin, {})
            gn_plas = sum(d.get("plasmid", {}).values())
            agree = d.get("plasmid", {}).get("plasmid", 0)
            gn_chrom = sum(d.get("chromosome", {}).values())
            chrom_fp = d.get("chromosome", {}).get("plasmid", 0)
            recall = agree / max(gn_plas, 1)
            fh.write(f"{lbin}\t{gn_plas}\t{agree}\t{recall:.3f}\t{chrom_fp}\n")

    # ── Disagreements sample ──────────────────────────────────────────────────
    disagreements.sort(key=lambda x: -x["length_bp"])
    with open(out_dir / "disagreements.tsv", "w") as fh:
        if disagreements:
            writer = csv.DictWriter(fh, fieldnames=list(disagreements[0].keys()),
                                    delimiter="\t")
            writer.writeheader()
            writer.writerows(disagreements[:5000])  # cap at 5000 rows

    # ── Summary ───────────────────────────────────────────────────────────────
    gn_plas_total = sum(confusion.get("plasmid", {}).values())
    pf2_plas_total = sum(confusion[gl].get("plasmid", 0) for gl in confusion)
    tp = confusion.get("plasmid", {}).get("plasmid", 0)
    fp = sum(confusion[gl].get("plasmid", 0) for gl in confusion if gl != "plasmid")
    fn = gn_plas_total - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    gn_chrom_total = sum(confusion.get("chromosome", {}).values())
    chrom_agree = confusion.get("chromosome", {}).get("chromosome", 0)
    chrom_acc = chrom_agree / max(gn_chrom_total, 1)

    overall_agree = sum(confusion[c].get(c, 0) for c in classes)
    overall_acc = overall_agree / max(len(shared), 1)

    summary_lines = [
        f"PlasFlow v2 vs geNomad comparison",
        f"geNomad min confidence threshold: {min_conf:.0%}",
        f"Shared contigs evaluated: {len(shared):,}",
        f"",
        f"=== Plasmid (treating geNomad ≥{min_conf:.0%} as ground truth) ===",
        f"  geNomad plasmid calls:   {gn_plas_total:,}",
        f"  PlasFlow v2 agrees (TP): {tp:,}",
        f"  PlasFlow v2 FP:          {fp:,}   (called plasmid when geNomad says chrom/phage)",
        f"  PlasFlow v2 FN:          {fn:,}   (missed plasmids geNomad found)",
        f"  Precision: {precision:.3f}",
        f"  Recall:    {recall:.3f}",
        f"  F1:        {f1:.3f}",
        f"",
        f"=== Chromosome ===",
        f"  geNomad chromosome calls:     {gn_chrom_total:,}",
        f"  PlasFlow v2 agree:            {chrom_agree:,} ({chrom_acc:.1%})",
        f"",
        f"=== Overall agreement ===",
        f"  Overall label agreement:      {overall_agree:,} / {len(shared):,} = {overall_acc:.1%}",
        f"",
        f"=== By length (plasmid recall) ===",
    ]
    for lbin in len_bins:
        d = by_length.get(lbin, {})
        gn_p = sum(d.get("plasmid", {}).values())
        agree = d.get("plasmid", {}).get("plasmid", 0)
        fp_l = sum(d.get("chromosome", {}).get("plasmid", 0) +
                   d.get("phage", {}).get("plasmid", 0)
                   for _ in [1])
        rec = agree / max(gn_p, 1)
        summary_lines.append(
            f"  {lbin:>8}  geNomad_plas={gn_p:5d}  pf2_agree={agree:5d}  "
            f"recall={rec:.3f}  chrom_fp={d.get('chromosome', {}).get('plasmid', 0)}"
        )

    summary_lines += [
        f"",
        f"Output files:",
        f"  {out_dir}/confusion_matrix.tsv",
        f"  {out_dir}/by_length.tsv",
        f"  {out_dir}/disagreements.tsv  ({len(disagreements):,} disagreements)",
    ]

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(out_dir / "comparison_summary.txt", "w") as fh:
        fh.write(summary_text + "\n")
    print(f"\nResults saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PlasFlow v2 predictions against geNomad pseudo-labels"
    )
    parser.add_argument("--plasflow2", type=Path, required=True,
                        help="PlasFlow v2 predictions TSV")
    parser.add_argument("--genomad", type=Path, required=True,
                        help="geNomad annotated_predictions.tsv OR results directory")
    parser.add_argument("--genomad-format", choices=["auto", "tsv", "dir", "v4"],
                        default="auto",
                        help="geNomad output format (default: auto-detect)")
    parser.add_argument("--min-conf", type=float, default=0.9,
                        help="Min geNomad confidence to use as pseudo-label (default 0.9)")
    parser.add_argument("--fasta", type=Path, default=None,
                        help="Original FASTA (for contig lengths). Auto-detected if omitted.")
    parser.add_argument("--out", type=Path, default=Path("results/comparison"),
                        help="Output directory")
    args = parser.parse_args()

    # Load geNomad labels
    fmt = args.genomad_format
    if fmt == "auto":
        fmt = "dir" if args.genomad.is_dir() else "tsv"

    if fmt == "dir":
        print(f"Loading geNomad labels from FASTA files in {args.genomad} …")
        gn_labels = load_genomad_fasta_labels(args.genomad, args.min_conf)
    else:
        print(f"Loading geNomad labels from {args.genomad} …")
        gn_labels = load_genomad_v4(args.genomad, args.min_conf)

    print(f"  Loaded {len(gn_labels):,} geNomad labels")

    # Load PlasFlow v2 predictions
    print(f"Loading PlasFlow v2 predictions from {args.plasflow2} …")
    pf2_preds = load_plasflow2(args.plasflow2)
    print(f"  Loaded {len(pf2_preds):,} PlasFlow v2 predictions")

    # Contig lengths
    lengths: dict[str, int] = {}
    if args.fasta and args.fasta.exists():
        print(f"Loading contig lengths from {args.fasta} …")
        lengths = _fasta_lengths(args.fasta)
    else:
        # Try to infer length from sequence_id if it has 'len=' field (MEGAHIT format)
        for sid in pf2_preds:
            if "len=" in sid:
                try:
                    lengths[sid] = int(sid.split("len=")[1].split()[0])
                except Exception:
                    pass

    compare(pf2_preds, gn_labels, lengths, args.out, args.min_conf)


if __name__ == "__main__":
    main()
