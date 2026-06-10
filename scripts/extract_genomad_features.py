"""Extract per-contig SPM features from geNomad annotate output.

Parses the `*_genes.tsv` file produced by `genomad annotate` and computes
12 per-contig summary features combining geNomad's 227k-profile SPM scores
with RBS and strand-switch signals.

These features complement MOB-suite DIAMOND hits and are designed to be
merged into the annotation TSV fed to the marker XGBoost second stage.

Usage
-----
  python scripts/extract_genomad_features.py \\
      --genes   data/benchmark/genomad_ann/benchmark_annotate/benchmark_genes.tsv \\
      --out     data/benchmark/genomad_features.tsv

  # Merge with existing MOB-suite annotations
  python scripts/extract_genomad_features.py \\
      --genes     data/benchmark/genomad_ann/benchmark_annotate/benchmark_genes.tsv \\
      --merge-tsv data/benchmark/annotations.tsv \\
      --out       data/benchmark/annotations_with_genomad.tsv

Output columns (per contig)
---------------------------
  contig_id
  p_marker_freq      fraction of genes with any plasmid-marker hit
  c_marker_freq      fraction of genes with any chromosome-marker hit
  v_marker_freq      fraction of genes with any virus-marker hit
  pp_marker_freq     fraction of genes with plasmid SPM > 0.5
  median_p_spm       median plasmid SPM across all genes on the contig
  median_c_spm       median chromosome SPM
  median_v_spm       median virus SPM
  p_vs_c_logistic    sigmoid(mean(p_spm - c_spm)) — compound plasmid score
  strand_switch_rate fraction of consecutive gene pairs with strand flips
  no_rbs_freq        fraction of genes with no RBS motif
  canonical_sd_freq  fraction with canonical Shine-Dalgarno motif
  n_plasmid_markers  raw count of plasmid-marker gene hits
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Canonical Shine-Dalgarno motifs (5′→3′, from geNomad / pyrodigal docs)
CANONICAL_SD = {
    "AGGAG", "GGAG", "GGAGG", "AGGA", "GAGG",
    "AGGAGG", "AGGAGGT", "AGGTG", "AGGUG",
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _detect_columns(header: list[str]) -> dict[str, str]:
    """Map canonical names to actual column names in this TSV.

    geNomad has renamed some columns across versions, so we search
    flexibly. Returns a dict: role → actual_col_name.
    """
    h = {c.lower(): c for c in header}

    def find(*candidates: str) -> str | None:
        for c in candidates:
            if c in h:
                return h[c]
        return None

    return {
        "gene":     find("gene", "gene_id", "id") or header[0],
        "strand":   find("strand") or "",
        "rbs":      find("rbs_motif", "rbs") or "",
        "marker":   find("marker", "marker_id") or "",
        "p_spm":    find("plasmid_hallmark_score", "plasmid_score", "p_spm") or "",
        "c_spm":    find("chromosome_hallmark_score", "chromosome_score", "c_spm") or "",
        "v_spm":    find("virus_hallmark_score", "virus_score", "v_spm") or "",
    }


def _contig_id_from_gene(gene_id: str) -> str:
    """Strip trailing `_N` ORF index to recover contig ID.

    geNomad uses `contig_id_1`, `contig_id_2`, ... for gene names.
    """
    parts = gene_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return gene_id


def load_genes_tsv(path: Path) -> dict[str, list[dict]]:
    """Parse geNomad genes TSV. Returns contig_id → list of gene dicts."""
    contigs: dict[str, list[dict]] = defaultdict(list)
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        cols = _detect_columns(list(reader.fieldnames or []))

        for row in reader:
            gene_id = row.get(cols["gene"], "").strip()
            contig = _contig_id_from_gene(gene_id)

            def _f(col: str) -> float:
                v = row.get(col, "")
                try:
                    return float(v) if v and v not in ("NA", "None", "") else 0.0
                except ValueError:
                    return 0.0

            contigs[contig].append({
                "strand":  row.get(cols["strand"], "").strip(),
                "rbs":     row.get(cols["rbs"], "").strip(),
                "marker":  row.get(cols["marker"], "").strip(),
                "p_spm":   _f(cols["p_spm"]),
                "c_spm":   _f(cols["c_spm"]),
                "v_spm":   _f(cols["v_spm"]),
            })

    return dict(contigs)


def compute_features(genes: list[dict]) -> dict[str, float]:
    """Compute 12 per-contig geNomad SPM features from a list of gene dicts."""
    n = len(genes)
    if n == 0:
        return {
            "p_marker_freq": 0.0, "c_marker_freq": 0.0, "v_marker_freq": 0.0,
            "pp_marker_freq": 0.0,
            "median_p_spm": 0.0, "median_c_spm": 0.0, "median_v_spm": 0.0,
            "p_vs_c_logistic": 0.5,
            "strand_switch_rate": 0.0,
            "no_rbs_freq": 0.0, "canonical_sd_freq": 0.0,
            "n_plasmid_markers": 0,
        }

    # SPM arrays
    p_spms = [g["p_spm"] for g in genes]
    c_spms = [g["c_spm"] for g in genes]
    v_spms = [g["v_spm"] for g in genes]

    # Marker presence
    has_marker  = [1 if g["marker"] else 0 for g in genes]
    # Classify markers by dominant SPM class
    p_markers   = [1 if g["marker"] and g["p_spm"] >= g["c_spm"] and g["p_spm"] >= g["v_spm"] else 0 for g in genes]
    c_markers   = [1 if g["marker"] and g["c_spm"] > g["p_spm"] and g["c_spm"] >= g["v_spm"] else 0 for g in genes]
    v_markers   = [1 if g["marker"] and g["v_spm"] > g["p_spm"] and g["v_spm"] > g["c_spm"] else 0 for g in genes]
    pp_markers  = [1 if g["p_spm"] > 0.5 else 0 for g in genes]

    # Strand switch rate
    strands = [g["strand"] for g in genes]
    switches = sum(
        1 for i in range(1, len(strands))
        if strands[i] and strands[i - 1] and strands[i] != strands[i - 1]
    )
    strand_switch_rate = switches / max(n - 1, 1) if n > 1 else 0.0

    # RBS features
    rbs_list = [g["rbs"] for g in genes]
    no_rbs    = sum(1 for r in rbs_list if not r or r in ("None", "NA", ""))
    canon_sd  = sum(1 for r in rbs_list if r.upper() in CANONICAL_SD)

    # Compound logistic score: sigmoid(mean(p - c))
    p_vs_c = [p - c for p, c in zip(p_spms, c_spms)]
    p_vs_c_logistic = _sigmoid(statistics.mean(p_vs_c)) if p_vs_c else 0.5

    return {
        "p_marker_freq":     sum(p_markers)  / n,
        "c_marker_freq":     sum(c_markers)  / n,
        "v_marker_freq":     sum(v_markers)  / n,
        "pp_marker_freq":    sum(pp_markers) / n,
        "median_p_spm":      statistics.median(p_spms),
        "median_c_spm":      statistics.median(c_spms),
        "median_v_spm":      statistics.median(v_spms),
        "p_vs_c_logistic":   p_vs_c_logistic,
        "strand_switch_rate": strand_switch_rate,
        "no_rbs_freq":       no_rbs  / n,
        "canonical_sd_freq": canon_sd / n,
        "n_plasmid_markers": sum(p_markers),
    }


GENOMAD_COLS = [
    "p_marker_freq", "c_marker_freq", "v_marker_freq", "pp_marker_freq",
    "median_p_spm", "median_c_spm", "median_v_spm", "p_vs_c_logistic",
    "strand_switch_rate", "no_rbs_freq", "canonical_sd_freq", "n_plasmid_markers",
]

_ZERO_FEATURES: dict[str, float] = {c: 0.0 for c in GENOMAD_COLS}
_ZERO_FEATURES["p_vs_c_logistic"] = 0.5  # neutral sigmoid value for contigs with no genes


def extract_all(genes_tsv: Path) -> dict[str, dict[str, float]]:
    """Return contig_id → feature_dict for every contig in genes_tsv."""
    contig_genes = load_genes_tsv(genes_tsv)
    return {cid: compute_features(gs) for cid, gs in contig_genes.items()}


def merge_with_annotations(
    genomad_features: dict[str, dict],
    annotations_tsv: Path,
    out_tsv: Path,
) -> None:
    """Merge geNomad features into an existing MOB-suite annotation TSV.

    Contigs absent from geNomad output (no predicted genes) get zero-filled
    geNomad features so the downstream XGBoost always receives a fixed-width
    feature vector.
    """
    with open(annotations_tsv) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        orig_fields = list(reader.fieldnames or [])
        rows = list(reader)

    out_fields = orig_fields + GENOMAD_COLS
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            cid = row.get("contig_id", "")
            gf = genomad_features.get(cid, _ZERO_FEATURES)
            for col in GENOMAD_COLS:
                v = gf.get(col, _ZERO_FEATURES[col])
                row[col] = f"{v:.6f}" if isinstance(v, float) else str(v)
            writer.writerow(row)

    n_matched = sum(1 for r in rows if r.get("contig_id", "") in genomad_features)
    print(f"Merged {n_matched:,} / {len(rows):,} contigs with geNomad features")
    print(f"Output: {out_tsv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-contig SPM features from geNomad annotate output"
    )
    parser.add_argument("--genes", type=Path, required=True,
                        help="Path to *_genes.tsv from genomad annotate")
    parser.add_argument("--merge-tsv", type=Path, default=None,
                        help="Existing annotations TSV to merge into (optional)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output TSV path")
    args = parser.parse_args()

    print(f"Loading geNomad genes from {args.genes} …")
    genomad_features = extract_all(args.genes)
    print(f"  {len(genomad_features):,} contigs with geNomad annotations")

    # Report summary stats
    n_with_p_markers = sum(1 for f in genomad_features.values() if f["p_marker_freq"] > 0)
    n_high_p_spm     = sum(1 for f in genomad_features.values() if f["median_p_spm"] > 0.3)
    print(f"  Contigs with plasmid marker hits: {n_with_p_markers:,}")
    print(f"  Contigs with median plasmid SPM > 0.3: {n_high_p_spm:,}")

    if args.merge_tsv and args.merge_tsv.exists():
        print(f"\nMerging with {args.merge_tsv} …")
        merge_with_annotations(genomad_features, args.merge_tsv, args.out)
    else:
        # Standalone output
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["contig_id"] + GENOMAD_COLS, delimiter="\t"
            )
            writer.writeheader()
            for cid, feat in sorted(genomad_features.items()):
                row = {"contig_id": cid}
                for col in GENOMAD_COLS:
                    v = feat.get(col, _ZERO_FEATURES[col])
                    row[col] = f"{v:.6f}" if isinstance(v, float) else str(v)
                writer.writerow(row)
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
