"""Gene-level TSV writer for PlasFlow v2.

Produces a per-ORF table with coordinates, functional annotations, and
ARG/VF/MGE flags — analogous to geNomad's *_genes.tsv output but enriched
with resistance and virulence metadata.

Columns
-------
contig_id   : parent contig identifier
gene_id     : ORF identifier (<contig>_<n>)
start       : 1-indexed start coordinate on the contig
end         : 1-indexed end coordinate on the contig
strand      : strand (1 = forward, -1 = reverse)
length_bp   : gene length in base pairs (end - start + 1)
contig_label: classification label of the parent contig
arg_flag    : 1 if this ORF was annotated as an ARG, else 0
vf_flag     : 1 if this ORF was annotated as a virulence factor, else 0
mge_flag    : 1 if this ORF was annotated as an MGE/IS element, else 0
gene_name   : functional name (ARG gene name, VF gene name, IS element, or "")
drug_class  : drug class(es) for ARG hits, else ""
amr_family  : AMR family for ARG hits, else ""
vf_category : virulence factor category, else ""
is_family   : IS element family, else ""
source      : annotation source (CARD, SARG, VFDB, ISfinder, or "")
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADER = [
    "contig_id",
    "gene_id",
    "start",
    "end",
    "strand",
    "length_bp",
    "contig_label",
    "arg_flag",
    "vf_flag",
    "mge_flag",
    "gene_name",
    "drug_class",
    "amr_family",
    "vf_category",
    "is_family",
    "source",
]


def write_genes_tsv(
    orfs,  # list[ORF]  — from annotate.args
    arg_hits,  # list[ARGHit]
    vf_hits,  # list[VFHit]
    mge_hits,  # list[MGEHit]
    label_by_contig: dict[str, str],  # contig_id → classification label
    output_path: Path | str,
    label_filter: str | None = None,  # if set, only write ORFs from contigs with this label
) -> Path:
    """Write a gene-level TSV with ARG/VF/MGE annotations.

    Args:
        orfs:             ORF objects (must have start/end/strand).
        arg_hits:         ARGHit list (with _orf_id filled in).
        vf_hits:          VFHit list (with _orf_id filled in).
        mge_hits:         MGEHit list (with _orf_id filled in).
        label_by_contig:  mapping from contig_id to its predicted label.
        output_path:      where to write the TSV.
        label_filter:     if given, only write ORFs whose parent contig has
                          this label (e.g. "plasmid").

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Index hits by orf_id for O(1) lookup
    arg_by_orf: dict[str, list] = defaultdict(list)
    for h in arg_hits:
        if h._orf_id:
            arg_by_orf[h._orf_id].append(h)

    vf_by_orf: dict[str, list] = defaultdict(list)
    for h in vf_hits:
        orf_id = getattr(h, "_orf_id", "")
        if orf_id:
            vf_by_orf[orf_id].append(h)

    mge_by_orf: dict[str, list] = defaultdict(list)
    for h in mge_hits:
        orf_id = getattr(h, "_orf_id", "")
        if orf_id:
            mge_by_orf[orf_id].append(h)

    rows_written = 0
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(_HEADER)

        for orf in orfs:
            contig_label = label_by_contig.get(orf.contig_id, "unclassified")
            if label_filter and contig_label != label_filter:
                continue

            a_hits = arg_by_orf.get(orf.orf_id, [])
            v_hits = vf_by_orf.get(orf.orf_id, [])
            m_hits = mge_by_orf.get(orf.orf_id, [])

            arg_flag = 1 if a_hits else 0
            vf_flag = 1 if v_hits else 0
            mge_flag = 1 if m_hits else 0

            # Pick the best annotation: ARG > VF > MGE > unannotated
            gene_name = drug_class = amr_family = vf_category = is_family = source = ""
            if a_hits:
                h = a_hits[0]
                gene_name = h.gene_name
                drug_class = h.drug_class
                amr_family = h.amr_family
                source = h.source
            elif v_hits:
                h = v_hits[0]
                gene_name = h.gene_name
                vf_category = getattr(h, "vf_category", "")
                source = "VFDB"
            elif m_hits:
                h = m_hits[0]
                gene_name = h.is_name
                is_family = h.is_family
                source = "ISfinder"

            writer.writerow(
                [
                    orf.contig_id,
                    orf.orf_id,
                    orf.start,
                    orf.end,
                    orf.strand,
                    abs(orf.end - orf.start) + 1,
                    contig_label,
                    arg_flag,
                    vf_flag,
                    mge_flag,
                    gene_name,
                    drug_class,
                    amr_family,
                    vf_category,
                    is_family,
                    source,
                ]
            )
            rows_written += 1

    logger.info("Wrote %d gene records to %s", rows_written, output_path)
    return output_path
