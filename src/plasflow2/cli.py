"""PlasFlow v2 CLI — built with Click.

Usage:
    # Full pipeline
    plasflow2 run --input assembly.fasta --output ./results/ \\
                  --threshold 0.7 --context clinical --threads 8

    # With taxonomy annotation
    plasflow2 run --input assembly.fasta --output ./results/ \\
                  --taxonomy-db data/databases/gtdb/gtdb_r220.dmnd \\
                  --taxon-map   data/databases/gtdb/taxon_map.tsv

    # Individual steps
    plasflow2 classify  --input assembly.fasta --output results/predictions.tsv
    plasflow2 annotate  --input plasmids.fasta  --output results/annotations/
    plasflow2 report    --input results/        --output results/report.html

    # Print setup / install instructions
    plasflow2 setup

Week 4 — Days 21-22 + 26 implementation.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import click

from plasflow2 import __version__
from plasflow2.annotate.args import annotate_contigs
from plasflow2.output.genes_tsv import write_genes_tsv
from plasflow2.annotate.mobility import annotate_mobility
from plasflow2.classify.predict import predict
from plasflow2.pipeline import PipelineResult, run_pipeline
from plasflow2.report.generator import (
    PlasmidRow,
    NonPlasmidRow,
    _arg_bar as _build_arg_chart,
    _pie as _build_pie_data,
    _risk_hist as _build_risk_histogram,
    _mobility_bar as _build_mobility_bar,
    _eskape_bar as _build_eskape_bar,
    build_report_data,
    generate_report,
    generate_reports,
)
from plasflow2.risk.scorer import score_plasmid
from plasflow2.utils.fasta import load_fasta, split_by_label, write_fasta

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DB_ROOT = Path(__file__).parent.parent.parent / "data" / "databases"
_DEFAULT_MODEL       = Path(__file__).parent.parent.parent / "data" / "models" / "mlp_v2.pt"
_DEFAULT_CARD_DB     = _DB_ROOT / "card" / "card.dmnd"
_DEFAULT_ARO_INDEX   = _DB_ROOT / "card" / "aro_index.tsv"
_DEFAULT_SARG_DB     = _DB_ROOT / "sarg" / "sarg.dmnd"
_DEFAULT_VFDB        = _DB_ROOT / "vfdb" / "vfdb.dmnd"
_DEFAULT_MGE_DB      = _DB_ROOT / "mge" / "isfinder.dmnd"
_DEFAULT_PLASMID_DB  = _DB_ROOT / "plasmids"   # dir; combined FASTA is built here
_DEFAULT_TAXONOMY_DB = _DB_ROOT / "taxonomy" / "refseq_taxonomy.dmnd"
_DEFAULT_TAXON_MAP   = _DB_ROOT / "taxonomy" / "taxon_map.tsv"
_DEFAULT_KAIJU_DIR   = _DB_ROOT / "kaiju"
_DEFAULT_KAIJU_NODES = _DEFAULT_KAIJU_DIR / "nodes.dmp"
_DEFAULT_KAIJU_NAMES = _DEFAULT_KAIJU_DIR / "names.dmp"


def _resolve_model(model_path: str | None) -> Path:
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise click.BadParameter(f"Model file not found: {p}", param_hint="--model")
        return p
    if _DEFAULT_MODEL.exists():
        return _DEFAULT_MODEL
    raise click.UsageError(
        "No model weights found. Either train a model with:\n"
        "  python scripts/train_model.py --mlp --data data/features.npy --labels data/labels.npy\n"
        "or specify --model <path>."
    )


def _write_predictions_tsv(pipeline_result: PipelineResult, output_path: Path) -> None:
    """Write comprehensive per-contig results to TSV — all classes, all annotations.

    Columns
    -------
    All contigs:
        contig_id, length, label, confidence,
        plasmid_score, chromosome_score, phage_score, archaea_score,
        taxonomy, taxonomy_rank, taxonomy_lineage

    Plasmid contigs (empty string for all other classes):
        num_args, drug_classes, arg_sources,
        mobility_class, replicon_type, relaxase_type, mpf_type,
        risk_score, mobility_score, arg_score, replicon_score,
        context_score, host_score, risk_evidence,
        eskape_host, eskape_genus
    """
    # Non-plasmid rows get empty strings for the 13 plasmid-specific mobility+risk columns
    PLASMID_EMPTY = [""] * 13
    PATHOGEN_EMPTY = ["", "", ""]  # filler when no pathogen detected
    # Non-plasmid rows that lack ARG/VF/MGE results (shouldn't happen if pipeline ran)
    ANNOT_EMPTY = [""] * 9  # 4 ARG + 2 VF + 3 MGE

    HEADER = [
        # ── universal ────────────────────────────────────────────────────
        "contig_id",
        "length",
        "label",
        "confidence",
        "plasmid_score",
        "chromosome_score",
        "phage_score",
        "archaea_score",
        "taxonomy",
        "taxonomy_rank",
        "taxonomy_lineage",
        # ── ARG annotation (all contig classes) ──────────────────────────
        "num_args",
        "arg_genes",         # e.g. "blaNDM-1; sul1; tetA"
        "drug_classes",
        "arg_sources",
        # ── VF annotation (all contig classes) ───────────────────────────
        "num_vf",
        "vf_genes",
        # ── MGE annotation (all contig classes) ──────────────────────────
        "num_mge",
        "mge_genes",         # IS element names e.g. "ISAba1; IS26"
        "mge_families",      # IS families e.g. "IS4; Tn3"
        # ── plasmid-specific mobility & risk ─────────────────────────────
        "mobility_class",
        "replicon_type",
        "relaxase_type",
        "mpf_type",
        "risk_score",
        "mobility_score",
        "arg_score",
        "replicon_score",
        "context_score",
        "host_score",
        "risk_evidence",
        "eskape_host",
        "eskape_genus",
        # ── topology & confidence ─────────────────────────────────────────
        "topology",          # circular / linear / too_short
        "low_confidence",    # True if confidence < 0.70 or argmax fallback used
        # ── plasmid-DB nucleotide match (plasmid contigs only) ────────────
        "plasmid_db_match",      # closest known plasmid accession (e.g. PLSDB_NZ_CP073379.1)
        "plasmid_db_source",     # PLSDB / RefSeq / COMPASS
        "plasmid_db_ani",        # approximate nucleotide identity % to DB hit
        "plasmid_db_cov",        # query coverage % of the DB hit alignment
        # ── pathogen detection (all classes) ─────────────────────────────
        "pathogen_species",
        "pathogen_threat",
        "pathogen_category",
        # ── BacMet biocide/metal resistance (all classes) ─────────────────
        "num_bacmet",
        "bacmet_genes",
        "bacmet_class",
        "bacmet_compounds",
        # ── ICE (all classes) ─────────────────────────────────────────────
        "num_ice",
        "ice_ids",
        "ice_functions",
    ]

    def _tax_fields(tax) -> list:
        if tax is None:
            return ["", "", ""]
        return [tax.display, tax.rank, tax.lineage]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Index results by contig_id for O(1) lookup
    plasmid_by_id     = {cr.record.id: cr for cr in pipeline_result.plasmid_results}
    non_plasmid_by_id = {cr.record.id: cr for cr in pipeline_result.non_plasmid_results}
    # Index all records by contig_id for length lookup
    record_by_id = {cr.record.id: cr.record for cr in pipeline_result.plasmid_results}
    record_by_id.update({cr.record.id: cr.record for cr in pipeline_result.non_plasmid_results})

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(HEADER)

        for pred in pipeline_result.all_predictions:
            cid = pred.sequence_id
            rec = record_by_id.get(cid)
            length = len(rec.seq) if rec else 0

            scores = pred.scores if hasattr(pred, "scores") and pred.scores else {}
            base_cols = [
                cid,
                length,
                pred.label,
                f"{pred.confidence:.4f}",
                f"{scores.get('plasmid', 0):.4f}",
                f"{scores.get('chromosome', 0):.4f}",
                f"{scores.get('phage', 0):.4f}",
                f"{scores.get('archaea', 0):.4f}",
            ]

            # ── Plasmid: full annotation columns ─────────────────────────
            if pred.label == "plasmid" and cid in plasmid_by_id:
                cr = plasmid_by_id[cid]
                tax_cols = _tax_fields(cr.taxonomy)

                # ARG
                arg_genes    = "; ".join(sorted({h.gene_name for h in cr.arg_hits})) if cr.arg_hits else ""
                unique_classes = sorted({
                    dc.strip()
                    for h in cr.arg_hits
                    for dc in h.drug_class.split(";")
                    if dc.strip() and dc.strip() != "unknown"
                })
                sources = sorted({h.source for h in cr.arg_hits if getattr(h, "source", "")})
                # VF
                vf_hits  = getattr(cr, "vf_hits",  [])
                vf_genes = "; ".join(sorted({h.gene_name for h in vf_hits})) if vf_hits else ""
                # MGE
                mge_hits     = getattr(cr, "mge_hits", [])
                mge_genes    = "; ".join(sorted({h.is_name   for h in mge_hits})) if mge_hits else ""
                mge_families = "; ".join(sorted({h.is_family for h in mge_hits})) if mge_hits else ""
                # BacMet
                bm_hits     = getattr(cr, "bacmet_hits", [])
                bm_genes    = "; ".join(sorted({h.gene_name for h in bm_hits})) if bm_hits else ""
                bm_classes  = "; ".join(sorted({h.resistance_class for h in bm_hits})) if bm_hits else ""
                bm_compounds= "; ".join(sorted({h.compound for h in bm_hits if h.compound})) if bm_hits else ""
                # ICE
                ice_hits_cr  = getattr(cr, "ice_hits", [])
                ice_ids_str  = "; ".join(sorted({h.ice_id for h in ice_hits_cr})) if ice_hits_cr else ""
                ice_funcs    = "; ".join(sorted({h.gene_function for h in ice_hits_cr if h.gene_function})) if ice_hits_cr else ""

                annot_cols = [
                    len(cr.arg_hits),
                    arg_genes,
                    "; ".join(unique_classes) if unique_classes else "",
                    ", ".join(sources) if sources else "",
                    len(vf_hits),
                    vf_genes,
                    len(mge_hits),
                    mge_genes,
                    mge_families,
                ]
                bacmet_cols = [len(bm_hits), bm_genes, bm_classes, bm_compounds]
                ice_cols    = [len(ice_hits_cr), ice_ids_str, ice_funcs]

                mob = cr.mobility
                risk = cr.risk
                plasmid_cols = [
                    mob.mobility_class if mob else "",
                    mob.replicon_type if mob else "",
                    mob.relaxase_type if mob else "",
                    mob.mpf_type if mob else "",
                    risk.score,
                    risk.mobility_score,
                    risk.arg_score,
                    risk.replicon_score,
                    risk.context_score,
                    risk.host_score,
                    "; ".join(risk.evidence) if risk.evidence else "",
                    risk.eskape_host,
                    risk.eskape_genus,
                ]

            else:
                # Non-plasmid: extract ARG/VF/MGE from NonPlasmidContigResult
                tax = pipeline_result.taxonomy.get(cid)
                tax_cols = _tax_fields(tax)
                plasmid_cols = PLASMID_EMPTY

                np_cr = non_plasmid_by_id.get(cid)
                if np_cr is not None:
                    np_arg_hits  = getattr(np_cr, "arg_hits",  [])
                    np_vf_hits   = getattr(np_cr, "vf_hits",   [])
                    np_mge_hits  = getattr(np_cr, "mge_hits",  [])
                    np_bm_hits   = getattr(np_cr, "bacmet_hits", [])
                    np_ice_hits  = getattr(np_cr, "ice_hits",   [])
                    np_arg_genes = "; ".join(sorted({h.gene_name for h in np_arg_hits})) if np_arg_hits else ""
                    np_drug_cls  = sorted({
                        dc.strip()
                        for h in np_arg_hits
                        for dc in h.drug_class.split(";")
                        if dc.strip() and dc.strip() != "unknown"
                    })
                    np_sources   = sorted({h.source for h in np_arg_hits if getattr(h, "source", "")})
                    np_vf_genes  = "; ".join(sorted({h.gene_name for h in np_vf_hits}))  if np_vf_hits  else ""
                    np_mge_genes = "; ".join(sorted({h.is_name   for h in np_mge_hits})) if np_mge_hits else ""
                    np_mge_fams  = "; ".join(sorted({h.is_family for h in np_mge_hits})) if np_mge_hits else ""
                    annot_cols = [
                        len(np_arg_hits), np_arg_genes,
                        "; ".join(np_drug_cls) if np_drug_cls else "",
                        ", ".join(np_sources) if np_sources else "",
                        len(np_vf_hits), np_vf_genes,
                        len(np_mge_hits), np_mge_genes, np_mge_fams,
                    ]
                    bacmet_cols = [
                        len(np_bm_hits),
                        "; ".join(sorted({h.gene_name for h in np_bm_hits})),
                        "; ".join(sorted({h.resistance_class for h in np_bm_hits})),
                        "; ".join(sorted({h.compound for h in np_bm_hits if h.compound})),
                    ]
                    ice_cols = [
                        len(np_ice_hits),
                        "; ".join(sorted({h.ice_id for h in np_ice_hits})),
                        "; ".join(sorted({h.gene_function for h in np_ice_hits if h.gene_function})),
                    ]
                else:
                    annot_cols  = ANNOT_EMPTY
                    bacmet_cols = ["", "", "", ""]
                    ice_cols    = ["", "", ""]

            # ── Topology & confidence flag ────────────────────────────────
            topology = pipeline_result.topology.get(cid, "")
            # Low-confidence: True when the model was uncertain (below 0.70
            # threshold or argmax fallback was the only reason it got a label)
            low_confidence = pred.confidence < 0.70
            topo_conf_cols = [topology, str(low_confidence)]

            # ── Plasmid-DB nucleotide match (plasmid contigs only) ────────
            pdb_hit = pipeline_result.plasmid_db_hits.get(cid)
            if pdb_hit:
                plasmid_db_cols = [
                    pdb_hit.match_acc,
                    pdb_hit.source_db,
                    str(pdb_hit.ani),
                    str(pdb_hit.query_cov),
                ]
            else:
                plasmid_db_cols = ["", "", "", ""]

            # ── Pathogen detection (all classes) ──────────────────────────
            path_hit = pipeline_result.pathogens.get(cid)
            if path_hit:
                pathogen_cols = [path_hit.species, path_hit.threat_level, path_hit.category]
            else:
                pathogen_cols = PATHOGEN_EMPTY

            writer.writerow(
                base_cols + tax_cols + annot_cols + plasmid_cols
                + topo_conf_cols + plasmid_db_cols + pathogen_cols
                + bacmet_cols + ice_cols
            )


def _write_annotated_tsv(pipeline_result: PipelineResult, output_path: Path) -> None:
    """Write a focused TSV of contigs that have at least one annotation.

    A contig is included if it has any of:
      - ARG hits (resistance genes)
      - MGE hits (mobile genetic elements)
      - VF hits (virulence factors)
      - mobility class other than non-mobilizable/unknown (plasmids only)
      - pathogen detection

    Columns are a curated subset of all_predictions.tsv, matching the user's
    requested schema:
      contig_id, prediction, arg_genes, drug_classes, mge_genes, mge_families,
      vf_genes, vf_families, mobility_class, risk_score, taxonomy_lineage,
      pathogen_category
    """
    HEADER = [
        "contig_id",
        "prediction",
        "arg_genes",
        "arg_drug_classes",
        "mge_genes",
        "mge_families",
        "vf_genes",
        "vf_categories",
        "bacmet_genes",
        "bacmet_class",
        "bacmet_compounds",
        "ice_ids",
        "ice_functions",
        "mobility_class",
        "risk_score",
        "taxonomy_lca",
        "pathogen_category",
    ]

    plasmid_by_id     = {cr.record.id: cr for cr in pipeline_result.plasmid_results}
    non_plasmid_by_id = {cr.record.id: cr for cr in pipeline_result.non_plasmid_results}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(HEADER)

        for pred in pipeline_result.all_predictions:
            cid = pred.sequence_id

            if pred.label == "plasmid" and cid in plasmid_by_id:
                cr = plasmid_by_id[cid]
                arg_hits    = cr.arg_hits or []
                vf_hits     = getattr(cr, "vf_hits",     []) or []
                mge_hits    = getattr(cr, "mge_hits",    []) or []
                bm_hits     = getattr(cr, "bacmet_hits", []) or []
                ice_hits_cr = getattr(cr, "ice_hits",    []) or []
                mob         = cr.mobility
                mob_class   = mob.mobility_class if mob else "unknown"
                risk_score  = cr.risk.score
                tax = cr.taxonomy
                tax_lca = tax.lineage if tax else ""
                path_hit = pipeline_result.pathogens.get(cid)
                pathogen_cat = path_hit.category if path_hit else ""
                is_mobile = mob_class not in ("non-mobilizable", "unknown", "")
            else:
                np_cr       = non_plasmid_by_id.get(cid)
                arg_hits    = getattr(np_cr, "arg_hits",     []) if np_cr else []
                vf_hits     = getattr(np_cr, "vf_hits",      []) if np_cr else []
                mge_hits    = getattr(np_cr, "mge_hits",     []) if np_cr else []
                bm_hits     = getattr(np_cr, "bacmet_hits",  []) if np_cr else []
                ice_hits_cr = getattr(np_cr, "ice_hits",     []) if np_cr else []
                mob_class   = ""
                risk_score  = ""
                is_mobile   = False
                tax = pipeline_result.taxonomy.get(cid)
                tax_lca = tax.lineage if tax else ""
                path_hit = pipeline_result.pathogens.get(cid)
                pathogen_cat = path_hit.category if path_hit else ""

            # Filter: include only if at least one annotation present
            if not (arg_hits or mge_hits or vf_hits or bm_hits or ice_hits_cr or is_mobile or pathogen_cat):
                continue

            arg_genes    = "; ".join(sorted({h.gene_name for h in arg_hits}))
            drug_classes = "; ".join(sorted({
                dc.strip() for h in arg_hits for dc in h.drug_class.split(";")
                if dc.strip() and dc.strip() != "unknown"
            }))
            mge_genes    = "; ".join(sorted({h.is_name   for h in mge_hits}))
            mge_families = "; ".join(sorted({h.is_family for h in mge_hits}))
            vf_genes     = "; ".join(sorted({h.gene_name for h in vf_hits}))
            vf_cats      = "; ".join(sorted({getattr(h, "vf_category", "") for h in vf_hits if getattr(h, "vf_category", "")}))
            bm_genes     = "; ".join(sorted({h.gene_name for h in bm_hits}))
            bm_class     = "; ".join(sorted({h.resistance_class for h in bm_hits}))
            bm_compounds = "; ".join(sorted({h.compound for h in bm_hits if h.compound}))
            ice_ids_str  = "; ".join(sorted({h.ice_id for h in ice_hits_cr}))
            ice_funcs    = "; ".join(sorted({h.gene_function for h in ice_hits_cr if h.gene_function}))

            writer.writerow([
                cid, pred.label,
                arg_genes, drug_classes,
                mge_genes, mge_families,
                vf_genes, vf_cats,
                bm_genes, bm_class, bm_compounds,
                ice_ids_str, ice_funcs,
                mob_class, risk_score, tax_lca, pathogen_cat,
            ])
            written += 1

    logger.info("Annotated predictions: %d contigs written to %s", written, output_path)


def _write_predictions_tsv_simple(predictions: list, output_path: Path) -> None:
    """Lightweight TSV writer used by 'plasflow2 classify' (no pipeline result)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "contig_id",
                "label",
                "confidence",
                "plasmid_score",
                "chromosome_score",
                "phage_score",
                "archaea_score",
            ]
        )
        for p in predictions:
            scores = p.scores if hasattr(p, "scores") and p.scores else {}
            writer.writerow(
                [
                    p.sequence_id,
                    p.label,
                    f"{p.confidence:.4f}",
                    f"{scores.get('plasmid', 0):.4f}",
                    f"{scores.get('chromosome', 0):.4f}",
                    f"{scores.get('phage', 0):.4f}",
                    f"{scores.get('archaea', 0):.4f}",
                ]
            )


def _write_annotations_json(plasmid_results: list, output_path: Path) -> None:
    """Serialise ARG + mobility + risk + taxonomy annotations to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for cr in plasmid_results:
        mob = cr.mobility
        tax = getattr(cr, "taxonomy", None)
        records.append(
            {
                "contig_id": cr.record.id,
                "length": len(cr.record.seq),
                "classification": {
                    "label": cr.prediction.label,
                    "confidence": cr.prediction.confidence,
                },
                "taxonomy": (
                    {
                        "lineage": tax.lineage,
                        "rank": tax.rank,
                        "taxon": tax.taxon,
                        "num_hits": tax.num_hits,
                        "agreement": tax.agreement,
                    }
                    if tax
                    else None
                ),
                "mobility": (
                    {
                        "mobility_class": mob.mobility_class if mob else "unknown",
                        "replicon_type": mob.replicon_type if mob else "unknown",
                        "relaxase_type": mob.relaxase_type if mob else "none",
                        "mpf_type": mob.mpf_type if mob else "none",
                    }
                    if mob
                    else None
                ),
                "arg_hits": [
                    {
                        "gene_name": h.gene_name,
                        "aro_accession": h.aro_accession,
                        "amr_family": h.amr_family,
                        "drug_class": h.drug_class,
                        "resistance_mechanism": h.resistance_mechanism,
                        "identity": h.identity,
                        "coverage": h.coverage,
                        "evalue": h.evalue,
                        "source": h.source,
                    }
                    for h in cr.arg_hits
                ],
                "risk": {
                    "score": cr.risk.score,
                    "mobility_score": cr.risk.mobility_score,
                    "arg_score": cr.risk.arg_score,
                    "replicon_score": cr.risk.replicon_score,
                    "context_score": cr.risk.context_score,
                    "host_score": cr.risk.host_score,
                    "evidence": cr.risk.evidence,
                    "eskape_host": cr.risk.eskape_host,
                    "eskape_genus": cr.risk.eskape_genus,
                },
            }
        )
    with open(output_path, "w") as fh:
        json.dump(records, fh, indent=2)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(__version__, prog_name="plasflow2")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """PlasFlow v2 — plasmid/chromosome/phage/archaea classifier and AMR risk scorer."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _configure_logging(verbose)


# ---------------------------------------------------------------------------
# plasflow2 run  (full pipeline)
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--input",
    "-i",
    "input_fasta",
    required=True,
    type=click.Path(exists=True),
    help="Input assembly FASTA file.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Output directory (created if absent).",
)
@click.option(
    "--model",
    "model_path",
    default=None,
    type=click.Path(),
    help="Path to trained .pt weights (default: data/models/mlp_v2.pt).",
)
@click.option(
    "--card-db",
    default=None,
    type=click.Path(),
    help="DIAMOND CARD database .dmnd (default: data/databases/card/card.dmnd).",
)
@click.option(
    "--aro-index",
    default=None,
    type=click.Path(),
    help="CARD aro_index.tsv (default: data/databases/card/aro_index.tsv).",
)
@click.option(
    "--threshold",
    default=0.7,
    show_default=True,
    help="Confidence threshold for chromosome/phage/archaea; sequences below this are 'unclassified'.",
)
@click.option(
    "--plasmid-threshold",
    "plasmid_threshold",
    default=0.95,
    show_default=True,
    help=(
        "Confidence threshold for plasmid calls (default 0.95). "
        "Higher than --threshold to correct for class-prior imbalance: "
        "the model trains on ~25% plasmid but real metagenomes have ~2-5% plasmid."
    ),
)
@click.option(
    "--context",
    default="unspecified",
    show_default=True,
    type=click.Choice(
        ["clinical", "wastewater", "environmental", "unspecified"], case_sensitive=False
    ),
    help="Sample source context for AMR risk scoring.",
)
@click.option("--threads", default=8, show_default=True, help="CPU threads for DIAMOND/MOB-suite.")
@click.option("--min-length", default=1000, show_default=True, help="Minimum contig length (bp).")
@click.option(
    "--min-confidence",
    "min_confidence",
    default=None,
    type=float,
    help=(
        "When set, contigs whose best class scores below this value receive the argmax "
        "label (best-guess) instead of 'unclassified'. "
        "Useful for reducing the unclassified rate without retraining. "
        "Example: --min-confidence 0.50 assigns every contig to its most-likely class "
        "regardless of confidence. "
        "Overrides --threshold / --plasmid-threshold for the fallback decision only; "
        "high-confidence calls are unaffected."
    ),
)
@click.option(
    "--skip-mobility",
    is_flag=True,
    default=False,
    help="Skip MOB-suite mobility typing (use when mob_typer is unavailable).",
)
@click.option(
    "--taxonomy-db",
    "taxonomy_db",
    default=None,
    type=click.Path(),
    help="DIAMOND database (.dmnd) built from GTDB-r220 / RefSeq proteins for taxonomy.",
)
@click.option(
    "--taxon-map",
    "taxon_map",
    default=None,
    type=click.Path(),
    help="2-column TSV mapping accession → GTDB lineage (optional, improves LCA accuracy).",
)
@click.option(
    "--skip-taxonomy",
    is_flag=True,
    default=False,
    help="Skip taxonomy annotation (use when no taxonomy DB is available).",
)
@click.option(
    "--taxonomy-engine",
    "taxonomy_engine",
    default="auto",
    type=click.Choice(["auto", "kaiju", "diamond"], case_sensitive=False),
    help=(
        "Taxonomy annotation engine. 'auto' uses Kaiju if installed and its DB is present, "
        "otherwise falls back to DIAMOND. 'kaiju' forces Kaiju (20–50× faster). "
        "'diamond' forces DIAMOND blastp.  [default: auto]"
    ),
)
@click.option(
    "--kaiju-db",
    "kaiju_db",
    default=None,
    type=click.Path(),
    help="Kaiju FM-index database (.fmi).  [auto-detected from data/databases/kaiju/]",
)
@click.option(
    "--kaiju-nodes",
    "kaiju_nodes",
    default=None,
    type=click.Path(),
    help="NCBI taxonomy nodes.dmp for Kaiju.  [auto-detected from data/databases/kaiju/]",
)
@click.option(
    "--kaiju-names",
    "kaiju_names",
    default=None,
    type=click.Path(),
    help="NCBI taxonomy names.dmp for Kaiju.  [auto-detected from data/databases/kaiju/]",
)
@click.option(
    "--sarg-db",
    "sarg_db",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database (.dmnd) built from the SARG (Structured ARG) database. "
        "When provided, ARG annotation runs against both CARD and SARG; CARD hits "
        "take precedence per ORF and SARG supplements with genes not found in CARD."
    ),
)
@click.option(
    "--amrprot-db",
    "amrprot_db",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database (.dmnd) built from the AMRFinderPlus AMRProt FASTA. "
        "Auto-detected from data/databases/amrfinder/amrprot.dmnd when present. "
        "Priority: CARD > AMRProt > SARG per ORF. "
        "Setup: diamond makedb --in AMRProt -d data/databases/amrfinder/amrprot"
    ),
)
@click.option(
    "--min-identity",
    "min_identity",
    default=80.0,
    show_default=True,
    help=(
        "Minimum amino-acid identity %% for DIAMOND ARG hits. "
        "80%% (default) is the standard for environmental/metagenomic samples. "
        "Use 90%% for clinical-isolate-grade precision."
    ),
)
@click.option(
    "--vfdb",
    "vfdb",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database (.dmnd) built from VFDB set A protein sequences. "
        "When provided, annotates plasmid contigs with virulence factors. "
        "Build with: diamond makedb --in VFDB_setA_pro.fas -d data/databases/vfdb/vfdb_setA"
    ),
)
@click.option(
    "--mge-db",
    "mge_db",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database (.dmnd) built from ISfinder transposase protein sequences. "
        "When provided, detects IS elements and transposons on plasmid contigs. "
        "Build with: diamond makedb --in ISfinder-sequences.fasta -d data/databases/mge/isfinder"
    ),
)
@click.pass_context
def run(
    ctx: click.Context,
    input_fasta: str,
    output_dir: str,
    model_path: str | None,
    card_db: str | None,
    aro_index: str | None,
    threshold: float,
    plasmid_threshold: float,
    context: str,
    threads: int,
    min_length: int,
    skip_mobility: bool,
    taxonomy_db: str | None,
    taxon_map: str | None,
    skip_taxonomy: bool,
    taxonomy_engine: str,
    kaiju_db: str | None,
    kaiju_nodes: str | None,
    kaiju_names: str | None,
    sarg_db: str | None,
    amrprot_db: str | None,
    min_identity: float,
    vfdb: str | None,
    mge_db: str | None,
    min_confidence: float | None,
) -> None:
    """Run the full PlasFlow v2 pipeline: classify → annotate → risk → report.

    \b
    Outputs written to OUTPUT_DIR:
        all_predictions.tsv       — per-sequence classification (all contigs, all annotations)
        annotated_predictions.tsv — filtered: contigs with ARGs / MGEs / VFs / mobility / pathogens
        plasmids.fasta            — classified plasmid sequences
        annotations.json          — ARG + mobility + risk per plasmid contig
        report_plasmid.html       — plasmid detail report (ARG/VF/MGE/risk)
        report_chromosome.html    — chromosome contig report
        report_phage.html         — phage contig report
        report_archaea.html       — archaea contig report
        report_unclassified.html  — unclassified contig report
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resolved_model = _resolve_model(model_path)

    card_db_path = Path(card_db) if card_db else _DEFAULT_CARD_DB
    aro_index_path = Path(aro_index) if aro_index else _DEFAULT_ARO_INDEX

    for p, name in [(card_db_path, "--card-db"), (aro_index_path, "--aro-index")]:
        if not p.exists():
            raise click.BadParameter(f"Not found: {p}", param_hint=name)

    # Auto-detect optional databases when not explicitly provided
    if sarg_db is None and _DEFAULT_SARG_DB.exists():
        sarg_db = str(_DEFAULT_SARG_DB)
        click.echo(f"[info] Auto-detected SARG database: {_DEFAULT_SARG_DB}")
    if vfdb is None and _DEFAULT_VFDB.exists():
        vfdb = str(_DEFAULT_VFDB)
        click.echo(f"[info] Auto-detected VFDB database: {_DEFAULT_VFDB}")
    if mge_db is None and _DEFAULT_MGE_DB.exists():
        mge_db = str(_DEFAULT_MGE_DB)
        click.echo(f"[info] Auto-detected MGE database: {_DEFAULT_MGE_DB}")
    if taxonomy_db is None and not skip_taxonomy and _DEFAULT_TAXONOMY_DB.exists():
        taxonomy_db = str(_DEFAULT_TAXONOMY_DB)
        click.echo(f"[info] Auto-detected taxonomy database: {_DEFAULT_TAXONOMY_DB}")
    if taxon_map is None and not skip_taxonomy and _DEFAULT_TAXON_MAP.exists():
        taxon_map = str(_DEFAULT_TAXON_MAP)
    plasmid_db_dir: str | None = None
    if _DEFAULT_PLASMID_DB.is_dir() and any(_DEFAULT_PLASMID_DB.iterdir()):
        plasmid_db_dir = str(_DEFAULT_PLASMID_DB)
        click.echo(f"[info] Auto-detected plasmid database: {_DEFAULT_PLASMID_DB}")

    # Kaiju auto-detection
    from plasflow2.annotate.taxonomy_kaiju import find_kaiju_db, kaiju_available
    if kaiju_db is None and _DEFAULT_KAIJU_DIR.is_dir():
        _found = find_kaiju_db(_DEFAULT_KAIJU_DIR)
        if _found:
            kaiju_db = str(_found)
    if kaiju_nodes is None and _DEFAULT_KAIJU_NODES.exists():
        kaiju_nodes = str(_DEFAULT_KAIJU_NODES)
    if kaiju_names is None and _DEFAULT_KAIJU_NAMES.exists():
        kaiju_names = str(_DEFAULT_KAIJU_NAMES)
    if kaiju_db and kaiju_nodes and kaiju_names:
        _engine_label = "kaiju" if taxonomy_engine in ("kaiju", "auto") else "diamond"
        if taxonomy_engine == "auto" and kaiju_available():
            click.echo(f"[info] Auto-detected Kaiju database: {kaiju_db} (will use kaiju for taxonomy)")
        elif taxonomy_engine == "kaiju":
            click.echo(f"[info] Kaiju database: {kaiju_db}")

    click.echo(f"[PlasFlow v2 v{__version__}] Running pipeline on {input_fasta}")

    # --min-confidence: when set, use argmax fallback below this threshold.
    # The lower of (min_confidence, threshold / plasmid_threshold) becomes the
    # effective floor; anything above the class-specific threshold is still a
    # normal high-confidence call.
    argmax_fallback = min_confidence is not None
    effective_threshold = min(threshold, min_confidence) if argmax_fallback else threshold
    effective_plasmid_threshold = (
        min(plasmid_threshold, min_confidence) if argmax_fallback else plasmid_threshold
    )
    if argmax_fallback:
        click.echo(
            f"[info] --min-confidence={min_confidence}: contigs below threshold will receive "
            f"argmax label instead of 'unclassified'"
        )

    pipeline_result = run_pipeline(
        fasta_path=input_fasta,
        model_path=resolved_model,
        card_db=card_db_path,
        aro_index=aro_index_path,
        work_dir=out / "work",
        source_context=context,
        confidence_threshold=effective_threshold,
        plasmid_threshold=effective_plasmid_threshold,
        argmax_fallback=argmax_fallback,
        min_contig_length=min_length,
        threads=threads,
        skip_mobility=skip_mobility,
        taxonomy_db=taxonomy_db,
        taxon_map_path=taxon_map,
        skip_taxonomy=skip_taxonomy,
        sarg_db=sarg_db,
        amrprot_db=amrprot_db,
        min_identity=min_identity,
        vfdb=vfdb,
        mge_db=mge_db,
        plasmid_db_dir=plasmid_db_dir,
        taxonomy_engine=taxonomy_engine,
        kaiju_db=kaiju_db,
        kaiju_nodes=kaiju_nodes,
        kaiju_names=kaiju_names,
    )

    # --- Write comprehensive predictions TSV (all contigs, all annotations) ---
    preds_tsv = out / "all_predictions.tsv"
    _write_predictions_tsv(pipeline_result, preds_tsv)
    click.echo(f"  Predictions → {preds_tsv}")

    # --- Write filtered annotated TSV (annotated contigs only) ---
    annot_tsv = out / "annotated_predictions.tsv"
    _write_annotated_tsv(pipeline_result, annot_tsv)
    click.echo(f"  Annotated   → {annot_tsv}")

    # --- Write per-class FASTAs (from all loaded records) ---
    # Use the work-dir copy as fallback in case the original input path no longer
    # exists (e.g. the source was a temp directory that got cleaned up).
    _fasta_source = Path(input_fasta)
    if not _fasta_source.exists():
        _fallback = out / "work" / "all_contigs.fasta"
        if _fallback.exists():
            logger.warning(
                "Original input %s not found — using work-dir copy %s",
                input_fasta, _fallback,
            )
            _fasta_source = _fallback
        else:
            raise click.ClickException(
                f"Input FASTA not found at {input_fasta} and no work-dir copy exists. "
                "Cannot write per-class FASTA files."
            )
    records = load_fasta(_fasta_source, min_length=min_length)
    pred_by_id = {p.sequence_id: p.label for p in pipeline_result.all_predictions}
    labels = [pred_by_id.get(r.id, "unclassified") for r in records]
    bins = split_by_label(records, labels)
    for label, recs in bins.items():
        fasta_out = out / f"{label}.fasta"
        write_fasta(recs, fasta_out)
        click.echo(f"  {label.capitalize()} sequences ({len(recs)}) → {fasta_out}")

    # --- Write gene-level TSV (all ORFs with ARG/VF/MGE flags + coordinates) ---
    if pipeline_result.orfs:
        label_by_contig = {p.sequence_id: p.label for p in pipeline_result.all_predictions}
        all_vf_hits  = [h for cr in pipeline_result.plasmid_results for h in cr.vf_hits] + \
                       [h for cr in pipeline_result.non_plasmid_results for h in cr.vf_hits]
        all_mge_hits = [h for cr in pipeline_result.plasmid_results for h in cr.mge_hits] + \
                       [h for cr in pipeline_result.non_plasmid_results for h in cr.mge_hits]
        all_arg_hits = [h for cr in pipeline_result.plasmid_results for h in cr.arg_hits] + \
                       [h for cr in pipeline_result.non_plasmid_results for h in cr.arg_hits]
        genes_tsv_path = out / "genes.tsv"
        write_genes_tsv(
            orfs=pipeline_result.orfs,
            arg_hits=all_arg_hits,
            vf_hits=all_vf_hits,
            mge_hits=all_mge_hits,
            label_by_contig=label_by_contig,
            output_path=genes_tsv_path,
        )
        click.echo(f"  Gene table   → {genes_tsv_path}")

    # --- Write annotations JSON ---
    ann_json = out / "annotations.json"
    _write_annotations_json(pipeline_result.plasmid_results, ann_json)
    click.echo(f"  Annotations → {ann_json}")

    # --- Write 5 separate HTML reports (one per class) ---
    report_data = build_report_data(pipeline_result, input_file=str(input_fasta))
    report_paths = generate_reports(report_data, out)
    click.echo("  HTML reports:")
    labels_order = ["plasmid", "chromosome", "phage", "archaea", "unclassified"]
    for key in labels_order:
        p = report_paths.get(key)
        if p:
            click.echo(f"    {key:<14} → {p}")

    click.echo(
        f"\nDone. {pipeline_result.total_sequences} sequences | "
        f"{pipeline_result.total_plasmids} plasmids | "
        f"{pipeline_result.total_args} ARGs detected."
    )


# ---------------------------------------------------------------------------
# plasflow2 classify
# ---------------------------------------------------------------------------


@main.command()
@click.option("--input", "-i", "input_fasta", required=True, type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    "output_tsv",
    required=True,
    type=click.Path(),
    help="Destination TSV file for predictions.",
)
@click.option("--model", "model_path", default=None, type=click.Path())
@click.option("--threshold", default=0.7, show_default=True)
@click.option("--min-length", "min_length", default=1000, show_default=True)
@click.pass_context
def classify(
    ctx: click.Context,
    input_fasta: str,
    output_tsv: str,
    model_path: str | None,
    threshold: float,
    min_length: int,
) -> None:
    """Classify sequences and write per-sequence predictions to TSV."""
    resolved_model = _resolve_model(model_path)

    records = load_fasta(input_fasta, min_length=min_length)
    if not records:
        click.echo(f"No sequences pass min_length={min_length} — nothing to classify.", err=True)
        return

    predictions = predict(
        [str(r.seq) for r in records],
        [r.id for r in records],
        resolved_model,
        threshold=threshold,
    )

    out_path = Path(output_tsv)
    _write_predictions_tsv_simple(predictions, out_path)

    counts: dict[str, int] = {}
    for p in predictions:
        counts[p.label] = counts.get(p.label, 0) + 1
    summary = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    click.echo(f"Classified {len(predictions)} sequences — {summary}")
    click.echo(f"Predictions → {out_path}")


# ---------------------------------------------------------------------------
# plasflow2 annotate
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--input",
    "-i",
    "input_fasta",
    required=True,
    type=click.Path(exists=True),
    help="Plasmid sequences FASTA (output of 'classify' or 'run').",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Output directory for intermediate files and annotations.json.",
)
@click.option("--card-db", default=None, type=click.Path())
@click.option("--aro-index", default=None, type=click.Path())
@click.option("--threads", default=8, show_default=True)
@click.option(
    "--skip-mobility",
    is_flag=True,
    default=False,
    help="Skip mob_typer mobility typing.",
)
@click.option(
    "--sarg-db",
    "sarg_db",
    default=None,
    type=click.Path(),
    help="DIAMOND database (.dmnd) built from SARG for dual-DB ARG annotation.",
)
@click.pass_context
def annotate(
    ctx: click.Context,
    input_fasta: str,
    output_dir: str,
    card_db: str | None,
    aro_index: str | None,
    threads: int,
    skip_mobility: bool,
    sarg_db: str | None,
) -> None:
    """Annotate plasmid sequences with ARGs (DIAMOND/CARD+SARG) and mobility (MOB-suite)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    card_db_path = Path(card_db) if card_db else _DEFAULT_CARD_DB
    aro_index_path = Path(aro_index) if aro_index else _DEFAULT_ARO_INDEX

    for p, name in [(card_db_path, "--card-db"), (aro_index_path, "--aro-index")]:
        if not p.exists():
            raise click.BadParameter(f"Not found: {p}", param_hint=name)

    db_label = "CARD + SARG" if sarg_db else "CARD"
    click.echo(f"Annotating ARGs on {input_fasta} ({db_label}) …")
    arg_hits = annotate_contigs(
        fasta_path=input_fasta,
        card_db=card_db_path,
        aro_index_path=aro_index_path,
        work_dir=out / "arg_work",
        threads=threads,
        sarg_db=sarg_db,
    )
    card_n = sum(1 for h in arg_hits if h.source == "CARD")
    sarg_n = sum(1 for h in arg_hits if h.source == "SARG")
    click.echo(f"  {len(arg_hits)} ARG hits detected (CARD: {card_n}, SARG: {sarg_n})")

    mobility_results = []
    if not skip_mobility:
        click.echo("Running mob_typer …")
        try:
            mobility_results = annotate_mobility(
                plasmid_fasta=input_fasta,
                work_dir=out / "mob_work",
                threads=threads,
            )
            click.echo(f"  {len(mobility_results)} mobility results")
        except (FileNotFoundError, RuntimeError) as exc:
            click.echo(f"  mob_typer unavailable: {exc} — skipping.", err=True)

    # Build a minimal annotation dict keyed by contig_id
    args_by_contig: dict[str, list] = {}
    for h in arg_hits:
        args_by_contig.setdefault(h.contig_id, []).append(h)
    mob_by_contig = {m.contig_id: m for m in mobility_results}

    all_contigs = sorted(set(args_by_contig) | set(mob_by_contig))
    records_out = []
    for cid in all_contigs:
        mob = mob_by_contig.get(cid)
        hits = args_by_contig.get(cid, [])
        records_out.append(
            {
                "contig_id": cid,
                "mobility": (
                    {
                        "mobility_class": mob.mobility_class if mob else "unknown",
                        "replicon_type": mob.replicon_type if mob else "unknown",
                        "relaxase_type": mob.relaxase_type if mob else "none",
                        "mpf_type": mob.mpf_type if mob else "none",
                    }
                    if mob
                    else None
                ),
                "arg_hits": [
                    {
                        "gene_name": h.gene_name,
                        "aro_accession": h.aro_accession,
                        "amr_family": h.amr_family,
                        "drug_class": h.drug_class,
                        "resistance_mechanism": h.resistance_mechanism,
                        "identity": h.identity,
                        "coverage": h.coverage,
                        "evalue": h.evalue,
                        "source": h.source,
                    }
                    for h in hits
                ],
            }
        )

    ann_json = out / "annotations.json"
    with open(ann_json, "w") as fh:
        json.dump(records_out, fh, indent=2)
    click.echo(f"Annotations → {ann_json}")


# ---------------------------------------------------------------------------
# plasflow2 report
# ---------------------------------------------------------------------------


@main.command("report")
@click.option(
    "--predictions",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="all_predictions.tsv (or legacy predictions.tsv) produced by 'plasflow2 run'.",
)
@click.option(
    "--output",
    "-o",
    "output_html",
    required=True,
    type=click.Path(),
    help="Output HTML file path.",
)
@click.option(
    "--annotations",
    "-a",
    default=None,
    type=click.Path(exists=False),
    hidden=True,
    help="[Deprecated] annotations.json — ignored; all data now comes from predictions.tsv.",
)
@click.option(
    "--context",
    default="unspecified",
    type=click.Choice(
        ["clinical", "wastewater", "environmental", "unspecified"], case_sensitive=False
    ),
)
@click.pass_context
def report_cmd(
    ctx: click.Context,
    predictions: str,
    output_html: str,
    annotations: str | None,
    context: str,
) -> None:
    """Regenerate an HTML report from all_predictions.tsv (no pipeline re-run needed).

    The all_predictions.tsv produced by 'plasflow2 run' contains all annotations
    (ARGs, mobility, risk scores, taxonomy, ESKAPE host) for every contig.
    This command reads that file and rebuilds the full interactive report.
    """
    from plasflow2.annotate.args import ARGHit
    from plasflow2.annotate.mobility import MobilityResult
    from plasflow2.annotate.taxonomy import TaxResult as _TaxResult
    from plasflow2.report.generator import NonPlasmidRow

    if annotations:
        click.echo(
            "Note: --annotations is deprecated. All data is now read from predictions.tsv.",
            err=True,
        )

    plasmid_rows: list[PlasmidRow] = []
    non_plasmid_rows: list[NonPlasmidRow] = []
    all_arg_hits_for_chart: list[ARGHit] = []
    risk_scores: list[int] = []
    class_counts: dict[str, int] = {}
    total = 0

    with open(predictions) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            total += 1
            label = row.get("label", "unclassified")
            class_counts[label] = class_counts.get(label, 0) + 1
            cid = row["contig_id"]
            length = int(row.get("length", 0) or 0)
            confidence = float(row.get("confidence", 0) or 0)

            # Taxonomy
            tax_display = row.get("taxonomy", "") or "—"
            tax_rank = row.get("taxonomy_rank", "") or ""
            tax_lineage = row.get("taxonomy_lineage", "") or ""
            tax_obj: _TaxResult | None = None
            if tax_rank and tax_rank not in ("", "unclassified"):
                tax_obj = _TaxResult(
                    contig_id=cid,
                    lineage=tax_lineage,
                    rank=tax_rank,
                    taxon=row.get("taxonomy", ""),
                    num_hits=int(row.get("taxonomy_num_hits", 0) or 0),
                    agreement=float(row.get("taxonomy_agreement", 0) or 0),
                )

            if label == "plasmid":
                # Reconstruct ARGHit objects for the drug-class chart
                # (stored compactly in TSV; we rebuild from aggregate columns)
                drug_classes_str = row.get("drug_classes", "") or ""
                arg_sources_str = row.get("arg_sources", "") or ""
                num_args = int(row.get("num_args", 0) or 0)
                # Create one synthetic ARGHit per drug class for charting purposes
                for dc in drug_classes_str.split(";"):
                    dc = dc.strip()
                    if dc:
                        all_arg_hits_for_chart.append(
                            ARGHit(
                                contig_id=cid,
                                gene_name="",
                                aro_accession="",
                                amr_family="",
                                drug_class=dc,
                                resistance_mechanism="",
                                identity=100.0,
                                coverage=100.0,
                                evalue=0.0,
                                source=arg_sources_str.split(",")[0].strip() or "CARD",
                            )
                        )

                mob_class = row.get("mobility_class", "") or "unknown"
                rep_type = row.get("replicon_type", "") or "unknown"
                risk_score = int(row.get("risk_score", 0) or 0)
                risk_scores.append(risk_score)

                mob = (
                    MobilityResult(
                        contig_id=cid,
                        mobility_class=mob_class,
                        replicon_type=rep_type,
                        relaxase_type=row.get("relaxase_type", "") or "none",
                        mpf_type=row.get("mpf_type", "") or "none",
                    )
                    if mob_class
                    else None
                )

                risk = score_plasmid(cid, mob, [], context, tax_obj)
                # Override computed risk with stored values from TSV (pipeline already scored)
                risk.score = risk_score
                risk.mobility_score = int(row.get("mobility_score", 0) or 0)
                risk.arg_score = int(row.get("arg_score", 0) or 0)
                risk.replicon_score = int(row.get("replicon_score", 0) or 0)
                risk.context_score = int(row.get("context_score", 0) or 0)
                risk.host_score = int(row.get("host_score", 0) or 0)
                risk.eskape_host = row.get("eskape_host", "False").lower() == "true"
                risk.eskape_genus = row.get("eskape_genus", "") or ""
                risk.evidence = [
                    e.strip() for e in row.get("risk_evidence", "").split(";") if e.strip()
                ]

                # ARG gene names (new column; absent in old TSV files)
                arg_genes_str = row.get("arg_genes", "") or ""
                # VF / MGE — read from TSV (gracefully absent in old files)
                num_vf        = int(row.get("num_vf",  0) or 0)
                vf_genes_str  = row.get("vf_genes",     "") or ""
                num_mge       = int(row.get("num_mge", 0) or 0)
                mge_genes_str = row.get("mge_genes", "") or ""
                mge_fam_str   = row.get("mge_families", "") or ""

                plasmid_rows.append(
                    PlasmidRow(
                        contig_id=cid,
                        contig_length=length,
                        confidence=confidence,
                        num_args=num_args,
                        arg_genes=arg_genes_str,
                        drug_classes=drug_classes_str if drug_classes_str else "—",
                        mobility_class=mob_class,
                        replicon_type=rep_type,
                        risk_score=risk_score,
                        taxonomy=tax_display,
                        risk_evidence=row.get("risk_evidence", "") or "—",
                        arg_sources=arg_sources_str,
                        eskape_host=risk.eskape_host,
                        eskape_genus=risk.eskape_genus,
                        num_vf=num_vf,
                        vf_genes=vf_genes_str,
                        num_mge=num_mge,
                        mge_genes=mge_genes_str,
                        mge_families=mge_fam_str,
                        topology=row.get("topology", "linear") or "linear",
                        low_confidence=(row.get("low_confidence", "False") or "False").lower() == "true",
                    )
                )
            else:
                scores = {
                    k: float(row.get(f"{k}_score", 0) or 0)
                    for k in ("plasmid", "chromosome", "phage", "archaea")
                }
                best_label = max(scores, key=scores.get) if scores else ""
                best_score = scores.get(best_label, 0.0) if best_label else 0.0
                non_plasmid_rows.append(
                    NonPlasmidRow(
                        contig_id=cid,
                        contig_length=length,
                        label=label,
                        confidence=confidence,
                        taxonomy=tax_display,
                        taxonomy_lineage=tax_lineage,
                        best_label=best_label,
                        best_score=best_score,
                        # ARG/VF/MGE — universal columns (present for all contig classes)
                        num_args=int(row.get("num_args", 0) or 0),
                        arg_genes=row.get("arg_genes", "") or "",
                        drug_classes=row.get("drug_classes", "") or "",
                        arg_sources=row.get("arg_sources", "") or "",
                        num_vf=int(row.get("num_vf", 0) or 0),
                        vf_genes=row.get("vf_genes", "") or "",
                        num_mge=int(row.get("num_mge", 0) or 0),
                        mge_genes=row.get("mge_genes", "") or "",
                        mge_families=row.get("mge_families", "") or "",
                        topology=row.get("topology", "linear") or "linear",
                        low_confidence=(row.get("low_confidence", "False") or "False").lower() == "true",
                    )
                )

    from plasflow2.report.generator import (
        _vf_bar as _build_vf_bar,
        _mge_bar as _build_mge_bar,
        _pathogen_bar as _build_pathogen_bar,
        _build_drug_cooccurrence_heatmap,
        _np_charts as _build_np_charts,
        _narrative_summary,
    )

    phage_rows        = [r for r in non_plasmid_rows if r.label == "phage"]
    chromosome_rows   = [r for r in non_plasmid_rows if r.label == "chromosome"]
    archaea_rows      = [r for r in non_plasmid_rows if r.label == "archaea"]
    unclassified_rows = [r for r in non_plasmid_rows
                         if r.label not in ("phage", "chromosome", "archaea")]
    # sort large lists by length so top-N truncation keeps longest contigs
    for lst in (chromosome_rows, unclassified_rows):
        lst.sort(key=lambda r: r.contig_length, reverse=True)

    # Build pathogen data from the pathogen_threat column in predictions.tsv
    from plasflow2.annotate.pathogens import PathogenResult as _PR
    _pathogens_from_tsv: dict[str, _PR] = {}
    # Re-read pathogens from the TSV (they were written as pathogen_species/threat/category)
    # We need to reparse since the tsv loop above doesn't track these
    # (simple rebuild from the row loop below)
    report_data = {
        "input_file":        predictions,
        "total":             total,
        "num_plasmids":      len(plasmid_rows),
        "total_args":        len(all_arg_hits_for_chart),
        "total_vf":          sum(r.num_vf  for r in plasmid_rows),
        "total_mge":         sum(r.num_mge for r in plasmid_rows),
        "tax_classified":    0,
        "total_pathogens":   0,  # filled below after re-reading pathogen cols
        "class_counts":      class_counts,
        # plasmid charts
        "pie_data":          _build_pie_data(class_counts),
        "arg_data":          _build_arg_chart(all_arg_hits_for_chart),
        "risk_data":         _build_risk_histogram(risk_scores),
        "vf_data":           _build_vf_bar(plasmid_rows),
        "mge_data":          _build_mge_bar(plasmid_rows),
        "mobility_data":     _build_mobility_bar(plasmid_rows),
        "eskape_data":       _build_eskape_bar(plasmid_rows),
        "pathogen_data":     _build_pathogen_bar({}),  # populated below
        "cooccurrence_data": _build_drug_cooccurrence_heatmap([]),
        "scatter_data":      {},
        "tax_bar_data":      {},
        # row lists
        "plasmid_rows":      plasmid_rows,
        "chromosome_rows":   chromosome_rows,
        "phage_rows":        phage_rows,
        "archaea_rows":      archaea_rows,
        "unclassified_rows": unclassified_rows,
        "other_rows":        archaea_rows + unclassified_rows,
        # per-class chart bundles
        "chrom_charts":      _build_np_charts(chromosome_rows,   "Chromosome",   "#27ae60"),
        "phage_charts":      _build_np_charts(phage_rows,        "Phage",        "#e67e22"),
        "arch_charts":       _build_np_charts(archaea_rows,      "Archaea",      "#8e44ad"),
        "unc_charts":        _build_np_charts(unclassified_rows, "Unclassified", "#95a5a6",
                                              show_best=True),
        # legacy flags
        "has_scatter":       False,
        "has_cooccurrence":  False,
        "has_phages":        bool(phage_rows),
        "has_chromosomes":   bool(chromosome_rows),
        "has_others":        bool(archaea_rows or unclassified_rows),
        # genome maps not available when rebuilding from TSV (no ORF data)
        "genome_maps":       {},
    }
    # narrative summary (computed after report_data is assembled)
    report_data["narrative"] = _narrative_summary(report_data)

    # Re-read pathogen columns from predictions.tsv and build pathogen chart
    _pathogen_hits: dict[str, _PR] = {}
    with open(predictions) as _pfh:
        _pr = csv.DictReader(_pfh, delimiter="\t")
        for _row in _pr:
            _sp  = _row.get("pathogen_species", "")
            _lv  = _row.get("pathogen_threat", "")
            _cat = _row.get("pathogen_category", "")
            if _sp and _lv:
                _pathogen_hits[_row["contig_id"]] = _PR(
                    contig_id=_row["contig_id"], genus=_sp.split()[0],
                    species=_sp, threat_level=_lv, category=_cat, note="",
                )
    if _pathogen_hits:
        report_data["pathogen_data"]   = _build_pathogen_bar(_pathogen_hits)
        report_data["total_pathogens"] = len(_pathogen_hits)

    import os as _os
    out_dir = _os.path.dirname(_os.path.abspath(output_html))
    report_paths = generate_reports(report_data, out_dir)
    click.echo("Reports written:")
    for key in ("plasmid", "chromosome", "phage", "archaea", "unclassified"):
        p = report_paths.get(key)
        if p:
            click.echo(f"  {key:<14} → {p}")


# ---------------------------------------------------------------------------
# plasflow2 setup
# ---------------------------------------------------------------------------

_SETUP_TEXT = """
PlasFlow v2 — External Dependency Setup
========================================

PlasFlow v2 requires the following external tools and databases.
Run the commands below once to get everything ready.

─────────────────────────────────────────
1. PYTHON DEPENDENCIES  (pip / Poetry)
─────────────────────────────────────────
    pip install poetry
    poetry install          # installs plasflow2 + all Python deps

─────────────────────────────────────────
2. SYSTEM TOOLS  (conda recommended)
─────────────────────────────────────────
    # DIAMOND  — ARG annotation + taxonomy search
    conda install -c bioconda diamond

    # MOB-suite — plasmid mobility typing
    conda install -c conda-forge -c bioconda mob_suite

    # Prodigal  — ORF prediction (Python wrapper bundled)
    # Already installed via:  pip install pyrodigal

─────────────────────────────────────────
3. CARD DATABASE  (ARG annotation)
─────────────────────────────────────────
    mkdir -p data/databases/card
    cd data/databases/card

    # Download the latest CARD data bundle:
    wget https://card.mcmaster.ca/latest/data -O card.tar.bz2

    # Extract and build DIAMOND database:
    python -c "
    from plasflow2.annotate.args import setup_card_db
    setup_card_db('data/databases/card')
    "

    # Expected output:
    #   data/databases/card/card.dmnd
    #   data/databases/card/aro_index.tsv

─────────────────────────────────────────
4. GTDB DATABASE  (taxonomy annotation)
─────────────────────────────────────────
    mkdir -p data/databases/gtdb
    cd data/databases/gtdb

    # Download GTDB-r220 representative protein sequences (~2 GB):
    wget https://data.ace.uq.edu.au/public/gtdb/data/releases/release220/220.0/\\
         genomic_files_reps/gtdb_proteins_aa_reps_r220.tar.gz

    tar xf gtdb_proteins_aa_reps_r220.tar.gz

    # Build DIAMOND protein database:
    diamond makedb \\
        --in gtdb_proteins_aa_reps_r220/gtdb_proteins_aa_reps_r220.faa \\
        -d data/databases/gtdb/gtdb_r220 \\
        --threads 8

    # Download GTDB taxonomy file and build accession→lineage map:
    wget https://data.ace.uq.edu.au/public/gtdb/data/releases/release220/220.0/\\
         bac120_taxonomy_r220.tsv.gz
    gunzip bac120_taxonomy_r220.tsv.gz

    python -c "
    from plasflow2.annotate.taxonomy import build_gtdb_taxon_map
    build_gtdb_taxon_map(
        'data/databases/gtdb/bac120_taxonomy_r220.tsv',
        'data/databases/gtdb/taxon_map.tsv'
    )
    "

    # Expected output:
    #   data/databases/gtdb/gtdb_r220.dmnd
    #   data/databases/gtdb/taxon_map.tsv

─────────────────────────────────────────
5. RUN THE FULL PIPELINE
─────────────────────────────────────────
    plasflow2 run \\
      --input      assembly.fasta \\
      --output     results/ \\
      --card-db    data/databases/card/card.dmnd \\
      --aro-index  data/databases/card/aro_index.tsv \\
      --taxonomy-db data/databases/gtdb/gtdb_r220.dmnd \\
      --taxon-map  data/databases/gtdb/taxon_map.tsv \\
      --context    wastewater \\
      --threads    8

    # Skip optional steps when databases are unavailable:
    plasflow2 run --input assembly.fasta --output results/ \\
      --skip-mobility --skip-taxonomy

─────────────────────────────────────────
6. CLASSIFY ONLY (no external databases needed)
─────────────────────────────────────────
    plasflow2 classify \\
      --input  assembly.fasta \\
      --output predictions.tsv

─────────────────────────────────────────
Tip: Run 'plasflow2 --help' for all commands and options.
"""


@main.command("setup")
def setup_cmd() -> None:
    """Print installation instructions for all external dependencies.

    Covers: Python deps, DIAMOND, MOB-suite, CARD database, GTDB database,
    and example commands for the full pipeline.
    """
    click.echo(_SETUP_TEXT)


if __name__ == "__main__":
    main()
