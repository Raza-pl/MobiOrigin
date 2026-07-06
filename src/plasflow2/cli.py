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

# ── macOS ARM segfault fix — MUST be before any numpy/torch import ──────────
import os as _os

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_v, "1")
# ────────────────────────────────────────────────────────────────────────────

import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402

from plasflow2 import __version__  # noqa: E402
from plasflow2.annotate.args import annotate_contigs  # noqa: E402
from plasflow2.annotate.mobility import annotate_mobility  # noqa: E402
from plasflow2.classify.predict import predict  # noqa: E402
from plasflow2.output.genes_tsv import write_genes_tsv  # noqa: E402
from plasflow2.pipeline import PipelineResult, run_pipeline  # noqa: E402
from plasflow2.report.generator import (  # noqa: E402
    NonPlasmidRow,
    PlasmidRow,
    build_report_data,
    generate_reports,
)
from plasflow2.report.generator import (  # noqa: E402
    _arg_bar as _build_arg_chart,
)
from plasflow2.report.generator import (  # noqa: E402
    _eskape_bar as _build_eskape_bar,
)
from plasflow2.report.generator import (  # noqa: E402
    _mobility_bar as _build_mobility_bar,
)
from plasflow2.report.generator import (  # noqa: E402
    _pie as _build_pie_data,
)
from plasflow2.report.generator import (  # noqa: E402
    _risk_hist as _build_risk_histogram,
)
from plasflow2.risk.scorer import score_plasmid  # noqa: E402
from plasflow2.utils.fasta import load_fasta, split_by_label, write_fasta  # noqa: E402

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
_DEFAULT_MODEL = Path(__file__).parent.parent.parent / "data" / "models" / "mlp_v2.pt"
_DEFAULT_MARKER_MODEL = Path(__file__).parent.parent.parent / "data" / "models" / "marker_xgb.pkl"

# Docker convention: models are mounted at /data/models/
# _resolve_model() falls back to these when repo-relative paths don't exist.
_DOCKER_MODEL = Path("/data/models/mlp_v2.pt")
_DOCKER_MARKER_MODEL = Path("/data/models/marker_xgb.pkl")

_DEFAULT_CARD_DB = _DB_ROOT / "card" / "card.dmnd"
_DEFAULT_ARO_INDEX = _DB_ROOT / "card" / "aro_index.tsv"
_DEFAULT_SARG_DB = _DB_ROOT / "sarg" / "sarg.dmnd"
_DEFAULT_VFDB = _DB_ROOT / "vfdb" / "vfdb.dmnd"
_DEFAULT_MGE_DB = _DB_ROOT / "mge" / "isfinder.dmnd"
_DEFAULT_PLASMID_DB = _DB_ROOT / "plasmids"  # dir; combined FASTA is built here
_DEFAULT_TAXONOMY_DB = _DB_ROOT / "taxonomy" / "refseq_taxonomy.dmnd"
_DEFAULT_TAXON_MAP = _DB_ROOT / "taxonomy" / "taxon_map.tsv"
_DEFAULT_KAIJU_DIR = _DB_ROOT / "kaiju"
_DEFAULT_KAIJU_NODES = _DEFAULT_KAIJU_DIR / "nodes.dmp"
_DEFAULT_KAIJU_NAMES = _DEFAULT_KAIJU_DIR / "names.dmp"


_DOCKER_MODEL = Path("/data/models/mlp_v2.pt")


def _resolve_model(model_path: str | None) -> Path:
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise click.BadParameter(f"Model file not found: {p}", param_hint="--model")
        return p
    # Check repo-relative path first, then Docker-convention path (/data/models/)
    for candidate in (_DEFAULT_MODEL, _DOCKER_MODEL):
        if candidate.exists():
            return candidate
    raise click.UsageError(
        "Model weights not found at data/models/mlp_v2.pt\n\n"
        "Download the pre-trained models by running:\n"
        "  bash scripts/setup_databases.sh\n\n"
        "Or if you only want the model files (skipping all databases):\n"
        "  bash scripts/setup_databases.sh \\\n"
        "    --skip-plsdb --skip-card --skip-sarg --skip-amrfinder \\\n"
        "    --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite\n\n"
        "Docker users: mount models with -v /path/to/data:/data\n"
        "To specify a custom model path: plasflow2 run --model /path/to/mlp_v2.pt ..."
    )


def _write_predictions_tsv(pipeline_result: PipelineResult, output_path: Path) -> None:
    """Write comprehensive per-contig results to TSV — all classes, all annotations.

    Columns
    -------
    All contigs:
        contig_id, length, label, confidence,
        plasmid_score, chromosome_score, phage_score,
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
        # ── prediction evidence (populated when marker XGBoost was used) ─
        "mlp_plasmid",  # raw MLP score before XGBoost blending
        "mlp_chromosome",
        "mlp_phage",
        "xgb_plasmid",  # XGBoost second-stage plasmid score
        "xgb_chromosome",
        "is_conjugative",  # biological markers from annotation TSV
        "is_mobilizable",
        "has_replicon",
        "has_ice",  # shown for users even though excluded from classification
        "has_rep_protein",
        "n_rep_per_kb",
        "evidence_type",  # mlp_only | xgb_blend | conjugative_override | hallmark_boost
        "taxonomy",
        "taxonomy_rank",
        "taxonomy_lineage",
        # ── ARG annotation (all contig classes) ──────────────────────────
        "num_args",
        "arg_genes",  # e.g. "blaNDM-1; sul1; tetA"
        "drug_classes",
        "arg_sources",
        # ── VF annotation (all contig classes) ───────────────────────────
        "num_vf",
        "vf_genes",
        # ── MGE annotation (all contig classes) ──────────────────────────
        "num_mge",
        "mge_genes",  # IS element names e.g. "ISAba1; IS26"
        "mge_families",  # IS families e.g. "IS4; Tn3"
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
        "topology",  # circular / linear / too_short
        "low_confidence",  # True if confidence < 0.70 or argmax fallback used
        # ── plasmid-DB nucleotide match (plasmid contigs only) ────────────
        "plasmid_db_match",  # closest known plasmid accession (e.g. PLSDB_NZ_CP073379.1)
        "plasmid_db_source",  # PLSDB / RefSeq / COMPASS
        "plasmid_db_ani",  # approximate nucleotide identity % to DB hit
        "plasmid_db_cov",  # query coverage % of the DB hit alignment
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
    plasmid_by_id = {cr.record.id: cr for cr in pipeline_result.plasmid_results}
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
            mlp = pred.mlp_scores or {}
            xgb = pred.xgb_scores or {}
            bio = pred.bio_evidence or {}
            base_cols = [
                cid,
                length,
                pred.label,
                f"{pred.confidence:.4f}",
                f"{scores.get('plasmid', 0):.4f}",
                f"{scores.get('chromosome', 0):.4f}",
                f"{scores.get('phage', 0):.4f}",
                # Evidence columns
                f"{mlp['plasmid']:.4f}" if "plasmid" in mlp else "",
                f"{mlp['chromosome']:.4f}" if "chromosome" in mlp else "",
                f"{mlp['phage']:.4f}" if "phage" in mlp else "",
                f"{xgb['plasmid']:.4f}" if "plasmid" in xgb else "",
                f"{xgb['chromosome']:.4f}" if "chromosome" in xgb else "",
                int(bio.get("is_conjugative", 0)) if bio else "",
                int(bio.get("is_mobilizable", 0)) if bio else "",
                int(bio.get("has_replicon", 0)) if bio else "",
                int(bio.get("has_ice", 0)) if bio else "",
                int(bio.get("has_rep_protein", 0)) if bio else "",
                f"{bio.get('n_rep_per_kb', 0):.3f}" if bio else "",
                pred.evidence_type or "",
            ]

            # ── Plasmid: full annotation columns ─────────────────────────
            if pred.label == "plasmid" and cid in plasmid_by_id:
                cr = plasmid_by_id[cid]
                tax_cols = _tax_fields(cr.taxonomy)

                # ARG
                arg_genes = (
                    "; ".join(sorted({h.gene_name for h in cr.arg_hits})) if cr.arg_hits else ""
                )
                unique_classes = sorted(
                    {
                        dc.strip()
                        for h in cr.arg_hits
                        for dc in h.drug_class.split(";")
                        if dc.strip() and dc.strip() != "unknown"
                    }
                )
                sources = sorted({h.source for h in cr.arg_hits if getattr(h, "source", "")})
                # VF
                vf_hits = getattr(cr, "vf_hits", [])
                vf_genes = "; ".join(sorted({h.gene_name for h in vf_hits})) if vf_hits else ""
                # MGE
                mge_hits = getattr(cr, "mge_hits", [])
                mge_genes = "; ".join(sorted({h.is_name for h in mge_hits})) if mge_hits else ""
                mge_families = (
                    "; ".join(sorted({h.is_family for h in mge_hits})) if mge_hits else ""
                )
                # BacMet
                bm_hits = getattr(cr, "bacmet_hits", [])
                bm_genes = "; ".join(sorted({h.gene_name for h in bm_hits})) if bm_hits else ""
                bm_classes = (
                    "; ".join(sorted({h.resistance_class for h in bm_hits})) if bm_hits else ""
                )
                bm_compounds = (
                    "; ".join(sorted({h.compound for h in bm_hits if h.compound}))
                    if bm_hits
                    else ""
                )
                # ICE
                ice_hits_cr = getattr(cr, "ice_hits", [])
                ice_ids_str = (
                    "; ".join(sorted({h.ice_id for h in ice_hits_cr})) if ice_hits_cr else ""
                )
                ice_funcs = (
                    "; ".join(sorted({h.gene_function for h in ice_hits_cr if h.gene_function}))
                    if ice_hits_cr
                    else ""
                )

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
                ice_cols = [len(ice_hits_cr), ice_ids_str, ice_funcs]

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
                    np_arg_hits = getattr(np_cr, "arg_hits", [])
                    np_vf_hits = getattr(np_cr, "vf_hits", [])
                    np_mge_hits = getattr(np_cr, "mge_hits", [])
                    np_bm_hits = getattr(np_cr, "bacmet_hits", [])
                    np_ice_hits = getattr(np_cr, "ice_hits", [])
                    np_arg_genes = (
                        "; ".join(sorted({h.gene_name for h in np_arg_hits})) if np_arg_hits else ""
                    )
                    np_drug_cls = sorted(
                        {
                            dc.strip()
                            for h in np_arg_hits
                            for dc in h.drug_class.split(";")
                            if dc.strip() and dc.strip() != "unknown"
                        }
                    )
                    np_sources = sorted({h.source for h in np_arg_hits if getattr(h, "source", "")})
                    np_vf_genes = (
                        "; ".join(sorted({h.gene_name for h in np_vf_hits})) if np_vf_hits else ""
                    )
                    np_mge_genes = (
                        "; ".join(sorted({h.is_name for h in np_mge_hits})) if np_mge_hits else ""
                    )
                    np_mge_fams = (
                        "; ".join(sorted({h.is_family for h in np_mge_hits})) if np_mge_hits else ""
                    )
                    annot_cols = [
                        len(np_arg_hits),
                        np_arg_genes,
                        "; ".join(np_drug_cls) if np_drug_cls else "",
                        ", ".join(np_sources) if np_sources else "",
                        len(np_vf_hits),
                        np_vf_genes,
                        len(np_mge_hits),
                        np_mge_genes,
                        np_mge_fams,
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
                        "; ".join(
                            sorted({h.gene_function for h in np_ice_hits if h.gene_function})
                        ),
                    ]
                else:
                    annot_cols = ANNOT_EMPTY
                    bacmet_cols = ["", "", "", ""]
                    ice_cols = ["", "", ""]

            # ── Topology & confidence flag ────────────────────────────────
            topology = pipeline_result.topology.get(cid, "")
            # low_confidence is set by the pipeline in two cases:
            #   (a) hallmark gate flagged (≥50 kb, no biological evidence)
            #   (b) confidence below threshold (weak model signal)
            low_confidence = pred.low_confidence or pred.confidence < 0.70
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
                base_cols
                + tax_cols
                + annot_cols
                + plasmid_cols
                + topo_conf_cols
                + plasmid_db_cols
                + pathogen_cols
                + bacmet_cols
                + ice_cols
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

    plasmid_by_id = {cr.record.id: cr for cr in pipeline_result.plasmid_results}
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
                arg_hits = cr.arg_hits or []
                vf_hits = getattr(cr, "vf_hits", []) or []
                mge_hits = getattr(cr, "mge_hits", []) or []
                bm_hits = getattr(cr, "bacmet_hits", []) or []
                ice_hits_cr = getattr(cr, "ice_hits", []) or []
                mob = cr.mobility
                mob_class = mob.mobility_class if mob else "unknown"
                risk_score = cr.risk.score
                tax = cr.taxonomy
                tax_lca = tax.lineage if tax else ""
                path_hit = pipeline_result.pathogens.get(cid)
                pathogen_cat = path_hit.category if path_hit else ""
                is_mobile = mob_class not in ("non-mobilizable", "unknown", "")
            else:
                np_cr = non_plasmid_by_id.get(cid)
                arg_hits = getattr(np_cr, "arg_hits", []) if np_cr else []
                vf_hits = getattr(np_cr, "vf_hits", []) if np_cr else []
                mge_hits = getattr(np_cr, "mge_hits", []) if np_cr else []
                bm_hits = getattr(np_cr, "bacmet_hits", []) if np_cr else []
                ice_hits_cr = getattr(np_cr, "ice_hits", []) if np_cr else []
                mob_class = ""
                risk_score = ""
                is_mobile = False
                tax = pipeline_result.taxonomy.get(cid)
                tax_lca = tax.lineage if tax else ""
                path_hit = pipeline_result.pathogens.get(cid)
                pathogen_cat = path_hit.category if path_hit else ""

            # Filter: include only if at least one annotation present
            if not (
                arg_hits
                or mge_hits
                or vf_hits
                or bm_hits
                or ice_hits_cr
                or is_mobile
                or pathogen_cat
            ):
                continue

            arg_genes = "; ".join(sorted({h.gene_name for h in arg_hits}))
            drug_classes = "; ".join(
                sorted(
                    {
                        dc.strip()
                        for h in arg_hits
                        for dc in h.drug_class.split(";")
                        if dc.strip() and dc.strip() != "unknown"
                    }
                )
            )
            mge_genes = "; ".join(sorted({h.is_name for h in mge_hits}))
            mge_families = "; ".join(sorted({h.is_family for h in mge_hits}))
            vf_genes = "; ".join(sorted({h.gene_name for h in vf_hits}))
            vf_cats = "; ".join(
                sorted(
                    {
                        getattr(h, "vf_category", "")
                        for h in vf_hits
                        if getattr(h, "vf_category", "")
                    }
                )
            )
            bm_genes = "; ".join(sorted({h.gene_name for h in bm_hits}))
            bm_class = "; ".join(sorted({h.resistance_class for h in bm_hits}))
            bm_compounds = "; ".join(sorted({h.compound for h in bm_hits if h.compound}))
            ice_ids_str = "; ".join(sorted({h.ice_id for h in ice_hits_cr}))
            ice_funcs = "; ".join(sorted({h.gene_function for h in ice_hits_cr if h.gene_function}))

            writer.writerow(
                [
                    cid,
                    pred.label,
                    arg_genes,
                    drug_classes,
                    mge_genes,
                    mge_families,
                    vf_genes,
                    vf_cats,
                    bm_genes,
                    bm_class,
                    bm_compounds,
                    ice_ids_str,
                    ice_funcs,
                    mob_class,
                    risk_score,
                    tax_lca,
                    pathogen_cat,
                ]
            )
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
    """PlasFlow v2 — metagenomic contig classifier and AMR risk scorer.

    Classifies contigs from metagenomic assemblies as plasmid, chromosome,
    or phage. Plasmid contigs are annotated with antibiotic resistance
    genes (ARGs), virulence factors, mobile genetic elements, mobility class
    (conjugative / mobilizable / non-mobilizable), and an AMR risk score (0-10).

    \b
    Typical workflow:
      plasflow2 run --input assembly.fasta --output results/ --threads 16

    \b
    Output files in results/:
      all_predictions.tsv    — every contig: label, scores, and all annotations
      plasmids.fasta         — extracted plasmid sequences
      report_plasmid.html    — interactive HTML report (open in browser)

    \b
    Quick classify (no databases needed, runs in seconds):
      plasflow2 classify --input assembly.fasta --output predictions.tsv

    \b
    Rebuild the HTML report from a saved TSV (no re-run needed):
      plasflow2 report --predictions results/all_predictions.tsv --output results/

    Run 'plasflow2 COMMAND --help' for details on any command.
    """
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
    help="Input assembly FASTA (.fasta / .fa / .fna / .gz / .bz2). Accepts compressed files.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Output directory. Created automatically if it does not exist.",
)
@click.option(
    "--model",
    "model_path",
    default=None,
    type=click.Path(),
    help=(
        "Path to MLP model weights (.pt). "
        "Auto-detected from data/models/mlp_v2.pt when not specified. "
        "Download via: bash scripts/setup_databases.sh"
    ),
)
@click.option(
    "--card-db",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND CARD database (.dmnd) for antibiotic resistance gene annotation. "
        "Auto-detected from data/databases/card/card.dmnd. "
        "Download via: bash scripts/setup_databases.sh"
    ),
)
@click.option(
    "--aro-index",
    default=None,
    type=click.Path(),
    help=(
        "CARD ARO index TSV (maps protein accessions to gene names and drug classes). "
        "Auto-detected from data/databases/card/aro_index.tsv."
    ),
)
@click.option(
    "--threshold",
    default=0.7,
    show_default=True,
    help=(
        "Minimum confidence score (0-1) to assign a label (plasmid, chromosome, or phage). "
        "Contigs below this threshold are labelled 'unclassified' rather than forced into a class. "
        "Lower values (e.g. 0.5) assign more contigs but increase misclassification."
    ),
)
@click.option(
    "--plasmid-threshold",
    "plasmid_threshold",
    default=0.95,
    show_default=True,
    help=(
        "Minimum score (0-1) to call a contig as plasmid. Set higher than --threshold (default 0.95) "
        "to compensate for class-prior imbalance — plasmids are rare in typical metagenomes. "
        "Lower to 0.80-0.90 if you expect a high plasmid fraction (e.g. plasmid-enriched samples)."
    ),
)
@click.option(
    "--context",
    default="unspecified",
    show_default=True,
    type=click.Choice(
        ["clinical", "wastewater", "environmental", "unspecified"], case_sensitive=False
    ),
    help=(
        "Sample source context for AMR risk scoring. "
        "Adds bonus points to the risk score: clinical (+3), wastewater (+2), environmental (+1). "
        "Use 'clinical' for hospital or patient-derived samples; "
        "'wastewater' for WWTP or sewage; 'environmental' for soil, water, etc."
    ),
)
@click.option(
    "--threads",
    default=8,
    show_default=True,
    help="CPU threads for DIAMOND searches and MOB-suite. More threads = faster annotation.",
)
@click.option(
    "--min-length",
    default=1000,
    show_default=True,
    help=(
        "Minimum contig length in base pairs. Shorter contigs are discarded before classification. "
        "1000 bp is the recommended minimum for reliable k-mer composition features."
    ),
)
@click.option(
    "--min-confidence",
    "min_confidence",
    default=None,
    type=float,
    help=(
        "When set, contigs below --threshold receive their best-guess label instead of 'unclassified'. "
        "Useful for maximising the number of classified contigs without retraining. "
        "Example: --min-confidence 0.50 assigns every contig to its most-likely class. "
        "Does not affect high-confidence calls."
    ),
)
@click.option(
    "--skip-mobility",
    is_flag=True,
    default=False,
    help=(
        "Skip MOB-suite mobility typing. "
        "Use when mob_typer is not installed or you only need ARG annotation. "
        "mobility_class, replicon_type, and relaxase_type columns will be empty."
    ),
)
@click.option(
    "--taxonomy-db",
    "taxonomy_db",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND protein database for host taxonomy prediction. "
        "Built from GTDB r220 or RefSeq representative genomes. "
        "Auto-detected from data/databases/taxonomy/refseq_taxonomy.dmnd."
    ),
)
@click.option(
    "--taxon-map",
    "taxon_map",
    default=None,
    type=click.Path(),
    help=(
        "TSV mapping sequence accession → GTDB lineage string. "
        "Improves LCA (lowest common ancestor) accuracy for taxonomy calls. "
        "Auto-detected from data/databases/taxonomy/taxon_map.tsv."
    ),
)
@click.option(
    "--skip-taxonomy",
    is_flag=True,
    default=False,
    help=(
        "Skip host taxonomy annotation. "
        "Saves 20-40 min on large datasets when no taxonomy database is available. "
        "taxonomy, taxonomy_rank, and taxonomy_lineage columns will be empty."
    ),
)
@click.option(
    "--lenient",
    is_flag=True,
    default=False,
    help=(
        "Disable the hallmark gate. Accept all MLP plasmid predictions directly, "
        "without requiring biological evidence (PLSDB match, relaxase, replicon, ICE, or rep protein). "
        "Useful when databases are not available or for exploratory runs. "
        "Increases sensitivity but also false positives."
    ),
)
@click.option(
    "--taxonomy-engine",
    "taxonomy_engine",
    default="auto",
    type=click.Choice(["auto", "kaiju", "diamond"], case_sensitive=False),
    help=(
        "Engine for host taxonomy annotation. "
        "'auto' uses Kaiju if installed (20-50x faster), otherwise falls back to DIAMOND. "
        "'kaiju' forces Kaiju (requires kaiju DB in data/databases/kaiju/). "
        "'diamond' forces DIAMOND blastp (works with any protein DB). "
        "[default: auto]"
    ),
)
@click.option(
    "--kaiju-db",
    "kaiju_db",
    default=None,
    type=click.Path(),
    help="Kaiju FM-index database (.fmi). Auto-detected from data/databases/kaiju/.",
)
@click.option(
    "--kaiju-nodes",
    "kaiju_nodes",
    default=None,
    type=click.Path(),
    help="NCBI taxonomy nodes.dmp for Kaiju. Auto-detected from data/databases/kaiju/.",
)
@click.option(
    "--kaiju-names",
    "kaiju_names",
    default=None,
    type=click.Path(),
    help="NCBI taxonomy names.dmp for Kaiju. Auto-detected from data/databases/kaiju/.",
)
@click.option(
    "--sarg-db",
    "sarg_db",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database built from the SARG (Structured ARG Database). "
        "Auto-detected from data/databases/sarg/sarg.dmnd. "
        "When present, ARG annotation runs against CARD + SARG together; "
        "CARD hits take precedence and SARG fills in genes not covered by CARD."
    ),
)
@click.option(
    "--amrprot-db",
    "amrprot_db",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database built from the NCBI AMRFinderPlus protein FASTA. "
        "Auto-detected from data/databases/amrfinder/amrprot.dmnd. "
        "Hit priority per ORF: CARD > AMRFinderPlus > SARG. "
        "Provides a third independent ARG source for higher recall."
    ),
)
@click.option(
    "--min-identity",
    "min_identity",
    default=80.0,
    show_default=True,
    help=(
        "Minimum amino-acid identity (%%) for DIAMOND ARG hits to be reported. "
        "80%% is standard for metagenomic / environmental samples. "
        "Raise to 90%% for stricter clinical-isolate-grade precision."
    ),
)
@click.option(
    "--vfdb",
    "vfdb",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database built from VFDB set A (experimentally validated virulence factors). "
        "Auto-detected from data/databases/vfdb/vfdb.dmnd. "
        "When present, annotates every contig with virulence factor hits."
    ),
)
@click.option(
    "--mge-db",
    "mge_db",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database built from ISfinder / Pärnänen MGE proteins. "
        "Auto-detected from data/databases/mge/isfinder.dmnd. "
        "When present, detects IS elements, transposons, and integrons on all contigs."
    ),
)
@click.option(
    "--genomad-db",
    "genomad_db",
    default=None,
    type=click.Path(),
    help=(
        "geNomad database directory for gene-marker SPM features. "
        "Auto-detected from data/databases/genomad_db/ when genomad is installed. "
        "geNomad runs automatically when available — no extra step needed. "
        "Download with: genomad download-database data/databases/"
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
    genomad_db: str | None,
    lenient: bool,
) -> None:
    """Run the full pipeline: classify contigs, annotate plasmids, score AMR risk, write reports.

    All databases are auto-detected from data/databases/. Run setup first if you haven't:

        bash scripts/setup_databases.sh

    geNomad runs automatically when installed, adding 12 gene-marker SPM features
    to the XGBoost stage-2 classifier. Install with:
        conda install -c conda-forge -c bioconda genomad

    \b
    Output files written to OUTPUT_DIR:
        all_predictions.tsv       — every contig: label, scores, ARGs, mobility, risk, taxonomy
        annotated_predictions.tsv — filtered view: contigs that have ARGs, MGEs, VFs, or pathogens
        plasmids.fasta            — classified plasmid sequences (FASTA)
        chromosome.fasta          — classified chromosome sequences
        phage.fasta               — classified phage sequences
        genes.tsv                 — gene-level table: all ORFs with ARG/VF/MGE flags and coordinates
        annotations.json          — full ARG + mobility + risk evidence per plasmid (machine-readable)
        report_plasmid.html       — interactive plasmid report with charts and AMR risk summary
        report_chromosome.html    — chromosome contig report
        report_phage.html         — phage contig report
        report_unclassified.html  — unclassified contig report

    \b
    Examples:
        # Standard run (geNomad features added automatically if installed)
        plasflow2 run --input assembly.fasta --output results/ --threads 16

        # Clinical sample (adjusts AMR risk scoring)
        plasflow2 run --input assembly.fasta --output results/ --context clinical --threads 16

        # Skip taxonomy annotation to save 20-40 min on large datasets
        plasflow2 run --input assembly.fasta --output results/ --skip-taxonomy --threads 16
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resolved_model = _resolve_model(model_path)

    # CARD: only error if the user explicitly passed a path that doesn't exist.
    # If not provided, auto-detect — and skip ARG annotation gracefully if absent.
    _card_explicit = card_db is not None
    card_db_path = Path(card_db) if card_db else _DEFAULT_CARD_DB
    aro_index_path = Path(aro_index) if aro_index else _DEFAULT_ARO_INDEX

    if _card_explicit:
        for p, name in [(card_db_path, "--card-db"), (aro_index_path, "--aro-index")]:
            if not p.exists():
                raise click.BadParameter(f"Not found: {p}", param_hint=name)
    elif not card_db_path.exists():
        click.echo(
            "[warn] CARD database not found — ARG annotation will be skipped.\n"
            "       Run: bash scripts/setup_databases.sh --skip-plsdb --skip-sarg "
            "--skip-amrfinder --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite"
        )
        card_db_path = None  # type: ignore[assignment]
        aro_index_path = None  # type: ignore[assignment]

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
            click.echo(
                f"[info] Auto-detected Kaiju database: {kaiju_db} (will use kaiju for taxonomy)"
            )
        elif taxonomy_engine == "kaiju":
            click.echo(f"[info] Kaiju database: {kaiju_db}")

    # geNomad auto-detection message
    import shutil as _shutil_cli

    _gn_db_resolved = Path(genomad_db) if genomad_db else (_DB_ROOT / "genomad_db")
    if _gn_db_resolved.is_dir() and _shutil_cli.which("genomad"):
        click.echo(
            f"[info] geNomad database found — SPM features will be added automatically ({_gn_db_resolved})"
        )
    elif not _shutil_cli.which("genomad"):
        click.echo(
            "[info] genomad not found on PATH — XGBoost will run without SPM features. "
            "Install with: conda install -c conda-forge -c bioconda genomad"
        )

    click.echo(f"[PlasFlow v2 v{__version__}] Running pipeline on {input_fasta}")

    # --min-confidence: when set, use argmax fallback below this threshold.
    # The lower of (min_confidence, threshold / plasmid_threshold) becomes the
    # effective floor; anything above the class-specific threshold is still a
    # normal high-confidence call.
    argmax_fallback = min_confidence is not None
    # --min-confidence lowers the general class threshold but must NOT override
    # --plasmid-threshold which is intentionally set higher to reduce FP plasmids.
    effective_threshold = min(threshold, min_confidence) if argmax_fallback else threshold
    effective_plasmid_threshold = plasmid_threshold  # always honour explicitly set value
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
        genomad_db_path=genomad_db if genomad_db else None,
        lenient=lenient,
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
                input_fasta,
                _fallback,
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
        all_vf_hits = [h for cr in pipeline_result.plasmid_results for h in cr.vf_hits] + [
            h for cr in pipeline_result.non_plasmid_results for h in cr.vf_hits
        ]
        all_mge_hits = [h for cr in pipeline_result.plasmid_results for h in cr.mge_hits] + [
            h for cr in pipeline_result.non_plasmid_results for h in cr.mge_hits
        ]
        all_arg_hits = [h for cr in pipeline_result.plasmid_results for h in cr.arg_hits] + [
            h for cr in pipeline_result.non_plasmid_results for h in cr.arg_hits
        ]
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
    labels_order = ["plasmid", "chromosome", "phage", "unclassified"]
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
@click.option(
    "--input",
    "-i",
    "input_fasta",
    required=True,
    type=click.Path(exists=True),
    help="Input assembly FASTA (.fasta / .fa / .fna / .gz / .bz2).",
)
@click.option(
    "--output",
    "-o",
    "output_tsv",
    required=True,
    type=click.Path(),
    help="Output TSV file for per-contig predictions.",
)
@click.option(
    "--model",
    "model_path",
    default=None,
    type=click.Path(),
    help="MLP model weights (.pt). Auto-detected from data/models/mlp_v2.pt.",
)
@click.option(
    "--threshold",
    default=0.7,
    show_default=True,
    help=(
        "Minimum confidence score (0-1) to assign a label. "
        "Contigs below this are returned as 'unclassified'."
    ),
)
@click.option(
    "--plasmid-threshold",
    "plasmid_threshold",
    default=0.95,
    show_default=True,
    help=(
        "Minimum score to call a contig as plasmid (default 0.95). "
        "Set higher than --threshold to reduce false-positive plasmid calls."
    ),
)
@click.option(
    "--min-length",
    "min_length",
    default=1000,
    show_default=True,
    help="Minimum contig length in bp. Shorter contigs are skipped.",
)
@click.option(
    "--annotation-tsv",
    "annotation_tsv",
    default=None,
    type=click.Path(exists=True),
    help=(
        "Pre-computed annotation TSV with biological marker features. "
        "Enables XGBoost stage-2 blending, adding conjugation proteins, replicon type, "
        "and coding density on top of k-mer scores. "
        "Generate by running 'plasflow2 run' and providing the work/arg_annotation output."
    ),
)
@click.option(
    "--marker-model",
    "marker_model_path",
    default=None,
    type=click.Path(),
    help="XGBoost stage-2 model (.pkl). Auto-detected from data/models/marker_xgb.pkl.",
)
@click.option(
    "--threads",
    default=4,
    show_default=True,
    help="CPU threads for pyrodigal ORF prediction (used when stage-2 XGBoost is active).",
)
@click.option(
    "--no-marker-model",
    "no_marker_model",
    is_flag=True,
    default=False,
    help="Force MLP-only mode — disable XGBoost stage-2 even if marker_xgb.pkl is present.",
)
@click.pass_context
def classify(
    ctx: click.Context,
    input_fasta: str,
    output_tsv: str,
    model_path: str | None,
    threshold: float,
    plasmid_threshold: float,
    min_length: int,
    annotation_tsv: str | None,
    marker_model_path: str | None,
    threads: int,
    no_marker_model: bool,
) -> None:
    """Classify contigs as plasmid / chromosome / phage / unclassified.

    Fast mode (no databases, runs in seconds):

        plasflow2 classify --input assembly.fasta --output predictions.tsv

    \b
    Output TSV columns:
        contig_id, label, confidence, plasmid_score, chromosome_score, phage_score

    When --annotation-tsv is provided, additional evidence columns are added:
        mlp_plasmid, xgb_plasmid, is_conjugative, is_mobilizable, has_replicon, evidence_type

    For the full pipeline (annotation + reports), use 'plasflow2 run' instead.
    """
    resolved_model = _resolve_model(model_path)

    # Resolve marker model (auto-detect unless explicitly disabled)
    resolved_marker: str | None = None
    if not no_marker_model:
        if marker_model_path:
            resolved_marker = marker_model_path
        elif _DEFAULT_MARKER_MODEL.exists():
            resolved_marker = str(_DEFAULT_MARKER_MODEL)
        elif _DOCKER_MARKER_MODEL.exists():
            resolved_marker = str(_DOCKER_MARKER_MODEL)

    if resolved_marker:
        click.echo(f"Stage-2 marker XGBoost: {resolved_marker}")
    if annotation_tsv:
        click.echo(f"Annotation TSV: {annotation_tsv}")

    records = load_fasta(input_fasta, min_length=min_length)
    if not records:
        click.echo(f"No sequences pass min_length={min_length} — nothing to classify.", err=True)
        return

    predictions = predict(
        [str(r.seq) for r in records],
        [r.id for r in records],
        resolved_model,
        threshold=threshold,
        plasmid_threshold=plasmid_threshold,
        argmax_fallback=False,
        marker_model_path=resolved_marker,
        annotation_tsv=annotation_tsv,
        use_pyrodigal=bool(resolved_marker),
        marker_alpha_base=0.3,
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

    # CARD: only error if the user explicitly passed a path that doesn't exist.
    # If not provided, auto-detect — and skip ARG annotation gracefully if absent.
    _card_explicit = card_db is not None
    card_db_path = Path(card_db) if card_db else _DEFAULT_CARD_DB
    aro_index_path = Path(aro_index) if aro_index else _DEFAULT_ARO_INDEX

    if _card_explicit:
        for p, name in [(card_db_path, "--card-db"), (aro_index_path, "--aro-index")]:
            if not p.exists():
                raise click.BadParameter(f"Not found: {p}", param_hint=name)
    elif not card_db_path.exists():
        click.echo(
            "[warn] CARD database not found — ARG annotation will be skipped.\n"
            "       Run: bash scripts/setup_databases.sh --skip-plsdb --skip-sarg "
            "--skip-amrfinder --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite"
        )
        card_db_path = None  # type: ignore[assignment]
        aro_index_path = None  # type: ignore[assignment]

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
    """Regenerate the interactive HTML report from a saved all_predictions.tsv.

    Useful when you want to change the --context (e.g. re-score risk as 'clinical')
    or simply re-render the report without re-running the full pipeline.

    \b
    Example:
        plasflow2 report \\
            --predictions results/all_predictions.tsv \\
            --output      results/ \\
            --context     clinical
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
                num_vf = int(row.get("num_vf", 0) or 0)
                vf_genes_str = row.get("vf_genes", "") or ""
                num_mge = int(row.get("num_mge", 0) or 0)
                mge_genes_str = row.get("mge_genes", "") or ""
                mge_fam_str = row.get("mge_families", "") or ""

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
                        low_confidence=(row.get("low_confidence", "False") or "False").lower()
                        == "true",
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
                        low_confidence=(row.get("low_confidence", "False") or "False").lower()
                        == "true",
                    )
                )

    from plasflow2.report.generator import (
        _build_drug_cooccurrence_heatmap,
        _narrative_summary,
    )
    from plasflow2.report.generator import (
        _mge_bar as _build_mge_bar,
    )
    from plasflow2.report.generator import (
        _np_charts as _build_np_charts,
    )
    from plasflow2.report.generator import (
        _pathogen_bar as _build_pathogen_bar,
    )
    from plasflow2.report.generator import (
        _vf_bar as _build_vf_bar,
    )

    phage_rows = [r for r in non_plasmid_rows if r.label == "phage"]
    chromosome_rows = [r for r in non_plasmid_rows if r.label == "chromosome"]
    archaea_rows = [r for r in non_plasmid_rows if r.label == "archaea"]
    unclassified_rows = [
        r for r in non_plasmid_rows if r.label not in ("phage", "chromosome", "archaea")
    ]
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
        "input_file": predictions,
        "total": total,
        "num_plasmids": len(plasmid_rows),
        "total_args": len(all_arg_hits_for_chart),
        "total_vf": sum(r.num_vf for r in plasmid_rows),
        "total_mge": sum(r.num_mge for r in plasmid_rows),
        "tax_classified": 0,
        "total_pathogens": 0,  # filled below after re-reading pathogen cols
        "class_counts": class_counts,
        # plasmid charts
        "pie_data": _build_pie_data(class_counts),
        "arg_data": _build_arg_chart(all_arg_hits_for_chart),
        "risk_data": _build_risk_histogram(risk_scores),
        "vf_data": _build_vf_bar(plasmid_rows),
        "mge_data": _build_mge_bar(plasmid_rows),
        "mobility_data": _build_mobility_bar(plasmid_rows),
        "eskape_data": _build_eskape_bar(plasmid_rows),
        "pathogen_data": _build_pathogen_bar({}),  # populated below
        "cooccurrence_data": _build_drug_cooccurrence_heatmap([]),
        "scatter_data": {},
        "tax_bar_data": {},
        # row lists
        "plasmid_rows": plasmid_rows,
        "chromosome_rows": chromosome_rows,
        "phage_rows": phage_rows,
        "archaea_rows": archaea_rows,
        "unclassified_rows": unclassified_rows,
        "other_rows": archaea_rows + unclassified_rows,
        # per-class chart bundles
        "chrom_charts": _build_np_charts(chromosome_rows, "Chromosome", "#27ae60"),
        "phage_charts": _build_np_charts(phage_rows, "Phage", "#e67e22"),
        "arch_charts": _build_np_charts(archaea_rows, "Archaea", "#8e44ad"),
        "unc_charts": _build_np_charts(
            unclassified_rows, "Unclassified", "#95a5a6", show_best=True
        ),
        # legacy flags
        "has_scatter": False,
        "has_cooccurrence": False,
        "has_phages": bool(phage_rows),
        "has_chromosomes": bool(chromosome_rows),
        "has_others": bool(archaea_rows or unclassified_rows),
        # genome maps not available when rebuilding from TSV (no ORF data)
        "genome_maps": {},
    }
    # narrative summary (computed after report_data is assembled)
    report_data["narrative"] = _narrative_summary(report_data)

    # Re-read pathogen columns from predictions.tsv and build pathogen chart
    _pathogen_hits: dict[str, _PR] = {}
    with open(predictions) as _pfh:
        _pr = csv.DictReader(_pfh, delimiter="\t")
        for _row in _pr:
            _sp = _row.get("pathogen_species", "")
            _lv = _row.get("pathogen_threat", "")
            _cat = _row.get("pathogen_category", "")
            if _sp and _lv:
                _pathogen_hits[_row["contig_id"]] = _PR(
                    contig_id=_row["contig_id"],
                    genus=_sp.split()[0],
                    species=_sp,
                    threat_level=_lv,
                    category=_cat,
                    note="",
                )
    if _pathogen_hits:
        report_data["pathogen_data"] = _build_pathogen_bar(_pathogen_hits)
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
PlasFlow v2 — Setup Guide
==========================

The easiest way to install everything is with the one-command installer
from the repo root:

    bash install.sh

This creates a conda environment, installs all tools, and downloads
all databases and model weights automatically.

─────────────────────────────────────────
MANUAL SETUP (if you prefer step-by-step)
─────────────────────────────────────────

Step 1 — Create the conda environment (Python 3.10, all tools):

    conda env create -f environment.yml
    conda activate plasflow2

Step 2 — Install PlasFlow v2 Python package:

    pip install -e .

Step 3 — Download all databases and model weights:

    bash scripts/setup_databases.sh

    # Or skip databases you don't need:
    bash scripts/setup_databases.sh --skip-plsdb --skip-iceberg

    # Or point to databases you already have:
    bash scripts/setup_databases.sh \\
      --card-path  /existing/card/card.dmnd \\
      --plsdb-path /existing/PLSDB.fna

─────────────────────────────────────────
WHAT GETS INSTALLED
─────────────────────────────────────────

Tools (via conda):
    diamond       — DIAMOND protein aligner (ARG/VF/taxonomy annotation)
    minimap2      — nucleotide aligner (closest known plasmid matching)
    mob-suite     — plasmid mobility typing (conjugative/mobilizable)
    genomad       — optional, adds 12 gene-signature features to XGBoost

Databases (via setup_databases.sh):
    CARD          — antibiotic resistance genes (primary ARG source)
    SARG          — supplementary ARG database
    AMRFinderPlus — NCBI ARG database (third ARG source)
    VFDB set A    — experimentally validated virulence factors
    BacMet2       — biocide and metal resistance genes
    MGE/ISfinder  — IS elements, transposons, integrons
    ICEberg3      — integrative conjugative elements (optional)
    PLSDB         — curated plasmid sequences for closest-match lookup
    MOB-suite DBs — replicon/relaxase typing databases

Model weights:
    data/models/mlp_v2.pt       — MLP classifier (~79 MB)
    data/models/marker_xgb.pkl  — XGBoost stage-2 model (~1 MB)
    data/models/k6_pca.pkl      — PCA transform for feature compression (~4 MB)

─────────────────────────────────────────
QUICK RUN (after setup)
─────────────────────────────────────────

    # Full pipeline — all databases auto-detected
    plasflow2 run --input assembly.fasta --output results/ --threads 16

    # Clinical sample
    plasflow2 run --input assembly.fasta --output results/ --context clinical

    # Fast classification — no databases needed, runs in seconds
    plasflow2 classify --input assembly.fasta --output predictions.tsv

─────────────────────────────────────────
TROUBLESHOOTING
─────────────────────────────────────────

"No model weights found"
    Run: bash scripts/setup_databases.sh --skip-plsdb --skip-card ...
    (add --skip-X for each database you want to skip)

"mob_typer not found"
    Run: conda install -c bioconda -c conda-forge mob_suite
    Or:  pip install mob-suite && mob_init
    Or:  plasflow2 run ... --skip-mobility

Python/pytz conflict
    Use the conda environment from environment.yml — it pins pytz explicitly.
    Run bash install.sh to set this up automatically.

Apple Silicon (M1-M5)
    All packages in environment.yml have arm64 conda builds.
    If mob-suite fails: pip install mob-suite && mob_init

WSL Ubuntu
    If 'conda activate' has no effect: conda init bash, then restart terminal.

─────────────────────────────────────────
Run 'plasflow2 --help' or 'plasflow2 COMMAND --help' for usage details.
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
