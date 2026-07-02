"""End-to-end PlasFlow v2 pipeline.

Week 3 — Day 18 implementation.

Orchestrates:
    FASTA → classify (MLP) → [plasmid contigs]
                           → annotate ARGs (DIAMOND/CARD)
                           → annotate mobility (MOB-suite)
                           → risk score (scorer.py)
                           → PipelineResult

Typical usage:
    from plasflow2.pipeline import run_pipeline
    result = run_pipeline(
        fasta_path="contigs.fasta",
        model_path="data/models/mlp_v2.pt",
        card_db="data/databases/card/card.dmnd",
        aro_index="data/databases/card/aro_index.tsv",
        work_dir="output/run1",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from Bio.SeqRecord import SeqRecord  # type: ignore[import]

from plasflow2.annotate.args import (
    ORF,
    ARGHit,
    annotate_contigs_with_orfs,
)
from plasflow2.annotate.bacmet import BacMetHit, annotate_bacmet
from plasflow2.annotate.ice import ICEHit, annotate_ice
from plasflow2.annotate.mge import MGEHit, annotate_mge
from plasflow2.annotate.mobility import (
    MobilityResult,
    index_by_contig,
    parse_mob_results,
    run_mob_typer,
)
from plasflow2.annotate.mobility_diamond import (
    annotate_mobility_diamond,
    find_mob_diamond_dbs,
)
from plasflow2.annotate.pathogens import PathogenResult, detect_pathogens
from plasflow2.annotate.plasmid_db import PlasmidDBHit, annotate_plasmid_db
from plasflow2.annotate.taxonomy import (
    TaxResult,
    assign_taxonomy,
    detect_archaeal_contigs,
    parse_diamond_taxonomy_output,
)
from plasflow2.annotate.taxonomy_kaiju import (
    assign_taxonomy_kaiju,
    kaiju_available,
)
from plasflow2.annotate.topology import Topology, detect_topologies
from plasflow2.annotate.vfdb import VFHit, annotate_vf
from plasflow2.classify.marker_classifier import (
    MarkerClassifier,
    aggregate_scores,
    extract_marker_features,
    marker_classifier_available,
)
from plasflow2.classify.predict import Prediction, predict
from plasflow2.risk.scorer import RiskScore, score_nonplasmid, score_plasmid
from plasflow2.utils.fasta import load_fasta, write_fasta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ContigResult:
    """All annotations for a single contig that passed the plasmid filter."""

    record: SeqRecord
    prediction: Prediction
    arg_hits: list[ARGHit]
    mobility: MobilityResult | None
    risk: RiskScore
    taxonomy: TaxResult | None = None  # LCA taxonomy from DIAMOND (optional)
    vf_hits: list[VFHit] = field(default_factory=list)
    mge_hits: list[MGEHit] = field(default_factory=list)
    bacmet_hits: list[BacMetHit] = field(default_factory=list)
    ice_hits: list[ICEHit] = field(default_factory=list)


@dataclass
class NonPlasmidContigResult:
    """Prediction + ARG/VF/MGE/taxonomy for chromosome / phage / archaea / unclassified."""

    record: SeqRecord
    prediction: Prediction
    taxonomy: TaxResult | None = None
    arg_hits: list[ARGHit] = field(default_factory=list)
    vf_hits: list[VFHit] = field(default_factory=list)
    mge_hits: list[MGEHit] = field(default_factory=list)
    bacmet_hits: list[BacMetHit] = field(default_factory=list)
    ice_hits: list[ICEHit] = field(default_factory=list)
    risk: RiskScore | None = None


@dataclass
class PipelineResult:
    """Aggregated results for one run_pipeline() call."""

    input_fasta: Path
    all_predictions: list[Prediction]  # every contig, all classes
    plasmid_results: list[ContigResult]  # plasmid contigs only, fully annotated
    # Non-plasmid contigs: chromosome, phage, archaea, unclassified — prediction + taxonomy only
    non_plasmid_results: list[NonPlasmidContigResult] = field(default_factory=list)
    # Taxonomy results for ALL contigs (keyed by contig_id); empty if skipped
    taxonomy: dict[str, TaxResult] = field(default_factory=dict)
    # Pathogen detection results (subset of taxonomy — only pathogenic contigs)
    pathogens: dict[str, PathogenResult] = field(default_factory=dict)
    # Gene-level ORF list (populated when annotation runs; empty otherwise)
    orfs: list[ORF] = field(default_factory=list)
    # Topology per contig: "circular", "linear", or "too_short"
    topology: dict[str, Topology] = field(default_factory=dict)
    # Plasmid-DB nucleotide match (closest known plasmid per contig; plasmid contigs only)
    plasmid_db_hits: dict[str, PlasmidDBHit] = field(default_factory=dict)
    # Convenience counts
    class_counts: dict[str, int] = field(default_factory=dict)
    total_sequences: int = 0
    total_plasmids: int = 0
    total_args: int = 0

    def __post_init__(self) -> None:
        self.total_sequences = len(self.all_predictions)
        self.total_plasmids = len(self.plasmid_results)
        # Count ARGs across ALL contig classes (plasmid + non-plasmid)
        self.total_args = sum(len(cr.arg_hits) for cr in self.plasmid_results) + sum(
            len(cr.arg_hits) for cr in self.non_plasmid_results
        )

        if not self.class_counts:
            counts: dict[str, int] = {}
            for p in self.all_predictions:
                counts[p.label] = counts.get(p.label, 0) + 1
            self.class_counts = counts


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    fasta_path: Path | str,
    model_path: Path | str,
    card_db: Path | str,
    aro_index: Path | str,
    work_dir: Path | str,
    source_context: str = "unspecified",
    confidence_threshold: float = 0.70,
    plasmid_threshold: float = 0.95,
    argmax_fallback: bool = False,
    min_contig_length: int = 1000,
    threads: int = 8,
    skip_mobility: bool = False,
    taxonomy_db: Path | str | None = None,
    taxon_map_path: Path | str | None = None,
    skip_taxonomy: bool = False,
    sarg_db: Path | str | None = None,
    amrprot_db: Path | str | None = None,
    min_identity: float = 80.0,
    vfdb: Path | str | None = None,
    mge_db: Path | str | None = None,
    plasmid_db_dir: Path | str | None = None,
    taxonomy_engine: str = "auto",
    kaiju_db: Path | str | None = None,
    kaiju_nodes: Path | str | None = None,
    kaiju_names: Path | str | None = None,
    genomad_db_path: Path | str | None = None,
    archaea_threshold: float | None = None,
    lenient: bool = False,
) -> PipelineResult:
    """Run the full PlasFlow v2 pipeline on a FASTA file.

    Steps
    -----
    1. Load and length-filter contigs from *fasta_path*.
    2. Classify every contig with the MLP (``predict()``).
    3. Write plasmid-classified contigs to ``work_dir/plasmids.fasta``.
    4. Annotate ARGs on plasmid contigs via DIAMOND + CARD.
    5. Annotate mobility on plasmid contigs via MOB-suite (unless
       *skip_mobility* is True — useful when mob_typer is unavailable).
    6. Score each plasmid contig with ``score_plasmid()``.

    Args:
        fasta_path: Input nucleotide FASTA (assembled contigs).
        model_path: Path to trained MLP weights (.pt file).
        card_db: Path to DIAMOND-formatted CARD database (.dmnd).
        aro_index: Path to CARD aro_index.tsv.
        work_dir: Directory for all intermediate and output files.
        source_context: Sample provenance for risk scoring — one of
            ``clinical``, ``wastewater``, ``environmental``,
            ``unspecified``.
        confidence_threshold: Minimum MLP confidence for chromosome / phage /
            archaea calls (sequences below this are labelled ``unclassified``).
        plasmid_threshold: Minimum MLP confidence for plasmid calls.  Defaults
            to 0.95 — higher than *confidence_threshold* — to compensate for
            class-prior imbalance: the model trains on ~25 % plasmid but real
            metagenomes contain only ~2–5 % plasmid.
        min_contig_length: Discard sequences shorter than this (bp).
        threads: CPU threads for DIAMOND and MOB-suite.
        skip_mobility: If True, skip mob_typer and set mobility to None
            for all contigs (use when mob_typer is not installed).
        taxonomy_db: Path to a DIAMOND database (.dmnd) built from GTDB-r220
            or RefSeq protein sequences for taxonomy annotation.  If ``None``
            and *skip_taxonomy* is False, taxonomy is skipped with a warning.
        taxon_map_path: Optional path to a 2-column accession→lineage TSV
            (output of ``build_gtdb_taxon_map``).  When None, lineage is
            extracted from DIAMOND ``stitle`` fields.
        skip_taxonomy: If True, skip taxonomy annotation entirely (useful when
            no GTDB/RefSeq database is available).
        sarg_db: Optional path to a DIAMOND .dmnd database built from the SARG
            (Structured ARG) database.  When provided, ARG annotation runs
            against both CARD and SARG; CARD hits are preferred per ORF and
            SARG contributes supplementary hits for genes not in CARD.
        min_identity: Minimum amino-acid identity % for DIAMOND ARG hits
            (default 80 %).  80 % is the standard for environmental/metagenomic
            samples; use 90 % for clinical-isolate precision.
        vfdb: Optional path to a DIAMOND .dmnd database built from VFDB set A
            protein sequences.  When provided, virulence factor annotation runs
            on plasmid contigs using the pre-predicted ORFs (no extra Prodigal
            pass needed).
        mge_db: Optional path to a DIAMOND .dmnd database built from ISfinder
            transposase protein sequences.  When provided, MGE/IS element
            annotation runs on plasmid contigs.

    Returns:
        :class:`PipelineResult` with all predictions and per-plasmid
        annotations.

    Raises:
        FileNotFoundError: If *fasta_path*, *model_path*, *card_db*, or
            *aro_index* do not exist.
    """
    fasta_path = Path(fasta_path)
    model_path = Path(model_path)
    card_db = Path(card_db)
    aro_index = Path(aro_index)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_db_path = Path(taxonomy_db) if taxonomy_db else None
    taxon_map = Path(taxon_map_path) if taxon_map_path else None

    for p, name in [
        (fasta_path, "fasta_path"),
        (model_path, "model_path"),
        (card_db, "card_db"),
        (aro_index, "aro_index"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    # ------------------------------------------------------------------
    # 1. Load contigs
    # ------------------------------------------------------------------
    logger.info("Loading contigs from %s (min_length=%d)", fasta_path, min_contig_length)
    records = load_fasta(fasta_path, min_length=min_contig_length)
    if not records:
        logger.warning("No sequences pass min_length=%d filter — aborting.", min_contig_length)
        return PipelineResult(
            input_fasta=fasta_path,
            all_predictions=[],
            plasmid_results=[],
        )

    sequences = [str(r.seq) for r in records]
    seq_ids = [r.id for r in records]

    # ------------------------------------------------------------------
    # 2. Classify
    # ------------------------------------------------------------------
    logger.info("Classifying %d contigs …", len(sequences))
    # In lenient mode lower plasmid threshold to match the general threshold so
    # that moderate-confidence plasmid calls reach the hallmark gate (which is
    # also skipped in lenient mode).  In normal mode keep the high plasmid
    # threshold (default 0.95) to suppress false positives from class imbalance.
    _effective_plasmid_threshold = confidence_threshold if lenient else plasmid_threshold
    predictions = predict(
        sequences,
        seq_ids,
        model_path,
        threshold=confidence_threshold,
        plasmid_threshold=_effective_plasmid_threshold,
        argmax_fallback=argmax_fallback,
        source_context=source_context,
        apply_prior=True,
    )
    pred_by_id = {p.sequence_id: p for p in predictions}

    # ------------------------------------------------------------------
    # 3. Extract plasmid contigs
    # ------------------------------------------------------------------
    plasmid_records = [r for r in records if pred_by_id[r.id].label == "plasmid"]
    logger.info("Plasmid contigs: %d / %d", len(plasmid_records), len(records))

    if not plasmid_records:
        logger.info(
            "No plasmid contigs found — continuing to annotate all %d contigs.", len(records)
        )

    # Write plasmid FASTA for mobility/plasmid-DB/geNomad steps (plasmid-specific)
    plasmid_fasta = work_dir / "plasmids.fasta"
    if plasmid_records:
        write_fasta(plasmid_records, plasmid_fasta)

    # ------------------------------------------------------------------
    # 3b. Topology detection (circular vs linear)
    # ------------------------------------------------------------------
    logger.info("Detecting contig topology (DTR-based circularity check) …")
    topology_by_contig = detect_topologies(records)

    # Write ALL contigs FASTA for ARG / VF / MGE annotation
    # (chromosomal ARG carriage is scientifically important to capture)
    all_contigs_fasta = work_dir / "all_contigs.fasta"
    write_fasta(records, all_contigs_fasta)

    # ------------------------------------------------------------------
    # 4. ARG annotation — ALL contigs (plasmid + chromosome + phage + archaea)
    # ------------------------------------------------------------------
    # Auto-detect AMRProt DIAMOND DB if not explicitly provided
    _amrprot_auto = (
        Path(__file__).parent.parent.parent / "data" / "databases" / "amrfinder" / "amrprot.dmnd"
    )
    _amrprot_db = amrprot_db or (_amrprot_auto if _amrprot_auto.exists() else None)

    dbs_label = "CARD" + (" + SARG" if sarg_db else "") + (" + AMRProt" if _amrprot_db else "")
    logger.info("Annotating ARGs on ALL %d contigs (%s) …", len(records), dbs_label)

    arg_hits, all_orfs = annotate_contigs_with_orfs(
        fasta_path=all_contigs_fasta,
        card_db=card_db,
        aro_index_path=aro_index,
        work_dir=work_dir / "arg_annotation",
        threads=threads,
        sarg_db=sarg_db,
        amrprot_db=_amrprot_db,
        min_identity=min_identity,
    )
    # Pre-predicted proteins path — reused by VFDB, MGE, mobility, taxonomy
    arg_proteins = work_dir / "arg_annotation" / "proteins.faa"

    # Group hits by contig_id for fast lookup
    args_by_contig: dict[str, list[ARGHit]] = {}
    for hit in arg_hits:
        args_by_contig.setdefault(hit.contig_id, []).append(hit)

    def _cached(path: Path) -> bool:
        """Return True if a work-dir result file exists and is non-empty (cache hit)."""
        if path.exists() and path.stat().st_size > 0:
            logger.info("  [cache] Reusing %s", path)
            return True
        return False

    # ------------------------------------------------------------------
    # 4b. Virulence factor annotation — ALL contigs
    # ------------------------------------------------------------------
    vf_by_contig: dict[str, list[VFHit]] = {}
    _vf_cache = work_dir / "vfdb_annotation" / "vfdb_hits.tsv"
    if vfdb is not None:
        vfdb_path = Path(vfdb)
        if vfdb_path.exists() or vfdb_path.with_suffix(".dmnd").exists():
            if not _cached(_vf_cache):
                logger.info("Annotating VFs on ALL %d contigs (VFDB) …", len(records))
            try:
                vf_hits = annotate_vf(
                    fasta_path=all_contigs_fasta,
                    vfdb=vfdb_path,
                    work_dir=work_dir / "vfdb_annotation",
                    threads=threads,
                    reuse_proteins=arg_proteins if arg_proteins.exists() else None,
                )
                for hit in vf_hits:
                    vf_by_contig.setdefault(hit.contig_id, []).append(hit)
                logger.info("VF hits: %d across %d contigs", len(vf_hits), len(vf_by_contig))
            except Exception as exc:
                logger.warning("VFDB annotation failed: %s — skipping.", exc)
        else:
            logger.warning("VFDB database not found at %s — skipping VF annotation.", vfdb)

    # ------------------------------------------------------------------
    # 4c. MGE / IS element annotation — ALL contigs
    # ------------------------------------------------------------------
    mge_by_contig: dict[str, list[MGEHit]] = {}
    _mge_cache = work_dir / "mge_annotation" / "mge_hits.tsv"
    if mge_db is not None:
        mge_db_path = Path(mge_db)
        if mge_db_path.exists() or mge_db_path.with_suffix(".dmnd").exists():
            if not _cached(_mge_cache):
                logger.info("Annotating MGEs on ALL %d contigs (ISfinder) …", len(records))
            try:
                mge_hits = annotate_mge(
                    fasta_path=all_contigs_fasta,
                    mge_db=mge_db_path,
                    work_dir=work_dir / "mge_annotation",
                    threads=threads,
                    reuse_proteins=arg_proteins if arg_proteins.exists() else None,
                )
                for hit in mge_hits:
                    mge_by_contig.setdefault(hit.contig_id, []).append(hit)
                logger.info("MGE hits: %d across %d contigs", len(mge_hits), len(mge_by_contig))
            except Exception as exc:
                logger.warning("MGE annotation failed: %s — skipping.", exc)
        else:
            logger.warning("MGE database not found at %s — skipping MGE annotation.", mge_db)

    # ------------------------------------------------------------------
    # 4c-2. BacMet annotation — ALL contigs (biocide/metal resistance)
    # ------------------------------------------------------------------
    bacmet_by_contig: dict[str, list[BacMetHit]] = {}
    _bacmet_auto = (
        Path(__file__).parent.parent.parent / "data" / "databases" / "bacmet" / "bacmet.dmnd"
    )
    if _bacmet_auto.exists() and arg_proteins.exists():
        try:
            bacmet_hits = annotate_bacmet(
                proteins_faa=arg_proteins,
                bacmet_db=_bacmet_auto,
                work_dir=work_dir / "bacmet_annotation",
                threads=threads,
            )
            for hit in bacmet_hits:
                bacmet_by_contig.setdefault(hit.contig_id, []).append(hit)
            logger.info(
                "BacMet hits: %d across %d contigs", len(bacmet_hits), len(bacmet_by_contig)
            )
        except Exception as exc:
            logger.warning("BacMet annotation failed: %s — skipping.", exc)

    # ------------------------------------------------------------------
    # 4c-3. ICE annotation — ALL contigs (integrative conjugative elements)
    # ------------------------------------------------------------------
    ice_by_contig: dict[str, list[ICEHit]] = {}
    _ice_auto = Path(__file__).parent.parent.parent / "data" / "databases" / "ice" / "ice.dmnd"
    if _ice_auto.exists() and arg_proteins.exists():
        try:
            ice_hits_all = annotate_ice(
                proteins_faa=arg_proteins,
                ice_db=_ice_auto,
                work_dir=work_dir / "ice_annotation",
                threads=threads,
            )
            for hit in ice_hits_all:
                ice_by_contig.setdefault(hit.contig_id, []).append(hit)
            logger.info("ICE hits: %d across %d contigs", len(ice_hits_all), len(ice_by_contig))
        except Exception as exc:
            logger.warning("ICE annotation failed: %s — skipping.", exc)

    # ------------------------------------------------------------------
    # 4d. Plasmid-DB nucleotide matching (plasmid contigs only)
    # ------------------------------------------------------------------
    plasmid_db_hits: dict[str, PlasmidDBHit] = {}
    _pdb_cache = work_dir / "plasmid_db" / "plasmid_db_hits.paf"
    if plasmid_db_dir is not None and plasmid_records:
        plasmid_db_path = Path(plasmid_db_dir)
        if plasmid_db_path.is_dir():
            if not _cached(_pdb_cache):
                logger.info(
                    "Running plasmid-DB match on %d plasmid contigs …", len(plasmid_records)
                )
            try:
                plasmid_db_hits = annotate_plasmid_db(
                    plasmid_fasta=plasmid_fasta,
                    plasmid_db_dir=plasmid_db_path,
                    work_dir=work_dir / "plasmid_db",
                    threads=threads,
                )
                logger.info(
                    "Plasmid-DB: %d / %d contigs matched",
                    len(plasmid_db_hits),
                    len(plasmid_records),
                )
            except Exception as exc:
                logger.warning("Plasmid-DB matching failed: %s — skipping.", exc)
        else:
            logger.warning(
                "Plasmid DB dir not found: %s — skipping plasmid-DB match.", plasmid_db_dir
            )

    # ------------------------------------------------------------------
    # 5. Mobility annotation — DIAMOND fast path or mob_typer fallback
    # ------------------------------------------------------------------
    mobility_by_contig: dict[str, MobilityResult] = {}
    if not skip_mobility and plasmid_records:
        _mob_diamond_dir = Path(__file__).parent.parent.parent / "data" / "databases" / "mob_suite"
        _mob_dmnd, _mpf_dmnd, _rep_protein_dmnd, _rep_fasta = find_mob_diamond_dbs(_mob_diamond_dir)
        _use_diamond_mob = _mob_dmnd is not None or _mpf_dmnd is not None

        if _use_diamond_mob:
            logger.info(
                "Mobility annotation: DIAMOND fast path on %d plasmid contigs …",
                len(plasmid_records),
            )
            try:
                mob_results = annotate_mobility_diamond(
                    plasmid_fasta=plasmid_fasta,
                    mob_suite_dir=_mob_diamond_dir,
                    work_dir=work_dir / "mob_diamond",
                    proteins_faa=arg_proteins if arg_proteins.exists() else None,
                    threads=threads,
                )
                mobility_by_contig = index_by_contig(mob_results)
            except Exception as exc:
                logger.warning("DIAMOND mobility failed: %s — falling back to mob_typer.", exc)
                _use_diamond_mob = False

        if not _use_diamond_mob:
            logger.info(
                "Mobility annotation: mob_typer on %d plasmid contigs …", len(plasmid_records)
            )
            try:
                mob_tsv = run_mob_typer(
                    plasmid_fasta,
                    work_dir / "mob_typer",
                    threads=threads,
                )
                mobility_results = parse_mob_results(mob_tsv)
                mobility_by_contig = index_by_contig(mobility_results)
            except (FileNotFoundError, RuntimeError) as exc:
                logger.warning("mob_typer unavailable or failed: %s — skipping mobility.", exc)
    else:
        logger.info("Mobility annotation skipped (skip_mobility=True)")

    # ------------------------------------------------------------------
    # 5b. Rep protein detection — hallmark evidence for non-mobile plasmids
    # ------------------------------------------------------------------
    # Every plasmid encodes a replication protein (RepA/RepB/RepC/…) even when
    # it carries no relaxase (non-mobilizable). Detecting rep proteins provides
    # biological evidence for plasmid identity independent of mobility class.
    # Build the DB with: bash scripts/setup_rep_diamond.sh
    rep_protein_hits: set[str] = set()
    _mob_diamond_dir_rep = Path(__file__).parent.parent.parent / "data" / "databases" / "mob_suite"
    _rep_prot_db = _mob_diamond_dir_rep / "rep_proteins.dmnd"
    if _rep_prot_db.exists() and arg_proteins.exists():
        try:
            import re as _re
            import subprocess as _sp

            _rep_out = work_dir / "mob_diamond" / "rep_protein_hits.tsv"
            _rep_out.parent.mkdir(parents=True, exist_ok=True)
            _rep_cmd = [
                "diamond",
                "blastp",
                "--query",
                str(arg_proteins),
                "--db",
                str(_rep_prot_db).removesuffix(".dmnd"),
                "--out",
                str(_rep_out),
                "--outfmt",
                "6",
                "qseqid",
                "sseqid",
                "pident",
                "qcovhsp",
                "evalue",
                "--id",
                "40.0",
                "--query-cover",
                "60.0",
                "--threads",
                str(threads),
                "--max-target-seqs",
                "1",
                "--sensitive",
                "--quiet",
            ]
            _rep_result = _sp.run(_rep_cmd, capture_output=True, text=True)
            if _rep_result.returncode == 0 and _rep_out.exists():
                with open(_rep_out) as _fh:
                    for _line in _fh:
                        _orf_id = _line.split("\t")[0].strip()
                        _cid = _re.sub(r"_\d+$", "", _orf_id)
                        rep_protein_hits.add(_cid)
                logger.info(
                    "Rep protein hits: %d contigs with replication proteins", len(rep_protein_hits)
                )
        except Exception as _exc:
            logger.warning("Rep protein DIAMOND failed: %s — skipping.", _exc)
    else:
        if not _rep_prot_db.exists():
            logger.debug("rep_proteins.dmnd not found — run scripts/setup_rep_diamond.sh to enable")

    # ------------------------------------------------------------------
    # 6. Taxonomy annotation — Kaiju (preferred) or DIAMOND blastp fallback
    # ------------------------------------------------------------------
    taxonomy_by_contig: dict[str, TaxResult] = {}
    if not skip_taxonomy:
        reuse_prot = arg_proteins if arg_proteins.exists() else None

        # Auto-detect Kaiju DBs: plasmids DB for plasmid contigs, refseq_ref for rest
        _kaiju_dir_auto = Path(__file__).parent.parent.parent / "data" / "databases" / "kaiju"
        _kaiju_nodes_auto = _kaiju_dir_auto / "nodes.dmp"
        _kaiju_names_auto = _kaiju_dir_auto / "names.dmp"

        def _find_fmi(name_substr: str) -> Path | None:
            """Find a .fmi file whose name contains name_substr."""
            for p in sorted(_kaiju_dir_auto.glob("*.fmi")):
                if name_substr in p.name:
                    return p
            return None

        _kaiju_plasmids_db = _find_fmi("plasmid")
        _kaiju_refseq_db = _find_fmi("refseq") or _find_fmi("nr")
        _kaiju_nodes = (
            Path(kaiju_nodes)
            if kaiju_nodes
            else (_kaiju_nodes_auto if _kaiju_nodes_auto.exists() else None)
        )
        _kaiju_names = (
            Path(kaiju_names)
            if kaiju_names
            else (_kaiju_names_auto if _kaiju_names_auto.exists() else None)
        )
        # Legacy: user-supplied single DB via --kaiju-db
        _kaiju_db_legacy = Path(kaiju_db) if kaiju_db else None

        _kaiju_ready = (
            kaiju_available()
            and _kaiju_nodes
            and _kaiju_nodes.exists()
            and _kaiju_names
            and _kaiju_names.exists()
            and reuse_prot
            and (taxonomy_engine in ("kaiju", "auto"))
        )

        _use_kaiju = False
        if _kaiju_ready and (_kaiju_plasmids_db or _kaiju_refseq_db or _kaiju_db_legacy):
            _use_kaiju = True
        elif taxonomy_engine == "kaiju":
            logger.warning(
                "taxonomy_engine='kaiju' requested but kaiju binary or DB files "
                "not found — falling back to DIAMOND."
            )

        if _use_kaiju:
            try:
                # ── Plasmid contigs: dedicated plasmids DB (fast, targeted) ──
                if _kaiju_plasmids_db and plasmid_records:
                    _plasmid_prot = work_dir / "arg_annotation" / "plasmid_proteins.faa"
                    # Write plasmid-only proteins subset
                    plasmid_ids = {r.id for r in plasmid_records}
                    if reuse_prot and reuse_prot.exists():
                        from Bio import SeqIO  # type: ignore

                        with open(_plasmid_prot, "w") as _pf:
                            for rec in SeqIO.parse(str(reuse_prot), "fasta"):
                                import re as _re

                                cid = _re.sub(r"_\d+$", "", rec.id)
                                if cid in plasmid_ids:
                                    _pf.write(f">{rec.id}\n{str(rec.seq)}\n")
                    if _plasmid_prot.exists() and _plasmid_prot.stat().st_size > 0:
                        logger.info(
                            "Taxonomy: Kaiju plasmids DB on %d plasmid contigs …",
                            len(plasmid_records),
                        )
                        plasmid_tax = assign_taxonomy_kaiju(
                            protein_fasta=_plasmid_prot,
                            kaiju_db=_kaiju_plasmids_db,
                            nodes_dmp=_kaiju_nodes,
                            names_dmp=_kaiju_names,
                            work_dir=work_dir / "taxonomy_plasmids",
                            threads=threads,
                        )
                        taxonomy_by_contig.update(plasmid_tax)
                        logger.info(
                            "Taxonomy (plasmids DB): %d plasmid contigs classified",
                            len(plasmid_tax),
                        )

                # ── All contigs: refseq_ref DB (or legacy single DB) ──────────
                _general_db = _kaiju_refseq_db or _kaiju_db_legacy
                if _general_db:
                    logger.info(
                        "Taxonomy: Kaiju %s on %d contigs (threads=%d) …",
                        _general_db.name,
                        len(records),
                        threads,
                    )
                    general_tax = assign_taxonomy_kaiju(
                        protein_fasta=reuse_prot,
                        kaiju_db=_general_db,
                        nodes_dmp=_kaiju_nodes,
                        names_dmp=_kaiju_names,
                        work_dir=work_dir / "taxonomy",
                        threads=threads,
                    )
                    # Plasmid contigs: prefer plasmids DB result if already classified
                    for cid, tax in general_tax.items():
                        if cid not in taxonomy_by_contig:
                            taxonomy_by_contig[cid] = tax

            except Exception as exc:
                logger.warning("Kaiju taxonomy failed: %s — falling back to DIAMOND.", exc)
                _use_kaiju = False  # trigger DIAMOND fallback below

        if not _use_kaiju:
            if taxonomy_db_path and taxonomy_db_path.exists():
                _tax_cache = work_dir / "taxonomy" / "diamond_taxonomy.tsv"
                if _cached(_tax_cache):
                    logger.info("Taxonomy: loading cached results from %s", _tax_cache)
                else:
                    logger.info(
                        "Taxonomy: DIAMOND blastp on %d contigs " "(threads=%d, block_size=4.0) …",
                        len(records),
                        threads,
                    )
                try:
                    taxonomy_by_contig, _raw_tax_hits = assign_taxonomy(
                        fasta_path=fasta_path,
                        taxonomy_db=taxonomy_db_path,
                        work_dir=work_dir / "taxonomy",
                        taxon_map_path=taxon_map,
                        threads=threads,
                        protein_fasta=reuse_prot,
                        block_size=4.0,
                        return_raw_hits=True,
                    )
                except Exception as exc:
                    logger.warning("DIAMOND taxonomy failed: %s — skipping.", exc)
            else:
                logger.info(
                    "No taxonomy DB found (--taxonomy-db or data/databases/taxonomy/). "
                    "Use --skip-taxonomy to suppress this message."
                )
    else:
        logger.info("Taxonomy annotation skipped (skip_taxonomy=True)")

    # ------------------------------------------------------------------
    # 5d. Kraken2 fallback taxonomy — classify contigs missing from DIAMOND
    # ------------------------------------------------------------------
    # DIAMOND classifies ~44% of contigs (those with detectable proteins at
    # ≥1000 bp). Kraken2 covers the remaining ~56% using nucleotide k-mers:
    # short contigs, repetitive regions, and sequences with no ORFs.
    # DIAMOND result always takes priority — Kraken2 only fills gaps.
    _kaiju_dir_auto = Path(__file__).parent.parent.parent / "data" / "databases" / "kaiju"
    from plasflow2.annotate.taxonomy_kraken2 import (
        assign_taxonomy_kraken2,
        find_kraken2_db,
        kraken2_available,
    )

    _kraken2_db = find_kraken2_db()
    _kraken2_nodes = _kaiju_dir_auto / "nodes.dmp"
    _kraken2_names = _kaiju_dir_auto / "names.dmp"

    if (
        not skip_taxonomy
        and _kraken2_db is not None
        and kraken2_available()
        and _kraken2_nodes.exists()
        and _kraken2_names.exists()
    ):
        try:
            logger.info(
                "Kraken2 fallback taxonomy: classifying contigs missing from DIAMOND "
                "(%d / %d without taxonomy) …",
                len(records) - len(taxonomy_by_contig),
                len(records),
            )
            _kraken2_new = assign_taxonomy_kraken2(
                fasta_path=all_contigs_fasta,
                db_dir=_kraken2_db,
                nodes_dmp=_kraken2_nodes,
                names_dmp=_kraken2_names,
                work_dir=work_dir / "taxonomy_kraken2",
                threads=threads,
                confidence=0.1,
                existing_taxonomy=taxonomy_by_contig,
            )
            taxonomy_by_contig.update(_kraken2_new)
            logger.info(
                "Kraken2: added %d new taxonomy assignments (total now %d / %d contigs)",
                len(_kraken2_new),
                len(taxonomy_by_contig),
                len(records),
            )
        except Exception as _exc:
            logger.warning("Kraken2 fallback taxonomy failed: %s — skipping.", _exc)
    elif not skip_taxonomy and _kraken2_db is None:
        logger.debug("Kraken2 DB not found — run scripts/setup_kraken2_db.sh for fallback taxonomy")

    # ------------------------------------------------------------------
    # 6b. Archaeal post-classification override
    # ------------------------------------------------------------------
    # The 3-class MLP does not output archaea. Instead, we detect archaeal
    # contigs from DIAMOND taxonomy using the paper criteria:
    #   (i)  archaeal ORF hits > bacterial ORF hits
    #   (ii) archaeal ORF hits >= 5
    # Only contigs labelled chromosome or unclassified are candidates —
    # plasmid and phage labels are left unchanged.
    _archaeal_ids: set[str] = set()
    # Use cached raw hits from assign_taxonomy() when available — avoids
    # re-parsing the 500 MB diamond_taxonomy.tsv (saves ~3 min per run).
    # Fall back to re-parsing if taxonomy ran via Kaiju or hits weren't cached.
    _raw_tax_hits: dict = locals().get("_raw_tax_hits", {})
    if not _raw_tax_hits:
        _diamond_tax_tsv = work_dir / "taxonomy" / "diamond_taxonomy.tsv"
        if _diamond_tax_tsv.exists() and _diamond_tax_tsv.stat().st_size > 0:
            try:
                _raw_tax_hits = parse_diamond_taxonomy_output(_diamond_tax_tsv)
            except Exception as _exc:
                logger.warning("Could not parse taxonomy TSV for archaeal detection: %s", _exc)
    if _raw_tax_hits:
        try:
            # Only consider chromosome/unclassified candidates to avoid
            # wrongly overriding plasmid or phage calls.
            _candidates = {
                cid
                for cid, pred in pred_by_id.items()
                if pred.label in ("chromosome", "unclassified")
            }
            _raw_hits_candidates = {k: v for k, v in _raw_tax_hits.items() if k in _candidates}
            _archaeal_ids = detect_archaeal_contigs(_raw_hits_candidates)
            # Apply archaea_threshold: only override when the MLP archaea score
            # meets the minimum (defaults to confidence_threshold when not set).
            _eff_arch_thresh = (
                archaea_threshold if archaea_threshold is not None else confidence_threshold
            )
            _archaeal_ids = {
                cid
                for cid in _archaeal_ids
                if pred_by_id[cid].scores.get("archaea", 0.0) >= _eff_arch_thresh
            }
            if _archaeal_ids:
                # Override predictions in pred_by_id
                for cid in _archaeal_ids:
                    old = pred_by_id[cid]
                    pred_by_id[cid] = Prediction(
                        sequence_id=old.sequence_id,
                        label="archaea",
                        confidence=old.confidence,
                        scores=old.scores,
                    )
                logger.info(
                    "Archaeal override: %d contigs relabelled archaea "
                    "(from chromosome/unclassified) using DIAMOND taxonomy "
                    "[archaea_threshold=%.2f]",
                    len(_archaeal_ids),
                    _eff_arch_thresh,
                )
        except Exception as _exc:
            logger.warning("Archaeal post-classification failed: %s — skipping.", _exc)

    # ------------------------------------------------------------------
    # 6c. Hallmark gate — require biological evidence for ALL plasmid calls
    # ------------------------------------------------------------------
    # 99.1% of plasmid calls in WWTP datasets have zero biological evidence
    # (no PLSDB match, no relaxase, no replicon). These are chromosomal fragments
    # with plasmid-like k-mer composition — the dominant false-positive source.
    #
    # Evidence types (any one is sufficient):
    #   1. PLSDB / RefSeq / COMPASS nucleotide match
    #   2. Relaxase gene (conjugative or mobilizable class)
    #   3. Replicon type (IncF, IncP, IncQ, …)
    #   4. ICE hit (integrative conjugative elements)
    #
    # Length tiers:
    #   < 50,000 bp  → demote to unclassified if no evidence
    #   ≥ 50,000 bp  → keep as plasmid but flag low_confidence (large novel plasmids
    #                  are plausible without PLSDB match; false positive rate is lower
    #                  at this length due to consistent k-mer composition)
    HALLMARK_HARD_THRESHOLD = 50_000  # bp — above this: trust MLP, flag low_confidence
    _hallmark_demoted = 0
    _hallmark_flagged = 0
    if lenient:
        logger.info("Hallmark gate disabled (--lenient): accepting all MLP plasmid predictions.")
    for record in list(plasmid_records) if not lenient else []:
        cid = record.id
        mob = mobility_by_contig.get(cid)
        has_mobility = mob is not None and mob.mobility_class in ("conjugative", "mobilizable")
        has_plsdb = cid in plasmid_db_hits
        has_replicon = mob is not None and bool(mob.replicon_type)
        has_ice = bool(ice_by_contig.get(cid))
        has_rep_protein = cid in rep_protein_hits
        has_evidence = has_mobility or has_plsdb or has_replicon or has_ice or has_rep_protein

        if has_evidence:
            continue  # good evidence — keep as plasmid

        contig_len = len(record.seq)
        old = pred_by_id[cid]
        if contig_len < HALLMARK_HARD_THRESHOLD:
            # Demote: k-mer signal alone is unreliable at this length
            pred_by_id[cid] = Prediction(
                sequence_id=old.sequence_id,
                label="unclassified",
                confidence=old.confidence,
                scores=old.scores,
            )
            _hallmark_demoted += 1
        else:
            # Keep but flag low_confidence — large contigs are plausible novel plasmids
            pred_by_id[cid] = Prediction(
                sequence_id=old.sequence_id,
                label="plasmid",
                confidence=old.confidence,
                scores=old.scores,
                low_confidence=True,
                mlp_scores=old.mlp_scores,
                xgb_scores=old.xgb_scores,
                bio_evidence=old.bio_evidence,
                evidence_type=old.evidence_type,
            )
            _hallmark_flagged += 1

    if _hallmark_demoted or _hallmark_flagged:
        logger.info(
            "Hallmark gate: demoted %d plasmid calls (no evidence, <%d bp) → unclassified; "
            "flagged %d (no evidence, ≥%d bp) → low_confidence plasmid",
            _hallmark_demoted,
            HALLMARK_HARD_THRESHOLD,
            _hallmark_flagged,
            HALLMARK_HARD_THRESHOLD,
        )
        plasmid_records = [r for r in records if pred_by_id[r.id].label == "plasmid"]

    # ------------------------------------------------------------------
    # 6d-pre. geNomad SPM feature extraction (automatic when available)
    # ------------------------------------------------------------------
    # geNomad annotate is run automatically when:
    #   (a) the `genomad` binary is on PATH, AND
    #   (b) a geNomad database directory is found (auto-detected or explicit)
    # The 12 per-contig SPM features it produces significantly improve XGBoost
    # accuracy — especially for non-mobilizable plasmids that lack relaxase hits.
    # Falls back gracefully (SPM features = 0) if geNomad is not installed.
    _gn_spm_by_contig: dict[str, dict[str, float]] = {}
    _gn_db: Path | None = None
    if genomad_db_path:
        _gn_db = Path(genomad_db_path)
    else:
        # Auto-detect standard install location
        _gn_db_auto = Path(__file__).parent.parent.parent / "data" / "databases" / "genomad_db"
        if _gn_db_auto.is_dir():
            _gn_db = _gn_db_auto

    if _gn_db is not None and _gn_db.is_dir() and plasmid_records:
        import shutil as _shutil
        import subprocess as _gn_sp
        import sys as _sys

        if _shutil.which("genomad") is not None:
            _gn_out_dir = work_dir / "genomad_annotate"
            _gn_genes_tsv_candidates = list(_gn_out_dir.glob("*_annotate/*_genes.tsv"))

            if _gn_genes_tsv_candidates:
                logger.info("geNomad: reusing cached genes TSV from %s", _gn_out_dir)
                _gn_genes_tsv = _gn_genes_tsv_candidates[0]
            else:
                logger.info(
                    "geNomad: running annotate on %d plasmid contigs (db=%s) …",
                    len(plasmid_records),
                    _gn_db,
                )
                _gn_out_dir.mkdir(parents=True, exist_ok=True)
                _gn_cmd = [
                    "genomad",
                    "annotate",
                    str(plasmid_fasta),
                    str(_gn_out_dir),
                    str(_gn_db),
                    "--threads",
                    str(threads),
                    "--quiet",
                ]
                try:
                    _gn_proc = _gn_sp.run(_gn_cmd, capture_output=True, text=True, timeout=600)
                    if _gn_proc.returncode != 0:
                        logger.warning(
                            "geNomad annotate failed (exit %d): %s — SPM features disabled.",
                            _gn_proc.returncode,
                            _gn_proc.stderr[:300],
                        )
                        _gn_genes_tsv = None
                    else:
                        _gn_genes_tsv_candidates = list(_gn_out_dir.glob("*_annotate/*_genes.tsv"))
                        _gn_genes_tsv = (
                            _gn_genes_tsv_candidates[0] if _gn_genes_tsv_candidates else None
                        )
                        if _gn_genes_tsv is None:
                            logger.warning(
                                "geNomad ran but no *_genes.tsv found in %s — SPM features disabled.",
                                _gn_out_dir,
                            )
                except _gn_sp.TimeoutExpired:
                    logger.warning("geNomad annotate timed out (600 s) — SPM features disabled.")
                    _gn_genes_tsv = None
                except Exception as _gn_exc:
                    logger.warning("geNomad annotate error: %s — SPM features disabled.", _gn_exc)
                    _gn_genes_tsv = None

            if _gn_genes_tsv and _gn_genes_tsv.exists():
                try:
                    # Import geNomad feature extractor from scripts/
                    _scripts_dir = Path(__file__).parent.parent.parent / "scripts"
                    if str(_scripts_dir) not in _sys.path:
                        _sys.path.insert(0, str(_scripts_dir))
                    from extract_genomad_features import (
                        extract_all as _gn_extract_all,  # type: ignore[import]
                    )

                    _gn_spm_by_contig = _gn_extract_all(_gn_genes_tsv)
                    logger.info(
                        "geNomad SPM features loaded for %d / %d plasmid contigs",
                        len(_gn_spm_by_contig),
                        len(plasmid_records),
                    )
                except Exception as _gn_feat_exc:
                    logger.warning(
                        "geNomad feature extraction failed: %s — SPM features disabled.",
                        _gn_feat_exc,
                    )
        else:
            logger.debug(
                "genomad binary not found on PATH — SPM features disabled. "
                "Install with: conda install -c conda-forge -c bioconda genomad"
            )
    else:
        logger.debug(
            "geNomad DB not found — SPM features disabled. "
            "Run: bash scripts/setup_databases.sh to download data/databases/genomad_db"
        )

    # ------------------------------------------------------------------
    # 6d. Marker XGBoost — post-annotation rescoring of plasmid candidates
    # ------------------------------------------------------------------
    # Runs AFTER all annotations so every feature has a real value:
    # mobility class, ARG/MGE/ICE hit density, ORF coding density.
    # PLSDB match is used as a hard override RULE (not a model feature)
    # to avoid training-time data leakage.
    _marker_model_path = Path(model_path).parent / "marker_xgb.pkl"
    _seq_by_id = dict(zip(seq_ids, sequences))
    # Pre-build ORF lookup dict — avoids O(n²) scan of 481k ORFs per contig
    _orfs_by_contig: dict[str, list] = {}
    for _orf in all_orfs:
        _cid = getattr(_orf, "contig_id", None)
        if _cid:
            _orfs_by_contig.setdefault(_cid, []).append(_orf)

    if _marker_model_path.exists() and marker_classifier_available():
        try:
            _marker_clf = MarkerClassifier.load(_marker_model_path)
            _n_xgb_promoted = 0
            _n_xgb_demoted = 0

            # Rescore all current plasmid predictions using real annotation data
            for record in list(plasmid_records):
                cid = record.id
                pred = pred_by_id[cid]
                seq = _seq_by_id.get(cid, "")

                feats = extract_marker_features(
                    contig_id=cid,
                    sequence=seq,
                    mlp_scores=pred.scores,
                    mobility=mobility_by_contig.get(cid),
                    arg_hits=args_by_contig.get(cid, []),
                    mge_hits=mge_by_contig.get(cid, []),
                    ice_hits=ice_by_contig.get(cid, []),
                    orfs=_orfs_by_contig.get(cid, []),
                    has_rep_protein=cid in rep_protein_hits,
                    n_rep_hits=len(
                        [h for h in _orfs_by_contig.get(cid, []) if cid in rep_protein_hits]
                    ),
                    genomad_spm=_gn_spm_by_contig.get(cid),
                )
                marker_scores = _marker_clf.predict_scores(feats)
                agg = aggregate_scores(pred.scores, marker_scores, feats.marker_gene_fraction)

                # PLSDB match: hard override — confirmed plasmid identity
                if cid in plasmid_db_hits:
                    agg = {k: (0.97 if k == "plasmid" else (1 - 0.97) / 2) for k in agg}

                best_class = max(agg, key=agg.get)
                best_conf = agg[best_class]
                thresh = plasmid_threshold if best_class == "plasmid" else confidence_threshold
                new_label = (
                    best_class
                    if best_conf >= thresh
                    else (best_class if argmax_fallback else "unclassified")
                )

                if new_label != pred.label:
                    if new_label == "plasmid":
                        _n_xgb_promoted += 1
                    else:
                        _n_xgb_demoted += 1
                    pred_by_id[cid] = Prediction(
                        sequence_id=cid,
                        label=new_label,
                        confidence=best_conf,
                        scores=agg,
                    )

            logger.info(
                "Marker XGBoost: promoted %d → plasmid, demoted %d → other",
                _n_xgb_promoted,
                _n_xgb_demoted,
            )
            # Rebuild plasmid_records after XGBoost rescoring
            plasmid_records = [r for r in records if pred_by_id[r.id].label == "plasmid"]
        except Exception as _exc:
            logger.warning("Marker XGBoost failed: %s — using MLP + hallmark gate only.", _exc)
    elif plasmid_db_hits:
        # No XGBoost model: still apply PLSDB hard override for confirmed matches
        for cid in plasmid_db_hits:
            if cid in pred_by_id and pred_by_id[cid].label != "plasmid":
                old = pred_by_id[cid]
                pred_by_id[cid] = Prediction(
                    sequence_id=cid,
                    label="plasmid",
                    confidence=0.97,
                    scores={**old.scores, "plasmid": 0.97},
                )
        plasmid_records = [r for r in records if pred_by_id[r.id].label == "plasmid"]

    # ------------------------------------------------------------------
    # 7. Risk scoring + assemble ContigResult list
    # ------------------------------------------------------------------
    plasmid_results: list[ContigResult] = []
    for record in plasmid_records:
        cid = record.id
        mobility = mobility_by_contig.get(cid)
        hits = args_by_contig.get(cid, [])
        risk = score_plasmid(cid, mobility, hits, source_context, taxonomy_by_contig.get(cid))
        plasmid_results.append(
            ContigResult(
                record=record,
                prediction=pred_by_id[cid],
                arg_hits=hits,
                mobility=mobility,
                risk=risk,
                taxonomy=taxonomy_by_contig.get(cid),
                vf_hits=vf_by_contig.get(cid, []),
                mge_hits=mge_by_contig.get(cid, []),
                bacmet_hits=bacmet_by_contig.get(cid, []),
                ice_hits=ice_by_contig.get(cid, []),
            )
        )

    # ------------------------------------------------------------------
    # 8. Build NonPlasmidContigResult list + risk score ALL non-plasmid contigs
    # ------------------------------------------------------------------
    non_plasmid_results: list[NonPlasmidContigResult] = []
    for record in records:
        cid = record.id
        if pred_by_id[cid].label != "plasmid":
            np_arg_hits = args_by_contig.get(cid, [])
            np_risk = score_nonplasmid(
                contig_id=cid,
                label=pred_by_id[cid].label,
                arg_hits=np_arg_hits,
                source_context=source_context,
                taxonomy=taxonomy_by_contig.get(cid),
            )
            non_plasmid_results.append(
                NonPlasmidContigResult(
                    record=record,
                    prediction=pred_by_id[cid],
                    taxonomy=taxonomy_by_contig.get(cid),
                    arg_hits=np_arg_hits,
                    vf_hits=vf_by_contig.get(cid, []),
                    mge_hits=mge_by_contig.get(cid, []),
                    bacmet_hits=bacmet_by_contig.get(cid, []),
                    ice_hits=ice_by_contig.get(cid, []),
                    risk=np_risk,
                )
            )

    # ── Pathogen detection (from taxonomy) ───────────────────────────────────
    pathogens_by_contig: dict[str, PathogenResult] = {}
    if taxonomy_by_contig:
        pathogens_by_contig = detect_pathogens(taxonomy_by_contig)
        if pathogens_by_contig:
            from collections import Counter

            by_level: Counter[str] = Counter(p.threat_level for p in pathogens_by_contig.values())
            logger.info(
                "Pathogen detection: %d pathogenic contigs " "(critical=%d, high=%d, medium=%d)",
                len(pathogens_by_contig),
                by_level.get("critical", 0),
                by_level.get("high", 0),
                by_level.get("medium", 0),
            )

    # Rebuild in original input order so all post-processing steps
    # (archaeal override, hallmark gate, XGBoost, PLSDB hard override) are reflected.
    final_predictions = [pred_by_id[r.id] for r in records]

    result = PipelineResult(
        input_fasta=fasta_path,
        all_predictions=final_predictions,
        plasmid_results=plasmid_results,
        non_plasmid_results=non_plasmid_results,
        taxonomy=taxonomy_by_contig,
        pathogens=pathogens_by_contig,
        orfs=all_orfs,
        topology=topology_by_contig,
        plasmid_db_hits=plasmid_db_hits,
    )
    tax_classified = sum(1 for r in taxonomy_by_contig.values() if r.rank != "unclassified")
    total_vf = sum(len(cr.vf_hits) for cr in plasmid_results)
    total_mge = sum(len(cr.mge_hits) for cr in plasmid_results)
    logger.info(
        "Pipeline complete — %d total | %d plasmid | %d ARGs | %d VFs | %d MGEs | "
        "%d/%d taxonomy-classified | %d pathogenic | risk scores %s",
        result.total_sequences,
        result.total_plasmids,
        result.total_args,
        total_vf,
        total_mge,
        tax_classified,
        len(taxonomy_by_contig),
        len(pathogens_by_contig),
        sorted({cr.risk.score for cr in plasmid_results}),
    )
    return result
