"""Publication-facing biological evidence and transparent priority tiers.

This module is deliberately downstream of MobiOrigin prediction.  Its evidence
tables never change an origin label, probability, or selective threshold.
"""

from __future__ import annotations

import csv
import html
import json
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from mobiorigin.annotate import ArgHit, Orf
from mobiorigin.fasta import FastaRecord
from mobiorigin.provenance import atomic_json, sha256_file

VFDB_MIN_IDENTITY = 60.0
VFDB_MIN_QUERY_COVERAGE = 80.0
MGE_MIN_IDENTITY = 70.0
MGE_MIN_QUERY_COVERAGE = 80.0
BACMET_MIN_IDENTITY = 80.0
BACMET_MIN_QUERY_COVERAGE = 80.0
MOB_MIN_IDENTITY = 50.0
MOB_MIN_QUERY_COVERAGE = 70.0


@dataclass(frozen=True)
class EvidenceHit:
    sequence_id: str
    orf_id: str
    orf_start: int
    orf_end: int
    orf_strand: int
    evidence_group: str
    source: str
    feature_type: str
    feature_name: str
    accession: str
    category: str
    description: str
    method: str
    identity: float | None
    query_coverage: float | None
    evalue: float | None
    bitscore: float | None


EVIDENCE_COLUMNS = tuple(EvidenceHit.__dataclass_fields__)
PREDICTION_FIELDS = (
    "prediction",
    "p_chromosome",
    "p_plasmid",
    "p_phage",
    "plasmid_score",
    "abstention_reason",
)


def run_evidence_diamond(
    *,
    diamond: Path,
    proteins: Path,
    database: Path,
    output: Path,
    threads: int,
    min_identity: float,
    min_query_coverage: float,
) -> None:
    """Run a frozen protein-homology search and retain one best target per ORF."""
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
            "stitle",
            "--id",
            str(min_identity),
            "--query-cover",
            str(min_query_coverage),
            "--threads",
            str(threads),
            "--sensitive",
            "--max-target-seqs",
            "1",
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


def _rows(path: Path) -> list[tuple[str, str, float, float, float, float, str]]:
    values: list[tuple[str, str, float, float, float, float, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t", 6)
            if len(parts) != 7:
                raise ValueError(f"Malformed evidence row {line_number} in {path.name}")
            try:
                values.append(
                    (
                        parts[0],
                        parts[1],
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                        parts[6],
                    )
                )
            except ValueError as error:
                raise ValueError(f"Invalid numeric evidence on row {line_number}") from error
    return values


def _orf(orfs: Mapping[str, Orf], identifier: str) -> Orf:
    try:
        return orfs[identifier]
    except KeyError as error:
        raise ValueError(f"Evidence contains an unknown ORF identifier: {identifier}") from error


def arg_evidence(hits: Sequence[ArgHit]) -> list[EvidenceHit]:
    return [
        EvidenceHit(
            sequence_id=hit.sequence_id,
            orf_id=hit.orf_id,
            orf_start=hit.orf_start,
            orf_end=hit.orf_end,
            orf_strand=hit.orf_strand,
            evidence_group="ARG",
            source=hit.source,
            feature_type="antimicrobial_resistance",
            feature_name=hit.gene_symbol,
            accession=hit.accession,
            category=hit.drug_class,
            description=hit.gene_name,
            method=hit.method,
            identity=hit.identity,
            query_coverage=hit.query_coverage,
            evalue=hit.evalue,
            bitscore=hit.bitscore,
        )
        for hit in hits
    ]


def parse_amrfinderplus_non_amr(path: Path, orfs: Mapping[str, Orf]) -> list[EvidenceHit]:
    """Retain official AMRFinderPlus VIRULENCE and STRESS rows outside ARG consensus."""
    hits: list[EvidenceHit] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Protein id", "Element symbol", "Element name", "Type", "Method"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("AMRFinderPlus output schema is unsupported")
        for row in reader:
            element_type = row.get("Type", "").strip().upper()
            if element_type not in {"VIRULENCE", "STRESS"}:
                continue
            item = _orf(orfs, row["Protein id"].strip())

            def number(name: str, current_row: Mapping[str, str] = row) -> float | None:
                value = current_row.get(name, "").strip()
                return None if not value or value.upper() == "NA" else float(value)

            subtype = row.get("Subtype", "").strip().lower()
            category = "; ".join(
                value
                for value in (row.get("Class", "").strip(), row.get("Subclass", "").strip())
                if value
            ).lower()
            hits.append(
                EvidenceHit(
                    item.sequence_id,
                    item.identifier,
                    item.start,
                    item.end,
                    item.strand,
                    element_type,
                    "AMRFINDERPLUS",
                    subtype or element_type.lower(),
                    row["Element symbol"].strip() or "unknown",
                    row.get("Closest reference accession", "").strip() or "unknown",
                    category or "unknown",
                    row["Element name"].strip() or "unknown",
                    row["Method"].strip() or "AMRFINDERPLUS",
                    number("% Identity to reference"),
                    number("% Coverage of reference"),
                    None,
                    None,
                )
            )
    return hits


def load_vfdb_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            category = parts[2].strip() or "unknown"
            vfg = re.match(r"(VFG\d+)", parts[0])
            accession = re.search(r"\((?:gb|ref)\|([^|)]+)", parts[0])
            if vfg:
                metadata[vfg.group(1)] = category
            if accession:
                metadata[accession.group(1)] = category
    if not metadata:
        raise ValueError(f"VFDB metadata is empty or unsupported: {path}")
    return metadata


_VFDB_HEADER = re.compile(
    r"^(?P<vfg>VFG\d+)\((?:gb|ref)\|(?P<accession>[^|)]+)\)\s+"
    r"(?P<gene>\S+)(?:\s+\[(?P<group>[^\]]+)\])?(?:\s+\[(?P<organism>[^\]]+)\])?"
)


def parse_vfdb(
    path: Path, orfs: Mapping[str, Orf], metadata: Mapping[str, str]
) -> list[EvidenceHit]:
    hits: list[EvidenceHit] = []
    for query, subject, identity, coverage, evalue, bitscore, title in _rows(path):
        item = _orf(orfs, query)
        match = _VFDB_HEADER.match(title) or _VFDB_HEADER.match(subject)
        if match:
            feature = match.group("gene")
            accession = match.group("vfg")
            category = metadata.get(accession) or metadata.get(match.group("accession"), "unknown")
            description = "; ".join(
                value for value in (match.group("group"), match.group("organism")) if value
            )
        else:
            feature, accession, category, description = subject, subject, "unknown", title
        hits.append(
            EvidenceHit(
                item.sequence_id,
                query,
                item.start,
                item.end,
                item.strand,
                "VIRULENCE",
                "VFDB_CORE",
                "virulence_factor_homolog",
                feature,
                accession,
                str(category or "unknown"),
                description or feature,
                "DIAMOND_BLASTP",
                identity,
                coverage,
                evalue,
                bitscore,
            )
        )
    return hits


def _load_tsv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="", errors="replace") as handle:
        rows = {
            row.get(key, "").strip(): dict(row) for row in csv.DictReader(handle, delimiter="\t")
        }
    rows.pop("", None)
    if not rows:
        raise ValueError(f"Metadata is empty or unsupported: {path}")
    return rows


def parse_mge(path: Path, orfs: Mapping[str, Orf], metadata_path: Path) -> list[EvidenceHit]:
    """Parse the optional legacy ISfinder-derived evidence route."""
    metadata = _load_tsv(metadata_path, "ID")
    hits: list[EvidenceHit] = []
    for query, subject, identity, coverage, evalue, bitscore, title in _rows(path):
        item = _orf(orfs, query)
        entry = metadata.get(subject, {})
        feature = entry.get("gene_name", "").strip()
        if not feature:
            tokens = subject.split("_")
            feature = tokens[1] if len(tokens) > 1 else subject
        hits.append(
            EvidenceHit(
                item.sequence_id,
                query,
                item.start,
                item.end,
                item.strand,
                "MGE",
                "ISFINDER_LEGACY",
                entry.get("Sub_class", "").strip() or "mobile_genetic_element",
                feature,
                subject,
                entry.get("Class", "").strip() or "unknown",
                title or feature,
                "DIAMOND_BLASTP",
                identity,
                coverage,
                evalue,
                bitscore,
            )
        )
    return hits


_MOBILEOG_MAJOR_CATEGORIES = {
    "IE": "integration_excision",
    "P": "phage_associated",
    "RRR": "replication_recombination_repair",
    "T": "transfer",
}


def parse_mobileog(path: Path, orfs: Mapping[str, Orf]) -> list[EvidenceHit]:
    """Parse mobileOG-db 2.x headers without requiring a second metadata payload."""
    hits: list[EvidenceHit] = []
    for query, subject, identity, coverage, evalue, bitscore, title in _rows(path):
        item = _orf(orfs, query)
        fields = subject.split("|")
        if len(fields) < 5 or not fields[0].startswith("mobileOG_"):
            title_fields = title.split(None, 1)[0].split("|")
            fields = title_fields if len(title_fields) >= 5 else fields
        if len(fields) < 5 or not fields[0].startswith("mobileOG_"):
            raise ValueError("mobileOG-db header is not the supported pipe-delimited schema")
        accession = fields[0]
        feature = fields[1].strip() or accession
        uniprot = fields[2].strip() if len(fields) > 2 else ""
        major = fields[3].strip().upper() if len(fields) > 3 else ""
        minor = fields[4].strip() if len(fields) > 4 else ""
        origin = fields[6].strip() if len(fields) > 6 else ""
        description = "; ".join(
            value
            for value in (
                f"mobileOG-db protein family {feature}",
                f"UniProt {uniprot}" if uniprot and uniprot != "N/A" else "",
                f"origin {origin}" if origin and origin != "N/A" else "",
            )
            if value
        )
        hits.append(
            EvidenceHit(
                item.sequence_id,
                query,
                item.start,
                item.end,
                item.strand,
                "MGE",
                "MOBILEOG_DB",
                _MOBILEOG_MAJOR_CATEGORIES.get(major, "mobile_genetic_element"),
                feature,
                accession,
                minor or major or "unknown",
                description,
                "DIAMOND_BLASTP",
                identity,
                coverage,
                evalue,
                bitscore,
            )
        )
    return hits


def parse_bacmet(path: Path, orfs: Mapping[str, Orf], metadata_path: Path) -> list[EvidenceHit]:
    metadata = _load_tsv(metadata_path, "BacMet_ID")
    hits: list[EvidenceHit] = []
    for query, subject, identity, coverage, evalue, bitscore, title in _rows(path):
        item = _orf(orfs, query)
        accession = subject.split("|", 1)[0]
        subject_fields = subject.split("|")
        entry = metadata.get(accession, {})
        compounds = entry.get("Compound", "").strip()
        hits.append(
            EvidenceHit(
                item.sequence_id,
                query,
                item.start,
                item.end,
                item.strand,
                "STRESS",
                "BACMET2_EXPERIMENTAL",
                entry.get("Class", "").strip().lower() or "biocide_metal_resistance",
                entry.get("Gene_name", "").strip()
                or (subject_fields[1] if len(subject_fields) > 1 else subject),
                accession,
                compounds.split(" [class:", 1)[0] or "unknown",
                title or compounds,
                "DIAMOND_BLASTP",
                identity,
                coverage,
                evalue,
                bitscore,
            )
        )
    return hits


def _marker_name(family: str, subject: str, title: str) -> str:
    text = f"{subject} {title}"
    if family == "rep":
        fields = subject.split("|")
        return fields[1] if len(fields) > 1 else fields[0]
    pattern = r"\b(MOB[FPHQCVM][A-Za-z0-9_]*)\b" if family == "mob" else r"\b(MPF[_ ]?[FTGIBC])\b"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).upper().replace(" ", "_") if match else subject


def parse_mob_marker(path: Path, orfs: Mapping[str, Orf], family: str) -> list[EvidenceHit]:
    definitions = {
        "rep": ("replication", "MOB_SUITE_REPLICON"),
        "mob": ("relaxase", "MOB_SUITE_RELAXASE"),
        "mpf": ("mating_pair_formation", "MOB_SUITE_MPF"),
    }
    feature_type, source = definitions[family]
    hits: list[EvidenceHit] = []
    for query, subject, identity, coverage, evalue, bitscore, title in _rows(path):
        item = _orf(orfs, query)
        hits.append(
            EvidenceHit(
                item.sequence_id,
                query,
                item.start,
                item.end,
                item.strand,
                "MOBILITY",
                source,
                feature_type,
                _marker_name(family, subject, title),
                subject,
                family,
                title or subject,
                "DIAMOND_BLASTP",
                identity,
                coverage,
                evalue,
                bitscore,
            )
        )
    return hits


def _format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def write_evidence(path: Path, hits: Sequence[EvidenceHit]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=EVIDENCE_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for hit in hits:
            row = asdict(hit)
            for key in ("identity", "query_coverage", "evalue", "bitscore"):
                row[key] = _format_number(row[key])
            writer.writerow(row)


def load_predictions(path: Path, records: Sequence[FastaRecord]) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sequence_id", "length_bp", *PREDICTION_FIELDS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("MobiOrigin prediction schema is unsupported")
        rows = list(reader)
    if len(rows) != len(records):
        raise ValueError("Prediction and FASTA row counts differ")
    for record, row in zip(records, rows, strict=True):
        if row["sequence_id"] != record.identifier or int(row["length_bp"]) != len(record.sequence):
            raise ValueError("Prediction and FASTA order or length differs")
    return {row["sequence_id"]: row for row in rows}


def _priority(hits: Sequence[EvidenceHit]) -> tuple[str, str, str]:
    args = [hit for hit in hits if hit.evidence_group == "ARG"]
    mobility = {hit.feature_type for hit in hits if hit.evidence_group == "MOBILITY"}
    has_mge = any(hit.evidence_group == "MGE" for hit in hits)
    if args and {"relaxase", "mating_pair_formation"}.issubset(mobility):
        return "A", "ARG plus relaxase and mating-pair-formation evidence", "conjugative"
    if args and (mobility or has_mge):
        return (
            "B",
            "ARG plus partial mobility, replication, or MGE evidence",
            "mobilizable_or_mobile_context",
        )
    if args:
        return "C", "ARG evidence without detected mobility context", "not_detected"
    if any(hit.evidence_group in {"VIRULENCE", "MGE", "STRESS", "MOBILITY"} for hit in hits):
        return "D", "non-ARG biological evidence only", "not_applicable"
    return "E", "no retained biological evidence", "not_applicable"


def write_integrated_results(
    path: Path,
    records: Sequence[FastaRecord],
    hits: Sequence[EvidenceHit],
    predictions: Mapping[str, Mapping[str, str]],
    consensus_arg_hits: Sequence[EvidenceHit] | None = None,
) -> list[dict[str, object]]:
    grouped: dict[str, list[EvidenceHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.sequence_id].append(hit)
    consensus_grouped: dict[str, list[EvidenceHit]] = defaultdict(list)
    for hit in consensus_arg_hits or ():
        consensus_grouped[hit.sequence_id].append(hit)
    fields = (
        "sequence_id",
        "length_bp",
        *PREDICTION_FIELDS,
        "consensus_arg_orfs",
        "arg_genes",
        "arg_drug_classes",
        "arg_evidence_sources",
        "virulence_hits",
        "virulence_features",
        "mge_hits",
        "mge_features",
        "stress_biocide_metal_hits",
        "mobility_marker_hits",
        "mobility_class",
        "evidence_priority_tier",
        "priority_rationale",
        "virulence_colocalized_with_arg",
    )
    output_rows: list[dict[str, object]] = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            selected = grouped[record.identifier]
            arg_hits = [hit for hit in selected if hit.evidence_group == "ARG"]
            consensus_args = consensus_grouped[record.identifier] or arg_hits
            virulence = [hit for hit in selected if hit.evidence_group == "VIRULENCE"]
            mge = [hit for hit in selected if hit.evidence_group == "MGE"]
            stress = [hit for hit in selected if hit.evidence_group == "STRESS"]
            mobility = [hit for hit in selected if hit.evidence_group == "MOBILITY"]
            tier, rationale, mobility_class = _priority(selected)
            prediction = predictions.get(record.identifier, {})
            row: dict[str, object] = {
                "sequence_id": record.identifier,
                "length_bp": len(record.sequence),
                **{field: prediction.get(field, "") for field in PREDICTION_FIELDS},
                "consensus_arg_orfs": len({hit.orf_id for hit in consensus_args}),
                "arg_genes": ";".join(sorted({hit.feature_name for hit in consensus_args})),
                "arg_drug_classes": ";".join(
                    sorted(
                        {
                            item.strip()
                            for hit in consensus_args
                            for item in hit.category.split(";")
                            if item.strip() and item.strip() != "unknown"
                        }
                    )
                ),
                "arg_evidence_sources": ";".join(sorted({hit.source for hit in arg_hits})),
                "virulence_hits": len(virulence),
                "virulence_features": ";".join(sorted({hit.feature_name for hit in virulence})),
                "mge_hits": len(mge),
                "mge_features": ";".join(sorted({hit.feature_name for hit in mge})),
                "stress_biocide_metal_hits": len(stress),
                "mobility_marker_hits": len(mobility),
                "mobility_class": mobility_class,
                "evidence_priority_tier": tier,
                "priority_rationale": rationale,
                "virulence_colocalized_with_arg": str(bool(arg_hits and virulence)).lower(),
            }
            writer.writerow(row)
            output_rows.append(row)
    return output_rows


def write_publication_summary(
    path: Path, rows: Sequence[Mapping[str, object]], evidence: Sequence[EvidenceHit]
) -> None:
    prediction_counts = Counter(str(row.get("prediction") or "not_provided") for row in rows)
    tier_counts = Counter(str(row["evidence_priority_tier"]) for row in rows)
    group_counts = Counter(hit.evidence_group for hit in evidence)
    source_counts = Counter(hit.source for hit in evidence)
    summary = {
        "schema_version": "mobiorigin-publication-summary-v1",
        "records": len(rows),
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "evidence_priority_tier_counts": dict(sorted(tier_counts.items())),
        "evidence_group_hit_counts": dict(sorted(group_counts.items())),
        "evidence_source_hit_counts": dict(sorted(source_counts.items())),
        "records_with_arg": sum(int(str(row["consensus_arg_orfs"])) > 0 for row in rows),
        "records_with_virulence_homolog": sum(int(str(row["virulence_hits"])) > 0 for row in rows),
        "records_with_mge": sum(int(str(row["mge_hits"])) > 0 for row in rows),
        "records_with_arg_and_virulence": sum(
            row["virulence_colocalized_with_arg"] == "true" for row in rows
        ),
        "interpretation": {
            "priority_is_clinical_risk_score": False,
            "homology_proves_phenotype": False,
            "annotation_changes_origin_prediction": False,
        },
    }
    atomic_json(path, summary)


def write_html_report(path: Path, rows: Sequence[Mapping[str, object]], summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prioritized = sorted(
        rows,
        key=lambda row: (
            str(row["evidence_priority_tier"]),
            -int(str(row["consensus_arg_orfs"])),
            str(row["sequence_id"]),
        ),
    )[:100]
    table_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[field]))}</td>"
            for field in (
                "sequence_id",
                "prediction",
                "evidence_priority_tier",
                "consensus_arg_orfs",
                "arg_genes",
                "mobility_class",
                "virulence_hits",
                "mge_hits",
            )
        )
        + "</tr>"
        for row in prioritized
    )
    cards = "".join(
        f"<div class='card'><strong>{html.escape(label)}</strong><span>{value}</span></div>"
        for label, value in (
            ("Sequences", summary["records"]),
            ("ARG-positive", summary["records_with_arg"]),
            ("Virulence homolog-positive", summary["records_with_virulence_homolog"]),
            ("MGE-positive", summary["records_with_mge"]),
        )
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MobiOrigin biological-evidence report</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#172033;background:#f6f8fb}}
h1,h2{{color:#123b57}} .notice{{background:#fff4d6;border-left:5px solid #d99b00;padding:1rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1rem;margin:1.5rem 0}}
.card{{background:white;border:1px solid #dce3ea;border-radius:8px;padding:1rem;display:flex;flex-direction:column}}
.card span{{font-size:1.7rem;color:#096b72}} table{{border-collapse:collapse;width:100%;background:white;font-size:.86rem}}
th,td{{border:1px solid #dce3ea;padding:.45rem;text-align:left;vertical-align:top}} th{{background:#e9f2f5}}
code{{background:#edf1f4;padding:.1rem .25rem}} footer{{margin-top:2rem;color:#52606d}}
</style></head><body><h1>MobiOrigin biological-evidence report</h1>
<p class="notice"><strong>Interpretation boundary:</strong> evidence tiers prioritize records for review. They are not clinical risk scores, do not prove phenotype or plasmid origin, and never alter MobiOrigin predictions.</p>
<div class="cards">{cards}</div><h2>Priority logic</h2>
<p><strong>A</strong>: ARG plus relaxase and mating-pair-formation evidence; <strong>B</strong>: ARG plus partial mobility, replication, or MGE evidence; <strong>C</strong>: ARG without detected mobility context; <strong>D</strong>: non-ARG biological evidence only; <strong>E</strong>: no retained evidence.</p>
<h2>Highest-priority records (up to 100)</h2><table><thead><tr><th>Sequence</th><th>Prediction</th><th>Tier</th><th>ARG ORFs</th><th>ARG genes</th><th>Mobility</th><th>VF hits</th><th>MGE hits</th></tr></thead><tbody>{table_rows}</tbody></table>
<footer>Generated deterministically by MobiOrigin. See <code>annotation_provenance.json</code>, <code>biological_evidence.tsv</code>, and <code>SHA256SUMS.txt</code> for methods and identities.</footer></body></html>"""
    path.write_text(document, encoding="utf-8")


def file_identity(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
