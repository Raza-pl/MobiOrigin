"""Deterministic, dependency-free visualization of MobiOrigin outputs."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PREDICTION_FIELDS = (
    "sequence_id",
    "length_bp",
    "prediction",
    "p_chromosome",
    "p_plasmid",
    "p_phage",
    "plasmid_score",
    "abstention_reason",
)
LABELS = ("chromosome", "plasmid", "phage", "unclassified")
COLORS = {
    "chromosome": "#355C7D",
    "plasmid": "#2A9D8F",
    "phage": "#E9C46A",
    "unclassified": "#8D99AE",
}
TIER_COLORS = {
    "A": "#B42318",
    "B": "#D97706",
    "C": "#CA8A04",
    "D": "#2563EB",
    "E": "#64748B",
}
TIER_LABELS = {
    "A": "ARG-bearing conjugative candidate",
    "B": "ARG-bearing mobile-context candidate",
    "C": "ARG-bearing candidate without detected mobility",
    "D": "Non-ARG biological-evidence candidate",
    "E": "No retained annotation evidence",
}
TIER_DESCRIPTIONS = {
    "A": "ARG with relaxase and mating-pair-formation markers",
    "B": "ARG with partial mobility, replication, or MGE context",
    "C": "ARG without detected mobility context",
    "D": "Non-ARG biological evidence only",
    "E": "No evidence retained at configured thresholds",
}
ANNOTATION_CLASS_FIELDS = {
    "ARG": ("arg_drug_classes", "ARG drug classes"),
    "MGE": ("mge_classes", "MGE classes"),
    "VFG": ("virulence_classes", "Virulence-factor classes"),
    "BACMET": ("bacmet_classes", "BacMet resistance categories"),
}
EVIDENCE_COLORS = {
    "ARG": "#B42318",
    "VFG": "#7C3AED",
    "MGE": "#15803D",
    "BACMET": "#B45309",
}
LENGTH_BINS = (
    ("1–<2 kb", 1_000, 2_000),
    ("2–<5 kb", 2_000, 5_000),
    ("5–<10 kb", 5_000, 10_000),
    ("10–<50 kb", 10_000, 50_000),
    ("≥50 kb", 50_000, 2**63),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != PREDICTION_FIELDS:
            raise ValueError("Unsupported MobiOrigin prediction schema")
        for raw in reader:
            identifier = raw["sequence_id"]
            if not identifier or identifier in identifiers:
                raise ValueError("Prediction identifiers must be non-empty and unique")
            identifiers.add(identifier)
            label = raw["prediction"]
            if label not in LABELS:
                raise ValueError(f"Unsupported prediction label: {label}")
            length = int(raw["length_bp"])
            probabilities = tuple(float(raw[field]) for field in PREDICTION_FIELDS[3:6])
            if length < 1 or any(value < 0 or value > 1 for value in probabilities):
                raise ValueError(f"Invalid prediction row for {identifier}")
            if abs(sum(probabilities) - 1.0) > 1e-5:
                raise ValueError(f"Probabilities do not sum to one for {identifier}")
            rows.append(
                {
                    "sequence_id": identifier,
                    "length_bp": length,
                    "prediction": label,
                    "p_chromosome": probabilities[0],
                    "p_plasmid": probabilities[1],
                    "p_phage": probabilities[2],
                    "plasmid_score": float(raw["plasmid_score"]),
                }
            )
    if not rows:
        raise ValueError("Prediction table is empty")
    return rows


def _annotation_rows(path: Path, identifiers: Sequence[str]) -> dict[str, dict[str, str]]:
    required = {
        "sequence_id",
        "prediction",
        "consensus_arg_orfs",
        "mge_hits",
        "mobility_marker_hits",
        "evidence_priority_tier",
    }
    observed: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("Unsupported MobiOrigin integrated-annotation schema")
        for row in reader:
            identifier = row["sequence_id"]
            if identifier in observed:
                raise ValueError("Annotation identifiers must be unique")
            observed[identifier] = row
    if list(observed) != list(identifiers):
        raise ValueError("Prediction and annotation identifiers or order differ")
    return observed


def _summaries(
    rows: Sequence[Mapping[str, Any]],
    annotation: Mapping[str, Mapping[str, str]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    total_records = len(rows)
    total_bases = sum(int(row["length_bp"]) for row in rows)
    label_counts = Counter(str(row["prediction"]) for row in rows)
    label_bases: Counter[str] = Counter()
    for row in rows:
        label_bases[str(row["prediction"])] += int(row["length_bp"])
    prediction_summary = [
        {
            "prediction": label,
            "contigs": label_counts[label],
            "contig_fraction": label_counts[label] / total_records,
            "bases": label_bases[label],
            "base_fraction": label_bases[label] / total_bases,
        }
        for label in LABELS
    ]

    bin_totals: Counter[str] = Counter()
    bin_plasmids: Counter[str] = Counter()
    bin_bases: Counter[str] = Counter()
    bin_plasmid_bases: Counter[str] = Counter()
    for row in rows:
        length = int(row["length_bp"])
        for label, lower, upper in LENGTH_BINS:
            if lower <= length < upper:
                bin_totals[label] += 1
                bin_bases[label] += length
                if row["prediction"] == "plasmid":
                    bin_plasmids[label] += 1
                    bin_plasmid_bases[label] += length
                break
    length_summary = [
        {
            "length_bin": label,
            "contigs": bin_totals[label],
            "plasmid_contigs": bin_plasmids[label],
            "plasmid_contig_fraction": (
                bin_plasmids[label] / bin_totals[label] if bin_totals[label] else 0.0
            ),
            "bases": bin_bases[label],
            "plasmid_bases": bin_plasmid_bases[label],
            "plasmid_base_fraction": (
                bin_plasmid_bases[label] / bin_bases[label] if bin_bases[label] else 0.0
            ),
        }
        for label, _, _ in LENGTH_BINS
    ]

    extra: dict[str, object] = {}
    if annotation is not None:
        tier_counts = Counter(row["evidence_priority_tier"] for row in annotation.values())
        arg_by_label: Counter[str] = Counter()
        for row in annotation.values():
            if int(row["consensus_arg_orfs"]):
                arg_by_label[row["prediction"]] += 1
        extra = {
            "evidence_priority_tier_counts": {
                tier: tier_counts[tier] for tier in ("A", "B", "C", "D", "E")
            },
            "arg_positive_records_by_prediction": {label: arg_by_label[label] for label in LABELS},
        }
    return (
        prediction_summary,
        length_summary,
        {
            "records": total_records,
            "bases": total_bases,
            **extra,
        },
    )


def _write_tsv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty table: {path.name}")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _terms(value: object) -> list[str]:
    """Split one deterministic semicolon-delimited annotation cell."""
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def _annotation_summaries(
    annotation: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build tier, class, and priority-candidate tables from integrated annotation."""
    rows = list(annotation.values())
    total_records = len(rows)
    tier_rows: list[dict[str, Any]] = []
    for tier in ("A", "B", "C", "D", "E"):
        selected = [row for row in rows if row["evidence_priority_tier"] == tier]
        tier_rows.append(
            {
                "tier": tier,
                "label": TIER_LABELS[tier],
                "description": TIER_DESCRIPTIONS[tier],
                "contigs": len(selected),
                "contig_fraction": len(selected) / total_records if total_records else 0.0,
                "bases": sum(int(row.get("length_bp") or 0) for row in selected),
                "plasmid_predictions": sum(row.get("prediction") == "plasmid" for row in selected),
                "conjugative_candidates": sum(
                    row.get("mobility_class") == "conjugative" for row in selected
                ),
            }
        )

    class_rows: list[dict[str, Any]] = []
    for group, (field, _) in ANNOTATION_CLASS_FIELDS.items():
        counts: Counter[str] = Counter()
        origin_counts: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
        for row in rows:
            terms = set(_terms(row.get(field, "")))
            counts.update(terms)
            origin_counts[str(row.get("prediction") or "unclassified")].update(terms)
        for annotation_class, contigs in sorted(
            counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0])
        ):
            class_rows.append(
                {
                    "evidence_group": group,
                    "class": annotation_class,
                    "contigs": contigs,
                    "contig_fraction": contigs / total_records if total_records else 0.0,
                    **{
                        f"{label}_contigs": origin_counts[label][annotation_class]
                        for label in LABELS
                    },
                }
            )

    candidate_fields = (
        "sequence_id",
        "length_bp",
        "prediction",
        "evidence_priority_tier",
        "evidence_priority_label",
        "arg_genes",
        "arg_gene_families",
        "arg_drug_classes",
        "arg_mechanisms",
        "mobility_class",
        "mobility_marker_types",
        "mge_classes",
        "virulence_classes",
        "bacmet_genes",
        "bacmet_gene_families",
        "bacmet_classes",
    )
    priority = [
        {field: row.get(field, "") for field in candidate_fields}
        for row in rows
        if row["evidence_priority_tier"] in {"A", "B", "C"}
        or row.get("mobility_class") == "conjugative"
    ]
    priority.sort(
        key=lambda row: (
            str(row["evidence_priority_tier"]),
            0 if row["prediction"] == "plasmid" else 1,
            -int(str(row["length_bp"] or 0)),
            str(row["sequence_id"]),
        )
    )
    return tier_rows, class_rows, priority


def _bar(x: float, y: float, width: float, height: float, color: str, label: str) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{color}" rx="3"><title>{html.escape(label)}</title></rect>'
    )


def _svg(
    prediction: Sequence[Mapping[str, Any]],
    length: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> str:
    width, height = 1200, 760
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F7F9FC"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#173B57}.title{font-size:28px;font-weight:700}.sub{font-size:14px;fill:#5D7184}.panel{font-size:19px;font-weight:700}.axis{font-size:12px}.value{font-size:12px;font-weight:700}</style>",
        '<text x="55" y="48" class="title">MobiOrigin prediction summary</text>',
        f'<text x="55" y="73" class="sub">{int(summary["records"]):,} sequences · {int(summary["bases"]):,} bp · descriptive output visualization</text>',
        '<text x="55" y="120" class="panel">A  Prediction composition</text>',
        '<text x="625" y="120" class="panel">B  Plasmid calls by contig length</text>',
    ]
    max_share = max(float(row["contig_fraction"]) for row in prediction) or 1.0
    for index, row in enumerate(prediction):
        label = str(row["prediction"])
        y = 155 + index * 70
        contig = float(row["contig_fraction"])
        base = float(row["base_fraction"])
        parts.extend(
            [
                f'<text x="55" y="{y + 17}" class="axis">{html.escape(label.title())}</text>',
                _bar(
                    155,
                    y,
                    350 * contig / max_share,
                    20,
                    COLORS[label],
                    f"Contig share {contig:.2%}",
                ),
                _bar(155, y + 25, 350 * base / max_share, 13, "#8EC9C1", f"Base share {base:.2%}"),
                f'<text x="515" y="{y + 17}" class="value" text-anchor="end">{contig:.1%}</text>',
                f'<text x="515" y="{y + 38}" class="axis" text-anchor="end">{base:.1%} bp</text>',
            ]
        )
    max_length_share = max(float(row["plasmid_contig_fraction"]) for row in length) or 1.0
    for index, row in enumerate(length):
        x = 630 + index * 100
        share = float(row["plasmid_contig_fraction"])
        bar_height = 250 * share / max_length_share
        parts.extend(
            [
                _bar(x, 430 - bar_height, 58, bar_height, "#2A9D8F", f"{share:.2%}"),
                f'<text x="{x + 29}" y="{450}" class="axis" text-anchor="middle">{html.escape(str(row["length_bin"]))}</text>',
                f'<text x="{x + 29}" y="{420 - bar_height}" class="value" text-anchor="middle">{share:.1%}</text>',
            ]
        )
    parts.extend(
        [
            '<rect x="55" y="515" width="1090" height="170" rx="10" fill="#FFFFFF" stroke="#D9E2EC"/>',
            '<text x="80" y="550" class="panel">Interpretation</text>',
            '<text x="80" y="580" class="sub">• Bar pairs show contig share (dark) and base-pair share (light).</text>',
            '<text x="80" y="608" class="sub">• Length-stratified bars show the fraction of contigs called plasmid within each fixed bin.</text>',
            '<text x="80" y="636" class="sub">• These plots summarize predictions; they do not estimate accuracy or prove plasmid origin.</text>',
            '<text x="80" y="664" class="sub">• Biological evidence tiers, when supplied, prioritize follow-up and are not clinical risk scores.</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _annotation_svg(
    tiers: Sequence[Mapping[str, Any]],
    classes: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> str:
    """Render evidence tiers and leading annotation classes as editable SVG."""
    width, height = 1200, 820
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:28px;font-weight:700}.sub{font-size:14px;fill:#52606d}.panel{font-size:19px;font-weight:700}.axis{font-size:12px}.value{font-size:12px;font-weight:700}</style>",
        '<text x="55" y="48" class="title">MobiOrigin annotation overview</text>',
        f'<text x="55" y="73" class="sub">{int(summary["records"]):,} contigs · retained homology and mobility evidence</text>',
        '<text x="55" y="120" class="panel">A  Evidence-priority tiers</text>',
        '<text x="620" y="120" class="panel">B  Leading annotation classes</text>',
    ]
    maximum_tier = max((int(row["contigs"]) for row in tiers), default=1) or 1
    for index, row in enumerate(tiers):
        tier = str(row["tier"])
        count = int(row["contigs"])
        y = 150 + index * 78
        parts.extend(
            [
                f'<circle cx="75" cy="{y + 16}" r="17" fill="{TIER_COLORS[tier]}"/>',
                f'<text x="75" y="{y + 21}" text-anchor="middle" style="fill:white;font-weight:700">{tier}</text>',
                _bar(
                    110,
                    y,
                    390 * count / maximum_tier,
                    32,
                    TIER_COLORS[tier],
                    f"Tier {tier}: {count:,} contigs",
                ),
                f'<text x="510" y="{y + 21}" class="value" text-anchor="end">{count:,}</text>',
                f'<text x="110" y="{y + 51}" class="axis">{html.escape(TIER_LABELS[tier])}</text>',
            ]
        )

    leading = sorted(
        classes,
        key=lambda row: (-int(row["contigs"]), str(row["class"]).casefold()),
    )[:12]
    maximum_class = max((int(row["contigs"]) for row in leading), default=1) or 1
    for index, row in enumerate(leading):
        group = str(row["evidence_group"])
        label = str(row["class"])
        shortened = label if len(label) <= 34 else f"{label[:31]}…"
        count = int(row["contigs"])
        y = 150 + index * 45
        parts.extend(
            [
                f'<text x="620" y="{y + 15}" class="axis">{html.escape(shortened)}</text>',
                _bar(
                    855,
                    y,
                    260 * count / maximum_class,
                    22,
                    EVIDENCE_COLORS.get(group, "#64748B"),
                    f"{group}: {label}; {count:,} contigs",
                ),
                f'<text x="1140" y="{y + 16}" class="value" text-anchor="end">{count:,}</text>',
            ]
        )
    if not leading:
        parts.append(
            '<text x="620" y="170" class="sub">No class-level annotation was retained.</text>'
        )
    parts.extend(
        [
            '<rect x="55" y="650" width="1090" height="105" rx="10" fill="#FFFFFF" stroke="#D9E2EC"/>',
            '<text x="80" y="686" class="panel">Interpretation boundary</text>',
            '<text x="80" y="716" class="sub">Class bars count contigs with at least one retained class; multiple classes may occur on one contig.</text>',
            '<text x="80" y="741" class="sub">Tiers prioritize review and do not prove origin, phenotype, transferability, or clinical risk.</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _priority_svg(
    tiers: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> str:
    """Render an editable summary focused on ARG-bearing and conjugative candidates."""
    width, height = 1200, 520
    tier_counts = {str(row["tier"]): int(row["contigs"]) for row in tiers}
    conjugative = sum(row.get("mobility_class") == "conjugative" for row in candidates)
    plasmid_conjugative = sum(
        row.get("mobility_class") == "conjugative" and row.get("prediction") == "plasmid"
        for row in candidates
    )
    cards = [
        ("Tier A", tier_counts.get("A", 0), "ARG + relaxase + MPF", TIER_COLORS["A"]),
        ("Tier B", tier_counts.get("B", 0), "ARG + partial mobile context", TIER_COLORS["B"]),
        ("Tier C", tier_counts.get("C", 0), "ARG without mobile context", TIER_COLORS["C"]),
        ("Conjugative candidates", conjugative, "marker-defined candidates", "#0369A1"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:28px;font-weight:700}.sub{font-size:14px;fill:#52606d}.count{font-size:42px;font-weight:700}.label{font-size:17px;font-weight:700}</style>",
        '<text x="55" y="48" class="title">Priority and mobility candidates</text>',
        '<text x="55" y="73" class="sub">Candidate counts for focused review and experimental follow-up</text>',
    ]
    for index, (label, count, detail, color) in enumerate(cards):
        x = 55 + index * 280
        parts.extend(
            [
                f'<rect x="{x}" y="115" width="250" height="185" rx="14" fill="#FFFFFF" stroke="#D9E2EC"/>',
                f'<rect x="{x}" y="115" width="250" height="10" rx="5" fill="{color}"/>',
                f'<text x="{x + 20}" y="167" class="label">{html.escape(label)}</text>',
                f'<text x="{x + 20}" y="225" class="count" style="fill:{color}">{count:,}</text>',
                f'<text x="{x + 20}" y="268" class="sub">{html.escape(detail)}</text>',
            ]
        )
    parts.extend(
        [
            '<rect x="55" y="335" width="1090" height="115" rx="12" fill="#FFFFFF" stroke="#D9E2EC"/>',
            '<text x="80" y="373" class="label">Conjugative-candidate context</text>',
            f'<text x="80" y="407" class="sub">{plasmid_conjugative:,} conjugative candidate(s) were also predicted as plasmid.</text>',
            '<text x="80" y="432" class="sub">Marker combinations are candidate evidence. Experimental transferability and complete plasmid structure require confirmation.</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _evidence_origin_svg(
    group: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Render one annotation category with stacked counts by predicted origin."""
    title = ANNOTATION_CLASS_FIELDS[group][1]
    selected = [row for row in rows if row["evidence_group"] == group][:15]
    width = 1200
    height = max(390, 175 + 43 * len(selected))
    maximum = max((int(row["contigs"]) for row in selected), default=1) or 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:27px;font-weight:700}.sub{font-size:14px;fill:#52606d}.axis{font-size:13px}.value{font-size:12px;font-weight:700}</style>",
        f'<text x="55" y="48" class="title">{html.escape(title)} by predicted origin</text>',
        '<text x="55" y="73" class="sub">Contigs with retained evidence; each bar is divided by the MobiOrigin prediction</text>',
    ]
    legend_x = 55
    for label in LABELS:
        parts.extend(
            [
                f'<rect x="{legend_x}" y="95" width="16" height="16" rx="3" fill="{COLORS[label]}"/>',
                f'<text x="{legend_x + 23}" y="108" class="axis">{html.escape(label.title())}</text>',
            ]
        )
        legend_x += 145
    for index, row in enumerate(selected):
        y = 140 + index * 43
        label = str(row["class"])
        shortened = label if len(label) <= 42 else f"{label[:39]}…"
        parts.append(f'<text x="55" y="{y + 17}" class="axis">{html.escape(shortened)}</text>')
        x = 360.0
        for origin in LABELS:
            count = int(row[f"{origin}_contigs"])
            segment = 720 * count / maximum
            if segment:
                parts.append(
                    _bar(
                        x,
                        y,
                        segment,
                        24,
                        COLORS[origin],
                        f"{label}; {origin}: {count:,} contigs",
                    )
                )
            x += segment
        parts.append(
            f'<text x="1110" y="{y + 17}" class="value" text-anchor="end">{int(row["contigs"]):,}</text>'
        )
    if not selected:
        parts.append(
            f'<text x="55" y="160" class="sub">No retained {html.escape(title.lower())} were detected.</text>'
        )
    parts.extend(
        [
            f'<text x="55" y="{height - 38}" class="sub">Counts describe retained homology evidence and are not prevalence or accuracy estimates.</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _html_report(
    svg: str,
    prediction: Sequence[Mapping[str, Any]],
    length: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    annotation_svg: str | None = None,
    priority_svg: str | None = None,
    tiers: Sequence[Mapping[str, Any]] = (),
    classes: Sequence[Mapping[str, Any]] = (),
    candidates: Sequence[Mapping[str, Any]] = (),
    result_rows: Sequence[Mapping[str, Any]] = (),
    evidence_svgs: Mapping[str, str] | None = None,
) -> str:
    prediction_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['prediction']))}</td>"
        f"<td>{int(row['contigs']):,}</td><td>{float(row['contig_fraction']):.2%}</td>"
        f"<td>{int(row['bases']):,}</td><td>{float(row['base_fraction']):.2%}</td></tr>"
        for row in prediction
    )
    length_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['length_bin']))}</td><td>{int(row['contigs']):,}</td>"
        f"<td>{int(row['plasmid_contigs']):,}</td>"
        f"<td>{float(row['plasmid_contig_fraction']):.2%}</td>"
        f"<td>{float(row['plasmid_base_fraction']):.2%}</td></tr>"
        for row in length
    )
    annotation_section = ""
    if annotation_svg is not None and priority_svg is not None:
        tier_cards = "".join(
            "<div class='tier-card' style='border-top-color:"
            f"{TIER_COLORS[str(row['tier'])]}'><div><strong>Tier {row['tier']}</strong>"
            f"<span>{int(row['contigs']):,}</span></div>"
            f"<p>{html.escape(str(row['label']))}</p>"
            f"<small>{html.escape(str(row['description']))}</small></div>"
            for row in tiers
        )
        class_rows = "".join(
            "<tr>"
            f"<td><span class='group group-{str(row['evidence_group']).lower()}'>{html.escape(str(row['evidence_group']))}</span></td>"
            f"<td>{html.escape(str(row['class']))}</td>"
            f"<td>{int(row['contigs']):,}</td>"
            f"<td>{int(row['chromosome_contigs']):,}</td>"
            f"<td>{int(row['plasmid_contigs']):,}</td>"
            f"<td>{int(row['phage_contigs']):,}</td>"
            f"<td>{int(row['unclassified_contigs']):,}</td>"
            f"<td>{float(row['contig_fraction']):.2%}</td></tr>"
            for row in classes[:100]
        )
        if not class_rows:
            class_rows = "<tr><td colspan='8'>No class-level annotation was retained.</td></tr>"
        result_table_rows = "".join(
            "<tr "
            f"data-tier='{html.escape(str(row['evidence_priority_tier']))}' "
            f"data-search='{html.escape(' '.join(str(value) for value in row.values()).lower())}'>"
            f"<td>{html.escape(str(row['sequence_id']))}</td>"
            f"<td>{html.escape(str(row['prediction']))}</td>"
            f"<td><span class='tier tier-{html.escape(str(row['evidence_priority_tier']).lower())}'>{html.escape(str(row['evidence_priority_tier']))}</span></td>"
            f"<td>{html.escape(str(row.get('arg_genes', '')))}</td>"
            f"<td>{html.escape(str(row.get('arg_gene_families', '')))}</td>"
            f"<td>{html.escape(str(row.get('arg_drug_classes', '')))}</td>"
            f"<td>{html.escape(str(row.get('arg_mechanisms', '')))}</td>"
            f"<td>{html.escape(str(row.get('mobility_class', '')))}</td>"
            f"<td>{html.escape(str(row.get('mobility_marker_types', '')))}</td>"
            f"<td>{html.escape(str(row.get('virulence_classes', '')))}</td>"
            f"<td>{html.escape(str(row.get('mge_classes', '')))}</td>"
            f"<td>{html.escape(str(row.get('bacmet_gene_families', '')))}</td>"
            f"<td>{html.escape(str(row.get('bacmet_classes', '')))}</td></tr>"
            for row in result_rows
        )
        if not result_table_rows:
            result_table_rows = "<tr><td colspan='14'>No annotation rows were supplied.</td></tr>"
        evidence_figures = "".join(
            f"<section class='evidence-figure'><div class='figure'>{evidence_svgs[group]}</div></section>"
            for group in ANNOTATION_CLASS_FIELDS
            if evidence_svgs and group in evidence_svgs
        )
        annotation_section = f"""
<section id="annotation"><h2>Annotation overview</h2>
<div class="figure">{annotation_svg}</div>
<h3>Evidence tiers explained</h3><div class="tier-grid">{tier_cards}</div>
<h3>Annotation plots</h3>{evidence_figures}
<h3>Annotation class summary</h3><p>Counts represent contigs with at least one retained annotation in each class. One contig may contribute to several classes.</p>
<div class="table-wrap"><table><thead><tr><th>Evidence group</th><th>Class or family</th><th>Total</th><th>Chromosome</th><th>Plasmid</th><th>Phage</th><th>Unclassified</th><th>All-contig share</th></tr></thead><tbody>{class_rows}</tbody></table></div>
</section>
<section id="results"><h2>Annotation results</h2>
<div class="figure">{priority_svg}</div>
<div class="controls"><label>Tier <select id="tier-filter"><option value="">All</option><option>A</option><option>B</option><option>C</option><option>D</option><option>E</option></select></label><label>Search <input id="candidate-search" type="search" placeholder="gene, class, contig or mobility"></label></div>
<div class="table-wrap"><table id="candidate-table"><thead><tr><th>Sequence</th><th>Prediction</th><th>Tier</th><th>ARG genes</th><th>ARG families</th><th>ARG classes</th><th>ARG mechanisms</th><th>Mobility</th><th>Marker types</th><th>VFG classes</th><th>MGE classes</th><th>BacMet families</th><th>BacMet classes</th></tr></thead><tbody>{result_table_rows}</tbody></table></div>
</section>"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MobiOrigin visualization</title><style>
body{{font-family:Inter,system-ui,sans-serif;margin:0 auto;max-width:1380px;color:#172033;background:#f7f9fc;padding:0 2rem 3rem}}
header{{background:linear-gradient(135deg,#0f3654,#096b72);color:white;margin:0 -2rem 2rem;padding:2.4rem}}
header h1{{color:white;margin:0 0 .4rem}} header p{{margin:0;color:#d8f3f1}}
h1,h2,h3{{color:#173b57}} nav{{display:flex;gap:1rem;flex-wrap:wrap;margin-top:1.2rem}} nav a{{color:white;text-decoration:none;font-weight:650}}
.notice{{background:#fff4d6;border-left:5px solid #d99b00;padding:1rem;border-radius:5px}}
.figure{{background:white;border:1px solid #d9e2ec;border-radius:10px;padding:1rem}} svg{{width:100%;height:auto}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1rem}}
.card{{background:white;border:1px solid #d9e2ec;border-radius:8px;padding:1rem;display:flex;flex-direction:column}}
.card span{{font-size:1.6rem;color:#2a9d8f}} .tier-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem;margin:1rem 0 2rem}}
.tier-card{{background:white;border:1px solid #d9e2ec;border-top:7px solid;border-radius:9px;padding:1rem}} .tier-card div{{display:flex;justify-content:space-between;align-items:center}} .tier-card span{{font-size:1.7rem;font-weight:750}}
.tier-card p{{font-weight:650;margin:.7rem 0 .4rem}} .tier-card small{{color:#52606d}}
.table-wrap{{overflow:auto;max-height:620px;background:white;border:1px solid #d9e2ec;border-radius:8px;margin-bottom:2rem}}
table{{border-collapse:collapse;width:100%;background:white}} th,td{{border-bottom:1px solid #e3e9ef;padding:.55rem;text-align:left;vertical-align:top}} th{{background:#e9f2f5;position:sticky;top:0;z-index:1;white-space:nowrap}}
.group,.tier{{display:inline-block;border-radius:999px;padding:.2rem .5rem;color:white;font-size:.76rem;font-weight:750}} .group-arg,.tier-a{{background:#b42318}} .group-vfg{{background:#7c3aed}} .group-mge{{background:#15803d}} .group-bacmet,.tier-b{{background:#b45309}} .tier-c{{background:#ca8a04}}
.evidence-figure{{margin:1rem 0}}
.controls{{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}} .controls label{{font-weight:650}} select,input{{margin-left:.4rem;padding:.45rem;border:1px solid #9fb3c8;border-radius:5px}}
</style></head><body><header><h1>MobiOrigin results</h1><p>Origin prediction and biological annotation</p><nav><a href="#prediction">Prediction</a><a href="#annotation">Annotation</a><a href="#results">Results table</a></nav></header>
<p class="notice"><strong>Interpretation boundary:</strong> prediction summaries and evidence tiers do not prove sequence origin, phenotype, transferability, or clinical risk.</p>
<section id="prediction"><div class="figure">{svg}</div>
<h2>Prediction composition</h2><table><thead><tr><th>Prediction</th><th>Contigs</th><th>Contig share</th><th>Bases</th><th>Base share</th></tr></thead><tbody>{prediction_rows}</tbody></table>
<h2>Length-stratified plasmid calls</h2><table><thead><tr><th>Length</th><th>Contigs</th><th>Plasmid calls</th><th>Contig share</th><th>Base share</th></tr></thead><tbody>{length_rows}</tbody></table>
</section>{annotation_section}<p>Generated deterministically by MobiOrigin.</p>
<script>
const tier=document.getElementById('tier-filter'); const search=document.getElementById('candidate-search');
function filterCandidates(){{if(!tier||!search)return; document.querySelectorAll('#candidate-table tbody tr[data-tier]').forEach(row=>{{const tierOK=!tier.value||row.dataset.tier===tier.value; const searchOK=!search.value||row.dataset.search.includes(search.value.toLowerCase()); row.style.display=tierOK&&searchOK?'':'none';}});}}
if(tier)tier.addEventListener('change',filterCandidates); if(search)search.addEventListener('input',filterCandidates);
</script></body></html>"""


def visualize(
    *,
    predictions_tsv: Path,
    output_dir: Path,
    annotated_results_tsv: Path | None = None,
) -> None:
    """Create deterministic publication-oriented tables, SVG, and HTML."""
    if output_dir.exists():
        raise FileExistsError("Visualization output directory already exists")
    rows = _prediction_rows(predictions_tsv)
    identifiers = [str(row["sequence_id"]) for row in rows]
    annotation = (
        _annotation_rows(annotated_results_tsv, identifiers)
        if annotated_results_tsv is not None
        else None
    )
    prediction, length, summary = _summaries(rows, annotation)
    tier_summary: list[dict[str, Any]] = []
    class_summary: list[dict[str, Any]] = []
    priority_candidates: list[dict[str, Any]] = []
    evidence_svgs: dict[str, str] = {}
    if annotation is not None:
        tier_summary, class_summary, priority_candidates = _annotation_summaries(annotation)
    summary.update(
        {
            "schema_version": "mobiorigin-visualization-summary-v2",
            "predictions_sha256": _sha256(predictions_tsv),
            "annotated_results_sha256": (
                _sha256(annotated_results_tsv) if annotated_results_tsv is not None else None
            ),
            "accuracy_metrics_calculated": False,
            "clinical_risk_scores_calculated": False,
            "annotation_class_counts_are_contig_presence_counts": True,
            "priority_candidates": len(priority_candidates),
            "conjugative_candidates": sum(
                row.get("mobility_class") == "conjugative" for row in priority_candidates
            ),
        }
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _write_tsv(temporary / "prediction_summary.tsv", prediction)
        _write_tsv(temporary / "prediction_by_length_bin.tsv", length)
        annotation_svg: str | None = None
        priority_svg: str | None = None
        if annotation is not None:
            _write_tsv(temporary / "evidence_tier_summary.tsv", tier_summary)
            _write_tsv(
                temporary / "annotation_class_summary.tsv",
                class_summary,
                fields=(
                    "evidence_group",
                    "class",
                    "contigs",
                    "chromosome_contigs",
                    "plasmid_contigs",
                    "phage_contigs",
                    "unclassified_contigs",
                    "contig_fraction",
                ),
            )
            _write_tsv(
                temporary / "priority_candidates.tsv",
                priority_candidates,
                fields=(
                    "sequence_id",
                    "length_bp",
                    "prediction",
                    "evidence_priority_tier",
                    "evidence_priority_label",
                    "arg_genes",
                    "arg_gene_families",
                    "arg_drug_classes",
                    "arg_mechanisms",
                    "mobility_class",
                    "mobility_marker_types",
                    "mge_classes",
                    "virulence_classes",
                    "bacmet_genes",
                    "bacmet_gene_families",
                    "bacmet_classes",
                ),
            )
            annotation_svg = _annotation_svg(tier_summary, class_summary, summary)
            priority_svg = _priority_svg(tier_summary, priority_candidates)
            (temporary / "mobiorigin_annotation_summary.svg").write_text(
                annotation_svg + "\n", encoding="utf-8"
            )
            (temporary / "mobiorigin_priority_candidates.svg").write_text(
                priority_svg + "\n", encoding="utf-8"
            )
            evidence_filenames = {
                "ARG": "mobiorigin_arg_classes.svg",
                "MGE": "mobiorigin_mge_classes.svg",
                "VFG": "mobiorigin_virulence_classes.svg",
                "BACMET": "mobiorigin_bacmet_categories.svg",
            }
            for group, filename in evidence_filenames.items():
                evidence_svgs[group] = _evidence_origin_svg(group, class_summary)
                (temporary / filename).write_text(evidence_svgs[group] + "\n", encoding="utf-8")
        (temporary / "visualization_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        svg = _svg(prediction, length, summary)
        (temporary / "mobiorigin_summary.svg").write_text(svg + "\n", encoding="utf-8")
        (temporary / "mobiorigin_dashboard.html").write_text(
            _html_report(
                svg,
                prediction,
                length,
                summary,
                annotation_svg=annotation_svg,
                priority_svg=priority_svg,
                tiers=tier_summary,
                classes=class_summary,
                candidates=priority_candidates,
                result_rows=list(annotation.values()) if annotation is not None else (),
                evidence_svgs=evidence_svgs,
            ),
            encoding="utf-8",
        )
        outputs = sorted(path for path in temporary.iterdir() if path.is_file())
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(f"{_sha256(path)}  {path.name}\n" for path in outputs), encoding="ascii"
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
