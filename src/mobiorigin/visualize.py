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


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def _html_report(
    svg: str,
    prediction: Sequence[Mapping[str, Any]],
    length: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
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
    tier_section = ""
    tiers = summary.get("evidence_priority_tier_counts")
    if isinstance(tiers, dict):
        tier_section = (
            "<h2>Biological-evidence priority tiers</h2><div class='cards'>"
            + "".join(
                f"<div class='card'><strong>Tier {tier}</strong><span>{int(tiers.get(tier, 0)):,}</span></div>"
                for tier in ("A", "B", "C", "D", "E")
            )
            + "</div>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MobiOrigin visualization</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1200px;color:#173b57;background:#f7f9fc}}
h1,h2{{color:#173b57}} .notice{{background:#fff4d6;border-left:5px solid #d99b00;padding:1rem}}
.figure{{background:white;border:1px solid #d9e2ec;border-radius:10px;padding:1rem}} svg{{width:100%;height:auto}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1rem}}
.card{{background:white;border:1px solid #d9e2ec;border-radius:8px;padding:1rem;display:flex;flex-direction:column}}
.card span{{font-size:1.6rem;color:#2a9d8f}} table{{border-collapse:collapse;width:100%;background:white;margin-bottom:2rem}}
th,td{{border:1px solid #d9e2ec;padding:.5rem;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#e9f2f5}}
</style></head><body><h1>MobiOrigin results</h1>
<p class="notice"><strong>Interpretation boundary:</strong> prediction summaries and evidence tiers do not prove sequence origin, phenotype, transferability, or clinical risk.</p>
<div class="figure">{svg}</div>
<h2>Prediction composition</h2><table><thead><tr><th>Prediction</th><th>Contigs</th><th>Contig share</th><th>Bases</th><th>Base share</th></tr></thead><tbody>{prediction_rows}</tbody></table>
<h2>Length-stratified plasmid calls</h2><table><thead><tr><th>Length</th><th>Contigs</th><th>Plasmid calls</th><th>Contig share</th><th>Base share</th></tr></thead><tbody>{length_rows}</tbody></table>
{tier_section}<p>Generated deterministically by MobiOrigin.</p></body></html>"""


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
    summary.update(
        {
            "schema_version": "mobiorigin-visualization-summary-v1",
            "predictions_sha256": _sha256(predictions_tsv),
            "annotated_results_sha256": (
                _sha256(annotated_results_tsv) if annotated_results_tsv is not None else None
            ),
            "accuracy_metrics_calculated": False,
            "clinical_risk_scores_calculated": False,
        }
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _write_tsv(temporary / "prediction_summary.tsv", prediction)
        _write_tsv(temporary / "prediction_by_length_bin.tsv", length)
        (temporary / "visualization_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        svg = _svg(prediction, length, summary)
        (temporary / "mobiorigin_summary.svg").write_text(svg + "\n", encoding="utf-8")
        (temporary / "mobiorigin_dashboard.html").write_text(
            _html_report(svg, prediction, length, summary), encoding="utf-8"
        )
        outputs = sorted(path for path in temporary.iterdir() if path.is_file())
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(f"{_sha256(path)}  {path.name}\n" for path in outputs), encoding="ascii"
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
