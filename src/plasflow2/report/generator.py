"""HTML report generator — produces 5 separate, interlinked HTML files.

One file per sequence class:
    report_plasmid.html      — full ARG / VF / MGE / risk analysis
    report_chromosome.html   — chromosome contig table
    report_phage.html        — phage contig table
    report_archaea.html      — archaea contig table
    report_unclassified.html — unclassified contig table

Each file contains a navigation bar linking to the other four.

Usage:
    from plasflow2.report.generator import build_report_data, generate_reports
    data = build_report_data(pipeline_result, input_file="contigs.fasta")
    paths = generate_reports(data, output_dir="results/")
    # paths = {"plasmid": Path(...), "chromosome": Path(...), ...}
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared CSS + JS libs (embedded in every page)
# ---------------------------------------------------------------------------

_COMMON_CSS = """
  body { font-family: -apple-system, Arial, sans-serif; margin: 0; color: #333; background: #f4f6f9; }
  .page-wrap { max-width: 1400px; margin: 0 auto; padding: 24px 28px; }
  h1 { color: #2c6fad; margin-bottom: 4px; font-size: 1.6rem; }
  h2 { color: #444; margin-top: 36px; border-bottom: 2px solid #e0e8f5; padding-bottom: 6px; }
  .meta { color: #666; font-size: 0.92rem; margin-bottom: 20px; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 20px 0; }
  .stat-card { background: #fff; padding: 14px 16px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .stat-card h3 { margin: 0 0 6px; font-size: 0.75rem; text-transform: uppercase; color: #888; letter-spacing: .5px; }
  .stat-card p  { margin: 0; font-size: 1.65rem; font-weight: 700; }
  .stat-plasmid      { border-left: 4px solid #2c6fad; } .stat-plasmid p      { color: #2c6fad; }
  .stat-chromosome   { border-left: 4px solid #27ae60; } .stat-chromosome p   { color: #27ae60; }
  .stat-phage        { border-left: 4px solid #e67e22; } .stat-phage p        { color: #e67e22; }
  .stat-archaea      { border-left: 4px solid #8e44ad; } .stat-archaea p      { color: #8e44ad; }
  .stat-unclassified { border-left: 4px solid #95a5a6; } .stat-unclassified p { color: #95a5a6; }
  .stat-default      { border-left: 4px solid #2c6fad; } .stat-default p      { color: #2c6fad; }
  .charts-row   { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 24px 0; }
  .charts-row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin: 24px 0; }
  .chart-box { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 4px; min-height: 320px; }
  table.dataTable { width: 100% !important; font-size: 0.88rem; }
  table.dataTable tbody tr:hover { background-color: #f0f6ff; }
  .dataTables_wrapper { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 16px; margin-top: 12px; }
  .risk-high   { color: #c0392b; font-weight: bold; }
  .risk-medium { color: #e67e22; font-weight: bold; }
  .risk-low    { color: #27ae60; font-weight: bold; }
  .src-badge   { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: .75rem; font-weight: 600; margin: 1px 2px; }
  .src-card    { background: #e8f0fe; color: #1a56db; }
  .src-sarg    { background: #fef3c7; color: #b45309; }
  .badge-vf  { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: .75rem; font-weight: 700;
               background: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }
  .badge-mge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: .75rem; font-weight: 700;
               background: #ede9fe; color: #5b21b6; border: 1px solid #8b5cf6; }
  .eskape-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: .75rem; font-weight: 700; white-space: nowrap; }
  .eskape-core  { background: #fde8e8; color: #c0392b; border: 1px solid #e74c3c; }
  .eskape-who   { background: #fef3c7; color: #b45309; border: 1px solid #f39c12; }
  .filter-bar  { margin: 12px 0 8px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .filter-btn  { padding: 6px 16px; border: none; border-radius: 20px; cursor: pointer;
                 font-size: 0.85rem; font-weight: 600; transition: opacity .15s; }
  .filter-btn:hover { opacity: 0.85; }
  .filter-btn.active { outline: 3px solid #333; }
  .btn-all    { background: #e0e0e0; color: #333; }
  .btn-high   { background: #c0392b; color: #fff; }
  .btn-medium { background: #e67e22; color: #fff; }
  .btn-low    { background: #27ae60; color: #fff; }
  .ellipsis   { max-width: 220px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: inline-block; vertical-align: middle; }
  .no-data    { color: #888; font-style: italic; font-size: 0.9rem; margin: 16px 0; }
  .table-note { color: #888; font-size: 0.82rem; margin: 4px 0 10px; }
  footer { margin-top: 48px; color: #aaa; font-size: 0.8rem; border-top: 1px solid #e5e5e5; padding-top: 12px; }
"""

_CDN_SCRIPTS = """
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <link  rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
"""

# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------

_NAV_PAGES = [
    ("plasmid",       "report_plasmid.html",       "Plasmid",       "#2c6fad"),
    ("chromosome",    "report_chromosome.html",     "Chromosome",    "#27ae60"),
    ("phage",         "report_phage.html",          "Phage",         "#e67e22"),
    ("archaea",       "report_archaea.html",        "Archaea",       "#8e44ad"),
    ("unclassified",  "report_unclassified.html",   "Unclassified",  "#95a5a6"),
]

_NAV_CSS = """
  .nav { display: flex; gap: 0; background: #fff; border-bottom: 2px solid #e0e8f5;
         box-shadow: 0 2px 6px rgba(0,0,0,.07); padding: 0 28px; flex-wrap: wrap; }
  .nav a { display: inline-block; padding: 13px 22px; text-decoration: none; font-weight: 600;
           font-size: 0.9rem; color: #555; border-bottom: 3px solid transparent;
           transition: color .15s, border-color .15s; }
  .nav a:hover  { color: #2c6fad; border-bottom-color: #2c6fad; }
  .nav a.active { color: var(--nav-color); border-bottom-color: var(--nav-color); }
  .nav .counts  { margin-left: auto; display: flex; align-items: center; gap: 16px;
                  font-size: 0.8rem; color: #888; padding: 0 4px; flex-wrap: wrap; }
  .nav .counts span { white-space: nowrap; }
"""


def _build_nav(active: str, class_counts: dict[str, int], nav_color: str) -> str:
    """Render the top navigation bar with class counts."""
    links = []
    for key, href, label, color in _NAV_PAGES:
        cls = "active" if key == active else ""
        links.append(f'<a href="{href}" class="{cls}">{label}</a>')

    counts = " &nbsp;|&nbsp; ".join(
        f'<span>{k.capitalize()}: <b>{v:,}</b></span>'
        for k, v in class_counts.items()
    )

    return (
        f'<nav class="nav" style="--nav-color:{nav_color}">'
        + "".join(links)
        + f'<div class="counts">{counts}</div>'
        + "</nav>"
    )


def _page_shell(
    title: str,
    active: str,
    class_counts: dict[str, int],
    nav_color: str,
    body: str,
    extra_css: str = "",
    extra_js: str = "",
) -> str:
    """Wrap page content in a full HTML document."""
    nav = _build_nav(active, class_counts, nav_color)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} — PlasFlow v2</title>
  {_CDN_SCRIPTS}
  <style>
{_COMMON_CSS}
{_NAV_CSS}
{extra_css}
  </style>
</head>
<body>
{nav}
<div class="page-wrap">
{body}
<footer>Generated by PlasFlow v2 &mdash; open in any modern browser, no server required.</footer>
</div>
{extra_js}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlasmidRow:
    """One row in the plasmid detail table."""

    contig_id: str
    contig_length: int
    confidence: float
    num_args: int
    drug_classes: str
    mobility_class: str
    replicon_type: str
    risk_score: int
    taxonomy: str
    risk_evidence: str
    arg_sources: str = ""
    eskape_host: bool = False
    eskape_genus: str = ""
    num_vf: int = 0
    vf_genes: str = ""
    num_mge: int = 0
    mge_families: str = ""


@dataclass
class NonPlasmidRow:
    """One row in the chromosome / phage / archaea / unclassified tables."""

    contig_id: str
    contig_length: int
    label: str
    confidence: float
    taxonomy: str = "—"
    taxonomy_lineage: str = "—"
    # Best scoring class name (shown for unclassified to give context)
    best_label: str = ""
    best_score: float = 0.0


# ---------------------------------------------------------------------------
# Chart builders  (unchanged from previous version)
# ---------------------------------------------------------------------------


def _build_pie_data(class_counts: dict[str, int]) -> dict:
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    colors = {
        "plasmid": "#2c6fad",
        "chromosome": "#27ae60",
        "phage": "#e67e22",
        "archaea": "#8e44ad",
        "unclassified": "#95a5a6",
    }
    return {
        "data": [{"type": "pie", "labels": labels, "values": values,
                  "marker": {"colors": [colors.get(l, "#aaa") for l in labels]},
                  "textinfo": "label+percent", "hole": 0.35}],
        "layout": {"title": {"text": "Sequence Classification", "font": {"size": 14}},
                   "margin": {"t": 50, "b": 20, "l": 20, "r": 20},
                   "showlegend": False,
                   "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"},
    }


def _build_arg_chart(arg_hits: list) -> dict:
    drug_class_counts: Counter[str] = Counter()
    for hit in arg_hits:
        for dc in hit.drug_class.split(";"):
            dc = dc.strip()
            if dc and dc != "unknown":
                drug_class_counts[dc] += 1
    if not drug_class_counts:
        return {"data": [{"type": "bar", "x": [], "y": [], "orientation": "h"}],
                "layout": {"title": {"text": "ARG Drug Classes (none detected)", "font": {"size": 14}},
                            "margin": {"t": 50, "b": 40, "l": 180, "r": 20},
                            "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}
    sorted_items = sorted(drug_class_counts.items(), key=lambda x: x[1])
    return {"data": [{"type": "bar", "x": [i[1] for i in sorted_items],
                      "y": [i[0] for i in sorted_items],
                      "orientation": "h", "marker": {"color": "#c0392b"}}],
            "layout": {"title": {"text": "ARGs by Drug Class", "font": {"size": 14}},
                       "xaxis": {"title": "Gene count"},
                       "margin": {"t": 50, "b": 40, "l": 180, "r": 20},
                       "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}


def _build_risk_histogram(risk_scores: list[int]) -> dict:
    counts_by_score = Counter(risk_scores)
    score_range = list(range(11))
    y_vals = [counts_by_score.get(s, 0) for s in score_range]
    bar_colors = ["#c0392b" if s >= 7 else "#e67e22" if s >= 4 else "#27ae60"
                  for s in score_range]
    return {"data": [{"type": "bar", "x": score_range, "y": y_vals,
                      "marker": {"color": bar_colors}}],
            "layout": {"title": {"text": "Risk Score Distribution", "font": {"size": 14}},
                       "xaxis": {"title": "Risk Score (0–10)", "dtick": 1},
                       "yaxis": {"title": "Plasmid count"},
                       "margin": {"t": 50, "b": 50, "l": 50, "r": 20},
                       "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}


def _build_scatter_data(plasmid_rows: list[PlasmidRow]) -> dict:
    mobility_classes = sorted({r.mobility_class for r in plasmid_rows})
    palette = ["#2c6fad", "#c0392b", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#d35400"]
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(mobility_classes)}
    traces = []
    for mob in mobility_classes:
        rows = [r for r in plasmid_rows if r.mobility_class == mob]
        if not rows:
            continue
        traces.append({"type": "scatter", "mode": "markers", "name": mob,
                        "x": [r.contig_length for r in rows],
                        "y": [r.risk_score for r in rows],
                        "text": [f"{r.contig_id}<br>Risk: {r.risk_score}<br>ARGs: {r.num_args}<br>"
                                 f"VFs: {r.num_vf}  MGEs: {r.num_mge}<br>{r.taxonomy}"
                                 for r in rows],
                        "hovertemplate": "%{text}<extra></extra>",
                        "marker": {"color": color_map[mob], "size": 7, "opacity": 0.75,
                                   "line": {"width": 0.5, "color": "#fff"}}})
    return {"data": traces,
            "layout": {"title": {"text": "Contig Length vs Risk Score", "font": {"size": 14}},
                       "xaxis": {"title": "Contig length (bp)", "type": "log"},
                       "yaxis": {"title": "Risk Score (0–10)", "dtick": 1, "range": [-0.5, 10.5]},
                       "legend": {"title": {"text": "Mobility"}},
                       "margin": {"t": 50, "b": 60, "l": 60, "r": 20},
                       "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}


def _build_taxonomy_bar(plasmid_rows: list[PlasmidRow]) -> dict:
    tax_counts: Counter[str] = Counter()
    for r in plasmid_rows:
        tax_counts[r.taxonomy if r.taxonomy and r.taxonomy != "—" else "unclassified"] += 1
    top15 = tax_counts.most_common(15)
    if not top15:
        return {"data": [{"type": "bar", "x": [], "y": []}],
                "layout": {"title": {"text": "Top Taxonomy (no data)", "font": {"size": 14}},
                            "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}
    labels = [item[0] for item in reversed(top15)]
    counts = [item[1] for item in reversed(top15)]
    return {"data": [{"type": "bar", "x": counts, "y": labels, "orientation": "h",
                      "marker": {"color": "#8e44ad"}}],
            "layout": {"title": {"text": "Top Taxonomy (plasmid contigs)", "font": {"size": 14}},
                       "xaxis": {"title": "Contig count"},
                       "margin": {"t": 50, "b": 40, "l": 220, "r": 20},
                       "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}


def _build_drug_cooccurrence_heatmap(plasmid_results: list) -> dict:
    contig_classes: list[frozenset[str]] = []
    for cr in plasmid_results:
        classes: set[str] = set()
        for hit in cr.arg_hits:
            for dc in hit.drug_class.split(";"):
                dc = dc.strip()
                if dc and dc not in ("unknown", ""):
                    classes.add(dc)
        if classes:
            contig_classes.append(frozenset(classes))
    all_classes = sorted({dc for classes in contig_classes for dc in classes})
    if len(all_classes) < 2:
        return {"data": [], "layout": {"title": {"text": "Drug-Class Co-occurrence (insufficient data)",
                                                  "font": {"size": 14}},
                                        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}
    n = len(all_classes)
    matrix = [[0] * n for _ in range(n)]
    for fset in contig_classes:
        for i, ci in enumerate(all_classes):
            if ci not in fset:
                continue
            for j, cj in enumerate(all_classes):
                if cj in fset:
                    matrix[i][j] += 1

    def _short(label: str, maxlen: int = 28) -> str:
        return label if len(label) <= maxlen else label[:maxlen - 1] + "…"

    short_labels = [_short(c) for c in all_classes]
    hover = [[f"{all_classes[i]}<br>∩ {all_classes[j]}<br>{matrix[i][j]} contig(s)"
              for j in range(n)] for i in range(n)]
    return {"data": [{"type": "heatmap", "z": matrix, "x": short_labels, "y": short_labels,
                      "text": hover, "hovertemplate": "%{text}<extra></extra>",
                      "colorscale": "Blues", "showscale": True,
                      "colorbar": {"title": "Contigs", "thickness": 14}}],
            "layout": {"title": {"text": "Drug-Class Co-occurrence (plasmid contigs)", "font": {"size": 14}},
                       "xaxis": {"title": "", "tickangle": -40, "tickfont": {"size": 10}, "automargin": True},
                       "yaxis": {"title": "", "tickfont": {"size": 10}, "automargin": True},
                       "margin": {"t": 60, "b": 120, "l": 160, "r": 40},
                       "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}


# ---------------------------------------------------------------------------
# VF / MGE bar charts  (new — used in plasmid report)
# ---------------------------------------------------------------------------


def _build_vf_bar(plasmid_rows: list[PlasmidRow]) -> dict:
    """Top 15 virulence factor genes across plasmid contigs."""
    gene_counts: Counter[str] = Counter()
    for r in plasmid_rows:
        for g in r.vf_genes.split(";"):
            g = g.strip()
            if g:
                gene_counts[g] += 1
    top = gene_counts.most_common(15)
    if not top:
        return {"data": [{"type": "bar", "x": [], "y": []}],
                "layout": {"title": {"text": "VF Genes (none detected)", "font": {"size": 14}},
                            "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}
    labels = [i[0] for i in reversed(top)]
    counts = [i[1] for i in reversed(top)]
    return {"data": [{"type": "bar", "x": counts, "y": labels, "orientation": "h",
                      "marker": {"color": "#f59e0b"}}],
            "layout": {"title": {"text": "Top VF Genes (plasmid contigs)", "font": {"size": 14}},
                       "xaxis": {"title": "Contig count"},
                       "margin": {"t": 50, "b": 40, "l": 180, "r": 20},
                       "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}


def _build_mge_bar(plasmid_rows: list[PlasmidRow]) -> dict:
    """Top 15 MGE families across plasmid contigs."""
    fam_counts: Counter[str] = Counter()
    for r in plasmid_rows:
        for f in r.mge_families.split(";"):
            f = f.strip()
            if f:
                fam_counts[f] += 1
    top = fam_counts.most_common(15)
    if not top:
        return {"data": [{"type": "bar", "x": [], "y": []}],
                "layout": {"title": {"text": "MGE Families (none detected)", "font": {"size": 14}},
                            "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}
    labels = [i[0] for i in reversed(top)]
    counts = [i[1] for i in reversed(top)]
    return {"data": [{"type": "bar", "x": counts, "y": labels, "orientation": "h",
                      "marker": {"color": "#8b5cf6"}}],
            "layout": {"title": {"text": "Top MGE Families (plasmid contigs)", "font": {"size": 14}},
                       "xaxis": {"title": "Contig count"},
                       "margin": {"t": 50, "b": 40, "l": 180, "r": 20},
                       "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}}


# ---------------------------------------------------------------------------
# HTML fragment builders
# ---------------------------------------------------------------------------


def _eskape_badge(row: PlasmidRow) -> str:
    if not row.eskape_host:
        return "—"
    genus = row.eskape_genus
    cls = "eskape-core" if genus in (
        "Enterococcus", "Staphylococcus", "Klebsiella", "Acinetobacter",
        "Pseudomonas", "Enterobacter", "Escherichia", "Enterobacteriaceae",
    ) else "eskape-who"
    prefix = "ESKAPE" if cls == "eskape-core" else "WHO"
    return f'<span class="eskape-badge {cls}">{prefix}: {genus}</span>'


def _src_badges(arg_sources: str) -> str:
    srcs = [s.strip() for s in arg_sources.split(",") if s.strip()]
    if not srcs:
        return "—"
    return " ".join(f'<span class="src-badge src-{s.lower()}">{s}</span>' for s in srcs)


def _plasmid_table_html(rows: list[PlasmidRow]) -> str:
    if not rows:
        return '<p class="no-data">No plasmid contigs detected.</p>'

    def _row(r: PlasmidRow) -> str:
        tier = "high" if r.risk_score >= 7 else "medium" if r.risk_score >= 4 else "low"
        vf_cell = (f'<span class="badge-vf" title="{r.vf_genes}">{r.num_vf} VF</span>'
                   if r.num_vf > 0 else "—")
        mge_cell = (f'<span class="badge-mge" title="{r.mge_families}">{r.num_mge} MGE</span>'
                    if r.num_mge > 0 else "—")
        return (
            f'<tr data-risk-tier="{tier}">'
            f'<td>{r.contig_id}</td>'
            f'<td>{r.contig_length:,}</td>'
            f'<td>{r.confidence:.3f}</td>'
            f'<td>{r.num_args}</td>'
            f'<td><span class="ellipsis" title="{r.drug_classes}">{r.drug_classes}</span></td>'
            f'<td>{_src_badges(r.arg_sources)}</td>'
            f'<td>{vf_cell}</td>'
            f'<td><span class="ellipsis" title="{r.vf_genes}">{r.vf_genes or "—"}</span></td>'
            f'<td>{mge_cell}</td>'
            f'<td><span class="ellipsis" title="{r.mge_families}">{r.mge_families or "—"}</span></td>'
            f'<td>{_eskape_badge(r)}</td>'
            f'<td>{r.mobility_class}</td>'
            f'<td>{r.replicon_type}</td>'
            f'<td class="risk-{tier}">{r.risk_score}</td>'
            f'<td><span class="ellipsis" title="{r.taxonomy}">{r.taxonomy}</span></td>'
            f'<td><span class="ellipsis" title="{r.risk_evidence}">{r.risk_evidence}</span></td>'
            f'</tr>'
        )

    tbody = "\n".join(_row(r) for r in rows)
    return f"""
<table id="plasmid-table" class="display" style="width:100%">
  <thead><tr>
    <th>Contig</th><th>Length (bp)</th><th>Conf.</th>
    <th>ARGs</th><th>Drug Classes</th><th>DB</th>
    <th>VFs</th><th>VF Genes</th>
    <th>MGEs</th><th>MGE Families</th>
    <th>Pathogen</th><th>Mobility</th><th>Replicon</th>
    <th>Risk</th><th>Taxonomy</th><th>Evidence</th>
  </tr></thead>
  <tbody>{tbody}</tbody>
</table>"""


def _simple_table_html(rows: list[NonPlasmidRow], table_id: str,
                       show_label: bool = False, show_best: bool = False) -> str:
    if not rows:
        return '<p class="no-data">No contigs in this category.</p>'

    header_extra = ""
    if show_label:
        header_extra += "<th>Label</th>"
    if show_best:
        header_extra += "<th>Best Label</th><th>Best Score</th>"

    def _row(r: NonPlasmidRow) -> str:
        extra = ""
        if show_label:
            extra += f"<td>{r.label}</td>"
        if show_best:
            extra += f"<td>{r.best_label}</td><td>{r.best_score:.3f}</td>"
        return (
            f"<tr>"
            f"<td>{r.contig_id}</td>"
            f"<td>{r.contig_length:,}</td>"
            f"{extra}"
            f"<td>{r.confidence:.3f}</td>"
            f'<td><span class="ellipsis" title="{r.taxonomy_lineage}">{r.taxonomy}</span></td>'
            f"</tr>"
        )

    tbody = "\n".join(_row(r) for r in rows)
    return f"""
<table id="{table_id}" class="display" style="width:100%">
  <thead><tr>
    <th>Contig</th><th>Length (bp)</th>
    {header_extra}
    <th>Confidence</th><th>Taxonomy (LCA)</th>
  </tr></thead>
  <tbody>{tbody}</tbody>
</table>"""


# ---------------------------------------------------------------------------
# Per-page renderers
# ---------------------------------------------------------------------------


def _render_plasmid_page(data: dict) -> str:
    """Full plasmid analysis report."""
    rows: list[PlasmidRow] = data["plasmid_rows"]
    class_counts = data["class_counts"]
    input_file = data["input_file"]
    total = data["total"]

    # stat cards
    stat_cards = (
        f'<div class="stat-card stat-default"><h3>Total Sequences</h3><p>{total:,}</p></div>'
        f'<div class="stat-card stat-plasmid"><h3>Plasmids</h3><p>{len(rows):,}</p></div>'
        f'<div class="stat-card stat-default"><h3>ARGs</h3><p>{data["total_args"]:,}</p></div>'
        f'<div class="stat-card stat-default" style="border-left-color:#f59e0b">'
        f'<h3>Virulence Factors</h3><p style="color:#92400e">{data["total_vf"]:,}</p></div>'
        f'<div class="stat-card stat-default" style="border-left-color:#8b5cf6">'
        f'<h3>MGEs</h3><p style="color:#5b21b6">{data["total_mge"]:,}</p></div>'
    )

    has_scatter = bool(rows)
    has_cooccurrence = (
        bool(data["cooccurrence_data"].get("data"))
        and data["cooccurrence_data"]["data"]
        and data["cooccurrence_data"]["data"][0].get("z")
    )

    table_html = _plasmid_table_html(rows)

    body = f"""
  <h1>PlasFlow v2 — Plasmid Report</h1>
  <p class="meta">Input: <code>{input_file}</code></p>
  <div class="stats-grid">{stat_cards}</div>

  <h2>Overview Charts</h2>
  <div class="charts-row-3">
    <div id="pie-chart"  class="chart-box"></div>
    <div id="arg-chart"  class="chart-box"></div>
    <div id="risk-chart" class="chart-box"></div>
  </div>

  <h2>Virulence Factors &amp; Mobile Genetic Elements</h2>
  <div class="charts-row">
    <div id="vf-chart"  class="chart-box"></div>
    <div id="mge-chart" class="chart-box"></div>
  </div>

  {"<h2>Contig Length vs Risk Score</h2><div class='charts-row'><div id='scatter-chart' class='chart-box' style='min-height:350px'></div><div id='tax-chart' class='chart-box' style='min-height:350px'></div></div>" if has_scatter else ""}

  {"<h2>Drug-Class Co-occurrence</h2><p style='color:#666;font-size:.88rem;margin:-8px 0 12px'>Each cell shows how many plasmid contigs carry both drug classes simultaneously.</p><div id='cooccurrence-chart' class='chart-box' style='min-height:420px'></div>" if has_cooccurrence else ""}

  <h2>Plasmid Predictions ({len(rows):,} contigs)</h2>
  <div class="filter-bar">
    <span style="font-size:.85rem;color:#555">Filter by risk tier:</span>
    <button class="filter-btn btn-all active" id="btn-all"    onclick="filterRisk('all')">All</button>
    <button class="filter-btn btn-high"       id="btn-high"   onclick="filterRisk('high')">High (&ge;7)</button>
    <button class="filter-btn btn-medium"     id="btn-medium" onclick="filterRisk('medium')">Medium (4–6)</button>
    <button class="filter-btn btn-low"        id="btn-low"    onclick="filterRisk('low')">Low (0–3)</button>
  </div>
  {table_html}
"""

    js = f"""
<script>
  var pieData   = {json.dumps(data["pie_data"])};
  var argData   = {json.dumps(data["arg_data"])};
  var riskData  = {json.dumps(data["risk_data"])};
  var vfData    = {json.dumps(data["vf_data"])};
  var mgeData   = {json.dumps(data["mge_data"])};
  Plotly.newPlot('pie-chart',  pieData.data,  pieData.layout,  {{responsive:true}});
  Plotly.newPlot('arg-chart',  argData.data,  argData.layout,  {{responsive:true}});
  Plotly.newPlot('risk-chart', riskData.data, riskData.layout, {{responsive:true}});
  Plotly.newPlot('vf-chart',   vfData.data,   vfData.layout,   {{responsive:true}});
  Plotly.newPlot('mge-chart',  mgeData.data,  mgeData.layout,  {{responsive:true}});
  {"var scatterData = " + json.dumps(data["scatter_data"]) + "; Plotly.newPlot('scatter-chart', scatterData.data, scatterData.layout, {responsive:true});" if has_scatter else ""}
  {"var taxData = " + json.dumps(data["tax_bar_data"]) + "; Plotly.newPlot('tax-chart', taxData.data, taxData.layout, {responsive:true});" if has_scatter else ""}
  {"var coData = " + json.dumps(data["cooccurrence_data"]) + "; Plotly.newPlot('cooccurrence-chart', coData.data, coData.layout, {responsive:true});" if has_cooccurrence else ""}

  var table = null;
  $(document).ready(function() {{
    table = $('#plasmid-table').DataTable({{order: [[13, 'desc']], pageLength: 25}});
  }});

  function filterRisk(tier) {{
    ['all','high','medium','low'].forEach(function(t) {{
      document.getElementById('btn-' + t).classList.toggle('active', t === tier);
    }});
    if (!table) return;
    $.fn.dataTable.ext.search = $.fn.dataTable.ext.search.filter(function(f) {{ return f.__riskFilter !== true; }});
    if (tier !== 'all') {{
      var fn = function(settings, data, dataIndex) {{
        return $(table.row(dataIndex).node()).data('risk-tier') === tier;
      }};
      fn.__riskFilter = true;
      $.fn.dataTable.ext.search.push(fn);
    }}
    table.draw();
  }}
</script>"""

    return _page_shell("Plasmid", "plasmid", class_counts, "#2c6fad", body, extra_js=js)


def _render_simple_page(
    data: dict,
    class_key: str,
    title: str,
    nav_color: str,
    rows: list[NonPlasmidRow],
    table_id: str,
    order_col: int = 1,
    show_label: bool = False,
    show_best: bool = False,
    extra_note: str = "",
) -> str:
    """Generic renderer for chromosome / phage / archaea / unclassified pages."""
    class_counts = data["class_counts"]
    input_file = data["input_file"]
    n_total = len(rows)
    MAX_ROWS = 10_000
    display_rows = rows[:MAX_ROWS]
    truncated = n_total > MAX_ROWS

    stat_cards = (
        f'<div class="stat-card" style="border-left:4px solid {nav_color}">'
        f'<h3>{title} Contigs</h3>'
        f'<p style="color:{nav_color}">{n_total:,}</p></div>'
        f'<div class="stat-card stat-default"><h3>Total Sequences</h3><p>{data["total"]:,}</p></div>'
    )

    note = ""
    if truncated:
        note = (f'<p class="table-note">⚠ Showing top {MAX_ROWS:,} contigs by length '
                f'(of {n_total:,} total). Full data available in predictions.tsv.</p>')
    if extra_note:
        note += f'<p class="table-note">{extra_note}</p>'

    table_html = _simple_table_html(display_rows, table_id,
                                    show_label=show_label, show_best=show_best)

    body = f"""
  <h1>PlasFlow v2 — {title} Report</h1>
  <p class="meta">Input: <code>{input_file}</code></p>
  <div class="stats-grid">{stat_cards}</div>
  <h2>{title} Contigs ({n_total:,})</h2>
  {note}
  {table_html}
"""

    js = f"""
<script>
$(document).ready(function() {{
  $('#{table_id}').DataTable({{order: [[{order_col}, 'desc']], pageLength: 25}});
}});
</script>"""

    return _page_shell(title, class_key, class_counts, nav_color, body, extra_js=js)


# ---------------------------------------------------------------------------
# Main data builder
# ---------------------------------------------------------------------------


def build_report_data(pipeline_result, input_file: str = "") -> dict:
    """Convert PipelineResult into the template data dict."""
    all_arg_hits = [hit for cr in pipeline_result.plasmid_results for hit in cr.arg_hits]
    taxonomy = getattr(pipeline_result, "taxonomy", {}) or {}

    plasmid_rows: list[PlasmidRow] = []
    for cr in pipeline_result.plasmid_results:
        unique_classes = sorted({
            dc.strip()
            for hit in cr.arg_hits
            for dc in hit.drug_class.split(";")
            if dc.strip() and dc.strip() != "unknown"
        })
        mob = cr.mobility
        tax = getattr(cr, "taxonomy", None) or taxonomy.get(cr.record.id)
        tax_display = tax.display if tax else "—"
        sources = sorted({h.source for h in cr.arg_hits if hasattr(h, "source") and h.source})
        vf_hits = getattr(cr, "vf_hits", [])
        vf_genes = "; ".join(sorted({h.gene_name for h in vf_hits})) if vf_hits else ""
        mge_hits = getattr(cr, "mge_hits", [])
        mge_families = "; ".join(sorted({h.is_family for h in mge_hits})) if mge_hits else ""

        plasmid_rows.append(PlasmidRow(
            contig_id=cr.record.id,
            contig_length=len(cr.record.seq),
            confidence=cr.prediction.confidence,
            num_args=len(cr.arg_hits),
            drug_classes="; ".join(unique_classes) if unique_classes else "—",
            mobility_class=mob.mobility_class if mob else "unknown",
            replicon_type=mob.replicon_type if mob else "unknown",
            risk_score=cr.risk.score,
            taxonomy=tax_display,
            risk_evidence="; ".join(cr.risk.evidence) if cr.risk.evidence else "—",
            arg_sources=", ".join(sources) if sources else "",
            eskape_host=cr.risk.eskape_host,
            eskape_genus=cr.risk.eskape_genus,
            num_vf=len(vf_hits),
            vf_genes=vf_genes,
            num_mge=len(mge_hits),
            mge_families=mge_families,
        ))

    risk_scores = [cr.risk.score for cr in pipeline_result.plasmid_results]
    tax_classified = sum(1 for r in taxonomy.values() if r.rank != "unclassified")

    cooccurrence_data = _build_drug_cooccurrence_heatmap(pipeline_result.plasmid_results)

    # ── Non-plasmid rows — split into 4 separate lists ──────────────────────
    non_plasmid_results = getattr(pipeline_result, "non_plasmid_results", [])
    phage_rows:         list[NonPlasmidRow] = []
    chromosome_rows:    list[NonPlasmidRow] = []
    archaea_rows:       list[NonPlasmidRow] = []
    unclassified_rows:  list[NonPlasmidRow] = []

    for npr in non_plasmid_results:
        tax = getattr(npr, "taxonomy", None) or taxonomy.get(npr.record.id)
        tax_display = tax.display if tax else "—"
        tax_lineage = tax.lineage if tax else "—"

        # For unclassified: capture the best-scoring class for context
        scores = {}
        if hasattr(npr, "prediction") and hasattr(npr.prediction, "scores"):
            scores = npr.prediction.scores or {}
        best_label = max(scores, key=scores.get) if scores else ""
        best_score = scores.get(best_label, 0.0) if best_label else 0.0

        row = NonPlasmidRow(
            contig_id=npr.record.id,
            contig_length=len(npr.record.seq),
            label=npr.prediction.label,
            confidence=npr.prediction.confidence,
            taxonomy=tax_display,
            taxonomy_lineage=tax_lineage,
            best_label=best_label,
            best_score=best_score,
        )
        label = npr.prediction.label
        if label == "phage":
            phage_rows.append(row)
        elif label == "chromosome":
            chromosome_rows.append(row)
        elif label == "archaea":
            archaea_rows.append(row)
        else:
            unclassified_rows.append(row)

    # Sort large lists by length descending so top-N truncation keeps longest
    for lst in (chromosome_rows, unclassified_rows):
        lst.sort(key=lambda r: r.contig_length, reverse=True)

    total_vf  = sum(r.num_vf  for r in plasmid_rows)
    total_mge = sum(r.num_mge for r in plasmid_rows)

    return {
        "input_file":        input_file or str(pipeline_result.input_fasta),
        "total":             pipeline_result.total_sequences,
        "num_plasmids":      pipeline_result.total_plasmids,
        "total_args":        pipeline_result.total_args,
        "total_vf":          total_vf,
        "total_mge":         total_mge,
        "tax_classified":    tax_classified,
        "class_counts":      pipeline_result.class_counts,
        # charts
        "pie_data":          _build_pie_data(pipeline_result.class_counts),
        "arg_data":          _build_arg_chart(all_arg_hits),
        "risk_data":         _build_risk_histogram(risk_scores),
        "scatter_data":      _build_scatter_data(plasmid_rows) if plasmid_rows else {},
        "tax_bar_data":      _build_taxonomy_bar(plasmid_rows) if plasmid_rows else {},
        "vf_data":           _build_vf_bar(plasmid_rows),
        "mge_data":          _build_mge_bar(plasmid_rows),
        "cooccurrence_data": cooccurrence_data,
        # row lists
        "plasmid_rows":      plasmid_rows,
        "chromosome_rows":   chromosome_rows,
        "phage_rows":        phage_rows,
        "archaea_rows":      archaea_rows,
        "unclassified_rows": unclassified_rows,
        # legacy flags (kept for backward compat with report_cmd)
        "has_scatter":       bool(plasmid_rows),
        "has_cooccurrence":  (bool(cooccurrence_data.get("data"))
                               and cooccurrence_data["data"]
                               and cooccurrence_data["data"][0].get("z")),
        "has_phages":       bool(phage_rows),
        "has_chromosomes":  bool(chromosome_rows),
        "has_others":       bool(archaea_rows or unclassified_rows),
        "other_rows":       archaea_rows + unclassified_rows,  # legacy compat
    }


# ---------------------------------------------------------------------------
# 5-file renderer (primary entry point)
# ---------------------------------------------------------------------------


def generate_reports(report_data: dict, output_dir: Path | str) -> dict[str, Path]:
    """Write 5 separate HTML reports, one per sequence class.

    Args:
        report_data: Dict produced by :func:`build_report_data`.
        output_dir:  Directory where files are written.

    Returns:
        Dict mapping class name → Path, e.g.
        ``{"plasmid": Path("results/report_plasmid.html"), ...}``
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pages = {
        "plasmid":       out / "report_plasmid.html",
        "chromosome":    out / "report_chromosome.html",
        "phage":         out / "report_phage.html",
        "archaea":       out / "report_archaea.html",
        "unclassified":  out / "report_unclassified.html",
    }

    html_map = {
        "plasmid": _render_plasmid_page(report_data),
        "chromosome": _render_simple_page(
            report_data, "chromosome", "Chromosome", "#27ae60",
            report_data["chromosome_rows"], "chromosome-table",
        ),
        "phage": _render_simple_page(
            report_data, "phage", "Phage", "#e67e22",
            report_data["phage_rows"], "phage-table",
        ),
        "archaea": _render_simple_page(
            report_data, "archaea", "Archaea", "#8e44ad",
            report_data["archaea_rows"], "archaea-table",
        ),
        "unclassified": _render_simple_page(
            report_data, "unclassified", "Unclassified", "#95a5a6",
            report_data["unclassified_rows"], "unclassified-table",
            show_best=True,
            extra_note=(
                "Contigs where no class scored ≥ 0.95 confidence. "
                "'Best Label' shows the top-scoring class even below threshold."
            ),
        ),
    }

    for key, path in pages.items():
        path.write_text(html_map[key], encoding="utf-8")
        logger.info("Report written to %s", path)

    return pages


# ---------------------------------------------------------------------------
# Legacy single-file renderer (kept for backward compat / 'plasflow2 report')
# ---------------------------------------------------------------------------

_TEMPLATE = ""  # kept as sentinel — generate_report() below builds HTML directly


def generate_report(report_data: dict, output_path: Path | str) -> Path:
    """Write 5 HTML files and return the plasmid report path.

    This keeps backward compatibility for callers that expect a single path.
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    paths = generate_reports(report_data, output_dir)
    return paths["plasmid"]
