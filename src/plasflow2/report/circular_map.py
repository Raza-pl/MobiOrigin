"""Circular plasmid genome maps — pure SVG, no JavaScript dependencies.

Generates one SVG per circular plasmid contig showing all annotated genes
colour-coded by type (ARG, VFG, MGE, BacMet, ICE, mobility, other).

Forward-strand genes appear on the outer track; reverse-strand on inner track.
Named genes (ARG/VFG/BacMet/ICE) are labelled with radial tick + text.

Output: a standalone HTML page (report_circular_plasmids.html) containing
all SVG maps, linked from the main plasmid report.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

COLOURS = {
    "arg": "#e74c3c",  # red        — antibiotic resistance
    "vf": "#e67e22",  # orange     — virulence factor
    "mge": "#8e44ad",  # purple     — mobile genetic element
    "bacmet": "#16a085",  # teal       — biocide/metal resistance
    "ice": "#2471a3",  # blue       — integrative conjugative element
    "mobility": "#1a5276",  # navy       — relaxase / MPF / TraG etc.
    "other": "#bdc3c7",  # light grey — unannotated ORF
}

LEGEND = [
    ("ARG", COLOURS["arg"]),
    ("Virulence", COLOURS["vf"]),
    ("MGE/IS", COLOURS["mge"]),
    ("BacMet", COLOURS["bacmet"]),
    ("ICE", COLOURS["ice"]),
    ("Mobility gene", COLOURS["mobility"]),
    ("Other ORF", COLOURS["other"]),
]

_MOBILITY_KW = frozenset(
    [
        "conjugat",
        "relaxase",
        "mpf",
        "trag",
        "trai",
        "traj",
        "mob",
        "orit",
        "virb",
        "vird",
        "trb",
    ]
)


# ---------------------------------------------------------------------------
# Gene type classification
# ---------------------------------------------------------------------------


def _gene_type(
    orf_id: str,
    arg_orf_ids: set[str],
    vf_orf_ids: set[str],
    mge_orf_ids: set[str],
    bacmet_orf_ids: set[str],
    ice_orf_ids: set[str],
    gene_name: str = "",
) -> str:
    if orf_id in arg_orf_ids:
        return "arg"
    if orf_id in bacmet_orf_ids:
        return "bacmet"
    if orf_id in ice_orf_ids:
        return "ice"
    if orf_id in mge_orf_ids:
        return "mge"
    if orf_id in vf_orf_ids:
        return "vf"
    if any(kw in gene_name.lower() for kw in _MOBILITY_KW):
        return "mobility"
    return "other"


# ---------------------------------------------------------------------------
# SVG geometry helpers
# ---------------------------------------------------------------------------


def _polar(cx: float, cy: float, r: float, angle_deg: float) -> tuple[float, float]:
    """Convert polar coordinates to Cartesian (SVG convention: 0° = top, CW)."""
    rad = math.radians(angle_deg - 90)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def _arc_path(
    cx: float,
    cy: float,
    r: float,
    start_deg: float,
    end_deg: float,
    width: float,
) -> str:
    """Return an SVG path string for a filled arc (annular sector)."""
    # Clamp arc to < 360° to avoid degenerate paths
    sweep = (end_deg - start_deg) % 360
    if sweep == 0:
        sweep = 359.99
    large = 1 if sweep > 180 else 0
    r_inner = r - width

    x1o, y1o = _polar(cx, cy, r, start_deg)
    x2o, y2o = _polar(cx, cy, r, start_deg + sweep)
    x1i, y1i = _polar(cx, cy, r_inner, start_deg + sweep)
    x2i, y2i = _polar(cx, cy, r_inner, start_deg)

    return (
        f"M {x1o:.2f} {y1o:.2f} "
        f"A {r:.2f} {r:.2f} 0 {large} 1 {x2o:.2f} {y2o:.2f} "
        f"L {x1i:.2f} {y1i:.2f} "
        f"A {r_inner:.2f} {r_inner:.2f} 0 {large} 0 {x2i:.2f} {y2i:.2f} "
        f"Z"
    )


# ---------------------------------------------------------------------------
# SVG builder for one contig
# ---------------------------------------------------------------------------


def build_circular_svg(
    contig_id: str,
    contig_length: int,
    orfs: list,  # list[ORF]
    arg_orf_ids: set[str],
    vf_orf_ids: set[str],
    mge_orf_ids: set[str],
    bacmet_orf_ids: set[str],
    ice_orf_ids: set[str],
    name_by_orf: dict[str, str],  # orf_id → gene name (for labels)
    size: int = 420,
) -> str:
    """Return a self-contained SVG string for one circular plasmid map."""
    cx = cy = size / 2
    r_outer_fwd = cx * 0.72  # forward-strand track outer radius
    r_outer_rev = cx * 0.62  # reverse-strand track outer radius
    track_w = cx * 0.09  # track width
    r_backbone = cx * 0.67  # backbone circle radius
    r_tick_out = cx * 0.77  # tick outer edge
    r_tick_in = cx * 0.74  # tick inner edge

    def bp_to_deg(bp: int) -> float:
        return 360.0 * bp / contig_length

    paths: list[str] = []
    labels: list[str] = []
    used_labels: set[str] = set()

    for orf in orfs:
        oid = orf.orf_id
        start = max(orf.start, 1)
        end = min(orf.end, contig_length)
        if end <= start:
            continue

        gtype = _gene_type(
            oid,
            arg_orf_ids,
            vf_orf_ids,
            mge_orf_ids,
            bacmet_orf_ids,
            ice_orf_ids,
        )
        colour = COLOURS[gtype]
        r_outer = r_outer_fwd if orf.strand >= 0 else r_outer_rev

        s_deg = bp_to_deg(start - 1)
        e_deg = bp_to_deg(end)
        path = _arc_path(cx, cy, r_outer, s_deg, e_deg, track_w)
        title = name_by_orf.get(oid, oid)
        paths.append(
            f'<path d="{path}" fill="{colour}" opacity="0.88">'
            f"<title>{title} ({start}–{end})</title></path>"
        )

        # Label named genes (ARG, VFG, BacMet, ICE) — avoid crowding
        gene_name = name_by_orf.get(oid, "")
        if gene_name and gtype in ("arg", "vf", "bacmet", "ice") and gene_name not in used_labels:
            used_labels.add(gene_name)
            mid_deg = bp_to_deg((start + end) // 2)
            lx1, ly1 = _polar(cx, cy, r_tick_in, mid_deg)
            lx2, ly2 = _polar(cx, cy, r_tick_out, mid_deg)
            # Text anchor based on position
            anchor = "start" if lx2 >= cx else "end"
            tx = lx2 + (6 if lx2 >= cx else -6)
            labels.append(
                f'<line x1="{lx1:.1f}" y1="{ly1:.1f}" '
                f'x2="{lx2:.1f}" y2="{ly2:.1f}" '
                f'stroke="{colour}" stroke-width="1.2"/>'
                f'<text x="{tx:.1f}" y="{ly2:.1f}" '
                f'font-size="9" font-family="monospace" fill="{colour}" '
                f'text-anchor="{anchor}" dominant-baseline="middle">'
                f"{gene_name}</text>"
            )

    # ── Position tick marks every ~1 kb ──────────────────────────────────
    tick_interval = max(1000, (contig_length // 10 // 1000) * 1000)
    ticks: list[str] = []
    pos = 0
    while pos <= contig_length:
        deg = bp_to_deg(pos)
        tx1, ty1 = _polar(cx, cy, r_backbone - 4, deg)
        tx2, ty2 = _polar(cx, cy, r_backbone + 4, deg)
        lx, ly = _polar(cx, cy, r_backbone + 13, deg)
        anchor = "middle"
        kb_label = f"{pos//1000}k" if pos > 0 else "0"
        ticks.append(
            f'<line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" '
            f'stroke="#999" stroke-width="0.8"/>'
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="8" fill="#888" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{kb_label}</text>'
        )
        pos += tick_interval

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items: list[str] = []
    lx_start = 6
    ly_start = size - 10 - len(LEGEND) * 13
    for i, (label, col) in enumerate(LEGEND):
        ly = ly_start + i * 13
        legend_items.append(
            f'<rect x="{lx_start}" y="{ly}" width="10" height="10" fill="{col}" rx="2"/>'
            f'<text x="{lx_start+13}" y="{ly+8}" font-size="9" fill="#555" '
            f'font-family="sans-serif">{label}</text>'
        )

    kb = contig_length / 1000
    _title_text = f"{contig_id}  ({kb:.1f} kb)"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <title>{contig_id}</title>
  <!-- backbone -->
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_backbone:.1f}"
          fill="none" stroke="#ccc" stroke-width="1.5"/>
  <!-- forward track outline -->
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_outer_fwd:.1f}"
          fill="none" stroke="#eee" stroke-width="{track_w:.1f}" opacity="0.4"/>
  <!-- reverse track outline -->
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_outer_rev:.1f}"
          fill="none" stroke="#eee" stroke-width="{track_w:.1f}" opacity="0.4"/>
  <!-- ticks -->
  {''.join(ticks)}
  <!-- genes -->
  {''.join(paths)}
  <!-- labels -->
  {''.join(labels)}
  <!-- title -->
  <text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" dominant-baseline="middle"
        font-size="9" font-family="monospace" fill="#444">{contig_id}</text>
  <text x="{cx:.1f}" y="{cy+12:.1f}" text-anchor="middle" dominant-baseline="middle"
        font-size="8" font-family="sans-serif" fill="#888">{kb:.1f} kb · circular</text>
  <!-- legend -->
  {''.join(legend_items)}
</svg>"""
    return svg


# ---------------------------------------------------------------------------
# Full HTML page builder
# ---------------------------------------------------------------------------


def build_circular_maps_page(
    pipeline_result,
    topology_map: dict[str, str],
    input_file: str = "",
) -> str:
    """Generate standalone HTML page with circular maps for all circular contigs.

    Includes plasmid, unclassified, and other contigs with circular topology so
    that contigs demoted by the hallmark gate (e.g. short plasmids without MOB
    evidence) still appear in the circular maps view.
    Returns empty-body HTML if no circular contigs are found.
    """
    from collections import defaultdict

    # Build lookup dicts from pipeline result
    orfs_by_contig: dict = defaultdict(list)
    all_orfs = getattr(pipeline_result, "orfs", []) or []
    for orf in all_orfs:
        orfs_by_contig[orf.contig_id].append(orf)

    # Build annotation lookup sets per ORF (empty when no ORF data available)
    arg_orf_ids: set[str] = set()
    vf_orf_ids: set[str] = set()
    mge_orf_ids: set[str] = set()
    bacmet_orf_ids: set[str] = set()
    ice_orf_ids: set[str] = set()
    name_by_orf: dict[str, str] = {}

    # Collect annotation hits from ALL contigs (plasmid + non-plasmid)
    all_results = list(getattr(pipeline_result, "plasmid_results", [])) + list(
        getattr(pipeline_result, "non_plasmid_results", [])
    )
    for cr in all_results:
        for h in getattr(cr, "arg_hits", []):
            oid = getattr(h, "_orf_id", "")
            if oid:
                arg_orf_ids.add(oid)
                name_by_orf[oid] = h.gene_name
        for h in getattr(cr, "vf_hits", []):
            oid = getattr(h, "_orf_id", "")
            if oid:
                vf_orf_ids.add(oid)
                name_by_orf[oid] = h.gene_name
        for h in getattr(cr, "mge_hits", []):
            oid = getattr(h, "_orf_id", "")
            if oid:
                mge_orf_ids.add(oid)
                name_by_orf[oid] = h.is_name
        for h in getattr(cr, "bacmet_hits", []):
            oid = getattr(h, "_orf_id", "")
            if oid:
                bacmet_orf_ids.add(oid)
                name_by_orf[oid] = h.gene_name
        for h in getattr(cr, "ice_hits", []):
            oid = getattr(h, "_orf_id", "")
            if oid:
                ice_orf_ids.add(oid)
                name_by_orf[oid] = h.gene_function

    # Find ALL circular contigs (plasmid + unclassified + chromosome)
    circular_results = [
        cr for cr in all_results if topology_map.get(cr.record.id, "") == "circular"
    ]

    if not circular_results:
        return _wrap_html(
            "",
            input_file,
            "<p style='color:#888;margin:40px'>"
            "No circular contigs detected in this assembly.</p>",
        )

    # Sort: plasmids first, then by length descending
    _label_order = {"plasmid": 0, "unclassified": 1, "chromosome": 2, "phage": 3}

    def _sort_key(cr):
        lbl = getattr(cr.prediction, "label", "unclassified")
        return (_label_order.get(lbl, 9), -len(cr.record.seq))

    circular_results.sort(key=_sort_key)

    svgs: list[str] = []
    for cr in circular_results:
        cid = cr.record.id
        clen = len(cr.record.seq)
        orfs = orfs_by_contig.get(cid, [])
        label = getattr(cr.prediction, "label", "unclassified")

        # Summary line for this contig
        n_args = len(getattr(cr, "arg_hits", []))
        n_vf = len(getattr(cr, "vf_hits", []))
        n_mge = len(getattr(cr, "mge_hits", []))
        n_bm = len(getattr(cr, "bacmet_hits", []))
        n_ice = len(getattr(cr, "ice_hits", []))
        risk_obj = getattr(cr, "risk", None)
        risk = risk_obj.score if risk_obj else 0
        mob_obj = getattr(cr, "mobility", None)
        mob = mob_obj.mobility_class if mob_obj else "unknown"
        orfs_note = f"{len(orfs)} ORFs · " if orfs else "no ORF data · "

        _label_colors = {
            "plasmid": "#1a7a4a",
            "unclassified": "#8B6914",
            "chromosome": "#6b6b6b",
            "phage": "#7b4a9a",
        }
        label_color = _label_colors.get(label, "#555")
        label_badge = (
            f'<span style="font-size:.7rem;font-weight:600;color:{label_color};'
            f"background:{label_color}18;border:1px solid {label_color}40;"
            f'border-radius:3px;padding:1px 5px;margin-left:6px">{label}</span>'
        )

        summary = (
            f"Risk {risk} · {mob} · {orfs_note}"
            f"{n_args} ARGs · {n_vf} VFs · {n_mge} MGEs · "
            f"{n_bm} BacMet · {n_ice} ICE"
        )

        svg = build_circular_svg(
            contig_id=cid,
            contig_length=clen,
            orfs=orfs,
            arg_orf_ids=arg_orf_ids,
            vf_orf_ids=vf_orf_ids,
            mge_orf_ids=mge_orf_ids,
            bacmet_orf_ids=bacmet_orf_ids,
            ice_orf_ids=ice_orf_ids,
            name_by_orf=name_by_orf,
        )

        svgs.append(
            f'<div class="map-card">'
            f'<p class="cid">{cid}{label_badge}</p>'
            f'<p class="meta">{summary}</p>'
            f"{svg}"
            f"</div>"
        )

    maps_html = "\n".join(svgs)
    gene_note = (
        "outer track = forward strand · inner track = reverse strand"
        if all_orfs
        else "gene tracks not shown (ARG databases were not configured at run time)"
    )
    n_plasmid = sum(
        1 for cr in circular_results if getattr(cr.prediction, "label", "") == "plasmid"
    )
    plasmid_note = f"{n_plasmid} plasmid · " if n_plasmid < len(circular_results) else ""
    return _wrap_html(
        maps_html,
        input_file,
        f"<p class='subtitle'>{len(circular_results)} circular contig"
        f"{'s' if len(circular_results)!=1 else ''} · {plasmid_note}{gene_note}</p>",
    )


def _wrap_html(maps_html: str, input_file: str, subtitle: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PlasFlow v2 — Circular Plasmid Maps</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #2c3e50; padding: 24px; }}
h1   {{ font-size: 1.3rem; margin-bottom: 4px; }}
.subtitle {{ color: #666; font-size: .88rem; margin-bottom: 4px; }}
.meta-line {{ color: #999; font-size: .8rem; margin-bottom: 20px; }}
a {{ color: #2c6fad; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
.map-card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px;
             padding: 12px; width: 440px; }}
.map-card .cid  {{ font-family: monospace; font-size: .82rem; color: #333;
                   margin-bottom: 2px; word-break: break-all; }}
.map-card .meta {{ font-size: .75rem; color: #777; margin-bottom: 8px; }}
</style>
</head>
<body>
<h1>PlasFlow v2 — Circular Plasmid Maps</h1>
{subtitle}
<p class="meta-line">Input: <code>{input_file}</code> &nbsp;·&nbsp;
<a href="report_plasmid.html">← Back to plasmid report</a></p>
<div class="grid">
{maps_html}
</div>
</body>
</html>"""
