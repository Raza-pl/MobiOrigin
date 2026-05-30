"""HTML report generator — 5 separate, interlinked HTML files.

Performance design
------------------
* Non-plasmid pages (chromosome / phage / archaea / unclassified):
  - No external JS libraries.  Row data stored as a compact JSON array;
    vanilla JS renders only the current page (50 rows at a time).
  - Result: files open instantly regardless of row count.
* Plasmid page:
  - Plotly loaded from CDN (cached after first visit).
  - Row data also stored as JSON; same lightweight paginator used.
  - Charts limited to the 4 most useful: pie, ARG bar, VF bar, MGE bar.
* All pages share a nav bar with class counts.
* Max rows shown: 2 000 (full data always available in predictions.tsv).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_TABLE_ROWS = 2_000   # rows shown in HTML; rest available in predictions.tsv

# ---------------------------------------------------------------------------
# Shared CSS (no external stylesheets needed)
# ---------------------------------------------------------------------------

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,Arial,sans-serif;background:#f4f6f9;color:#333}
.nav{display:flex;align-items:stretch;background:#fff;border-bottom:2px solid #e0e8f5;
     box-shadow:0 2px 6px rgba(0,0,0,.07);padding:0 24px;flex-wrap:wrap;gap:0}
.nav a{display:inline-flex;align-items:center;padding:12px 18px;text-decoration:none;
       font-weight:600;font-size:.88rem;color:#666;border-bottom:3px solid transparent;
       white-space:nowrap;transition:color .15s,border-color .15s}
.nav a:hover{color:#2c6fad;border-bottom-color:#2c6fad}
.nav a.active{color:var(--nc);border-bottom-color:var(--nc)}
.nav .pill{margin-left:auto;display:flex;align-items:center;gap:10px;
           font-size:.78rem;color:#888;padding:0 4px;flex-wrap:wrap}
.nav .pill span{background:#f0f4f8;border-radius:12px;padding:2px 9px;white-space:nowrap}
.wrap{max-width:1380px;margin:0 auto;padding:24px 26px}
h1{color:#2c6fad;font-size:1.5rem;margin-bottom:4px}
h2{color:#444;margin-top:32px;border-bottom:2px solid #e0e8f5;padding-bottom:5px;font-size:1.1rem}
.meta{color:#777;font-size:.88rem;margin:6px 0 18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:16px 0}
.card{background:#fff;padding:13px 15px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.card h3{font-size:.72rem;text-transform:uppercase;color:#999;letter-spacing:.4px;margin-bottom:5px}
.card p{font-size:1.6rem;font-weight:700}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:20px 0}
.chart-grid.three{grid-template-columns:1fr 1fr 1fr}
.cbox{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);
      padding:4px;min-height:280px}
/* lightweight table */
.tbl-wrap{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);
          padding:16px;margin-top:14px}
.tbl-ctrl{display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap}
.tbl-ctrl input{padding:6px 10px;border:1px solid #d0d8e4;border-radius:6px;
                font-size:.85rem;width:260px;outline:none}
.tbl-ctrl input:focus{border-color:#2c6fad}
.tbl-ctrl .pg-info{margin-left:auto;font-size:.82rem;color:#888}
.tbl-ctrl .pg-btn{padding:5px 12px;border:1px solid #d0d8e4;border-radius:6px;
                  background:#fff;cursor:pointer;font-size:.82rem}
.tbl-ctrl .pg-btn:hover{background:#f0f4f8}
.tbl-ctrl .pg-btn:disabled{opacity:.4;cursor:default}
table{width:100%;border-collapse:collapse;font-size:.83rem}
th{background:#f8fafc;text-align:left;padding:8px 10px;font-size:.75rem;
   text-transform:uppercase;color:#777;border-bottom:2px solid #e0e8f5;
   cursor:pointer;user-select:none;white-space:nowrap}
th:hover{background:#eef2f7}
th.sort-asc::after{content:" ▲"}
th.sort-desc::after{content:" ▼"}
td{padding:7px 10px;border-bottom:1px solid #f0f4f8;vertical-align:middle}
tr:hover td{background:#f7fbff}
.ellipsis{max-width:180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
          display:inline-block;vertical-align:middle}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.73rem;font-weight:700}
.bvf {background:#fef3c7;color:#92400e;border:1px solid #f59e0b}
.bmge{background:#ede9fe;color:#5b21b6;border:1px solid #8b5cf6}
.bcard{background:#e8f0fe;color:#1a56db}
.bsarg{background:#fef3c7;color:#b45309}
.besk{background:#fde8e8;color:#c0392b;border:1px solid #e74c3c}
.bwho{background:#fef3c7;color:#b45309;border:1px solid #f39c12}
.risk-h{color:#c0392b;font-weight:700}
.risk-m{color:#e67e22;font-weight:700}
.risk-l{color:#27ae60;font-weight:700}
.note{color:#888;font-size:.8rem;margin:4px 0 10px;font-style:italic}
.filter-bar{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap}
.fbtn{padding:5px 14px;border:none;border-radius:16px;cursor:pointer;
      font-size:.82rem;font-weight:600;opacity:.85}
.fbtn:hover{opacity:1}
.fbtn.active{outline:2px solid #333;opacity:1}
footer{margin-top:40px;color:#bbb;font-size:.78rem;border-top:1px solid #e5e5e5;padding-top:10px}
"""

# ---------------------------------------------------------------------------
# Vanilla JS lightweight paginator (no jQuery, no DataTables)
# ---------------------------------------------------------------------------

_PAGINATOR_JS = """
function LightTable(cfg) {
  // cfg: {tableId, data, cols, pageSize, searchId, pgInfoId, prevId, nextId, accentColor}
  var t = this;
  t.data = cfg.data; t.filtered = cfg.data.slice();
  t.cols = cfg.cols; t.page = 0; t.ps = cfg.pageSize || 50;
  t.sortCol = -1; t.sortDir = 1;
  t.tbody = document.querySelector('#'+cfg.tableId+' tbody');
  t.ths   = document.querySelectorAll('#'+cfg.tableId+' th');
  t.pgInfo = document.getElementById(cfg.pgInfoId);
  t.prev   = document.getElementById(cfg.prevId);
  t.next   = document.getElementById(cfg.nextId);
  t.search = document.getElementById(cfg.searchId);
  // sort on header click
  t.ths.forEach(function(th, ci) {
    th.addEventListener('click', function() {
      if (t.sortCol === ci) { t.sortDir *= -1; }
      else { t.sortCol = ci; t.sortDir = 1; }
      t.ths.forEach(function(h) { h.className=''; });
      th.className = t.sortDir===1 ? 'sort-asc' : 'sort-desc';
      t.applySort(); t.page=0; t.render();
    });
  });
  // search
  if (t.search) {
    t.search.addEventListener('input', function() {
      var q = t.search.value.toLowerCase();
      t.filtered = q ? t.data.filter(function(r) {
        return r.some(function(c){ return String(c).toLowerCase().indexOf(q) !== -1; });
      }) : t.data.slice();
      t.applySort(); t.page=0; t.render();
    });
  }
  if (t.prev) t.prev.addEventListener('click', function(){ if(t.page>0){t.page--;t.render();} });
  if (t.next) t.next.addEventListener('click', function(){
    if((t.page+1)*t.ps<t.filtered.length){t.page++;t.render();}
  });
  t.applySort = function() {
    if (t.sortCol < 0) return;
    var ci = t.sortCol, dir = t.sortDir;
    t.filtered.sort(function(a,b){
      var av=a[ci], bv=b[ci];
      var an=parseFloat(av), bn=parseFloat(bv);
      if(!isNaN(an)&&!isNaN(bn)){ return dir*(an-bn); }
      return dir*(String(av).localeCompare(String(bv)));
    });
  };
  t.render = function() {
    var start=t.page*t.ps, end=Math.min(start+t.ps, t.filtered.length);
    var html='';
    for(var i=start;i<end;i++){
      var r=t.filtered[i];
      html += '<tr>' + t.cols.map(function(c,ci){
        return '<td>'+c.render(r[ci],r)+'</td>';
      }).join('') + '</tr>';
    }
    t.tbody.innerHTML = html || '<tr><td colspan="'+t.cols.length+'" style="color:#999;text-align:center;padding:20px">No results</td></tr>';
    if(t.pgInfo) t.pgInfo.textContent = t.filtered.length===0 ? '0 rows' :
      'Showing '+(start+1)+'–'+end+' of '+t.filtered.length.toLocaleString();
    if(t.prev) t.prev.disabled = t.page===0;
    if(t.next) t.next.disabled = end>=t.filtered.length;
  };
  t.render();
}
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function ellipsis(s,max){ s=esc(s); return s.length>max?'<span class="ellipsis" title="'+s+'">'+s.substring(0,max)+'…</span>':s; }
"""

# ---------------------------------------------------------------------------
# Nav bar
# ---------------------------------------------------------------------------

_NAV_PAGES = [
    ("plasmid",      "report_plasmid.html",      "Plasmid",      "#2c6fad"),
    ("chromosome",   "report_chromosome.html",    "Chromosome",   "#27ae60"),
    ("phage",        "report_phage.html",         "Phage",        "#e67e22"),
    ("archaea",      "report_archaea.html",       "Archaea",      "#8e44ad"),
    ("unclassified", "report_unclassified.html",  "Unclassified", "#95a5a6"),
]


def _nav(active: str, class_counts: dict[str, int], color: str) -> str:
    links = "".join(
        f'<a href="{href}" class="{"active" if k==active else ""}">{label}</a>'
        for k, href, label, _ in _NAV_PAGES
    )
    pills = "".join(
        f'<span>{k.capitalize()}: <b>{v:,}</b></span>'
        for k, v in class_counts.items()
    )
    return (f'<nav class="nav" style="--nc:{color}">{links}'
            f'<div class="pill">{pills}</div></nav>')


def _page(title: str, active: str, counts: dict, color: str, body: str, extra_js: str = "") -> str:
    return (
        f'<!DOCTYPE html><html lang="en"><head>'
        f'<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{title} — PlasFlow v2</title>'
        f'<style>{_CSS}</style></head><body>'
        f'{_nav(active, counts, color)}'
        f'<div class="wrap">{body}'
        f'<footer>PlasFlow v2 — open in any browser, no server required.</footer>'
        f'</div>'
        f'<script>{_PAGINATOR_JS}{extra_js}</script>'
        f'</body></html>'
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlasmidRow:
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
    contig_id: str
    contig_length: int
    label: str
    confidence: float
    taxonomy: str = "—"
    taxonomy_lineage: str = "—"
    best_label: str = ""
    best_score: float = 0.0


# ---------------------------------------------------------------------------
# Chart builders (Plotly — plasmid page only)
# ---------------------------------------------------------------------------


def _pie(class_counts: dict[str, int]) -> dict:
    colors = {"plasmid":"#2c6fad","chromosome":"#27ae60","phage":"#e67e22",
              "archaea":"#8e44ad","unclassified":"#95a5a6"}
    labels, values = list(class_counts.keys()), list(class_counts.values())
    return {"data":[{"type":"pie","labels":labels,"values":values,
                     "marker":{"colors":[colors.get(l,"#aaa") for l in labels]},
                     "textinfo":"label+percent","hole":0.35}],
            "layout":{"title":{"text":"Classification","font":{"size":13}},
                      "margin":{"t":45,"b":15,"l":15,"r":15},"showlegend":False,
                      "paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"}}


def _arg_bar(arg_hits: list) -> dict:
    dc: Counter[str] = Counter()
    for h in arg_hits:
        for c in h.drug_class.split(";"):
            c = c.strip()
            if c and c != "unknown":
                dc[c] += 1
    if not dc:
        return {"data":[{"type":"bar","x":[],"y":[],"orientation":"h"}],
                "layout":{"title":{"text":"ARGs (none)","font":{"size":13}},
                           "margin":{"t":45,"b":30,"l":160,"r":15},
                           "paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"}}
    items = sorted(dc.items(), key=lambda x: x[1])
    return {"data":[{"type":"bar","x":[i[1] for i in items],"y":[i[0] for i in items],
                     "orientation":"h","marker":{"color":"#c0392b"}}],
            "layout":{"title":{"text":"ARGs by Drug Class","font":{"size":13}},
                      "xaxis":{"title":"Count"},"margin":{"t":45,"b":30,"l":170,"r":15},
                      "paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"}}


def _vf_bar(rows: list[PlasmidRow]) -> dict:
    gc: Counter[str] = Counter()
    for r in rows:
        for g in r.vf_genes.split(";"):
            g = g.strip()
            if g:
                gc[g] += 1
    top = gc.most_common(12)
    if not top:
        return {"data":[{"type":"bar","x":[],"y":[]}],
                "layout":{"title":{"text":"VF Genes (none)","font":{"size":13}},
                           "paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"}}
    labels,counts = [i[0] for i in reversed(top)],[i[1] for i in reversed(top)]
    return {"data":[{"type":"bar","x":counts,"y":labels,"orientation":"h",
                     "marker":{"color":"#f59e0b"}}],
            "layout":{"title":{"text":"Top VF Genes","font":{"size":13}},
                      "xaxis":{"title":"Contigs"},"margin":{"t":45,"b":30,"l":160,"r":15},
                      "paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"}}


def _mge_bar(rows: list[PlasmidRow]) -> dict:
    fc: Counter[str] = Counter()
    for r in rows:
        for f in r.mge_families.split(";"):
            f = f.strip()
            if f:
                fc[f] += 1
    top = fc.most_common(12)
    if not top:
        return {"data":[{"type":"bar","x":[],"y":[]}],
                "layout":{"title":{"text":"MGE Families (none)","font":{"size":13}},
                           "paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"}}
    labels,counts = [i[0] for i in reversed(top)],[i[1] for i in reversed(top)]
    return {"data":[{"type":"bar","x":counts,"y":labels,"orientation":"h",
                     "marker":{"color":"#8b5cf6"}}],
            "layout":{"title":{"text":"Top MGE Families","font":{"size":13}},
                      "xaxis":{"title":"Contigs"},"margin":{"t":45,"b":30,"l":160,"r":15},
                      "paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"}}


def _risk_hist(risk_scores: list[int]) -> dict:
    c = Counter(risk_scores)
    sr = list(range(11))
    y = [c.get(s,0) for s in sr]
    cols = ["#c0392b" if s>=7 else "#e67e22" if s>=4 else "#27ae60" for s in sr]
    return {"data":[{"type":"bar","x":sr,"y":y,"marker":{"color":cols}}],
            "layout":{"title":{"text":"Risk Score Distribution","font":{"size":13}},
                      "xaxis":{"title":"Risk Score","dtick":1},"yaxis":{"title":"Plasmids"},
                      "margin":{"t":45,"b":40,"l":45,"r":15},
                      "paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"}}


def _build_drug_cooccurrence_heatmap(plasmid_results: list) -> dict:
    """Co-occurrence heatmap — kept for backward compat with report_cmd."""
    contig_classes = []
    for cr in plasmid_results:
        classes: set[str] = set()
        for hit in cr.arg_hits:
            for dc in hit.drug_class.split(";"):
                dc = dc.strip()
                if dc and dc not in ("unknown", ""):
                    classes.add(dc)
        if classes:
            contig_classes.append(frozenset(classes))
    all_classes = sorted({dc for fset in contig_classes for dc in fset})
    if len(all_classes) < 2:
        return {"data":[],"layout":{"title":{"text":"Co-occurrence (insufficient data)","font":{"size":13}},
                                     "paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"}}
    n = len(all_classes)
    matrix = [[0]*n for _ in range(n)]
    for fset in contig_classes:
        for i,ci in enumerate(all_classes):
            if ci not in fset: continue
            for j,cj in enumerate(all_classes):
                if cj in fset: matrix[i][j] += 1
    short = [c if len(c)<=26 else c[:25]+"…" for c in all_classes]
    return {"data":[{"type":"heatmap","z":matrix,"x":short,"y":short,
                     "colorscale":"Blues","showscale":True}],
            "layout":{"title":{"text":"Drug-Class Co-occurrence","font":{"size":13}},
                      "margin":{"t":50,"b":110,"l":150,"r":30},
                      "paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"}}


# ---------------------------------------------------------------------------
# Plasmid page renderer
# ---------------------------------------------------------------------------


def _plasmid_row_to_arr(r: PlasmidRow) -> list:
    """Compact row as JSON array for the vanilla paginator."""
    eskape = ""
    if r.eskape_host:
        eskape = r.eskape_genus
    return [
        r.contig_id,           # 0
        r.contig_length,       # 1
        round(r.confidence, 4),# 2
        r.num_args,            # 3
        r.drug_classes,        # 4
        r.arg_sources,         # 5
        r.num_vf,              # 6
        r.vf_genes,            # 7
        r.num_mge,             # 8
        r.mge_families,        # 9
        eskape,                # 10
        r.mobility_class,      # 11
        r.replicon_type,       # 12
        r.risk_score,          # 13
        r.taxonomy,            # 14
        r.risk_evidence,       # 15
    ]


def _render_plasmid_page(data: dict) -> str:
    rows: list[PlasmidRow] = data["plasmid_rows"]
    counts = data["class_counts"]
    n_total = len(rows)
    display = rows[:MAX_TABLE_ROWS]
    truncated = n_total > MAX_TABLE_ROWS

    stat_cards = (
        f'<div class="card" style="border-left:4px solid #2c6fad">'
        f'<h3>Plasmids</h3><p style="color:#2c6fad">{n_total:,}</p></div>'
        f'<div class="card" style="border-left:4px solid #c0392b">'
        f'<h3>ARGs</h3><p style="color:#c0392b">{data["total_args"]:,}</p></div>'
        f'<div class="card" style="border-left:4px solid #f59e0b">'
        f'<h3>VF Genes</h3><p style="color:#92400e">{data["total_vf"]:,}</p></div>'
        f'<div class="card" style="border-left:4px solid #8b5cf6">'
        f'<h3>MGEs</h3><p style="color:#5b21b6">{data["total_mge"]:,}</p></div>'
    )

    note = (f'<p class="note">Showing top {MAX_TABLE_ROWS:,} of {n_total:,} plasmid contigs. '
            f'Full data in predictions.tsv.</p>') if truncated else ""

    row_data = json.dumps([_plasmid_row_to_arr(r) for r in display])

    body = f"""
<h1>PlasFlow v2 — Plasmid Report</h1>
<p class="meta">Input: <code>{data["input_file"]}</code></p>
<div class="cards">{stat_cards}</div>

<h2>Overview</h2>
<div class="chart-grid three">
  <div id="cpie"  class="cbox"></div>
  <div id="carg"  class="cbox"></div>
  <div id="crisk" class="cbox"></div>
</div>
<div class="chart-grid">
  <div id="cvf"  class="cbox"></div>
  <div id="cmge" class="cbox"></div>
</div>

<h2>Plasmid Predictions ({n_total:,} contigs)</h2>
<div class="filter-bar">
  <button class="fbtn active" id="fa" onclick="setRisk('')"  style="background:#e0e0e0;color:#333">All</button>
  <button class="fbtn"        id="fh" onclick="setRisk('h')" style="background:#c0392b;color:#fff">High ≥7</button>
  <button class="fbtn"        id="fm" onclick="setRisk('m')" style="background:#e67e22;color:#fff">Medium 4–6</button>
  <button class="fbtn"        id="fl" onclick="setRisk('l')" style="background:#27ae60;color:#fff">Low 0–3</button>
</div>
{note}
<div class="tbl-wrap">
  <div class="tbl-ctrl">
    <input id="psearch" placeholder="Search…">
    <span class="pg-info" id="ppg"></span>
    <button class="pg-btn" id="pprev">◀ Prev</button>
    <button class="pg-btn" id="pnext">Next ▶</button>
  </div>
  <table id="ptable">
    <thead><tr>
      <th>Contig</th><th>Length</th><th>Conf.</th><th>ARGs</th><th>Drug Classes</th><th>DB</th>
      <th>VFs</th><th>VF Genes</th><th>MGEs</th><th>MGE Families</th>
      <th>Pathogen</th><th>Mobility</th><th>Replicon</th><th>Risk</th><th>Taxonomy</th><th>Evidence</th>
    </tr></thead>
    <tbody></tbody>
  </table>
</div>"""

    js = f"""
(function(){{
var Plotly=window.Plotly;
Plotly.newPlot('cpie',  {json.dumps(data['pie_data']['data'])},  {json.dumps(data['pie_data']['layout'])},  {{responsive:true,displayModeBar:false}});
Plotly.newPlot('carg',  {json.dumps(data['arg_data']['data'])},  {json.dumps(data['arg_data']['layout'])},  {{responsive:true,displayModeBar:false}});
Plotly.newPlot('crisk', {json.dumps(data['risk_data']['data'])}, {json.dumps(data['risk_data']['layout'])}, {{responsive:true,displayModeBar:false}});
Plotly.newPlot('cvf',   {json.dumps(data['vf_data']['data'])},   {json.dumps(data['vf_data']['layout'])},   {{responsive:true,displayModeBar:false}});
Plotly.newPlot('cmge',  {json.dumps(data['mge_data']['data'])},  {json.dumps(data['mge_data']['layout'])},  {{responsive:true,displayModeBar:false}});

var ALL={row_data};
var cur=ALL;
var risk_filter='';

function riskCls(v){{return v>=7?'h':v>=4?'m':'l';}}
function renderR(v){{var c=riskCls(v);return '<span class="risk-'+c+'">'+v+'</span>';}}
function srcBadges(s){{if(!s)return'—';
  return s.split(',').map(function(x){{x=x.trim();return x?'<span class="badge b'+x.toLowerCase()+'">'+esc(x)+'</span>':'';}}).join(' ');}}
function eskBadge(e){{if(!e)return'—';
  var ek=['Enterococcus','Staphylococcus','Klebsiella','Acinetobacter','Pseudomonas','Enterobacter','Escherichia'];
  var cls=ek.indexOf(e)>=0?'besk':'bwho';
  return '<span class="badge '+cls+'">'+esc(e)+'</span>';}}

var cols=[
  {{render:function(v){{return ellipsis(v,30);}}}},
  {{render:function(v){{return Number(v).toLocaleString();}}}},
  {{render:function(v){{return v;}}}} ,
  {{render:function(v){{return v;}}}} ,
  {{render:function(v){{return ellipsis(v,35);}}}},
  {{render:function(v){{return srcBadges(v);}}}},
  {{render:function(v,r){{return v>0?'<span class="badge bvf">'+v+' VF</span>':'—';}}}},
  {{render:function(v){{return ellipsis(v,30);}}}},
  {{render:function(v,r){{return v>0?'<span class="badge bmge">'+v+' MGE</span>':'—';}}}},
  {{render:function(v){{return ellipsis(v,30);}}}},
  {{render:function(v){{return eskBadge(v);}}}},
  {{render:function(v){{return esc(v);}}}},
  {{render:function(v){{return esc(v);}}}},
  {{render:function(v){{return renderR(v);}}}},
  {{render:function(v){{return ellipsis(v,28);}}}},
  {{render:function(v){{return ellipsis(v,35);}}}},
];
var tbl=new LightTable({{tableId:'ptable',data:ALL,cols:cols,pageSize:50,
  searchId:'psearch',pgInfoId:'ppg',prevId:'pprev',nextId:'pnext'}});

function setRisk(r){{
  risk_filter=r;
  ['fa','fh','fm','fl'].forEach(function(id){{document.getElementById(id).classList.remove('active');}});
  document.getElementById(r===''?'fa':r==='h'?'fh':r==='m'?'fm':'fl').classList.add('active');
  cur = r==='' ? ALL : ALL.filter(function(row){{return riskCls(row[13])===r;}});
  tbl.data=cur; tbl.filtered=cur.slice(); tbl.page=0; tbl.applySort(); tbl.render();
  document.getElementById('psearch').value='';
}}
window.setRisk=setRisk;
}})();"""

    extra_head = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" defer></script>'
    return (
        f'<!DOCTYPE html><html lang="en"><head>'
        f'<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Plasmid — PlasFlow v2</title>'
        f'{extra_head}'
        f'<style>{_CSS}</style></head><body>'
        f'{_nav("plasmid", counts, "#2c6fad")}'
        f'<div class="wrap">{body}'
        f'<footer>PlasFlow v2 — open in any browser, no server required.</footer>'
        f'</div>'
        f'<script>{_PAGINATOR_JS}{js}</script>'
        f'</body></html>'
    )


# ---------------------------------------------------------------------------
# Non-plasmid page renderer (no Plotly, no jQuery, no DataTables)
# ---------------------------------------------------------------------------


def _nonplasmid_row_to_arr(r: NonPlasmidRow, show_best: bool = False) -> list:
    row = [r.contig_id, r.contig_length, round(r.confidence, 4), r.taxonomy, r.taxonomy_lineage]
    if show_best:
        row += [r.best_label, round(r.best_score, 4)]
    return row


def _render_nonplasmid_page(
    data: dict,
    class_key: str,
    title: str,
    color: str,
    rows: list[NonPlasmidRow],
    table_id: str,
    show_best: bool = False,
    extra_note: str = "",
) -> str:
    n_total = len(rows)
    display = rows[:MAX_TABLE_ROWS]
    truncated = n_total > MAX_TABLE_ROWS
    counts = data["class_counts"]

    stat_cards = (
        f'<div class="card" style="border-left:4px solid {color}">'
        f'<h3>{title} Contigs</h3><p style="color:{color}">{n_total:,}</p></div>'
        f'<div class="card" style="border-left:4px solid #2c6fad">'
        f'<h3>Total Sequences</h3><p style="color:#2c6fad">{data["total"]:,}</p></div>'
    )

    note_parts = []
    if truncated:
        note_parts.append(
            f'Showing top {MAX_TABLE_ROWS:,} of {n_total:,} contigs (sorted by length). '
            f'Full data available in <code>predictions.tsv</code>.'
        )
    if extra_note:
        note_parts.append(extra_note)
    note_html = "".join(f'<p class="note">{n}</p>' for n in note_parts)

    # Column definitions
    base_cols_html = "<th>Contig</th><th>Length (bp)</th><th>Confidence</th><th>Taxonomy (LCA)</th>"
    if show_best:
        base_cols_html += "<th>Best Label</th><th>Best Score</th>"

    row_data = json.dumps([_nonplasmid_row_to_arr(r, show_best) for r in display])

    if show_best:
        col_js = """[
  {render:function(v){return ellipsis(v,40);}},
  {render:function(v){return Number(v).toLocaleString();}},
  {render:function(v){return v;}},
  {render:function(v,r){return '<span class="ellipsis" title="'+esc(r[4])+'">'+esc(v)+'</span>';}},
  {render:function(v){return esc(v);}},
  {render:function(v){return v;}}
]"""
    else:
        col_js = """[
  {render:function(v){return ellipsis(v,40);}},
  {render:function(v){return Number(v).toLocaleString();}},
  {render:function(v){return v;}},
  {render:function(v,r){return '<span class="ellipsis" title="'+esc(r[4])+'">'+esc(v)+'</span>';}},
]"""

    js = f"""
(function(){{
var DATA={row_data};
var cols={col_js};
new LightTable({{tableId:'{table_id}',data:DATA,cols:cols,pageSize:50,
  searchId:'{table_id}_search',pgInfoId:'{table_id}_pg',
  prevId:'{table_id}_prev',nextId:'{table_id}_next'}});
}})();"""

    body = f"""
<h1>PlasFlow v2 — {title} Report</h1>
<p class="meta">Input: <code>{data["input_file"]}</code></p>
<div class="cards">{stat_cards}</div>
<h2>{title} Contigs ({n_total:,})</h2>
{note_html}
<div class="tbl-wrap">
  <div class="tbl-ctrl">
    <input id="{table_id}_search" placeholder="Search contigs…">
    <span class="pg-info" id="{table_id}_pg"></span>
    <button class="pg-btn" id="{table_id}_prev">◀ Prev</button>
    <button class="pg-btn" id="{table_id}_next">Next ▶</button>
  </div>
  <table id="{table_id}">
    <thead><tr>{base_cols_html}</tr></thead>
    <tbody></tbody>
  </table>
</div>"""

    return _page(title, class_key, counts, color, body, extra_js=js)


# ---------------------------------------------------------------------------
# Main data builder
# ---------------------------------------------------------------------------


def build_report_data(pipeline_result, input_file: str = "") -> dict:
    all_arg_hits = [h for cr in pipeline_result.plasmid_results for h in cr.arg_hits]
    taxonomy = getattr(pipeline_result, "taxonomy", {}) or {}

    plasmid_rows: list[PlasmidRow] = []
    for cr in pipeline_result.plasmid_results:
        unique_classes = sorted({
            dc.strip()
            for h in cr.arg_hits
            for dc in h.drug_class.split(";")
            if dc.strip() and dc.strip() != "unknown"
        })
        mob = cr.mobility
        tax = getattr(cr, "taxonomy", None) or taxonomy.get(cr.record.id)
        vf_hits = getattr(cr, "vf_hits", [])
        mge_hits = getattr(cr, "mge_hits", [])
        sources = sorted({h.source for h in cr.arg_hits if getattr(h, "source", "")})
        plasmid_rows.append(PlasmidRow(
            contig_id=cr.record.id,
            contig_length=len(cr.record.seq),
            confidence=cr.prediction.confidence,
            num_args=len(cr.arg_hits),
            drug_classes="; ".join(unique_classes) or "—",
            mobility_class=mob.mobility_class if mob else "unknown",
            replicon_type=mob.replicon_type if mob else "unknown",
            risk_score=cr.risk.score,
            taxonomy=tax.display if tax else "—",
            risk_evidence="; ".join(cr.risk.evidence) if cr.risk.evidence else "—",
            arg_sources=", ".join(sources),
            eskape_host=cr.risk.eskape_host,
            eskape_genus=cr.risk.eskape_genus,
            num_vf=len(vf_hits),
            vf_genes="; ".join(sorted({h.gene_name for h in vf_hits})),
            num_mge=len(mge_hits),
            mge_families="; ".join(sorted({h.is_family for h in mge_hits})),
        ))

    non_plasmid_results = getattr(pipeline_result, "non_plasmid_results", [])
    phage_rows: list[NonPlasmidRow] = []
    chromosome_rows: list[NonPlasmidRow] = []
    archaea_rows: list[NonPlasmidRow] = []
    unclassified_rows: list[NonPlasmidRow] = []

    for npr in non_plasmid_results:
        tax = getattr(npr, "taxonomy", None) or taxonomy.get(npr.record.id)
        scores = getattr(npr.prediction, "scores", {}) or {}
        best_label = max(scores, key=scores.get) if scores else ""
        best_score = scores.get(best_label, 0.0) if best_label else 0.0
        row = NonPlasmidRow(
            contig_id=npr.record.id,
            contig_length=len(npr.record.seq),
            label=npr.prediction.label,
            confidence=npr.prediction.confidence,
            taxonomy=tax.display if tax else "—",
            taxonomy_lineage=tax.lineage if tax else "—",
            best_label=best_label,
            best_score=best_score,
        )
        lbl = npr.prediction.label
        if lbl == "phage":          phage_rows.append(row)
        elif lbl == "chromosome":   chromosome_rows.append(row)
        elif lbl == "archaea":      archaea_rows.append(row)
        else:                       unclassified_rows.append(row)

    for lst in (chromosome_rows, unclassified_rows):
        lst.sort(key=lambda r: r.contig_length, reverse=True)

    total_vf  = sum(r.num_vf  for r in plasmid_rows)
    total_mge = sum(r.num_mge for r in plasmid_rows)
    risk_scores = [cr.risk.score for cr in pipeline_result.plasmid_results]
    tax_classified = sum(1 for r in taxonomy.values() if r.rank != "unclassified")

    return {
        "input_file":        input_file or str(pipeline_result.input_fasta),
        "total":             pipeline_result.total_sequences,
        "num_plasmids":      pipeline_result.total_plasmids,
        "total_args":        pipeline_result.total_args,
        "total_vf":          total_vf,
        "total_mge":         total_mge,
        "tax_classified":    tax_classified,
        "class_counts":      pipeline_result.class_counts,
        "pie_data":          _pie(pipeline_result.class_counts),
        "arg_data":          _arg_bar(all_arg_hits),
        "risk_data":         _risk_hist(risk_scores),
        "vf_data":           _vf_bar(plasmid_rows),
        "mge_data":          _mge_bar(plasmid_rows),
        "cooccurrence_data": _build_drug_cooccurrence_heatmap(pipeline_result.plasmid_results),
        "scatter_data":      {},
        "tax_bar_data":      {},
        "plasmid_rows":      plasmid_rows,
        "chromosome_rows":   chromosome_rows,
        "phage_rows":        phage_rows,
        "archaea_rows":      archaea_rows,
        "unclassified_rows": unclassified_rows,
        "other_rows":        archaea_rows + unclassified_rows,
        "has_scatter":       False,
        "has_cooccurrence":  False,
        "has_phages":        bool(phage_rows),
        "has_chromosomes":   bool(chromosome_rows),
        "has_others":        bool(archaea_rows or unclassified_rows),
    }


# ---------------------------------------------------------------------------
# 5-file renderer
# ---------------------------------------------------------------------------


def generate_reports(report_data: dict, output_dir: Path | str) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pages = {
        "plasmid":      out / "report_plasmid.html",
        "chromosome":   out / "report_chromosome.html",
        "phage":        out / "report_phage.html",
        "archaea":      out / "report_archaea.html",
        "unclassified": out / "report_unclassified.html",
    }

    html_map = {
        "plasmid": _render_plasmid_page(report_data),
        "chromosome": _render_nonplasmid_page(
            report_data, "chromosome", "Chromosome", "#27ae60",
            report_data["chromosome_rows"], "ctable",
        ),
        "phage": _render_nonplasmid_page(
            report_data, "phage", "Phage", "#e67e22",
            report_data["phage_rows"], "phtable",
        ),
        "archaea": _render_nonplasmid_page(
            report_data, "archaea", "Archaea", "#8e44ad",
            report_data["archaea_rows"], "artable",
        ),
        "unclassified": _render_nonplasmid_page(
            report_data, "unclassified", "Unclassified", "#95a5a6",
            report_data["unclassified_rows"], "utable",
            show_best=True,
            extra_note=(
                "Contigs where no class scored ≥ threshold. "
                "Best Label shows the top-scoring class. "
                "Use <code>--min-confidence</code> to assign these instead of leaving them unclassified."
            ),
        ),
    }

    for key, path in pages.items():
        path.write_text(html_map[key], encoding="utf-8")
        logger.info("Report written to %s", path)

    return pages


# ---------------------------------------------------------------------------
# Legacy single-file entry point (backward compat)
# ---------------------------------------------------------------------------


def generate_report(report_data: dict, output_path: Path | str) -> Path:
    """Write 5 HTML files; return the plasmid report path (backward compat)."""
    paths = generate_reports(report_data, Path(output_path).parent)
    return paths["plasmid"]
