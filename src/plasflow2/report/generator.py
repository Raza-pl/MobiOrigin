"""HTML report generator — 5 interlinked, chart-equipped HTML files.

Performance targets
-------------------
* All pages: Plotly loaded from CDN (cached after first visit, ~1 MB).
* Row data stored as compact JSON; vanilla JS renders 50 rows at a time.
* All pages stay under ~400 KB even with 2 000 table rows.
* Plasmid table: checkbox multi-select, Download Selected TSV.
* Risk-score 0 plasmid contigs excluded from HTML (in predictions.tsv).

Charts per page
---------------
* Plasmid     : Overview pie | ARG drug classes | VF genes | MGE families
                Mobility classes | ESKAPE & other pathogens | Risk histogram
* Chromosome  : Length distribution | Confidence distribution | Top taxonomy
* Phage       : Length distribution | Confidence distribution | Top taxonomy
* Archaea     : Length distribution | Confidence distribution | Top taxonomy
* Unclassified: Best-label distribution | Confidence distribution | Length dist
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_TABLE_ROWS = 2_000

# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,Arial,sans-serif;background:#f4f6f9;color:#333;font-size:14px}
.nav{display:flex;align-items:stretch;background:#fff;border-bottom:2px solid #e0e8f5;
     box-shadow:0 2px 6px rgba(0,0,0,.07);padding:0 22px;flex-wrap:wrap}
.nav a{display:inline-flex;align-items:center;padding:11px 16px;text-decoration:none;
       font-weight:600;font-size:.86rem;color:#666;border-bottom:3px solid transparent;
       white-space:nowrap;transition:color .15s,border-color .15s}
.nav a:hover{color:#2c6fad;border-bottom-color:#2c6fad}
.nav a.active{color:var(--nc);border-bottom-color:var(--nc)}
.nav .pill{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:.76rem;
           color:#888;padding:0 4px;flex-wrap:wrap}
.nav .pill span{background:#f0f4f8;border-radius:12px;padding:2px 8px;white-space:nowrap}
.wrap{max-width:1380px;margin:0 auto;padding:22px 24px}
h1{color:#2c6fad;font-size:1.4rem;margin-bottom:4px}
h2{color:#444;margin-top:28px;border-bottom:2px solid #e0e8f5;padding-bottom:5px;font-size:1.05rem}
.meta{color:#777;font-size:.86rem;margin:5px 0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:14px 0}
.card{background:#fff;padding:12px 14px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.card h3{font-size:.7rem;text-transform:uppercase;color:#999;letter-spacing:.4px;margin-bottom:4px}
.card p{font-size:1.55rem;font-weight:700}
.chart-grid{display:grid;gap:12px;margin:16px 0}
.g2{grid-template-columns:1fr 1fr}
.g3{grid-template-columns:1fr 1fr 1fr}
.g4{grid-template-columns:1fr 1fr 1fr 1fr}
.cbox{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);
      padding:4px;min-height:260px}
/* table */
.tbl-wrap{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);
          padding:14px;margin-top:12px}
.tbl-ctrl{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}
.tbl-ctrl input[type=text]{padding:5px 9px;border:1px solid #d0d8e4;border-radius:6px;
                font-size:.84rem;width:220px;outline:none}
.tbl-ctrl input[type=text]:focus{border-color:#2c6fad}
.pg-info{margin-left:auto;font-size:.8rem;color:#888;white-space:nowrap}
.pg-btn{padding:4px 11px;border:1px solid #d0d8e4;border-radius:6px;background:#fff;
        cursor:pointer;font-size:.8rem}
.pg-btn:hover{background:#f0f4f8}
.pg-btn:disabled{opacity:.35;cursor:default}
.dl-btn{padding:5px 14px;border:none;border-radius:6px;cursor:pointer;font-size:.82rem;
        font-weight:600;color:#fff}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{background:#f8fafc;text-align:left;padding:7px 8px;font-size:.72rem;
   text-transform:uppercase;color:#777;border-bottom:2px solid #e0e8f5;
   cursor:pointer;user-select:none;white-space:nowrap}
th:hover{background:#eef2f7}
th.sort-asc::after{content:" ▲"}th.sort-desc::after{content:" ▼"}
th.no-sort{cursor:default}th.no-sort:hover{background:#f8fafc}
td{padding:6px 8px;border-bottom:1px solid #f0f4f8;vertical-align:middle}
tr:hover td{background:#f7fbff}
.ellipsis{max-width:170px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
          display:inline-block;vertical-align:middle}
.badge{display:inline-block;padding:1px 6px;border-radius:10px;font-size:.71rem;font-weight:700}
.bvf {background:#fef3c7;color:#92400e;border:1px solid #f59e0b}
.bmge{background:#ede9fe;color:#5b21b6;border:1px solid #8b5cf6}
.bcard{background:#e8f0fe;color:#1a56db}.bsarg{background:#fef3c7;color:#b45309}
.besk{background:#fde8e8;color:#c0392b;border:1px solid #e74c3c}
.bwho{background:#fef3c7;color:#b45309;border:1px solid #f39c12}
.bcirc{background:#e0f2fe;color:#0369a1;border:1px solid #38bdf8}
.blowconf{background:#fef9c3;color:#854d0e;border:1px solid #fbbf24}
.risk-h{color:#c0392b;font-weight:700}.risk-m{color:#e67e22;font-weight:700}
.risk-l{color:#27ae60;font-weight:700}
.filter-bar{display:flex;gap:7px;margin:8px 0;flex-wrap:wrap;align-items:center}
.fbtn{padding:4px 12px;border:none;border-radius:14px;cursor:pointer;
      font-size:.8rem;font-weight:600;opacity:.85;white-space:nowrap}
.fbtn:hover,.fbtn.active{opacity:1}
.fbtn.active{outline:2px solid #333}
.note{color:#888;font-size:.78rem;margin:3px 0 8px;font-style:italic}
.narrative{background:linear-gradient(135deg,#eef6ff,#f0fff4);border-left:4px solid #2c6fad;
           border-radius:0 8px 8px 0;padding:14px 18px;margin:12px 0 20px;font-size:.93rem;
           line-height:1.6;color:#2d3748}
footer{margin-top:36px;color:#bbb;font-size:.76rem;border-top:1px solid #e5e5e5;padding-top:8px}
input[type=checkbox]{width:14px;height:14px;cursor:pointer;accent-color:#2c6fad}
.sel-count{font-size:.8rem;color:#2c6fad;font-weight:600}
"""

# ---------------------------------------------------------------------------
# Shared Plotly config
# ---------------------------------------------------------------------------

_PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

_LAYOUT_BASE = {
    "margin": {"t": 40, "b": 30, "l": 150, "r": 15},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"size": 11},
}


def _layout(**extra) -> dict:
    d = dict(_LAYOUT_BASE)
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# Vanilla JS — shared paginator + checkbox support
# ---------------------------------------------------------------------------

_PAGINATOR_JS = r"""
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function ellipsis(s,mx){s=esc(s);return s.length>mx?'<span class="ellipsis" title="'+s+'">'+s.slice(0,mx)+'…</span>':s;}

function LightTable(cfg){
  var t=this;
  t.data=cfg.data; t.filtered=cfg.data.slice();
  t.cols=cfg.cols; t.page=0; t.ps=cfg.pageSize||50;
  t.sortCol=-1; t.sortDir=1;
  t.hasCheck=!!cfg.onCheck;
  t.selected=cfg.selected||null; // external Set for cross-page selection
  t.tbody=document.querySelector('#'+cfg.tableId+' tbody');
  t.ths=document.querySelectorAll('#'+cfg.tableId+' thead th');
  t.pgInfo=document.getElementById(cfg.pgInfoId);
  t.prevBtn=document.getElementById(cfg.prevId);
  t.nextBtn=document.getElementById(cfg.nextId);
  t.searchEl=document.getElementById(cfg.searchId);
  t.selCount=cfg.selCountId?document.getElementById(cfg.selCountId):null;

  // sort
  t.ths.forEach(function(th,ci){
    if(th.classList.contains('no-sort'))return;
    th.addEventListener('click',function(){
      if(t.sortCol===ci){t.sortDir*=-1;}else{t.sortCol=ci;t.sortDir=1;}
      t.ths.forEach(function(h){h.className=h.classList.contains('no-sort')?'no-sort':'';});
      th.className=(t.sortDir===1?'sort-asc':'sort-desc');
      t.applySort();t.page=0;t.render();
    });
  });
  // search
  if(t.searchEl)t.searchEl.addEventListener('input',function(){
    var q=t.searchEl.value.toLowerCase();
    t.filtered=q?t.data.filter(function(r){
      return r.some(function(c){return String(c).toLowerCase().indexOf(q)!==-1;});
    }):t.data.slice();
    t.applySort();t.page=0;t.render();
  });
  if(t.prevBtn)t.prevBtn.addEventListener('click',function(){if(t.page>0){t.page--;t.render();}});
  if(t.nextBtn)t.nextBtn.addEventListener('click',function(){
    if((t.page+1)*t.ps<t.filtered.length){t.page++;t.render();}
  });

  t.applySort=function(){
    if(t.sortCol<0)return;
    var ci=t.sortCol,dir=t.sortDir;
    t.filtered.sort(function(a,b){
      var av=a[ci],bv=b[ci],an=parseFloat(av),bn=parseFloat(bv);
      if(!isNaN(an)&&!isNaN(bn))return dir*(an-bn);
      return dir*String(av).localeCompare(String(bv));
    });
  };

  t.render=function(){
    var start=t.page*t.ps,end=Math.min(start+t.ps,t.filtered.length);
    var html='';
    for(var i=start;i<end;i++){
      var r=t.filtered[i];
      var rowId=r[0]; // first col used as unique key
      var chk=t.hasCheck?(
        '<td><input type="checkbox" data-id="'+esc(rowId)+'" '+(t.selected&&t.selected.has(rowId)?'checked':'')+
        ' onchange="tblCheckChange(this)"></td>'):'' ;
      html+='<tr>'+chk+t.cols.map(function(c,ci){
        return '<td>'+c.render(r[ci],r)+'</td>';
      }).join('')+'</tr>';
    }
    t.tbody.innerHTML=html||'<tr><td colspan="'+(t.cols.length+(t.hasCheck?1:0))+
      '" style="color:#999;text-align:center;padding:20px">No results</td></tr>';
    if(t.pgInfo)t.pgInfo.textContent=t.filtered.length===0?'0 rows':
      'Showing '+(start+1)+'–'+end+' of '+t.filtered.length.toLocaleString();
    if(t.prevBtn)t.prevBtn.disabled=t.page===0;
    if(t.nextBtn)t.nextBtn.disabled=end>=t.filtered.length;
    if(t.selCount&&t.selected)t.selCount.textContent=t.selected.size+' selected';
  };
  t.render();
}
"""

# ---------------------------------------------------------------------------
# Nav bar
# ---------------------------------------------------------------------------

_NAV_PAGES = [
    ("plasmid",      "report_plasmid.html",      "Plasmid",      "#2c6fad"),
    ("chromosome",   "report_chromosome.html",    "Chromosome",   "#27ae60"),
    ("phage",        "report_phage.html",         "Phage",        "#e67e22"),
    ("archaea",      "report_archaea.html",        "Archaea",      "#8e44ad"),
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


def _full_page(title: str, active: str, counts: dict, color: str, body: str, js: str = "") -> str:
    return (
        f'<!DOCTYPE html><html lang="en"><head>'
        f'<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{title} — PlasFlow v2</title>'
        f'{_PLOTLY_CDN}'
        f'<style>{_CSS}</style></head><body>'
        f'{_nav(active, counts, color)}'
        f'<div class="wrap">{body}'
        f'<footer>PlasFlow v2 — open in any browser, no server required.</footer></div>'
        f'<script>{_PAGINATOR_JS}{js}</script>'
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
    arg_genes: str = ""      # e.g. "blaNDM-1; sul1"
    eskape_host: bool = False
    eskape_genus: str = ""
    num_vf: int = 0
    vf_genes: str = ""
    num_mge: int = 0
    mge_genes: str = ""      # actual IS element names e.g. "ISAba1; IS26"
    mge_families: str = ""   # IS families e.g. "IS4; Tn3"
    topology: str = "linear"        # "circular" | "linear" | "too_short"
    low_confidence: bool = False    # True if confidence < 0.70


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
    # ARG annotation (universal — all contig classes)
    num_args: int = 0
    arg_genes: str = ""
    drug_classes: str = ""
    arg_sources: str = ""
    # VF annotation
    num_vf: int = 0
    vf_genes: str = ""
    # MGE annotation
    num_mge: int = 0
    mge_genes: str = ""
    mge_families: str = ""
    topology: str = "linear"
    low_confidence: bool = False


# ---------------------------------------------------------------------------
# Plasmid chart builders
# ---------------------------------------------------------------------------


def _pie(class_counts: dict[str, int]) -> dict:
    colors = {"plasmid":"#2c6fad","chromosome":"#27ae60","phage":"#e67e22",
              "archaea":"#8e44ad","unclassified":"#95a5a6"}
    labels, values = list(class_counts.keys()), list(class_counts.values())
    return {"data":[{"type":"pie","labels":labels,"values":values,
                     "marker":{"colors":[colors.get(l,"#aaa") for l in labels]},
                     "textinfo":"label+percent","hole":0.35}],
            "layout":{**_layout(margin={"t":40,"b":10,"l":10,"r":10}),
                      "title":{"text":"Classification Overview","font":{"size":12}},
                      "showlegend":False}}


def _arg_bar(arg_hits: list) -> dict:
    dc: Counter[str] = Counter()
    for h in arg_hits:
        for c in h.drug_class.split(";"):
            c = c.strip()
            if c and c != "unknown":
                dc[c] += 1
    if not dc:
        return {"data":[{"type":"bar","x":[],"y":[],"orientation":"h"}],
                "layout":{**_layout(),"title":{"text":"ARG Drug Classes (none detected)","font":{"size":12}}}}
    items = sorted(dc.items(), key=lambda x: x[1])
    return {"data":[{"type":"bar","x":[i[1] for i in items],"y":[i[0] for i in items],
                     "orientation":"h","marker":{"color":"#c0392b"}}],
            "layout":{**_layout(),"title":{"text":"ARGs by Drug Class","font":{"size":12}},
                      "xaxis":{"title":"Count"}}}


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
                "layout":{**_layout(),"title":{"text":"VF Genes (none detected)","font":{"size":12}}}}
    return {"data":[{"type":"bar","x":[i[1] for i in reversed(top)],
                     "y":[i[0] for i in reversed(top)],
                     "orientation":"h","marker":{"color":"#f59e0b"}}],
            "layout":{**_layout(),"title":{"text":"Top VF Genes","font":{"size":12}},
                      "xaxis":{"title":"Contigs"}}}


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
                "layout":{**_layout(),"title":{"text":"MGE Families (none detected)","font":{"size":12}}}}
    return {"data":[{"type":"bar","x":[i[1] for i in reversed(top)],
                     "y":[i[0] for i in reversed(top)],
                     "orientation":"h","marker":{"color":"#8b5cf6"}}],
            "layout":{**_layout(),"title":{"text":"Top MGE Families","font":{"size":12}},
                      "xaxis":{"title":"Contigs"}}}


def _mobility_bar(rows: list[PlasmidRow]) -> dict:
    mc: Counter[str] = Counter(r.mobility_class for r in rows)
    items = sorted(mc.items(), key=lambda x: x[1])
    colors_map = {"conjugative":"#c0392b","mobilizable":"#e67e22","non-mobilizable":"#27ae60",
                  "unknown":"#95a5a6"}
    bar_colors = [colors_map.get(i[0], "#2c6fad") for i in items]
    return {"data":[{"type":"bar","x":[i[1] for i in items],"y":[i[0] for i in items],
                     "orientation":"h","marker":{"color":bar_colors}}],
            "layout":{**_layout(margin={"t":40,"b":30,"l":140,"r":15}),
                      "title":{"text":"Mobility Classes","font":{"size":12}},
                      "xaxis":{"title":"Plasmid contigs"}}}


def _eskape_bar(rows: list[PlasmidRow]) -> dict:
    gc: Counter[str] = Counter()
    for r in rows:
        if r.eskape_host and r.eskape_genus:
            gc[r.eskape_genus] += 1
    if not gc:
        return {"data":[{"type":"bar","x":[],"y":[]}],
                "layout":{**_layout(),"title":{"text":"ESKAPE / Pathogen Hosts (none detected)","font":{"size":12}}}}
    items = sorted(gc.items(), key=lambda x: x[1])
    eskape_set = {"Enterococcus","Staphylococcus","Klebsiella","Acinetobacter",
                  "Pseudomonas","Enterobacter","Escherichia","Enterobacteriaceae"}
    bar_colors = ["#c0392b" if i[0] in eskape_set else "#e67e22" for i in items]
    return {"data":[{"type":"bar","x":[i[1] for i in items],"y":[i[0] for i in items],
                     "orientation":"h","marker":{"color":bar_colors},
                     "name":""}],
            "layout":{**_layout(margin={"t":40,"b":30,"l":150,"r":15}),
                      "title":{"text":"ESKAPE & Other Pathogen Hosts<br><sup>Red = ESKAPE, Orange = WHO priority</sup>",
                               "font":{"size":12}},
                      "xaxis":{"title":"Plasmid contigs"}}}


def _risk_hist(risk_scores: list[int]) -> dict:
    c = Counter(risk_scores)
    sr = list(range(1, 11))  # exclude 0
    y = [c.get(s, 0) for s in sr]
    cols = ["#c0392b" if s >= 7 else "#e67e22" if s >= 4 else "#27ae60" for s in sr]
    return {"data":[{"type":"bar","x":sr,"y":y,"marker":{"color":cols}}],
            "layout":{**_layout(margin={"t":40,"b":40,"l":45,"r":15}),
                      "title":{"text":"Risk Score Distribution (score ≥ 1)","font":{"size":12}},
                      "xaxis":{"title":"Risk Score","dtick":1},
                      "yaxis":{"title":"Plasmids"}}}


# ---------------------------------------------------------------------------
# Non-plasmid chart builders
# ---------------------------------------------------------------------------

_LENGTH_EDGES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, float("inf")]
_LENGTH_LABELS = ["1–3 kb", "3–10 kb", "10–30 kb", "30–100 kb", "100–300 kb", "300 kb–1 Mb", ">1 Mb"]


def _length_hist(rows: list[NonPlasmidRow], title: str, color: str) -> dict:
    bins = [0] * len(_LENGTH_EDGES)
    for r in rows:
        for i, edge in enumerate(_LENGTH_EDGES):
            if r.contig_length < edge:
                bins[i] += 1
                break
    return {"data":[{"type":"bar","x":_LENGTH_LABELS,"y":bins,
                     "marker":{"color":color}}],
            "layout":{**_layout(margin={"t":40,"b":60,"l":55,"r":15}),
                      "title":{"text":f"Contig Length Distribution — {title}","font":{"size":12}},
                      "xaxis":{"tickangle":-35},"yaxis":{"title":"Contigs"}}}


def _confidence_hist(rows: list[NonPlasmidRow], title: str, color: str) -> dict:
    bins_x = [f"{i/10:.1f}–{(i+1)/10:.1f}" for i in range(10)]
    bins_y = [0] * 10
    for r in rows:
        i = min(int(r.confidence * 10), 9)
        bins_y[i] += 1
    return {"data":[{"type":"bar","x":bins_x,"y":bins_y,
                     "marker":{"color":color,"opacity":0.8}}],
            "layout":{**_layout(margin={"t":40,"b":50,"l":55,"r":15}),
                      "title":{"text":f"Classification Confidence — {title}","font":{"size":12}},
                      "xaxis":{"title":"Confidence","tickangle":-30},
                      "yaxis":{"title":"Contigs"}}}


def _taxonomy_bar_nonplasmid(rows: list[NonPlasmidRow], title: str, color: str) -> dict:
    tc: Counter[str] = Counter()
    for r in rows:
        t = r.taxonomy
        if t and t != "—":
            tc[t] += 1
    top = tc.most_common(12)
    if not top:
        return {"data":[{"type":"bar","x":[],"y":[]}],
                "layout":{**_layout(),"title":{"text":f"Taxonomy — {title} (no data)","font":{"size":12}}}}
    return {"data":[{"type":"bar","x":[i[1] for i in reversed(top)],
                     "y":[i[0] for i in reversed(top)],
                     "orientation":"h","marker":{"color":color}}],
            "layout":{**_layout(),"title":{"text":f"Top Taxonomy — {title}","font":{"size":12}},
                      "xaxis":{"title":"Contigs"}}}


def _best_label_bar(rows: list[NonPlasmidRow]) -> dict:
    bc: Counter[str] = Counter(r.best_label for r in rows if r.best_label)
    if not bc:
        return {"data":[{"type":"bar","x":[],"y":[]}],
                "layout":{**_layout(),"title":{"text":"Best Label Distribution","font":{"size":12}}}}
    colors_map = {"plasmid":"#2c6fad","chromosome":"#27ae60","phage":"#e67e22","archaea":"#8e44ad"}
    items = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    return {"data":[{"type":"bar",
                     "x":[i[0] for i in items],
                     "y":[i[1] for i in items],
                     "marker":{"color":[colors_map.get(i[0],"#95a5a6") for i in items]}}],
            "layout":{**_layout(margin={"t":40,"b":40,"l":45,"r":15}),
                      "title":{"text":"Best-Guess Class Distribution<br><sup>What the model would assign if forced</sup>",
                               "font":{"size":12}},
                      "xaxis":{"title":"Class"},"yaxis":{"title":"Contigs"}}}


def _pathogen_bar(pathogens: dict) -> dict:
    """Horizontal bar: top pathogenic species detected across all contigs."""
    from collections import Counter
    sc: Counter[str] = Counter()
    level_map: dict[str, str] = {}
    for pr in pathogens.values():
        sc[pr.species] += 1
        level_map[pr.species] = getattr(pr, "threat_level", "medium")
    if not sc:
        return {"data":[{"type":"bar","x":[],"y":[],"orientation":"h"}],
                "layout":{**_layout(),
                           "title":{"text":"Pathogenic Species (none detected — run with --taxonomy-db)",
                                    "font":{"size":11}}}}
    items = sorted(sc.items(), key=lambda x: x[1])[-20:]  # top 20
    level_colors = {"critical":"#c0392b","high":"#e67e22","medium":"#f1c40f"}
    return {"data":[{"type":"bar",
                     "x":[i[1] for i in items],
                     "y":[i[0] for i in items],
                     "orientation":"h",
                     "text":[level_map.get(i[0],"") for i in items],
                     "textposition":"auto",
                     "marker":{"color":[level_colors.get(level_map.get(i[0],"medium"),"#aaa")
                                        for i in items]}}],
            "layout":{**_layout(),
                      "title":{"text":"Pathogenic Species Detected<br>"
                                      "<sup>Red=critical · Orange=high · Yellow=medium</sup>",
                               "font":{"size":12}},
                      "xaxis":{"title":"Contigs"}}}


def _build_drug_cooccurrence_heatmap(plasmid_rows: list) -> dict:
    """Build drug-class co-occurrence heatmap from PlasmidRow objects.

    Each cell (i,j) shows how many plasmids carry both drug class i and j.
    Only includes plasmids with ≥2 drug classes (co-occurrence requires ≥2).
    """
    from collections import Counter
    # Collect drug classes per plasmid
    drug_sets: list[list[str]] = []
    for r in plasmid_rows:
        if not hasattr(r, "drug_classes") or not r.drug_classes or r.drug_classes == "—":
            continue
        classes = [c.strip() for c in r.drug_classes.replace(";", ",").split(",")
                   if c.strip() and c.strip() != "—"]
        if len(classes) >= 2:
            drug_sets.append(classes)

    if not drug_sets:
        return {"data": [], "layout": {"title": {"text": "Drug Class Co-occurrence", "font": {"size": 12}}}}

    # Build co-occurrence matrix
    all_classes = sorted({c for ds in drug_sets for c in ds})
    if len(all_classes) < 2:
        return {"data": [], "layout": {"title": {"text": "Drug Class Co-occurrence", "font": {"size": 12}}}}

    n = len(all_classes)
    idx = {c: i for i, c in enumerate(all_classes)}
    matrix = [[0] * n for _ in range(n)]
    for ds in drug_sets:
        for i, a in enumerate(ds):
            for b in ds[i + 1:]:
                ia, ib = idx[a], idx[b]
                matrix[ia][ib] += 1
                matrix[ib][ia] += 1

    # Shorten long class names for display
    def _short(name: str) -> str:
        subs = {
            "antibiotic": "", "beta-lactam": "β-lact", "aminoglycoside": "aminogl.",
            "fluoroquinolone": "FQ", "tetracycline": "Tet", "macrolide": "Macr.",
            "carbapenem": "Carb.", "cephalosporin": "Ceph.", "sulfonamide": "Sulf.",
            "trimethoprim": "TMP", "diaminopyrimidine": "DMP", "phenicol": "Phen.",
            "lincosamide": "Linc.", "streptogramin": "Strept.", "rifamycin": "Rif.",
            "glycopeptide": "Glyc.", "peptide": "Pept.",
        }
        for k, v in subs.items():
            name = name.replace(k, v).strip()
        return name[:20]

    labels = [_short(c) for c in all_classes]
    z_text = [[str(matrix[i][j]) if matrix[i][j] > 0 else "" for j in range(n)] for i in range(n)]

    return {
        "data": [{
            "type": "heatmap",
            "z": matrix,
            "x": labels,
            "y": labels,
            "text": z_text,
            "texttemplate": "%{text}",
            "colorscale": [[0, "#fff5f0"], [0.5, "#fc8d59"], [1, "#b30000"]],
            "showscale": True,
            "colorbar": {"title": "Plasmids", "thickness": 12},
        }],
        "layout": {
            "title": {"text": "Drug Class Co-occurrence on Plasmids", "font": {"size": 12}},
            "xaxis": {"tickangle": -35, "tickfont": {"size": 9}},
            "yaxis": {"tickfont": {"size": 9}},
            "margin": {"l": 130, "b": 130, "t": 40, "r": 20},
        },
    }


# ---------------------------------------------------------------------------
# Genome map — per-plasmid horizontal track diagram
# ---------------------------------------------------------------------------

_GENE_COLORS = {
    "arg":      "#e74c3c",   # red
    "vf":       "#e67e22",   # orange
    "mge":      "#8e44ad",   # purple
    "mobility": "#2980b9",   # blue
    "other":    "#bdc3c7",   # light grey
}

_MOBILITY_KEYWORDS = frozenset([
    "conjugat", "relaxase", "mpf", "traG", "traI", "traJ",
    "replic", "mob", "oriT", "virB", "virD",
])


def _gene_type(gene_name: str, arg_flag: int, vf_flag: int, mge_flag: int) -> str:
    """Classify an ORF into a display category for the genome map."""
    if arg_flag:
        return "arg"
    if mge_flag:
        return "mge"
    if vf_flag:
        return "vf"
    name_lower = gene_name.lower()
    if any(kw in name_lower for kw in _MOBILITY_KEYWORDS):
        return "mobility"
    return "other"


def _genome_map_data(
    contig_id: str,
    contig_length: int,
    orfs: list,           # list of ORF objects (filtered to this contig)
    arg_orf_ids: set[str],
    vf_orf_ids: set[str],
    mge_orf_ids: set[str],
    arg_name_by_orf: dict[str, str],
    vf_name_by_orf: dict[str, str],
    mge_name_by_orf: dict[str, str],
) -> dict:
    """Build a Plotly figure dict for a single contig genome map."""
    traces = {t: {"x": [], "y": [], "text": [], "color": []} for t in _GENE_COLORS}

    for orf in orfs:
        oid = orf.orf_id
        gene_type = _gene_type(
            arg_name_by_orf.get(oid, vf_name_by_orf.get(oid, mge_name_by_orf.get(oid, ""))),
            1 if oid in arg_orf_ids else 0,
            1 if oid in vf_orf_ids  else 0,
            1 if oid in mge_orf_ids else 0,
        )
        mid   = (orf.start + orf.end) / 2
        width = max(orf.end - orf.start, 1)
        label = (arg_name_by_orf.get(oid)
                 or vf_name_by_orf.get(oid)
                 or mge_name_by_orf.get(oid)
                 or orf.orf_id)
        hover = f"{orf.orf_id}<br>{orf.start}–{orf.end} ({'+' if orf.strand >= 0 else '-'})<br>{label}"
        traces[gene_type]["x"].append(mid)
        traces[gene_type]["y"].append(1 if orf.strand >= 0 else -1)
        traces[gene_type]["text"].append(hover)
        # store width for marker sizing (approx pixels — capped)
        traces[gene_type]["color"].append(width)

    plotly_traces = []
    type_labels = {"arg": "ARG", "vf": "Virulence Factor",
                   "mge": "MGE/IS", "mobility": "Mobility Gene", "other": "Other ORF"}
    for gtype, color in _GENE_COLORS.items():
        d = traces[gtype]
        if not d["x"]:
            continue
        plotly_traces.append({
            "type": "scatter",
            "mode": "markers",
            "name": type_labels[gtype],
            "x": d["x"],
            "y": d["y"],
            "text": d["text"],
            "hoverinfo": "text",
            "marker": {
                "color": color,
                "symbol": "square",
                "size": 12,
                "line": {"width": 0.5, "color": "#fff"},
            },
        })

    # Backbone line
    plotly_traces.insert(0, {
        "type": "scatter",
        "mode": "lines",
        "name": "Contig",
        "x": [0, contig_length],
        "y": [0, 0],
        "line": {"color": "#555", "width": 2},
        "hoverinfo": "skip",
        "showlegend": False,
    })

    return {
        "data": plotly_traces,
        "layout": {
            "height": 180,
            "margin": {"l": 40, "r": 20, "t": 30, "b": 30},
            "title": {"text": f"{contig_id} ({contig_length:,} bp)", "font": {"size": 12}},
            "xaxis": {"title": "Position (bp)", "range": [0, contig_length]},
            "yaxis": {"visible": False, "range": [-2.5, 2.5]},
            "showlegend": True,
            "legend": {"orientation": "h", "y": -0.3},
            "plot_bgcolor": "#f9f9f9",
            "paper_bgcolor": "transparent",
        },
    }


# ---------------------------------------------------------------------------
# Plain-English summary narrative
# ---------------------------------------------------------------------------

def _narrative_summary(data: dict) -> str:
    """Generate a 3–4 sentence natural-language interpretation for the plasmid report."""
    total         = data.get("total", 0)
    num_plasmids  = data.get("num_plasmids", 0)
    total_args    = data.get("total_args", 0)
    plasmid_rows: list[PlasmidRow] = data.get("plasmid_rows", [])

    if total == 0:
        return "No sequences were provided for analysis."

    pct = f"{100 * num_plasmids / total:.1f}" if total else "0"

    # Resistance summary
    arg_contigs = [r for r in plasmid_rows if r.num_args > 0]
    all_drugs: list[str] = []
    for r in arg_contigs:
        all_drugs.extend(d.strip() for d in r.drug_classes.split(";") if d.strip() and d != "—")
    from collections import Counter
    drug_counts = Counter(all_drugs)
    top_drugs = [d for d, _ in drug_counts.most_common(3) if d]

    # High-risk conjugative + ARG
    high_risk = [r for r in plasmid_rows if r.mobility_class == "conjugative" and r.num_args > 0]

    # Circular
    circular = sum(1 for r in plasmid_rows if r.topology == "circular")

    sentences: list[str] = []

    # Sentence 1: overview
    sentences.append(
        f"PlasFlow v2 classified {num_plasmids:,} of {total:,} contigs ({pct}%) as plasmids."
    )

    # Sentence 2: resistance
    if total_args > 0 and arg_contigs:
        drug_str = ", ".join(top_drugs[:2]) if top_drugs else "multiple drug classes"
        sentences.append(
            f"{len(arg_contigs)} plasmid contig{'s' if len(arg_contigs) != 1 else ''} "
            f"carr{'y' if len(arg_contigs) != 1 else 'ies'} antimicrobial resistance genes "
            f"({total_args} total ARGs; resistance to {drug_str} detected)."
        )
    else:
        sentences.append("No antimicrobial resistance genes were detected on plasmid contigs.")

    # Sentence 3: high-risk
    if high_risk:
        sentences.append(
            f"<strong>{len(high_risk)} conjugative plasmid{'s' if len(high_risk) != 1 else ''}"
            f"</strong> co-carr{'y' if len(high_risk) != 1 else 'ies'} resistance genes — "
            f"these represent the highest horizontal transfer risk."
        )

    # Sentence 4: topology / circularity
    if circular > 0:
        sentences.append(
            f"{circular} plasmid contig{'s' if circular != 1 else ''} "
            f"{'show' if circular != 1 else 'shows'} circular topology (direct terminal repeats detected)."
        )

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Plasmid page
# ---------------------------------------------------------------------------


def _render_genome_map_js(genome_maps: dict) -> str:
    """Return JS snippet that renders all per-plasmid genome map charts."""
    if not genome_maps:
        return ""
    lines = []
    for i, (cid, fig) in enumerate(genome_maps.items()):
        div_id = f"gmap_{i}"
        lines.append(
            f"P.newPlot('{div_id}',{json.dumps(fig['data'])},{json.dumps(fig['layout'])},dm);"
        )
    return "\n".join(lines)


def _p_row(r: PlasmidRow) -> list:
    """Compact array for the vanilla JS paginator."""
    return [
        r.contig_id, r.contig_length, round(r.confidence, 4),
        r.num_args, r.arg_genes, r.drug_classes, r.arg_sources,
        r.num_vf, r.vf_genes, r.num_mge, r.mge_genes, r.mge_families,
        r.eskape_genus if r.eskape_host else "",
        r.mobility_class, r.replicon_type,
        r.risk_score, r.taxonomy, r.risk_evidence,
        r.topology, r.low_confidence,
    ]


_PLASMID_COL_HEADERS = [
    "Contig", "Length (bp)", "Conf.",
    "ARGs", "ARG Names", "Drug Classes", "DB",
    "VFs", "VF Genes", "MGEs", "MGE Elements", "MGE Families",
    "Pathogen", "Mobility", "Replicon", "Risk", "Taxonomy", "Evidence",
    "Topology", "Low Conf.",
]

_PLASMID_DOWNLOAD_HEADERS = [
    "contig_id", "length_bp", "confidence",
    "num_args", "arg_genes", "drug_classes", "db_source",
    "num_vf", "vf_genes", "num_mge", "mge_genes", "mge_families",
    "pathogen_host", "mobility_class", "replicon_type", "risk_score", "taxonomy", "risk_evidence",
    "topology", "low_confidence",
]


def _render_plasmid_page(data: dict) -> str:
    all_rows: list[PlasmidRow] = data["plasmid_rows"]
    # Exclude risk score = 0 from HTML display
    rows = [r for r in all_rows if r.risk_score > 0]
    counts = data["class_counts"]
    n_total = len(all_rows)
    n_shown = len(rows)
    n_zero  = n_total - n_shown
    display = rows[:MAX_TABLE_ROWS]
    truncated = n_shown > MAX_TABLE_ROWS

    stat_cards = (
        f'<div class="card" style="border-left:4px solid #2c6fad">'
        f'<h3>Plasmids (risk≥1)</h3><p style="color:#2c6fad">{n_shown:,}</p></div>'
        f'<div class="card" style="border-left:4px solid #95a5a6">'
        f'<h3>Risk = 0 (hidden)</h3><p style="color:#95a5a6">{n_zero:,}</p></div>'
        f'<div class="card" style="border-left:4px solid #c0392b">'
        f'<h3>ARGs</h3><p style="color:#c0392b">{data["total_args"]:,}</p></div>'
        f'<div class="card" style="border-left:4px solid #f59e0b">'
        f'<h3>VF Genes</h3><p style="color:#92400e">{data["total_vf"]:,}</p></div>'
        f'<div class="card" style="border-left:4px solid #8b5cf6">'
        f'<h3>MGEs</h3><p style="color:#5b21b6">{data["total_mge"]:,}</p></div>'
        f'<div class="card" style="border-left:4px solid #e74c3c">'
        f'<h3>Pathogenic Contigs</h3><p style="color:#c0392b">{data.get("total_pathogens",0):,}</p></div>'
    )

    note = ""
    if truncated:
        note = (f'<p class="note">Showing top {MAX_TABLE_ROWS:,} of {n_shown:,} '
                f'risk≥1 plasmid contigs. Full data in predictions.tsv.</p>')
    if n_zero:
        note += f'<p class="note">{n_zero:,} contigs with risk score = 0 are hidden. Full list in predictions.tsv.</p>'

    row_data_json = json.dumps([_p_row(r) for r in display])
    headers_json  = json.dumps(_PLASMID_DOWNLOAD_HEADERS)

    th_row = '<th class="no-sort"><input type="checkbox" id="chk-all" title="Select all on page"></th>'
    th_row += "".join(f"<th>{h}</th>" for h in _PLASMID_COL_HEADERS)

    # Narrative summary block
    narrative_html = (
        f'<div class="narrative">{data.get("narrative", "")}</div>'
        if data.get("narrative") else ""
    )

    # Genome map section — link to separate page instead of embedding
    genome_maps: dict = data.get("genome_maps", {})
    if genome_maps:
        genome_section = (
            f'<div style="border:1px solid #d5e8f5;border-radius:6px;padding:12px 16px;'
            f'margin-bottom:20px;background:#f0f7ff">'
            f'<h2 style="margin:0 0 6px">Plasmid Genome Maps</h2>'
            f'<p style="margin:0;color:#555;font-size:.9rem">'
            f'{len(genome_maps):,} contigs with ≥3 genes or risk &gt; 4 have genome maps. '
            f'<a href="report_genome_maps.html" style="color:#2c6fad;font-weight:bold">'
            f'Open genome maps →</a>'
            f'</p></div>'
        )
    else:
        genome_section = ""

    high_risk_html  = data.get("high_risk_table", "")
    pathogen_sum_html = data.get("pathogen_table", "")

    body = f"""
<h1>PlasFlow v2 — Plasmid Report</h1>
<p class="meta">Input: <code>{data["input_file"]}</code></p>
{narrative_html}
<div class="cards">{stat_cards}</div>
{high_risk_html}
{pathogen_sum_html}

<h2>Overview</h2>
<div class="chart-grid g3">
  <div id="cpie"  class="cbox"></div>
  <div id="carg"  class="cbox"></div>
  <div id="crisk" class="cbox"></div>
</div>

<h2>Virulence Factors &amp; Mobile Genetic Elements</h2>
<div class="chart-grid g2">
  <div id="cvf"  class="cbox"></div>
  <div id="cmge" class="cbox"></div>
</div>

<h2>Mobility &amp; Pathogen Hosts</h2>
<div class="chart-grid g2">
  <div id="cmob"  class="cbox"></div>
  <div id="cesk"  class="cbox"></div>
</div>

{genome_section}
<h2>Plasmid Predictions — risk ≥ 1 ({n_shown:,} contigs)</h2>
<div class="filter-bar">
  <button class="fbtn active" id="fa" onclick="setRisk('')"  style="background:#ddd;color:#333">All</button>
  <button class="fbtn" id="fh" onclick="setRisk('h')" style="background:#c0392b;color:#fff">High ≥7</button>
  <button class="fbtn" id="fm" onclick="setRisk('m')" style="background:#e67e22;color:#fff">Medium 4–6</button>
  <button class="fbtn" id="fl" onclick="setRisk('l')" style="background:#27ae60;color:#fff">Low 1–3</button>
  <span style="margin-left:8px">
    <button class="dl-btn" style="background:#2c6fad" onclick="downloadSel()">⬇ Download Selected</button>
    <button class="dl-btn" style="background:#555;margin-left:4px" onclick="downloadFiltered()">⬇ Download All Filtered</button>
  </span>
  <span class="sel-count" id="sel-count"></span>
</div>
{note}
<div class="tbl-wrap">
  <div class="tbl-ctrl">
    <input type="text" id="psearch" placeholder="Search…">
    <span class="pg-info" id="ppg"></span>
    <button class="pg-btn" id="pprev">◀ Prev</button>
    <button class="pg-btn" id="pnext">Next ▶</button>
  </div>
  <table id="ptable">
    <thead><tr>{th_row}</tr></thead>
    <tbody></tbody>
  </table>
</div>"""

    js = f"""
(function(){{
var P=window.Plotly,dm={{responsive:true,displayModeBar:false}};
P.newPlot('cpie', {json.dumps(data['pie_data']['data'])},{json.dumps(data['pie_data']['layout'])},dm);
P.newPlot('carg', {json.dumps(data['arg_data']['data'])},{json.dumps(data['arg_data']['layout'])},dm);
P.newPlot('crisk',{json.dumps(data['risk_data']['data'])},{json.dumps(data['risk_data']['layout'])},dm);
P.newPlot('cvf',  {json.dumps(data['vf_data']['data'])}, {json.dumps(data['vf_data']['layout'])}, dm);
P.newPlot('cmge', {json.dumps(data['mge_data']['data'])},{json.dumps(data['mge_data']['layout'])},dm);
P.newPlot('cmob', {json.dumps(data['mobility_data']['data'])},{json.dumps(data['mobility_data']['layout'])},dm);
P.newPlot('cesk', {json.dumps(data['eskape_data']['data'])},{json.dumps(data['eskape_data']['layout'])},dm);

var ALL={row_data_json};
var HEADERS={headers_json};
var cur=ALL.slice();
var SEL=new Set(); // selected contig IDs

window.tblCheckChange=function(cb){{
  var id=cb.getAttribute('data-id');
  if(cb.checked)SEL.add(id);else SEL.delete(id);
  document.getElementById('sel-count').textContent=SEL.size>0?SEL.size+' selected':'';
}};
document.getElementById('chk-all').addEventListener('change',function(){{
  var chk=this.checked;
  document.querySelectorAll('#ptable tbody input[type=checkbox]').forEach(function(c){{
    c.checked=chk;
    var id=c.getAttribute('data-id');
    if(chk)SEL.add(id);else SEL.delete(id);
  }});
  document.getElementById('sel-count').textContent=SEL.size>0?SEL.size+' selected':'';
}});

function riskCls(v){{return v>=7?'h':v>=4?'m':'l';}}
function renderR(v){{var c=riskCls(v);return '<span class="risk-'+c+'">'+v+'</span>';}}
function srcBadges(s){{if(!s)return'—';
  return s.split(',').map(function(x){{x=x.trim();
    return x?'<span class="badge b'+x.toLowerCase()+'">'+esc(x)+'</span>':'';}}).join(' ')}}
function eskBadge(e){{if(!e)return'—';
  var ek=['Enterococcus','Staphylococcus','Klebsiella','Acinetobacter','Pseudomonas','Enterobacter','Escherichia'];
  return'<span class="badge '+(ek.indexOf(e)>=0?'besk':'bwho')+'">'+esc(e)+'</span>';}}

var COLS=[
  {{render:function(v){{return ellipsis(v,28);}}}},           // 0 contig_id
  {{render:function(v){{return Number(v).toLocaleString();}}}}, // 1 length
  {{render:function(v){{return v;}}}},                        // 2 confidence
  {{render:function(v){{return v;}}}},                        // 3 num_args
  {{render:function(v){{return ellipsis(v,36);}}}},           // 4 arg_genes
  {{render:function(v){{return ellipsis(v,32);}}}},           // 5 drug_classes
  {{render:function(v){{return srcBadges(v);}}}},             // 6 arg_sources (DB)
  {{render:function(v){{return v>0?'<span class="badge bvf">'+v+' VF</span>':'—';}}}}, // 7 num_vf
  {{render:function(v){{return ellipsis(v,28);}}}},           // 8 vf_genes
  {{render:function(v){{return v>0?'<span class="badge bmge">'+v+' MGE</span>':'—';}}}}, // 9 num_mge
  {{render:function(v){{return ellipsis(v,28);}}}},           // 10 mge_genes
  {{render:function(v){{return ellipsis(v,22);}}}},           // 11 mge_families
  {{render:function(v){{return eskBadge(v);}}}},              // 12 pathogen
  {{render:function(v){{return esc(v);}}}},                   // 13 mobility_class
  {{render:function(v){{return esc(v);}}}},                   // 14 replicon_type
  {{render:function(v){{return renderR(v);}}}},               // 15 risk_score
  {{render:function(v){{return ellipsis(v,25);}}}},           // 16 taxonomy
  {{render:function(v){{return ellipsis(v,32);}}}},           // 17 risk_evidence
  {{render:function(v){{                                     // 18 topology
    if(v==='circular')return'<span class="badge bcirc">⭕ circular</span>';
    if(v==='too_short')return'<span style="color:#95a5a6">too short</span>';
    return'<span style="color:#7f8c8d">— linear</span>';
  }}}},
  {{render:function(v){{                                     // 19 low_confidence
    return v?'<span class="badge blowconf">⚠ low conf</span>':'<span style="color:#27ae60">✓</span>';
  }}}},
];

var tbl=new LightTable({{tableId:'ptable',data:ALL,cols:COLS,pageSize:50,
  searchId:'psearch',pgInfoId:'ppg',prevId:'pprev',nextId:'pnext',
  onCheck:true,selected:SEL,selCountId:'sel-count'}});

function setRisk(r){{
  ['fa','fh','fm','fl'].forEach(function(id){{document.getElementById(id).classList.remove('active');}});
  document.getElementById(r===''?'fa':r==='h'?'fh':r==='m'?'fm':'fl').classList.add('active');
  cur=r===''?ALL.slice():ALL.filter(function(row){{return riskCls(row[15])===r;}});
  tbl.data=cur;tbl.filtered=cur.slice();tbl.page=0;tbl.applySort();tbl.render();
  document.getElementById('psearch').value='';
}}
window.setRisk=setRisk;

function toTSV(rows){{
  var lines=[HEADERS.join('\\t')];
  rows.forEach(function(r){{
    lines.push(r.map(function(v){{return '"'+String(v).replace(/"/g,'""')+'"';}}).join('\\t'));
  }});
  return lines.join('\\n');
}}
function triggerDownload(tsv,fname){{
  var blob=new Blob([tsv],{{type:'text/tab-separated-values'}});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');a.href=url;a.download=fname;a.click();
  URL.revokeObjectURL(url);
}}
window.downloadSel=function(){{
  if(SEL.size===0){{alert('No rows selected. Use the checkboxes to select rows.');return;}}
  var rows=ALL.filter(function(r){{return SEL.has(r[0]);}});
  triggerDownload(toTSV(rows),'plasflow_selected_'+SEL.size+'.tsv');
}};
window.downloadFiltered=function(){{
  triggerDownload(toTSV(tbl.filtered),'plasflow_filtered_'+tbl.filtered.length+'.tsv');
}};
}})();"""

    return _full_page("Plasmid", "plasmid", counts, "#2c6fad", body, js=js)


# ---------------------------------------------------------------------------
# Non-plasmid page renderer
# ---------------------------------------------------------------------------


def _np_row(r: NonPlasmidRow, show_best: bool) -> list:
    row = [
        r.contig_id, r.contig_length, round(r.confidence, 4),
        r.taxonomy, r.taxonomy_lineage,
        r.num_args, r.arg_genes, r.drug_classes,
        r.num_vf, r.vf_genes,
        r.num_mge, r.mge_genes, r.mge_families,
    ]
    if show_best:
        row += [r.best_label, round(r.best_score, 4)]
    return row


def _build_high_risk_table(
    plasmid_rows: list["PlasmidRow"],
    pathogens: dict,
) -> str:
    """Return an HTML table of plasmids that are mobile AND carry ARGs AND
    match a pathogenic taxonomy — the three-signal intersection.

    Returns an empty string when no such plasmids exist.
    """
    hits = [
        r for r in plasmid_rows
        if r.mobility_class in ("conjugative", "mobilizable")
        and r.num_args > 0
        and (r.eskape_host or r.contig_id in pathogens)
    ]
    if not hits:
        return ""

    hits.sort(key=lambda r: (-r.risk_score, -r.num_args))

    def _mob_badge(m: str) -> str:
        color = "#c0392b" if m == "conjugative" else "#e67e22"
        return f'<span style="background:{color};color:#fff;padding:1px 6px;border-radius:3px;font-size:.78rem">{m}</span>'

    def _threat_badge(cid: str, r: "PlasmidRow") -> str:
        pr = pathogens.get(cid)
        if pr:
            colors = {"critical": "#c0392b", "high": "#e67e22", "medium": "#f39c12"}
            c = colors.get(pr.threat_level, "#888")
            label = pr.species or pr.genus
            return f'<span style="background:{c};color:#fff;padding:1px 6px;border-radius:3px;font-size:.78rem">{label}</span>'
        if r.eskape_host and r.eskape_genus:
            return f'<span style="background:#c0392b;color:#fff;padding:1px 6px;border-radius:3px;font-size:.78rem">{r.eskape_genus}</span>'
        return "—"

    rows_html = "".join(
        f"<tr>"
        f"<td style='font-family:monospace;font-size:.8rem'>{r.contig_id}</td>"
        f"<td>{r.contig_length:,}</td>"
        f"<td>{_mob_badge(r.mobility_class)}</td>"
        f"<td>{r.replicon_type if r.replicon_type not in ('-','unknown','') else '—'}</td>"
        f"<td style='color:#c0392b'><strong>{r.num_args}</strong></td>"
        f"<td style='font-size:.8rem'>{r.arg_genes or '—'}</td>"
        f"<td style='font-size:.8rem'>{r.drug_classes or '—'}</td>"
        f"<td>{_threat_badge(r.contig_id, r)}</td>"
        f"<td><strong style='color:{'#c0392b' if r.risk_score>=7 else '#e67e22' if r.risk_score>=4 else '#27ae60'}'>{r.risk_score}</strong></td>"
        f"</tr>"
        for r in hits
    )

    return f"""
<div style="border:2px solid #c0392b;border-radius:6px;padding:12px 16px;margin-bottom:20px;background:#fff5f5">
<h2 style="color:#c0392b;margin-top:0">&#9888; Priority Alert — {len(hits)} High-Risk Plasmid{'s' if len(hits)!=1 else ''}</h2>
<p style="margin:0 0 10px;color:#555;font-size:.9rem">
  Plasmids that are <strong>mobile</strong> (conjugative or mobilizable),
  carry <strong>resistance genes</strong>, and match a <strong>pathogenic host</strong> taxonomy.
  These represent the highest-priority AMR dissemination risk.
</p>
<div style="overflow-x:auto">
<table style="width:100%;border-collapse:collapse;font-size:.85rem">
<thead><tr style="background:#fde8e8;text-align:left">
  <th style="padding:6px 8px">Contig</th>
  <th style="padding:6px 8px">Length</th>
  <th style="padding:6px 8px">Mobility</th>
  <th style="padding:6px 8px">Replicon</th>
  <th style="padding:6px 8px">ARGs</th>
  <th style="padding:6px 8px">ARG Genes</th>
  <th style="padding:6px 8px">Drug Classes</th>
  <th style="padding:6px 8px">Pathogen Host</th>
  <th style="padding:6px 8px">Risk</th>
</tr></thead>
<tbody style="border-top:1px solid #e0c0c0">
{rows_html}
</tbody>
</table>
</div>
</div>"""


def _build_pathogen_table(
    plasmid_rows: list["PlasmidRow"],
    pathogens: dict,
) -> str:
    """Return an HTML summary table of pathogenic plasmid contigs by threat level."""
    # Pull only plasmid contigs that are in the pathogens dict
    plas_ids = {r.contig_id for r in plasmid_rows}
    plas_pathogens = {cid: pr for cid, pr in pathogens.items() if cid in plas_ids}
    if not plas_pathogens:
        return ""

    by_level: dict[str, list] = {"critical": [], "high": [], "medium": []}
    for cid, pr in plas_pathogens.items():
        by_level.setdefault(pr.threat_level, []).append(pr)

    rows_html = ""
    for level in ("critical", "high", "medium"):
        prs = by_level.get(level, [])
        if not prs:
            continue
        colors = {"critical": "#c0392b", "high": "#e67e22", "medium": "#f39c12"}
        c = colors[level]
        # Group by species
        from collections import Counter
        species_counts = Counter(pr.species or pr.genus for pr in prs)
        for sp, cnt in species_counts.most_common():
            rows_html += (
                f"<tr>"
                f"<td><span style='background:{c};color:#fff;padding:1px 6px;border-radius:3px;font-size:.78rem'>{level}</span></td>"
                f"<td style='font-style:italic'>{sp}</td>"
                f"<td style='text-align:center'><strong>{cnt}</strong></td>"
                f"</tr>"
            )

    if not rows_html:
        return ""

    total = len(plas_pathogens)
    return f"""
<div style="border:1px solid #e0c0c0;border-radius:6px;padding:12px 16px;margin-bottom:20px;background:#fffaf9">
<h2 style="margin-top:0">Pathogenic Host Summary — {total} plasmid contig{'s' if total!=1 else ''}</h2>
<p style="margin:0 0 10px;color:#555;font-size:.9rem">
  Plasmid contigs whose taxonomy matches known pathogenic species (WHO BPPL 2024 / ESKAPE / CDC AR Threat Report).
</p>
<table style="border-collapse:collapse;font-size:.85rem;min-width:360px">
<thead><tr style="background:#f5ece8;text-align:left">
  <th style="padding:6px 10px">Threat</th>
  <th style="padding:6px 10px">Species</th>
  <th style="padding:6px 10px">Contigs</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""


def _render_genome_maps_page(genome_maps: dict, input_file: str = "") -> str:
    """Standalone HTML page containing all filtered plasmid genome map charts."""
    if not genome_maps:
        body = "<p style='color:#888;margin:40px'>No genome maps generated (no contigs with ≥3 genes or risk score &gt; 4).</p>"
        js_block = ""
    else:
        divs = "\n".join(
            f'<div style="margin-bottom:24px">'
            f'<p style="font-family:monospace;font-size:.85rem;color:#555;margin:0 0 4px">{cid}</p>'
            f'<div id="gmap_{i}" style="height:160px"></div>'
            f'</div>'
            for i, cid in enumerate(genome_maps)
        )
        body = f"""
<h1 style="font-size:1.3rem;color:#2c3e50">Plasmid Genome Maps</h1>
<p style="color:#555;font-size:.9rem;margin-bottom:4px">
  Input: <code>{input_file}</code> &nbsp;·&nbsp;
  {len(genome_maps):,} contigs shown (≥3 genes or risk &gt; 4) &nbsp;·&nbsp;
  <a href="report_plasmid.html">← Back to plasmid report</a>
</p>
<p style="font-size:.82rem;color:#888;margin-top:0">
  Colours: <span style="color:#e74c3c">■ ARG</span>
  <span style="color:#e67e22">■ Virulence</span>
  <span style="color:#8e44ad">■ MGE/IS</span>
  <span style="color:#2980b9">■ Mobility</span>
  <span style="color:#bdc3c7">■ Other</span>
</p>
<div style="max-width:1100px">{divs}</div>"""

        plot_calls = "\n".join(
            f"P.newPlot('gmap_{i}',{json.dumps(fig['data'])},{json.dumps(fig['layout'])},dm);"
            for i, (cid, fig) in enumerate(genome_maps.items())
        )
        js_block = f"""
<script>
(function(){{
var P=window.Plotly,dm={{responsive:true,displayModeBar:false}};
{plot_calls}
}})();
</script>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PlasFlow v2 — Genome Maps</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     margin:24px 32px;background:#fafafa;color:#2c3e50}}
code{{background:#eee;padding:1px 4px;border-radius:3px;font-size:.85rem}}
a{{color:#2c6fad}}
</style>
</head>
<body>
{body}
{js_block}
</body>
</html>"""


def _render_nonplasmid_page(
    data: dict,
    class_key: str,
    title: str,
    color: str,
    rows: list[NonPlasmidRow],
    table_id: str,
    chart_data: dict,
    show_best: bool = False,
    extra_note: str = "",
) -> str:
    n_total = len(rows)
    display  = rows[:MAX_TABLE_ROWS]
    truncated = n_total > MAX_TABLE_ROWS
    counts = data["class_counts"]

    stat_cards = (
        f'<div class="card" style="border-left:4px solid {color}">'
        f'<h3>{title} Contigs</h3><p style="color:{color}">{n_total:,}</p></div>'
        f'<div class="card" style="border-left:4px solid #2c6fad">'
        f'<h3>Total Sequences</h3><p style="color:#2c6fad">{data["total"]:,}</p></div>'
    )

    notes = []
    if truncated:
        notes.append(f'Showing top {MAX_TABLE_ROWS:,} of {n_total:,} contigs (by length). Full data in predictions.tsv.')
    if extra_note:
        notes.append(extra_note)
    note_html = "".join(f'<p class="note">{n}</p>' for n in notes)

    # Charts — 3 charts for non-plasmid pages; add pathogen bar on chromosome page
    ch = chart_data
    pathogen_data = data.get("pathogen_data")
    show_pathogen = class_key == "chromosome" and pathogen_data
    if show_pathogen:
        chart_html = f"""
<h2>Summary Charts</h2>
<div class="chart-grid g3">
  <div id="{table_id}_c1" class="cbox"></div>
  <div id="{table_id}_c2" class="cbox"></div>
  <div id="{table_id}_c3" class="cbox"></div>
</div>
<h2>Pathogenic Bacteria Detected</h2>
<p class="note">Contigs from known pathogenic species (WHO/ESKAPE/CDC priority lists). Requires --taxonomy-db.</p>
<div class="chart-grid g2">
  <div id="{table_id}_cpat" class="cbox" style="min-height:320px"></div>
  <div class="card" style="padding:16px;font-size:.84rem">
    <h3 style="margin-bottom:8px">Threat Levels</h3>
    <p><span class="badge besk">Critical</span> WHO critical / ESKAPE pathogens — pandrug-resistant, highest clinical risk</p>
    <br><p><span class="badge bwho">High</span> WHO high-priority — MDR, significant public health threat</p>
    <br><p><span class="badge bcard">Medium</span> WHO medium / regionally important — monitor for resistance</p>
    <br><p style="margin-top:12px;color:#888;font-size:.78rem">Pathogen DB covers 50+ pathogenic genera from WHO BPPL 2024, ESKAPE, and CDC AR Threat Report.</p>
  </div>
</div>"""
    else:
        chart_html = f"""
<h2>Summary Charts</h2>
<div class="chart-grid g3">
  <div id="{table_id}_c1" class="cbox"></div>
  <div id="{table_id}_c2" class="cbox"></div>
  <div id="{table_id}_c3" class="cbox"></div>
</div>"""

    # Table columns
    # _np_row layout: [contig_id(0), length(1), conf(2), taxonomy(3), lineage(4),
    #                  num_args(5), arg_genes(6), drug_classes(7),
    #                  num_vf(8), vf_genes(9), num_mge(10), mge_genes(11), mge_families(12),
    #                  best_label(13)*, best_score(14)*]
    base_th = (
        "<th>Contig</th><th>Length (bp)</th><th>Confidence</th><th>Taxonomy (LCA)</th>"
        "<th>ARGs</th><th>ARG Names</th><th>Drug Classes</th>"
        "<th>VFs</th><th>VF Genes</th>"
        "<th>MGEs</th><th>MGE Elements</th><th>MGE Families</th>"
    )
    if show_best:
        base_th += "<th>Best Label</th><th>Best Score</th>"

    _arg_cols_js = """\
  {render:function(v){return v>0?'<span class="badge bsarg">'+v+' ARG</span>':'—';}},
  {render:function(v){return ellipsis(v,36);}},
  {render:function(v){return ellipsis(v,32);}},
  {render:function(v){return v>0?'<span class="badge bvf">'+v+' VF</span>':'—';}},
  {render:function(v){return ellipsis(v,28);}},
  {render:function(v){return v>0?'<span class="badge bmge">'+v+' MGE</span>':'—';}},
  {render:function(v){return ellipsis(v,28);}},
  {render:function(v){return ellipsis(v,22);}},"""

    if show_best:
        col_js = """[
  {render:function(v){return ellipsis(v,38);}},
  {render:function(v){return Number(v).toLocaleString();}},
  {render:function(v){return v;}},
  {render:function(v,r){return '<span class="ellipsis" title="'+esc(r[4])+'">'+esc(v)+'</span>';}},
""" + _arg_cols_js + """
  {render:function(v){return esc(v);}},
  {render:function(v){return v;}}
]"""
    else:
        col_js = """[
  {render:function(v){return ellipsis(v,38);}},
  {render:function(v){return Number(v).toLocaleString();}},
  {render:function(v){return v;}},
  {render:function(v,r){return '<span class="ellipsis" title="'+esc(r[4])+'">'+esc(v)+'</span>';}},
""" + _arg_cols_js + """
]"""

    row_json = json.dumps([_np_row(r, show_best) for r in display])

    pathogen_js = ""
    if show_pathogen and pathogen_data:
        pathogen_js = (
            f"P.newPlot('{table_id}_cpat',"
            f"{json.dumps(pathogen_data['data'])},"
            f"{json.dumps(pathogen_data['layout'])},dm);"
        )

    js = f"""
(function(){{
var P=window.Plotly,dm={{responsive:true,displayModeBar:false}};
P.newPlot('{table_id}_c1',{json.dumps(ch['c1']['data'])},{json.dumps(ch['c1']['layout'])},dm);
P.newPlot('{table_id}_c2',{json.dumps(ch['c2']['data'])},{json.dumps(ch['c2']['layout'])},dm);
P.newPlot('{table_id}_c3',{json.dumps(ch['c3']['data'])},{json.dumps(ch['c3']['layout'])},dm);
{pathogen_js}
var DATA={row_json};
new LightTable({{tableId:'{table_id}',data:DATA,cols:{col_js},pageSize:50,
  searchId:'{table_id}_s',pgInfoId:'{table_id}_pg',
  prevId:'{table_id}_prev',nextId:'{table_id}_next'}});
}})();"""

    body = f"""
<h1>PlasFlow v2 — {title} Report</h1>
<p class="meta">Input: <code>{data["input_file"]}</code></p>
<div class="cards">{stat_cards}</div>
{chart_html}
<h2>{title} Contigs ({n_total:,})</h2>
{note_html}
<div class="tbl-wrap">
  <div class="tbl-ctrl">
    <input type="text" id="{table_id}_s" placeholder="Search contigs…">
    <span class="pg-info" id="{table_id}_pg"></span>
    <button class="pg-btn" id="{table_id}_prev">◀ Prev</button>
    <button class="pg-btn" id="{table_id}_next">Next ▶</button>
  </div>
  <table id="{table_id}">
    <thead><tr>{base_th}</tr></thead>
    <tbody></tbody>
  </table>
</div>"""

    return _full_page(title, class_key, counts, color, body, js=js)


# ---------------------------------------------------------------------------
# Main data builder
# ---------------------------------------------------------------------------


def _np_charts(rows: list, title: str, color: str, show_best: bool = False) -> dict:
    """Build the 3-chart bundle for a non-plasmid class page (length, confidence, taxonomy/best-label)."""
    c3 = _best_label_bar(rows) if show_best else _taxonomy_bar_nonplasmid(rows, title, color)
    return {
        "c1": _length_hist(rows, title, color),
        "c2": _confidence_hist(rows, title, color),
        "c3": c3,
    }


def build_report_data(pipeline_result, input_file: str = "") -> dict:  # noqa: C901
    all_arg_hits = [h for cr in pipeline_result.plasmid_results for h in cr.arg_hits]
    taxonomy = getattr(pipeline_result, "taxonomy", {}) or {}
    topology_map = getattr(pipeline_result, "topology", {}) or {}

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
        vf_hits  = getattr(cr, "vf_hits",  [])
        mge_hits = getattr(cr, "mge_hits", [])
        sources  = sorted({h.source for h in cr.arg_hits if getattr(h, "source", "")})
        arg_genes_str  = "; ".join(sorted({h.gene_name for h in cr.arg_hits})) if cr.arg_hits else ""
        mge_genes_str  = "; ".join(sorted({h.is_name   for h in mge_hits}))    if mge_hits   else ""
        mge_fams_str   = "; ".join(sorted({h.is_family for h in mge_hits}))    if mge_hits   else ""
        cid = cr.record.id
        plasmid_rows.append(PlasmidRow(
            contig_id=cid,
            contig_length=len(cr.record.seq),
            confidence=cr.prediction.confidence,
            num_args=len(cr.arg_hits),
            arg_genes=arg_genes_str,
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
            mge_genes=mge_genes_str,
            mge_families=mge_fams_str,
            topology=topology_map.get(cid, "linear"),
            low_confidence=cr.prediction.confidence < 0.70,
        ))

    non_plasmid_results = getattr(pipeline_result, "non_plasmid_results", [])
    phage_rows:         list[NonPlasmidRow] = []
    chromosome_rows:    list[NonPlasmidRow] = []
    archaea_rows:       list[NonPlasmidRow] = []
    unclassified_rows:  list[NonPlasmidRow] = []

    for npr in non_plasmid_results:
        tax = getattr(npr, "taxonomy", None) or taxonomy.get(npr.record.id)
        scores = getattr(npr.prediction, "scores", {}) or {}
        best_label = max(scores, key=scores.get) if scores else ""
        best_score = scores.get(best_label, 0.0) if best_label else 0.0

        np_arg_hits  = getattr(npr, "arg_hits",  [])
        np_vf_hits   = getattr(npr, "vf_hits",   [])
        np_mge_hits  = getattr(npr, "mge_hits",  [])

        np_arg_genes = "; ".join(sorted({h.gene_name for h in np_arg_hits})) if np_arg_hits else ""
        np_drug_classes = sorted({
            dc.strip()
            for h in np_arg_hits
            for dc in h.drug_class.split(";")
            if dc.strip() and dc.strip() != "unknown"
        })
        np_sources = sorted({h.source for h in np_arg_hits if getattr(h, "source", "")})
        np_vf_genes  = "; ".join(sorted({h.gene_name for h in np_vf_hits}))  if np_vf_hits  else ""
        np_mge_genes = "; ".join(sorted({h.is_name   for h in np_mge_hits})) if np_mge_hits else ""
        np_mge_fams  = "; ".join(sorted({h.is_family for h in np_mge_hits})) if np_mge_hits else ""

        np_cid = npr.record.id
        row = NonPlasmidRow(
            contig_id=np_cid,
            contig_length=len(npr.record.seq),
            label=npr.prediction.label,
            confidence=npr.prediction.confidence,
            taxonomy=tax.display if tax else "—",
            taxonomy_lineage=tax.lineage if tax else "—",
            best_label=best_label,
            best_score=best_score,
            num_args=len(np_arg_hits),
            arg_genes=np_arg_genes,
            drug_classes="; ".join(np_drug_classes),
            arg_sources=", ".join(np_sources),
            num_vf=len(np_vf_hits),
            vf_genes=np_vf_genes,
            num_mge=len(np_mge_hits),
            mge_genes=np_mge_genes,
            mge_families=np_mge_fams,
            topology=topology_map.get(np_cid, "linear"),
            low_confidence=npr.prediction.confidence < 0.70,
        )
        lbl = npr.prediction.label
        if lbl == "phage":        phage_rows.append(row)
        elif lbl == "chromosome": chromosome_rows.append(row)
        elif lbl == "archaea":    archaea_rows.append(row)
        else:                     unclassified_rows.append(row)

    for lst in (chromosome_rows, unclassified_rows):
        lst.sort(key=lambda r: r.contig_length, reverse=True)

    total_vf  = sum(r.num_vf  for r in plasmid_rows)
    total_mge = sum(r.num_mge for r in plasmid_rows)
    risk_scores = [cr.risk.score for cr in pipeline_result.plasmid_results if cr.risk.score > 0]
    tax_classified = sum(1 for r in taxonomy.values() if r.rank != "unclassified")

    # Pathogen detection results (populated when taxonomy DB was used)
    pathogens = getattr(pipeline_result, "pathogens", {}) or {}

    # Genome maps — per-plasmid contig (only when ORF data is available)
    genome_maps: dict[str, dict] = {}
    all_orfs = getattr(pipeline_result, "orfs", []) or []
    if all_orfs:
        from collections import defaultdict as _dd
        orfs_by_contig: dict = _dd(list)
        for orf in all_orfs:
            orfs_by_contig[orf.contig_id].append(orf)

        # Build lookup sets/dicts for ARG/VF/MGE by orf_id
        arg_orf_ids:    set[str] = {h._orf_id for cr in pipeline_result.plasmid_results for h in cr.arg_hits if h._orf_id}
        vf_orf_ids:     set[str] = {h._orf_id for cr in pipeline_result.plasmid_results for h in cr.vf_hits  if getattr(h, "_orf_id", "")}
        mge_orf_ids:    set[str] = {h._orf_id for cr in pipeline_result.plasmid_results for h in cr.mge_hits if getattr(h, "_orf_id", "")}
        arg_name_by_orf: dict[str, str] = {h._orf_id: h.gene_name for cr in pipeline_result.plasmid_results for h in cr.arg_hits if h._orf_id}
        vf_name_by_orf:  dict[str, str] = {h._orf_id: h.gene_name for cr in pipeline_result.plasmid_results for h in cr.vf_hits  if getattr(h, "_orf_id", "")}
        mge_name_by_orf: dict[str, str] = {h._orf_id: h.is_name   for cr in pipeline_result.plasmid_results for h in cr.mge_hits if getattr(h, "_orf_id", "")}

        # Build a risk_score lookup so we can apply the filter below
        risk_by_contig = {cr.record.id: cr.risk.score for cr in pipeline_result.plasmid_results}

        for cr in pipeline_result.plasmid_results:
            cid = cr.record.id
            contig_orfs = orfs_by_contig.get(cid, [])
            # Only generate maps for contigs with ≥3 ORFs or risk score > 4
            if contig_orfs and (len(contig_orfs) >= 3 or risk_by_contig.get(cid, 0) > 4):
                genome_maps[cid] = _genome_map_data(
                    contig_id=cid,
                    contig_length=len(cr.record.seq),
                    orfs=contig_orfs,
                    arg_orf_ids=arg_orf_ids,
                    vf_orf_ids=vf_orf_ids,
                    mge_orf_ids=mge_orf_ids,
                    arg_name_by_orf=arg_name_by_orf,
                    vf_name_by_orf=vf_name_by_orf,
                    mge_name_by_orf=mge_name_by_orf,
                )

    # Build the return dict first (narrative needs it)
    result_dict = {
        "input_file":        input_file or str(pipeline_result.input_fasta),
        "total":             pipeline_result.total_sequences,
        "num_plasmids":      pipeline_result.total_plasmids,
        "total_args":        pipeline_result.total_args,
        "total_vf":          total_vf,
        "total_mge":         total_mge,
        "tax_classified":    tax_classified,
        "total_pathogens":   len(pathogens),
        "class_counts":      pipeline_result.class_counts,
        # plasmid charts
        "pie_data":          _pie(pipeline_result.class_counts),
        "arg_data":          _arg_bar(all_arg_hits),
        "risk_data":         _risk_hist(risk_scores),
        "vf_data":           _vf_bar(plasmid_rows),
        "mge_data":          _mge_bar(plasmid_rows),
        "mobility_data":     _mobility_bar(plasmid_rows),
        "eskape_data":       _eskape_bar(plasmid_rows),
        "pathogen_data":     _pathogen_bar(pathogens),
        "cooccurrence_data": _build_drug_cooccurrence_heatmap(plasmid_rows),
        "scatter_data":      {}, "tax_bar_data": {},
        # row lists
        "plasmid_rows":      plasmid_rows,
        "chromosome_rows":   chromosome_rows,
        "phage_rows":        phage_rows,
        "archaea_rows":      archaea_rows,
        "unclassified_rows": unclassified_rows,
        "other_rows":        archaea_rows + unclassified_rows,
        # per-class chart bundles
        "chrom_charts":      _np_charts(chromosome_rows, "Chromosome", "#27ae60"),
        "phage_charts":      _np_charts(phage_rows,      "Phage",      "#e67e22"),
        "arch_charts":       _np_charts(archaea_rows,    "Archaea",    "#8e44ad"),
        "unc_charts":        _np_charts(unclassified_rows, "Unclassified", "#95a5a6", show_best=True),
        # legacy flags
        "has_scatter": False, "has_cooccurrence": bool(plasmid_rows),
        "has_phages":  bool(phage_rows), "has_chromosomes": bool(chromosome_rows),
        "has_others":  bool(archaea_rows or unclassified_rows),
        # genome maps (per-plasmid contig → Plotly figure dict)
        "genome_maps": genome_maps,
        # high-risk intersection: conjugative + ARGs + ESKAPE/pathogen
        "high_risk_table": _build_high_risk_table(plasmid_rows, pathogens),
        # pathogen summary: breakdown by threat level
        "pathogen_table": _build_pathogen_table(plasmid_rows, pathogens),
    }
    result_dict["narrative"] = _narrative_summary(result_dict)
    return result_dict


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
        "genome_maps":  out / "report_genome_maps.html",
    }

    html_map = {
        "plasmid": _render_plasmid_page(report_data),
        "genome_maps": _render_genome_maps_page(
            report_data.get("genome_maps", {}),
            input_file=report_data.get("input_file", ""),
        ),
        "chromosome": _render_nonplasmid_page(
            report_data, "chromosome", "Chromosome", "#27ae60",
            report_data["chromosome_rows"], "ctable",
            report_data["chrom_charts"],
        ),
        "phage": _render_nonplasmid_page(
            report_data, "phage", "Phage", "#e67e22",
            report_data["phage_rows"], "phtable",
            report_data["phage_charts"],
        ),
        "archaea": _render_nonplasmid_page(
            report_data, "archaea", "Archaea", "#8e44ad",
            report_data["archaea_rows"], "artable",
            report_data["arch_charts"],
        ),
        "unclassified": _render_nonplasmid_page(
            report_data, "unclassified", "Unclassified", "#95a5a6",
            report_data["unclassified_rows"], "utable",
            report_data["unc_charts"],
            show_best=True,
            extra_note=(
                "Contigs where no class scored above threshold. "
                "Best Label = top-scoring class even below threshold. "
                "Re-run with <code>--min-confidence 0.50</code> to assign these."
            ),
        ),
    }

    for key, path in pages.items():
        path.write_text(html_map[key], encoding="utf-8")
        logger.info("Report written to %s", path)

    return pages


# ---------------------------------------------------------------------------
# Backward compat
# ---------------------------------------------------------------------------


def generate_report(report_data: dict, output_path: Path | str) -> Path:
    paths = generate_reports(report_data, Path(output_path).parent)
    return paths["plasmid"]


# ---------------------------------------------------------------------------
# Backward-compat aliases (used by tests and report_cmd imports)
# ---------------------------------------------------------------------------

_build_pie_data       = _pie
_build_arg_chart      = _arg_bar
_build_risk_histogram = _risk_hist
