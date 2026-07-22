"""Unit tests for the HTML report generator (report/generator.py).

Covers the JSON-in-<script> escaping fix added in response to the second
external code review (docs/CODE_REVIEW_FINDINGS_2026-07.md, Round 8):
json.dumps() doesn't escape "/", so a contig ID, gene name, or other report
data containing a literal "</script>" could break out of the surrounding
<script> tag. _json_for_script() now escapes "</" as "<\\/" everywhere
JSON is embedded inside a <script> block.
"""

from __future__ import annotations

import json

from plasflow2.report.generator import (
    PlasmidRow,
    _json_for_script,
    _render_plasmid_page,
)

# ---------------------------------------------------------------------------
# _json_for_script
# ---------------------------------------------------------------------------


class TestJsonForScript:
    def test_escapes_closing_script_tag(self):
        payload = {"contig_id": "c1</script><script>alert(1)</script>"}
        out = _json_for_script(payload)
        assert "</script>" not in out
        assert "<\\/script>" in out

    def test_round_trips_to_same_value(self):
        payload = {"a": 1, "b": ["x", "y</z>"], "c": None}
        out = _json_for_script(payload)
        # "<\/" in a JS string literal evaluates to "</" -- simulate that
        # here to confirm the escaped form decodes back to the original.
        assert json.loads(out.replace("<\\/", "</")) == payload

    def test_no_op_when_no_closing_tag_present(self):
        payload = {"contig_id": "plain_contig_1", "n": 3}
        assert _json_for_script(payload) == json.dumps(payload)


# ---------------------------------------------------------------------------
# _render_plasmid_page — end-to-end escaping check
# ---------------------------------------------------------------------------


def _minimal_plasmid_row(contig_id: str) -> PlasmidRow:
    return PlasmidRow(
        contig_id=contig_id,
        contig_length=50_000,
        confidence=0.9,
        num_args=0,
        drug_classes="",
        mobility_class="",
        replicon_type="",
        risk_score=1,
        taxonomy="",
        risk_evidence="",
    )


def _chart(label: str = "x") -> dict:
    return {"data": [{"labels": [label]}], "layout": {"title": label}}


def _minimal_report_data(rows: list[PlasmidRow]) -> dict:
    return {
        "plasmid_rows": rows,
        "class_counts": {"plasmid": len(rows)},
        "total_args": 0,
        "total_vf": 0,
        "total_mge": 0,
        "input_file": "test.fasta",
        "pie_data": _chart("pie"),
        "arg_data": _chart("arg"),
        "risk_data": _chart("risk"),
        "vf_data": _chart("vf"),
        "mge_data": _chart("mge"),
        "mobility_data": _chart("mob"),
        "eskape_data": _chart("esk"),
    }


def test_render_plasmid_page_escapes_malicious_contig_id():
    malicious_id = "c1</script><script>alert(document.cookie)</script>"
    data = _minimal_report_data([_minimal_plasmid_row(malicious_id)])
    html = _render_plasmid_page(data)

    # Baseline is 2 legitimate "</script>" tags: the Plotly CDN <script src=
    # ...></script> in <head>, and the single closing tag for the page's
    # inline <script> block (_full_page()). The injected payload must not
    # add a third.
    clean_data = _minimal_report_data([_minimal_plasmid_row("plain_id")])
    baseline_html = _render_plasmid_page(clean_data)
    # The malicious id's "</script>" sequences must not add any extra
    # (unescaped) closing tags beyond the page's own legitimate ones -- that
    # would be the actual break-out. An unescaped literal "<script>" (the
    # *opening* half) embedded inside a JS string is harmless on its own,
    # since it can't terminate anything by itself.
    assert html.count("</script>") == baseline_html.count("</script>")
    # The escaped form (used inside the embedded JS/JSON) should be present.
    assert "<\\/script>" in html
    assert "alert(document.cookie)" in html


def test_render_plasmid_page_normal_contig_id_unaffected():
    data = _minimal_report_data([_minimal_plasmid_row("NZ_CP012345.1")])
    html = _render_plasmid_page(data)
    assert "NZ_CP012345.1" in html
