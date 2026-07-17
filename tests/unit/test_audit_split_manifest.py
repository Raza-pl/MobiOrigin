"""Tests for split-manifest and benchmark-lockout auditing."""

from __future__ import annotations

from scripts.audit_split_manifest import audit_split_manifest


def test_audit_split_manifest_detects_overlap_and_split_leakage(tmp_path) -> None:
    manifest = tmp_path / "manifest.tsv"
    lockout = tmp_path / "lockout.txt"
    report = tmp_path / "report.json"
    manifest.write_text(
        "row_index\tfeature_row_index\tsequence_id\tsource_group\tlabel\tsplit\n"
        "0\t10\ta\tGROUP_A\t0\ttrain\n"
        "1\t11\tb\tGROUP_A\t0\ttest\n"
        "2\t12\tc\tLOCKED\t1\tvalidation\n"
    )
    lockout.write_text("LOCKED\n")

    issues, summary = audit_split_manifest(manifest, lockout, report)

    assert any("cross split" in issue for issue in issues)
    assert any("locked benchmark" in issue for issue in issues)
    assert summary["locked_overlap"] == 1


def test_audit_split_manifest_accepts_clean_manifest(tmp_path) -> None:
    manifest = tmp_path / "manifest.tsv"
    lockout = tmp_path / "lockout.txt"
    report = tmp_path / "report.json"
    manifest.write_text(
        "row_index\tfeature_row_index\tsequence_id\tsource_group\tlabel\tsplit\n"
        "0\t10\ta\tGROUP_A\t0\ttrain\n"
        "1\t11\tb\tGROUP_B\t1\tvalidation\n"
        "2\t12\tc\tGROUP_C\t2\ttest\n"
    )
    lockout.write_text("LOCKED\n")

    issues, summary = audit_split_manifest(manifest, lockout, report)

    assert issues == []
    assert summary["split_conflicts"] == 0
