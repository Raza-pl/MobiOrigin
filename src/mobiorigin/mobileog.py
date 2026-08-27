"""Shared mobileOG-db header parsing and source-compatibility auditing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

MOBILEOG_PARSER_SCHEMA = "mobiorigin-mobileog-header-parser-v2"
MOBILEOG_COMPATIBILITY_SCHEMA = "mobiorigin-mobileog-compatibility-v1"

_MOBILEOG_TOKEN = re.compile(r"mobileOG_[^|\s]+(?:\|[^|\s]*){4,}")


def mobileog_fields(*candidates: str) -> tuple[str, ...] | None:
    """Return a supported pipe-delimited mobileOG identifier from any candidate text."""
    for candidate in candidates:
        for match in _MOBILEOG_TOKEN.finditer(candidate):
            fields = tuple(match.group(0).split("|"))
            if len(fields) >= 5 and fields[0].startswith("mobileOG_"):
                return fields
    return None


def audit_mobileog_fasta(source: Path, output: Path) -> dict[str, Any]:
    """Stream every FASTA header and record compatibility without loading sequences."""
    total = 0
    supported = 0
    unsupported_examples: list[str] = []
    with source.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if not raw.startswith(">"):
                continue
            total += 1
            header = raw[1:].rstrip("\r\n")
            if mobileog_fields(header) is not None:
                supported += 1
            elif len(unsupported_examples) < 10:
                unsupported_examples.append(header[:500])
    if total == 0:
        raise ValueError("mobileOG-db FASTA contains no sequence headers")
    result: dict[str, Any] = {
        "schema_version": MOBILEOG_COMPATIBILITY_SCHEMA,
        "parser_schema": MOBILEOG_PARSER_SCHEMA,
        "status": "PASS" if supported == total else "PASS_WITH_EXCLUSIONS",
        "headers_total": total,
        "headers_supported": supported,
        "headers_excluded": total - supported,
        "unsupported_examples": unsupported_examples,
        "policy": (
            "Unsupported headers are excluded from mobileOG biological evidence and reported; "
            "they are never interpreted by inference."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
