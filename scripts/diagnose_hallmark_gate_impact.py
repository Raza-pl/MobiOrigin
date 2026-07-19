#!/usr/bin/env python3
"""Isolate how much of a recall change is attributable to the hallmark-gate
fix (removing bare ICE hits / raw PLSDB matches as sufficient evidence),
versus other factors (e.g. geNomad timing out and disabling SPM features)
that can also differ between two `plasflow2 run` invocations.

Works entirely from a completed run's all_predictions.tsv + a ground-truth
TSV -- no need for the original run's terminal log.

How it works
------------
pipeline.py's hallmark gate runs BEFORE the marker-XGBoost rescoring stage,
and only contigs that survive the gate get rescored. So for any contig:

  - `plasmid_score` in the output TSV is the Stage-1 MLP score from BEFORE
    the gate ran (the gate never touches `scores`, only `label`) -- so
    whether the contig started out as a Stage-1 plasmid candidate is
    reconstructable from plasmid_score vs. the same per-length threshold
    table predict.py uses.
  - `xgb_plasmid` is populated only for contigs that reached the marker
    rescoring stage -- i.e. survived the hallmark gate.

So: Stage-1 candidate + empty xgb_plasmid + final label != plasmid means
the hallmark gate demoted it. Among those, this script re-evaluates each
row's biological-evidence columns (already in the TSV for every contig)
under both the OLD gate policy (mobility OR replicon OR rep-protein OR ICE
OR PLSDB match) and the NEW one (mobility OR replicon OR rep-protein only)
to find rows the gate fix specifically flipped.

Usage
-----
    python scripts/diagnose_hallmark_gate_impact.py \\
        --predictions /tmp/pf2_default/all_predictions.tsv \\
        --ground-truth data/benchmark/ground_truth.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

LENGTH_THRESHOLD_TIERS = [
    (2_000, 0.862, 0.95, 0.75),
    (4_999, 0.864, 0.92, 0.68),
    (9_999, 0.859, 0.90, 0.65),
    (19_999, 0.857, 0.90, 0.63),
    (float("inf"), 0.809, 0.90, 0.62),
]
LEGACY_PLASMID_FLOOR = 0.95  # historical CLI default, applied below 5kb


def _tier_plasmid_threshold(length: int) -> float:
    for max_len, plas_t, _phage_t, _chr_t in LENGTH_THRESHOLD_TIERS:
        if length <= max_len:
            if length < 5_000:
                return max(plas_t, LEGACY_PLASMID_FLOOR)
            return plas_t
    return LEGACY_PLASMID_FLOOR if length < 5_000 else 0.80


def _flt(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()

    gt: dict[str, str] = {}
    with open(args.ground_truth) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            gt[row["contig_id"]] = row["true_label"]

    n_true_plasmid = 0
    n_fn = 0  # true plasmid, final label != plasmid
    n_never_candidate = 0  # Stage-1 MLP itself didn't call it plasmid
    n_reached_xgboost_but_lost = 0  # candidate, reached XGBoost, lost there
    n_hallmark_demoted = 0  # candidate, never reached XGBoost -> gate demoted it
    n_would_survive_old_gate = 0  # of those, had ICE/PLSDB evidence only
    examples: list[dict] = []

    with open(args.predictions) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing_cols = {"plasmid_score", "chromosome_score", "phage_score", "xgb_plasmid"} - set(
            reader.fieldnames or []
        )
        if missing_cols:
            raise SystemExit(
                f"predictions TSV is missing expected columns: {sorted(missing_cols)}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            cid = row["contig_id"]
            true = gt.get(cid)
            if true != "plasmid":
                continue
            n_true_plasmid += 1

            if row.get("label") == "plasmid":
                continue  # correctly classified, not an FN
            n_fn += 1

            length = int(_flt(row.get("length", "0")))
            plas_s = _flt(row.get("plasmid_score"))
            chr_s = _flt(row.get("chromosome_score"))
            phg_s = _flt(row.get("phage_score"))
            was_candidate = (
                plas_s >= chr_s and plas_s >= phg_s and plas_s >= _tier_plasmid_threshold(length)
            )

            if not was_candidate:
                n_never_candidate += 1
                continue

            reached_xgb = bool((row.get("xgb_plasmid") or "").strip())
            if reached_xgb:
                n_reached_xgboost_but_lost += 1
                continue

            # Candidate, never reached XGBoost -> hallmark gate demoted it.
            n_hallmark_demoted += 1

            has_mobility_evidence = (
                _flt(row.get("is_conjugative")) > 0
                or _flt(row.get("is_mobilizable")) > 0
                or _flt(row.get("has_replicon")) > 0
                or _flt(row.get("has_rep_protein")) > 0
            )
            has_ice = _flt(row.get("has_ice")) > 0 or _flt(row.get("num_ice", "0")) > 0
            has_plsdb = bool((row.get("plasmid_db_match") or "").strip())

            would_survive_old_gate = (not has_mobility_evidence) and (has_ice or has_plsdb)
            if would_survive_old_gate:
                n_would_survive_old_gate += 1
                if len(examples) < 15:
                    examples.append(
                        {
                            "contig_id": cid,
                            "length": length,
                            "plasmid_score": plas_s,
                            "has_ice": has_ice,
                            "has_plsdb": has_plsdb,
                        }
                    )

    print(f"True plasmids in ground truth (matched in predictions): {n_true_plasmid}")
    print(f"False negatives (true plasmid, final label != plasmid): {n_fn}")
    print()
    print(f"  Never a Stage-1 candidate (MLP itself missed it):        {n_never_candidate}")
    print(
        f"  Reached XGBoost rescoring, lost there:                   {n_reached_xgboost_but_lost}"
    )
    print(f"  Hallmark-gate demoted (candidate, never reached XGBoost): {n_hallmark_demoted}")
    print()
    print(
        f"    Of those gate-demoted, would have SURVIVED the OLD gate "
        f"(ICE/PLSDB evidence only, no mobility/replicon/rep-protein):  "
        f"{n_would_survive_old_gate}"
    )
    print(
        "    -> this count is the hallmark-gate fix's direct, isolated "
        "contribution to the recall drop; the rest of n_hallmark_demoted "
        "would have been demoted under the OLD gate too."
    )
    if examples:
        print("\n  Example contigs flipped by the gate fix (up to 15):")
        for ex in examples:
            print(f"    {ex}")


if __name__ == "__main__":
    main()
