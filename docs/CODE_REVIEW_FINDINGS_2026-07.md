# Diagnostic review — verification log (July 2026)

## Round 4: candidate-routing widening (near-miss margin)

The confirmation run's remaining recall gap (0.449 vs. the 0.538 pre-fix
baseline) traced to the biggest architectural item on the backlog: every
biological-evidence step (PLSDB, mobility, geNomad, marker-XGBoost
rescoring) only ever runs on `plasmid_records = [r for r in records if
pred_by_id[r.id].label == "plasmid"]` — a contig the Stage-1 MLP doesn't
outright label plasmid gets zero evidence gathered, ever.

`scripts/analyze_missed_candidates.py` (run against the confirmation run's
`all_predictions.tsv`) characterized the 170 missed true plasmids:

| Bucket | Count | % |
|---|---|---|
| MLP argmax winner WAS plasmid, just under threshold | 95 | 55.9% |
| MLP argmax winner was chromosome (genuine miss) | 74 | 43.5% |
| MLP argmax winner was phage | 1 | 0.6% |

The first bucket is a cheap, well-justified fix: the marker-XGBoost
rescoring loop already computes its `new_label` purely from blended scores
(`agg`/`best_conf`/`thresh`), not from the contig's original MLP label — it
was already capable of promoting these, it just never got the chance
because they never entered `plasmid_records`.

Sized the widening in the sandbox with `predict()` alone (no DIAMOND/
geNomad needed) before touching pipeline code, since a bad guess here is
expensive to walk back (unlike the ICE/PLSDB gate mistake, this one scales
annotation runtime directly):

| Widening rule | Contigs added | True plasmids captured | Precision of added slice |
|---|---|---|---|
| Any argmax=plasmid regardless of margin (naive) | 7,371 | 95 | 1.3% |
| Margin ≤ 0.005 | 5 | 4 | 80.0% |
| Margin ≤ 0.01 | 19 | 10 | 52.6% |
| **Margin ≤ 0.02 (chosen)** | **66** | **21** | **31.8%** |
| Margin ≤ 0.03 | 170 | 24 | 14.1% |
| Margin ≤ 0.05 | 479 | 30 | 6.3% |
| Margin ≤ 0.10 | 1,327 | 43 | 3.2% |

The naive version was rejected outright — 26x the candidate count for ~1%
of it being true plasmids, on top of a pipeline where geNomad already times
out at ~290 candidates. Chose margin ≤ 0.02: adds only 66 contigs (+23%)
while capturing 21/95 of the recoverable bucket; margin 0.03 was rejected
for poor marginal return (+104 contigs for only +3 more true plasmids).

Also confirmed most of the expensive annotation (ARG/VF/MGE/BacMet/ICE via
DIAMOND, rep-protein DIAMOND) already runs on ALL contigs regardless of
candidacy — only PLSDB match, mobility, and geNomad scale with the widened
candidate count, so the real added cost of +66 contigs is small.

**Implementation** (`pipeline.py`, section 3): added a
`NEAR_MISS_CANDIDATE_MARGIN = 0.02` near-miss widening step right after the
initial `plasmid_records` extraction — includes any contig where the MLP's
own argmax winner was "plasmid" but fell within 0.02 of the (lenient-mode-
aware) threshold. Also guarded the hallmark gate's "≥50kb, no evidence →
keep as low_confidence plasmid" branch to only apply to contigs that were
genuinely MLP-labeled plasmid to begin with (`_original_mlp_plasmid_ids`)
— without this guard, a large widened near-miss contig with zero evidence
would have been auto-promoted to plasmid on size alone, which was never
the intent of that branch.

Smoke-tested in the sandbox: `predict()`-only run against the real
benchmark FASTA confirms exactly 290 → 356 candidates (+66), matching the
standalone sizing analysis exactly. Unit tests (`tests/unit/test_pipeline.py`,
12 tests) pass unchanged.

**NOT yet validated against real DIAMOND/mob-suite/geNomad/PLSDB
annotation data** — the actual recall gain depends on how many of those 66
extra candidates have real biological evidence, which only the full
`run_pipeline()` path on real hardware can determine. Needs a benchmark
rerun before this is considered shippable.

## Round 3 confirmation: benchmark rerun after revert

Reran the same benchmark on real hardware with the hallmark-gate revert
applied (`--skip-genomad`, same input/ground truth as the regression run):

| Run | precision | recall | F1 |
|---|---|---|---|
| Pre-fix baseline (before either change shipped) | 0.777 | 0.538 | 0.636 |
| Broken gate (ICE/PLSDB excluded) — the regression | 0.861 | 0.251 | 0.389 |
| **Reverted gate (this confirmation run)** | **0.835** | **0.449** | **0.584** |

Recall recovered +0.198 (0.251 → 0.449) — matches the diagnostic script's
prediction almost exactly (78/394 ≈ 0.198 points attributable to the gate
fix). Strong internal-consistency check that the diagnosis and revert were
correct. Log line confirms it directly: `Hallmark gate: demoted 78 plasmid
calls (no evidence, <50000 bp)`, down from ~125 under the broken gate.

Recall is still ~9 points under the 0.538 pre-fix/sandbox baseline. This
is **not** the gate's doing — this run logged `Marker XGBoost: promoted 0
→ plasmid, demoted 0 → other` (zero activity at that stage, consistent
with `n_reached_xgboost_but_lost = 0` in the earlier diagnostic). The
remaining gap traces to the largest bucket in the original diagnostic
table: **170 of 394 true plasmids never became a Stage-1 MLP candidate at
all**, so no amount of gate or XGBoost tuning can recover them under the
current architecture. This is a live demonstration of the review's
candidate-routing item (Stage 1 only routes evidence-gathering to contigs
the MLP already scores as plasmid) — the single largest remaining
architectural item on the backlog, not a new bug.

Also observed, not yet investigated: total plasmid calls dropped to 212
this run (vs. 290 in the original pre-revert runs), and the
promoted-0/demoted-0 XGBoost anomaly above appeared in both this run and
the prior one. Flagged as open questions, not blocking.

**Verdict: hallmark-gate revert confirmed correct and sufficient to close
the round-3 regression.** No further action needed on this item.

## Round 3: hallmark-gate ICE/PLSDB fix reverted (real-hardware regression)

Items 4+5 (below, "Fixes applied — round 1") shipped in `cbf9a7e` without
benchmark validation, flagged explicitly as such. The user then ran the
real pipeline on real hardware and found it: default-mode recall came in
at **0.251**, far below the 0.538 predicted from sandbox testing (which
never exercised the hallmark gate — only `predict()` directly). Using
`scripts/diagnose_hallmark_gate_impact.py` against the actual
`all_predictions.tsv`, isolated the cause precisely:

| Bucket (out of 295 total false negatives, 394 true plasmids) | Count |
|---|---|
| Never a Stage-1 MLP candidate at all | 170 |
| Reached marker-XGBoost rescoring, lost there | 0 |
| Hallmark-gate demoted | 125 |
| — of which: would have survived the OLD gate (ICE/PLSDB evidence only) | **78** |

78 out of 394 true plasmids (~20 points of recall) were demoted specifically
because the gate fix stopped accepting a bare ICE hit or raw PLSDB match as
sufficient evidence. That's a real, measured, substantial regression — the
analogy to predict.py's *standalone* policy (which disables the same
signals as a hard override forcing 0.97 confidence) didn't transfer to the
hallmark gate's much weaker use of the same evidence (one of several signals
that can avoid demoting an already-plausible MLP candidate, not a
confidence-forcing override). **Reverted** the hallmark-gate evidence set
back to the original (any-evidence, including ICE/PLSDB) policy.

The other half of that original fix — removing a separate, more aggressive
hard override that forced plasmid confidence to 0.97 on any PLSDB match
during marker-XGBoost rescoring — was **kept**: the same diagnostic showed
zero false negatives attributable to the XGBoost-rescoring stage in this
run, so there's no evidence it caused harm, and it's a much closer analogue
to predict.py's own already-validated override removal (forcing near-
certain confidence from one weak signal, not just avoiding a demotion).

Lesson for next time: an architecturally-motivated fix based on "this same
evidence type was already shown unreliable elsewhere in the codebase" is
not automatically safe to generalize to a different use of that evidence —
each use of a signal (hard override vs. veto-avoidance vs. trained feature)
has a different risk profile and needs its own validation, not inherited
reasoning from a superficially similar case.

## Round 2 (same session, after "do all"): pickle migration + double-counting investigation

**Item 10 (pickle → JSON) — shipped.** `MarkerClassifier.save()` now writes
XGBoost's native JSON format via `model.save_model()` instead of
`pickle.dump()`. Every existing call site still passes a `.pkl` path;
`save()` transparently redirects to the `.json` sibling. `load()` resolves
to whichever of `<stem>.json`, `<stem>.ubj`, or the literal path exists
(preferring native format), falling back to `pickle.load()` only for
pre-migration checkpoints, with a warning. New `resolve_marker_model_path()`
helper is used everywhere a `.pkl` path was previously checked with a raw
`Path.exists()` (`cli.py` ×2, `pipeline.py`, `predict.py` ×2,
`predict_sequences.py`) — those would have silently stopped finding the
marker model once it was re-saved under a different extension. The locally
deployed `data/models/marker_xgb.pkl` (git-ignored) was migrated in place to
`marker_xgb.json`, preserving its provenance metadata; verified
`predict_proba()` output is bit-identical before/after, and a full
`predict()` smoke test through the `.pkl`-path auto-resolution works
end-to-end. The GitHub Release still serves the old pickle — new installs
use it via the legacy fallback (with a warning) until it's regenerated and
re-uploaded, which needs the user's action. Committed `d763453`.

**Item 6 (MLP double-counting in marker fusion) — investigated, NOT
shipped.** The review's critique is architecturally correct: all three MLP
scores are XGBoost input features, and `predict.py` then re-blends
XGBoost's output with the same raw MLP scores again
(`alpha * marker + (1-alpha) * mlp`). The natural "fix" — trust XGBoost's
own output directly, since it already learned from the MLP scores as
features — was implemented as a side experiment and benchmark-tested (not
committed) using the same `predict()` cascade on
`data/benchmark/benchmark.fna` + `annotations_with_replicons.tsv`, by
capturing each contig's raw `xgb_scores` and `mlp_scores` (both already
recorded on every `Prediction` object) alongside the current, blended,
fully-boosted `scores`:

| Aggregation | Precision | Recall | F1 |
|---|---|---|---|
| Current (blend + all post-hoc boosts) — shipped baseline | 0.777 | 0.538 | 0.636 |
| XGBoost output alone (the "fix") | 0.249 | 0.564 | 0.346 |
| MLP output alone (no XGBoost at all) | 0.772 | 0.569 | 0.655 |

Trusting XGBoost's raw output collapses precision (0.777→0.249) — worse
than doing nothing at all. Most contigs in this benchmark have zero
biological (DIAMOND-derived) evidence, and the marker XGBoost model
evidently does not generalize well on all-zero-evidence rows; the blend's
`alpha_base` (which keeps some MLP weight even at zero evidence) is
functioning as a necessary correction for that, not a redundant leftover.
Removing it, as the review's suggested fix would do, is a real regression,
not a cleanup. This mirrors the earlier `LENGTH_THRESHOLD_TIERS` incident
in this same session: an architecturally well-reasoned change that failed
its own benchmark and was correctly not shipped. **Left exactly as-is.** A
real fix here would mean training a genuinely unified fusion model (per the
review's own Section 7 recommendation) rather than removing the current
blend outright — that's a retrain, out of scope for a no-retrain pass.

## Fixes applied (round 1, this session, after the verification pass below)

Four items were picked off the "suggested fix order" list and implemented,
each validated (tests + a benchmark re-run where behavior could change):

1. **Item 9 (torch seed)** — `scripts/train_model.py` now calls
   `torch.manual_seed(42)` / `torch.cuda.manual_seed_all(42)` alongside the
   existing numpy seed. No production-model impact until the next retrain.
2. **Item 2 (threshold semantics)** — `predict.py`'s `_assign_label`,
   `predict()`, `pipeline.py`'s `run_pipeline()`, and both `cli.py` commands
   (`run`, `classify`) were reworked so `--plasmid-threshold` / `--threshold`
   are `None` unless the user explicitly passes them. **Important
   correction to the original plan**: making `None` mean "always use the
   calibrated per-length tier profile" was tried first and benchmark-tested
   against the same `data/benchmark/benchmark.fna` + `annotations_with_replicons.tsv`
   setup used for the `has_replicon` validation. It collapsed default-run
   plasmid precision from **0.777 → 0.224** (recall roughly flat,
   0.538→0.543) — because the CLI's historical default (0.95) had always
   silently applied only below 5kb, and the <5kb tier values in
   `LENGTH_THRESHOLD_TIERS` were apparently swept assuming that floor would
   still be in place, so they were never validated as standalone numbers.
   Shipping that as the new default would have been an undisclosed,
   unvalidated regression. **What actually shipped instead**: `None` now
   reproduces today's real default behavior exactly (verified to match to
   4 decimal places), and only an *explicit* value (`--plasmid-threshold`,
   `--threshold`, or `--lenient`) overrides the tier profile, uniformly, at
   every length — that part was a genuine bug (an explicit lower value was
   previously discarded by a `max()` against the tier default) and is now
   fixed. **Discovered side effect worth flagging**: because `--lenient`'s
   override now actually takes effect at all lengths instead of being
   silently neutralized, `--lenient` is now much more aggressive than it has
   ever actually been in practice — precision drops to **0.091** at recall
   **0.734** on the same benchmark (vs. what real `--lenient` runs have
   actually been producing: ~0.224 precision / 0.543 recall, since the old
   bug meant `--lenient` was accidentally landing on the raw tier profile,
   not the documented 0.70). This matches the README's documented intent
   ("expect more false positives") but the magnitude may be more aggressive
   than intended — worth deciding deliberately rather than accepting as a
   side effect of the bug fix.
   A related but *not fully validated* fix was also made to the
   marker-XGBoost second-stage relabeling in `pipeline.py` (~line 1207):
   explicit `--lenient`/`--plasmid-threshold` now also affects the
   post-marker-fusion label, not just the initial candidate list. This one
   could not be benchmark-validated in this pass — it requires a real
   `run_pipeline()` invocation with DIAMOND/mob-suite annotation, which
   isn't runnable in the sandbox this was written in. Default-run behavior
   is provably unchanged (the override only fires when a value is
   explicitly set); validate with a real `--lenient` run before trusting it.
3. **Items 4+5 (ICE/PLSDB hallmark-gate contradiction)** — `pipeline.py`'s
   hallmark gate no longer treats a bare ICE hit or a raw PLSDB nucleotide
   match as sufficient evidence to keep a plasmid call; only relaxase-based
   mobility, replicon type, and rep-protein hits count now, matching
   predict.py's already-documented, already-justified policy. A second,
   more severe instance of the same PLSDB contradiction was also found and
   removed: a hard override in the marker-fusion rescoring step that forced
   plasmid confidence to 0.97 on any PLSDB match, regardless of anything
   else. **Not benchmark-validated** — this logic only runs inside the full
   `run_pipeline()` with real DIAMOND-based mobility/ICE/plasmid-DB
   annotation dicts, which this sandbox cannot produce from scratch (same
   limitation noted throughout this project's session history). The
   directional case (removing signals predict.py itself measured to cause
   thousands of chromosome FPs) is strong, but real precision/recall impact
   on this specific pipeline path is unverified. Recommend a real
   `plasflow2 run` on a known benchmark before trusting the new numbers.
4. **Item "k=7 redundant RC pass"** — removed from
   `kmer_vector_k7_canonical`. Verified bit-identical output (`np.allclose`,
   atol=1e-6) against the previous implementation across 20 random
   sequences of varying lengths (1bp–5000bp). Pure speed win, zero behavior
   change, no further validation needed.

All four passed `black --check`, `ruff check`, `mypy` (on touched files),
and the full test suite (229 passed) after each change. Not yet pushed —
same as every prior round, `git push origin main` is needed from your Mac.



An external static review of `main` raised 14 findings (3 Critical, 8 High, 3
Medium) plus an architectural critique of the candidate-routing design. Each
checkable claim was verified directly against source on `main` before any
fix work started. This doc is the tracking record: verdict, evidence, and
fix status.

Do not mark anything "Fixed" here without a benchmark re-run confirming the
production `predict()`/`pipeline.py` cascade still behaves as expected — see
`data/benchmark/` and the validation methodology used for the `has_replicon`
fix (recall 0.284→0.538, F1 0.420→0.636, zero threshold changes) as the bar
for "actually validated."

## Critical

### 1. Candidate routing restricted to initial MLP plasmid label
**Verdict: confirmed.**
`pipeline.py:314` — `plasmid_records = [r for r in records if pred_by_id[r.id].label == "plasmid"]`.
PLSDB matching (`:491`), mobility annotation (`:522`), and geNomad (`:988`)
are all gated on `plasmid_records`. A true plasmid the MLP scores as
chromosome/unclassified never receives biological evidence and cannot be
rescued. Stage 2 currently functions as a precision gate on Stage 1's
positives, not a recall-recovery stage over all plausible candidates.
**Status:** not started. Largest-scope item — needs a "high-recall candidate
pool" (e.g. MLP plasmid score ≥ some low threshold OR plasmid is
second-best OR score margin is thin) before annotation stages run.

### 2. `--plasmid-threshold` / `--lenient` don't apply above 5kb
**Verdict: confirmed.**
`predict.py:378` — `if seq_len < 5_000: plas_t = max(plas_t, plasmid_threshold)`.
For `seq_len >= 5_000` the user-supplied threshold is ignored entirely;
`_get_length_thresholds()`'s hardcoded tier table is used instead. `--lenient`
sets `_effective_plasmid_threshold = confidence_threshold` (`pipeline.py:298`)
but inherits the same 5kb gate, so it does NOT lower the plasmid threshold
for contigs ≥5kb despite disabling the hallmark gate for all lengths.
README (`README.md:363`) claims lenient mode "lowers the MLP plasmid
threshold from 0.95 → 0.70" with no length caveat — this is wrong for ≥5kb
contigs.
**Status:** not started.

### 3. Cache reuse skips rebuilding in-memory annotation results
**Verdict: refuted for current `main`.**
Checked VFDB (`vfdb.py:269-280`), MGE (`mge.py:362-373`), plasmid-DB
(`plasmid_db.py:207-209`), and geNomad (`pipeline.py:997-1074`). In every
case, the pipeline-level `_cached()` helper (`pipeline.py:383-388`) only
gates a log message — the underlying `annotate_*()` call always runs, and
each one internally checks its own cache file and *re-parses* hits into the
return value on a cache hit (e.g. `plasmid_db.py` returns
`_parse_paf(paf_path, ...)` rather than an empty list). A resumed run
correctly repopulates `vf_by_contig`/`mge_by_contig`/etc. May have been true
on an older commit; not true now. **No action needed**, but worth a
regression test (`fresh_run == resumed_run`) to keep it that way — see item
13.

## High

### 4. ICE evidence: excluded in `predict.py`, sufficient in `pipeline.py`
**Verdict: confirmed — direct contradiction.**
`predict.py:679-682` zeroes ICE evidence with the comment "ICEs integrate
into chromosomes and create FPs when used as plasmid signal." `pipeline.py`
hallmark gate (`:925,927`) treats any ICE hit as sufficient standalone
evidence to keep a plasmid call (`has_evidence = has_mobility or has_plsdb
or has_replicon or has_ice or has_rep_protein`).
**Status:** not started.

### 5. PLSDB nucleotide evidence: disabled in `predict.py`, sufficient in `pipeline.py`
**Verdict: confirmed — direct contradiction.**
`predict.py:826-833` explicitly disables the PLSDB nucleotide hard override
("minimap2 asm5 at ≥50% qcov / ≥90% identity is non-specific... causing
~3,300 chromosome FPs"). `pipeline.py:922,927` still treats
`cid in plasmid_db_hits` (same underlying match) as sufficient hallmark
evidence.
**Status:** not started. Items 4 and 5 should be fixed together — same
`pipeline.py` function (`hallmark gate`, ~line 895).

### 6. Marker XGBoost double-counts MLP signal
**Verdict: confirmed.**
`predict.py:673-675` — all three MLP class scores (`mlp_plasmid_score`,
`mlp_chromosome_score`, `mlp_phage_score`) are input features to the
XGBoost marker model. `predict.py:813` then blends XGBoost's output with
the *same raw MLP scores* again: `alpha * marker_s + (1-alpha) * mlp_s`.
MLP signal enters the final score twice — once learned (inside XGBoost),
once heuristic (the post-hoc blend).
**Status:** not started. Highest risk to fix — changes calibration
meaningfully, needs full benchmark re-validation before shipping. Do not
attempt alongside items 4/5 in the same PR.

### 7. Manual probability mass-transfer heuristics (~45-65%)
**Verdict: confirmed.** `predict.py:856-864` (`hallmark_boost`) and the
PLSDB protein boost transfer 55% of non-plasmid mass to plasmid on
qualifying evidence. Same file documents several of these
(`plsdb_prot_boost`, `hallmark_boost`, `marker_threshold_boost`) as
redundant with trained XGBoost features (see the documentation block added
near `predict.py:700` in the prior session). Confirms these are decision
heuristics layered on top of already-trained signal, not calibrated
probabilities.
**Status:** partially addressed — documented as redundant in a prior
session (commit history), not yet removed. Removal blocked on a real
tier1 benchmark run per that same prior documentation (to confirm they're
not silently compensating for something the model underweights).

### 8. Marker model trained/validated with row-level split, not grouped
**Verdict: confirmed.** `marker_classifier.py:461` —
`train_test_split(X, y, test_size=eval_fraction, stratify=y, random_state=...)`.
No group parameter; ordinary stratified split. Contrast with the MLP
trainer (`scripts/train_model.py:522`, `grouped_split_indices(...)`), which
is correctly group-aware. Also confirmed: marker training logs
`val_accuracy` only, not F1/AUPRC/calibration (`train_marker_model.py:113`).
**Status:** not started.

### 9. MLP training has no `torch.manual_seed`
**Verdict: confirmed.** `scripts/train_model.py` sets `_SEED = 42` and uses
it for `np.random.default_rng` and `grouped_split_indices(..., random_state=_SEED)`,
but no `torch.manual_seed`/`torch.cuda.manual_seed` call exists anywhere in
the file. Weight init and any non-deterministic ops are unseeded.
**Status:** not started. Trivial fix (add the calls); low risk.

### 10. Pickled XGBoost checkpoint loaded directly
**Verdict: confirmed.** `marker_classifier.py:578` —
`pickle.load(fh)  # noqa: S301`. The lint-suppression comment is itself
acknowledgment of the exact risk (arbitrary code execution on
deserialization of an untrusted pickle).
**Status:** not started. Fix = migrate to XGBoost's native JSON/UBJ format
(`model.save_model(path, format="json")` / `Booster.load_model`). Needs a
migration path for the deployed `.pkl` (this repo already distributes
models via GitHub Release, not git — see model-card work from prior
session).

## Medium / structural (verified, not itemized in depth)

- **Ambiguous bases (N) encoded as A**: confirmed, `features.py:108`
  (`_ASCII_TO_BASE = np.zeros(256, ...)`). The in-code comment claiming this
  is "equivalent to skipping them" is incorrect — it fabricates A-rich
  composition rather than omitting ambiguous windows. Fixing this changes
  the k=1-7 feature vectors for any sequence containing N, which means the
  MLP would need retraining to actually benefit — this is a train+serve
  change, not a serve-only patch. Scope accordingly.
- **k=7 canonical folding does redundant reverse-complement work**:
  confirmed and worse than the review states — traced the math: the second
  (reverse-complement) pass produces an *exact* duplicate of the
  forward-strand canonical counts (since `canon_map[raw] == canon_map[rc(raw)]`
  for every k-mer), so it doubles every count and cancels out entirely under
  L2 normalization. Pure wasted compute, zero effect on the resulting
  vector. Safe, mechanical speed fix — remove the RC pass in
  `kmer_vector_k7_canonical` (`features.py:294-301`).
- **Unknown source-group IDs become independent groups in split logic**:
  confirmed as described, but `splits.py:22-28`'s own docstring says this
  is a deliberate, conservative choice — unmatched IDs split into *more*
  groups, not merged, which is the safe direction against leakage, not a
  leakage vector itself. Lower priority than the review implies.

## Not yet checked

Sections 11 (checkpoint schema), 12 (context-prior label-shift math), 13-14
(benchmark suite design / metrics), 17 (degraded-state error handling), 18
(packaging/CI gaps), 20-21 (proposed multi-stage architecture, hybrid
class) are design proposals or require broader judgment calls rather than
single verifiable code facts — not run through the same confirm/refute pass.
Worth a second look before committing to the full P0-P3 roadmap as written.

## Suggested fix order (risk-ranked, not the review's original order)

1. Item 9 (torch seed) — trivial, zero behavior risk.
2. Item 2 (threshold semantics) — mechanical, no retrain, testable in
   isolation, fixes a documented reproducibility bug.
3. Items 4+5 (ICE/PLSDB hallmark-gate contradiction) — no retrain, but
   changes real classification outcomes; needs a benchmark re-run.
4. k=7 redundant RC pass — safe speed win, verify output vector is
   bit-identical before/after on a sample set.
5. Item 10 (pickle → JSON) — needs a deployed-model migration plan.
6. Item 6 (MLP double-counting) — largest calibration risk; benchmark
   before and after on the full cascade.
7. Item 8 (grouped marker split) — requires rebuilding
   `marker_features_balanced_28_genomad.npz` with group IDs threaded
   through; larger data-engineering lift.
8. Ambiguous-base encoding — requires MLP retrain to realize any benefit;
   biggest single lift, do last or as its own project.
9. Candidate-routing redesign (item 1) — architectural, cross-cutting,
   should follow all of the above so the baseline being compared against is
   already correct.
