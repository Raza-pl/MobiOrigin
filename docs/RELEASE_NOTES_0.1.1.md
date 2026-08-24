# MobiOrigin 0.1.1 release notes

MobiOrigin 0.1.1 is the publication and biological-annotation update for the
frozen `mobiorigin-dev1-mob-selective-v1` candidate. The three model checkpoints,
training-only marker normalization, ensemble behavior, and selective threshold
are byte-identical to version 0.1.0.

## Classification evidence

- Development locked test: macro-F1 0.8467, balanced accuracy 0.8403, plasmid
  precision 0.8648, plasmid sensitivity 0.7550, and coverage 0.9833.
- Prospective external cohort: 3,000 records from 3,000 distinct versioned source
  accessions.
- Against the preregistered geNomad 1.12.0/database 1.9 comparator, MobiOrigin's
  macro-F1 difference was +0.0315 (95% CI +0.0135 to +0.0497; Holm-adjusted
  *P*=0.00120) and plasmid binary F1 difference was +0.0578 (95% CI +0.0300 to
  +0.0852; Holm-adjusted *P*=0.000400).
- Required trade-off disclosure: MobiOrigin had lower plasmid precision and lower
  coverage but higher plasmid sensitivity than geNomad on this cohort.

## New in 0.1.1

- `mobiorigin annotate` provides prediction-independent CARD, SARG, official
  AMRFinderPlus, VFDB, MGE, BacMet2, and MOB-suite evidence.
- Annotation outputs include per-hit evidence, per-ORF consensus, per-contig
  summaries, provenance, checksums, and a self-contained HTML report.
- A–E biological evidence-priority tiers support transparent follow-up triage;
  they are not clinical risk scores.
- A separate post-hoc exploratory analysis compares MobiOrigin with PlasClass,
  PlasFlow v1, PLASMe, and Platon on the frozen external cohort.
- A label-free operational study reports runtime, call rate, coverage, agreement,
  and biological evidence for two deterministic real-assembly subsets. All 12
  dataset–tool routes and all 10 pairwise operational comparisons completed.
- Validation tables, editable SVG figures, methods, and limitations have been
  updated.

## Distribution contents

- Standalone `mobiorigin predict`, `mobiorigin annotate`, and
  `mobiorigin setup-databases` commands.
- Three frozen tensor-only model state dictionaries and training-only marker
  normalization.
- Atomic prediction and annotation artifacts with provenance and SHA-256
  checksums.
- User documentation, citation metadata, changelog, and aggregate validation
  evidence.
- No third-party biological database payloads; users retrieve licensed databases
  locally following the documented setup routes.

## Validation gates

- Black, Ruff, and mypy passed.
- Focused tests: 30 passed.
- Full unit tests: 30 passed.
- All four packaged model and normalization artifacts matched their frozen
  identities.
- The wheel and source distribution passed the final local publication/package
  gate without bundling biological databases.

## Scientific boundaries

- The locked development test and prospective external cohort remain closed to
  retrospective tuning and record-level error mining.
- geNomad and secondary-comparator outputs are not MobiOrigin features or training
  targets.
- Secondary-comparator findings are explicitly exploratory and post hoc.
- Real-assembly operational results have no frozen record-level ground truth and
  therefore do not support accuracy or superiority claims.
- Biological evidence tiers prioritize review and do not establish clinical risk,
  host pathogenicity, transfer, or causality.

## External release actions

The GitHub release page and archival DOI remain external service actions. Neither
may change the frozen scientific candidate or reported results.
