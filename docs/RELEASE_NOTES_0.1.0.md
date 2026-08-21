# MobiOrigin 0.1.0 release notes

MobiOrigin 0.1.0 is the initial research release of the frozen `mobiorigin-dev1-mob-selective-v1` candidate. Model weights, marker normalization, ensemble behavior, and the selective threshold are unchanged from the qualified development and prospective external evaluations.

## Validation summary

- Development locked test: macro-F1 0.8467, balanced accuracy 0.8403, plasmid precision 0.8648, plasmid sensitivity 0.7550, and coverage 0.9833.
- Prospective external cohort: 3,000 records from 3,000 distinct versioned source accessions.
- External co-primary comparison versus geNomad 1.12.0/database 1.9:
  - Macro-F1 difference +0.0315 (95% CI +0.0135 to +0.0497; Holm-adjusted *P*=0.00120).
  - Plasmid binary F1 difference +0.0578 (95% CI +0.0300 to +0.0852; Holm-adjusted *P*=0.000400).
- Required trade-off disclosure: MobiOrigin had lower plasmid precision and lower coverage but higher plasmid sensitivity than geNomad on this cohort.

## Distribution contents

- Standalone `mobiorigin` command.
- Three frozen tensor-only model state dictionaries and training-only marker normalization.
- Atomic exact-hash marker-database setup command; third-party biological databases are not bundled.
- Atomic prediction outputs with provenance and checksums.
- User documentation, citation metadata, changelog, manuscript draft, and aggregate publication artifacts.

## Reproducibility gates

- Full unit suite: 728 tests passed.
- MobiOrigin critical-core coverage: 87.4%.
- Black, Ruff, mypy, compile, Poetry metadata, wheel, and source-distribution gates passed.
- Clean wheel installed into an isolated target and reproduced byte-identical real synthetic-sequence prediction outputs in two runs.
- Wheel model and normalization bytes match their frozen SHA-256 identities.
- Wheel contains no `.dmnd` database payloads and exposes no broken historical console entry point.

## Scientific boundaries

- The locked development test and prospective external cohort are closed to retrospective tuning and record-level error mining.
- geNomad outputs are not used as MobiOrigin features or training targets.
- No hard biological label overrides or post-hoc probability transfer are used.
- External subgroup and precision/sensitivity/coverage comparisons are descriptive unless explicitly identified as co-primary.

## External actions after repository freeze

The repository has been renamed to `Raza-pl/MobiOrigin`. Creating the GitHub release page, minting an archival DOI, and submitting to a journal still require external-service actions and author confirmation. Those actions must not change the frozen scientific candidate or reported results.
