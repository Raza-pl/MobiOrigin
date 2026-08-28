# MobiOrigin 0.1.5 release notes

MobiOrigin 0.1.5 improves installation diagnostics, model transport, and
biological-evidence reporting without changing the frozen classifier or any
scientific decision threshold.

## User-visible changes

- The complete guided Conda or Mamba installation is presented first and runs
  an eight-contig prediction, comprehensive annotation, and visualization test.
- Setup and `mobiorigin doctor` are concise by default. `--verbose` retains the
  full checksum inventory for audit.
- `mobiorigin doctor` verifies software, frozen models, marker databases, and
  comprehensive annotation databases together.
- The example output is written outside the repository by default, leaving a
  clean Git working tree.
- Apple Silicon documentation distinguishes native arm64 prediction and
  annotation from the isolated Rosetta helper used only to reconstruct frozen
  MOB-suite marker databases.
- External-tool discovery prefers DIAMOND, AMRFinderPlus, and updater commands
  inside the active MobiOrigin environment. A host tool such as
  `~/bin/diamond` can no longer shadow the conda-managed executable.
- Frozen marker reconstruction receives the absolute DIAMOND 2.0.15 path from
  its isolated helper environment. Explicit user-supplied paths are still
  honored.

## Model transport

Source checkouts no longer contain duplicate checkpoint binaries. The
versioned `mobiorigin-models-dev1.tar` release asset remains the canonical model
transport source. Setup retrieves the archive once and verifies the archive,
three checkpoints, marker normalization array, and model manifest before atomic
installation.

The model archive remains pinned to the v0.1.3 release because its bytes are
unchanged. MobiOrigin 0.1.5 does not retrain, replace, or modify any model
artifact.

## Normalized gene reporting

`biological_evidence.tsv` now provides additive canonical fields across CARD,
AMRFinderPlus, SARG, VFDB, mobileOG-db, BacMet, MOB-suite, and the optional
legacy ISfinder route:

- `gene_symbol`
- `gene_name`
- `gene_family`
- `functional_class`
- `functional_subclass`
- `mechanism`

The original `feature_type`, `feature_name`, `category`, `description`, source,
accession, coordinate, identity, coverage, E-value, and bit-score fields remain
available. Missing values are normalized to `unknown`; biological aliases and
unsupported mechanisms are not inferred.

`mobiorigin_annotated_results.tsv` and the HTML report now summarize normalized
gene symbols, names, families, classes, subclasses, mechanisms, and contributing
databases for each contig. Annotation provenance advances to version 4, the
publication summary advances to version 2, and both identify the normalized
gene vocabulary as `mobiorigin-normalized-gene-v1`.

## Verification

- GitHub CI passed on Python 3.10 and 3.11.
- The local production suite passed 67 tests with coverage above the enforced
  80% threshold.
- Comprehensive macOS validation produced 62 retained evidence hits across
  eight demonstration contigs and verified every normalized schema field.
- Guided installation and verification completed successfully on WSL/Linux.

The bundled demonstration is a software test. Its sequences and annotations
must not be interpreted as prevalence, accuracy, or biological validation.

## Scientific boundary

This release does not change model checkpoints, sequence features, marker
features, normalization, ensemble weights, abstention threshold, supported
length range, annotation thresholds, database payloads, origin predictions, ARG
consensus calls, or A to E evidence-priority rules. Normalized fields are
source-preserving report columns. They do not prove phenotype, plasmid origin,
transferability, pathogenicity, or clinical risk.
