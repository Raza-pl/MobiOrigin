# Changelog

All notable MobiOrigin changes are documented here. The project uses semantic versioning for the standalone `mobiorigin` package interface.

## Unreleased

## 0.1.2 — 2026-08-26

Installation, database automation, and integrated analysis update for the
unchanged frozen `mobiorigin-dev1-mob-selective-v1` classifier.

### Added

- Guided Conda or Mamba installation with a post-installation doctor check and
  bundled deterministic demonstration.
- Automatic, resumable setup and cryptographic verification of the comprehensive
  annotation resources, with explicit acceptance of applicable third-party terms.
- mobileOG-db as the default MGE resource, while retaining legacy ISfinder as an
  optional user-supplied resource.
- A bundled eight-contig assembly example that demonstrates chromosome, plasmid,
  phage, and unclassified outputs without presenting an accuracy claim.
- Deterministic SVG, HTML, and tabular visualizations for prediction and annotation
  outputs.

### Changed

- `mobiorigin run` now performs prediction, comprehensive biological annotation,
  and integrated visualization by default in one atomic output directory. A
  `--skip-annotation` option preserves the lightweight prediction-only route.
- Expanded `--threads` from the original validated 1–8 range to 1–128 for
  DIAMOND, AMRFinderPlus, and other external searches. Deterministic neural-network
  inference remains single-threaded, and the model and scientific policy are unchanged.

### Fixed

- Separated the CPU-only MobiOrigin runtime from MOB-suite's incompatible legacy
  NumPy/pandas database-building stack.
- Added a non-overwriting database setup helper with Linux/WSL, Intel macOS, and
  Apple Silicon/Rosetta handling plus actionable failure messages.
- Prevented Conda from selecting CUDA by pinning the documented runtime to a
  cross-platform CPU PyTorch build.
- Added CI installation smoke tests, an isolated MOB-suite dependency solve, and
  installation-contract unit tests.

### Scientific boundaries

- The classifier checkpoints, feature definitions, ensemble, selective threshold,
  and frozen external-validation results are unchanged from version 0.1.1.
- Biological annotations remain independent supporting evidence. They do not
  override prediction labels or probabilities and are not clinical risk scores.
- The bundled assembly is a software demonstration. It is not an accuracy,
  prevalence, or biological-discovery dataset.

## 0.1.1 — 2026-08-23

Publication and biological-annotation update for the unchanged frozen
`mobiorigin-dev1-mob-selective-v1` classifier.

### Added

- Prediction-independent `mobiorigin annotate` workflow integrating CARD, SARG,
  official AMRFinderPlus, VFDB, MGE, BacMet2, and MOB-suite evidence without
  changing classifier labels or probabilities.
- Publication-quality annotation tables, provenance, checksums, and HTML reports,
  including transparent A–E biological evidence-priority tiers that are explicitly
  not clinical risk scores.
- Post-hoc exploratory external comparisons with PlasClass, PlasFlow v1, PLASMe,
  and Platon under a separately frozen statistical contract.
- Label-free operational evidence from two deterministic real-assembly subsets,
  covering all 12 dataset–tool runs and 10 pairwise operational comparisons.
- Validation tables, editable vector figures, methods, limitations, and updated
  repository documentation.

### Changed

- Package and citation metadata now identify the expanded publication bundle as
  version 0.1.1.
- Distribution metadata includes the annotation and operational-validation
  documentation.

### Scientific boundaries

- The frozen classifier, three model checkpoints, marker normalization, ensemble,
  and selective threshold are unchanged from version 0.1.0.
- Secondary comparator findings are exploratory and do not alter the preregistered
  MobiOrigin-versus-geNomad co-primary evidence.
- Real-assembly results support runtime, call-rate, coverage, agreement, and
  biological-evidence reporting only; they do not support ground-truth accuracy or
  superiority claims.

## 0.1.0 — 2026-08-21

Initial research release of the frozen `mobiorigin-dev1-mob-selective-v1` candidate.

### Added

- Standalone `mobiorigin predict` interface for chromosome, plasmid, phage, and explicit unclassified predictions.
- Deterministic 9,557-dimensional sequence-feature extraction and 17-dimensional MOB protein-marker extraction.
- Three frozen neural-network checkpoints combined by an equal-weight softmax mean.
- Frozen plasmid selective-abstention rule with threshold `0.19835489988327026`.
- Safe tensor-only checkpoint loading and exact model, normalization, and database identity verification.
- Atomic prediction outputs with provenance and SHA-256 checksums.
- `mobiorigin setup-databases` for atomic retrieval or offline installation of the exact marker databases.
- Prospective external validation against geNomad 1.12.0/database 1.9 using 3,000 source-disjoint records.
- Aggregate validation tables, vector figure, methods, and claim boundaries.

### Scientific boundaries

- The frozen external cohort is closed to retrospective tuning and record-level error mining.
- geNomad outputs are not model features or training targets.
- MobiOrigin does not use hard biological overrides or post-hoc probability transfer.
- Third-party biological database records are retrieved for local use and are not bundled in the Python distribution.
