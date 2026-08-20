# Changelog

All notable MobiOrigin changes are documented here. The project uses semantic versioning for the standalone `mobiorigin` package interface.

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
- Publication-facing aggregate tables, vector figure, methods, claim boundaries, and manuscript draft.

### Scientific boundaries

- The frozen external cohort is closed to retrospective tuning and record-level error mining.
- geNomad outputs are not model features or training targets.
- MobiOrigin does not use hard biological overrides or post-hoc probability transfer.
- Third-party biological database records are retrieved for local use and are not bundled in the Python distribution.

### Historical compatibility

- The historical broader `plasflow2` source remains in the repository, but the MobiOrigin distribution intentionally installs only the standalone `mobiorigin` command.
