# MobiOrigin 0.1.2 release notes

MobiOrigin 0.1.2 makes the complete analysis route easier to install and use.
The classifier itself is unchanged. The release integrates prediction,
independent biological annotation, and publication-quality visualization behind
one command. It also automates verified preparation of the required marker and
annotation databases.

## Main user workflow

After installation and database setup, a complete analysis is run with:

```bash
mobiorigin run \
  --input-fasta assembly.fasta \
  --output-dir mobiorigin_results \
  --threads 8
```

The command creates three clearly separated output areas:

- `predictions/` contains the four-class prediction table and provenance.
- `annotation/` contains ARG, virulence, MGE, stress-response, and mobility
  evidence with checksums and an HTML report.
- `visualization/` contains deterministic summary tables, SVG figures, and an
  integrated HTML dashboard.

Use `--skip-annotation` when only prediction and visualization are required.

## Installation and database improvements

- Guided Conda or Mamba installation supports macOS, Linux, and WSL.
- `mobiorigin doctor` checks the installed software and marker databases.
- Annotation resources are downloaded, built, and verified through
  `mobiorigin setup-databases --component annotation`.
- Downloadable resources use resumable transport and exact identity checks.
- CARD, SARG, VFDB, BacMet, mobileOG-db, AMRFinderPlus, and MOB-suite evidence
  are staged without bundling third-party database payloads in the Python wheel.
- mobileOG-db is the default MGE source. Legacy ISfinder data can be supplied
  explicitly when its separate access terms are satisfied.

## Demonstration and visualization

The release includes a small assembly example with two records for each output
class. The example validates installation, prediction, comprehensive annotation,
and visualization. It is a software demonstration and not an accuracy or
prevalence benchmark.

## Runtime control

External searches accept 1 to 128 workers through `--threads`. The useful value
depends on available CPU cores, memory, database storage speed, and the input
assembly. Neural-network inference remains deterministic and does not change with
the external-search worker count.

## Scientific continuity

Version 0.1.2 does not retrain or tune the classifier. The following identities
remain frozen:

- three model checkpoints and their equal-weight ensemble
- sequence and MOB-marker feature definitions
- normalization artifacts
- selective plasmid threshold `0.19835489988327026`
- prospective external validation and its statistical policy

Annotation evidence is reported independently. It can prioritize biological
follow-up but does not prove sequence origin, pathogenicity, transferability, or
clinical risk. Annotation results do not alter MobiOrigin predictions.
