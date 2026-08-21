# MobiOrigin

[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10%E2%80%933.11-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

MobiOrigin is a CPU-oriented sequence-and-marker classifier for assigning bacterial DNA fragments to chromosome, plasmid, phage, or an explicit unclassified state. The frozen dev1 candidate combines 9,557 sequence features with 17 MOB-suite-derived protein-marker features and an equal-weight ensemble of three independently trained neural networks.

MobiOrigin is the successor research classifier developed from the PlasFlow2 project. Historical PlasFlow2 pipeline documentation is preserved in [`docs/PLASFLOW2_LEGACY_README.md`](docs/PLASFLOW2_LEGACY_README.md); it is not the usage guide for the standalone MobiOrigin classifier.

## Status

- Package version: `0.1.0` (initial research release of the frozen dev1 candidate).
- Supported input length: 1,000–500,000 bp. Records outside this range remain explicitly unclassified.
- Runtime: deterministic CPU inference; 1–8 requested DIAMOND threads.
- Network access during prediction: none.
- Models: three frozen checkpoints distributed with the package. Each is governed by a cryptographic manifest and SHA-256 verified before use.
- Marker databases: not redistributed. Users must provide the exact identity-verified MOB-suite-derived research databases described in [`docs/MOBIORIGIN_DATABASE_SETUP.md`](docs/MOBIORIGIN_DATABASE_SETUP.md).

## Install from this repository

MobiOrigin currently targets Python 3.10 or 3.11 and requires DIAMOND on `PATH`.

```bash
git clone https://github.com/Raza-pl/MobiOrigin.git
cd MobiOrigin
python -m pip install .
mobiorigin --help
```

The repository URL retains its historical name until a separately authorized remote rename. The installed package and primary command are `mobiorigin`.

## Prepare marker databases

MobiOrigin does not bundle third-party biological database records. Retrieve and verify the exact research databases locally:

```bash
mobiorigin setup-databases --output-dir mobiorigin_mob_databases
```

The command publishes the directory atomically only after all three database hashes match. An offline source-directory route and complete provenance details are documented in [`docs/MOBIORIGIN_DATABASE_SETUP.md`](docs/MOBIORIGIN_DATABASE_SETUP.md). Prediction fails closed if any database hash differs.

## Run

```bash
mobiorigin predict   --input-fasta assembly.fasta   --output-dir mobiorigin_results   --database-dir /path/to/mobiorigin_mob_databases   --threads 8
```

The output directory must not already exist. Input FASTA identifiers must be unique first-token identifiers, and sequences may contain standard IUPAC DNA symbols.

## Outputs

The output directory contains:

- `predictions.tsv`: ordered per-record labels, probabilities, plasmid margin, and abstention reason.
- `provenance.json`: package version, input identity, model/database identities, threshold, and prediction identity.
- `SHA256SUMS.txt`: output checksums.

See [`docs/MOBIORIGIN_OUTPUT_SCHEMA.md`](docs/MOBIORIGIN_OUTPUT_SCHEMA.md) for the exact schema and interpretation.

## Prospective external validation

The frozen prospective external cohort contained 3,000 fragments from 3,000 distinct versioned source accessions. MobiOrigin and geNomad 1.12.0/database 1.9 received identical class-hidden FASTA bytes, and both prediction sets were frozen before label release.

| Metric | MobiOrigin | geNomad | Difference |
|---|---:|---:|---:|
| Three-class macro-F1¹ | 0.7889 | 0.7574 | +0.0315 |
| Plasmid binary F1¹ | 0.7453 | 0.6876 | +0.0578 |
| Plasmid precision² | 0.7518 | 0.8930 | −0.1412 |
| Plasmid sensitivity² | 0.7390 | 0.5590 | +0.1800 |
| Prediction coverage² | 0.9737 | 0.9993 | −0.0257 |

¹ Preregistered co-primary endpoints. Both MobiOrigin advantages had positive paired source-bootstrap 95% intervals and remained significant after Holm correction across the two endpoints.
² Descriptive metrics, not preregistered superiority endpoints.

The result supports higher macro-F1 and plasmid binary F1 on this frozen cohort. It does **not** support higher plasmid precision, higher coverage, or universal superiority. MobiOrigin traded lower precision and modestly lower coverage for substantially higher plasmid sensitivity. The external cohort is closed to retrospective tuning and record-level error mining.

Publication-facing aggregate tables, methods, figure data, and frozen claim boundaries are in [`docs/manuscript/mobiorigin_external_validation`](docs/manuscript/mobiorigin_external_validation/README.md).

## Reproducibility and scope

- External evaluation: 10,000 paired bootstrap replicates over `source_accession`, seed `20260818`.
- Multiplicity: Holm correction across three-class macro-F1 and plasmid binary F1.
- Candidate: `mobiorigin-dev1-mob-selective-v1` with frozen selective threshold `0.19835489988327026`.
- geNomad output is not used as a MobiOrigin feature or teacher.
- Hard biological overrides and post-hoc probability transfer are not used.
- The frozen external cohort cannot be used for further model or threshold tuning.

## Licensing and citation

MobiOrigin source code is distributed under GPL-3.0. Third-party MOB-suite-derived database records are not bundled and retain their own provenance and licensing conditions. The official MOB-suite repository is Apache-2.0 licensed, but this project does not assume that license establishes redistribution rights for every record in its separately hosted database archive.

Use the versioned metadata in [`CITATION.cff`](CITATION.cff) when citing the software. An archival DOI can be added after the tagged release is deposited; until then, include the repository URL, version, and commit used for analysis. Release changes are recorded in [`CHANGELOG.md`](CHANGELOG.md).

## Historical compatibility

The repository still contains the historical broader `plasflow2` source tree, but the MobiOrigin distribution intentionally installs only the standalone `mobiorigin` command. See [`docs/MOBIORIGIN_MIGRATION.md`](docs/MOBIORIGIN_MIGRATION.md) before replacing an existing PlasFlow2 workflow.
