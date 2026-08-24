# MobiOrigin

[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10%E2%80%933.11-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

MobiOrigin is a CPU-oriented sequence-and-marker classifier for assigning bacterial DNA fragments to chromosome, plasmid, phage, or an explicit unclassified state. The frozen dev1 candidate combines 9,557 sequence features with 17 MOB-suite-derived protein-marker features and an equal-weight ensemble of three independently trained neural networks.

## Status

- Package version: `0.1.1` (publication bundle for the frozen dev1 candidate).
- Supported input length: 1,000–500,000 bp. Records outside this range remain explicitly unclassified.
- Runtime: deterministic CPU inference; 1–8 requested DIAMOND threads.
- Network access during prediction: none.
- Models: three frozen checkpoints distributed with the package. Each is governed by a cryptographic manifest and SHA-256 verified before use.
- Marker databases: not redistributed. Users must provide the exact identity-verified MOB-suite-derived research databases described in [`docs/MOBIORIGIN_DATABASE_SETUP.md`](docs/MOBIORIGIN_DATABASE_SETUP.md).

## Install

The recommended installation uses a small CPU-only runtime environment. MOB-suite is intentionally kept in a separate database-bootstrap environment because its supported NumPy range does not overlap MobiOrigin's runtime requirement.

```bash
git clone https://github.com/Raza-pl/MobiOrigin.git
cd MobiOrigin
mamba env create -f environment.yml
conda activate mobiorigin
mobiorigin --help
```

The runtime pins a cross-platform CPU build of PyTorch and does not install CUDA. Use `conda env create -f environment.yml` when Mamba is unavailable. Windows users should run these commands inside WSL2. A source-only route, WSL notes, and diagnostics are provided in the complete [installation and analysis tutorial](docs/INSTALLATION_AND_TUTORIAL.md).

MobiOrigin is not yet published on PyPI or Bioconda. The README will expose those one-line installation routes only after their external release pages exist.

## Prepare marker databases

MobiOrigin does not bundle third-party biological database records. From the repository root, one helper creates the isolated MOB-suite environment, retrieves the official data, and verifies the three frozen MobiOrigin databases:

```bash
bash scripts/setup_mobiorigin_databases.sh "$HOME/mobiorigin_databases"
```

The helper never installs MOB-suite into the `mobiorigin` runtime environment and never overwrites an existing output directory. MobiOrigin publishes its database directory atomically only after all three hashes match. Complete manual steps and provenance details are documented in [`docs/MOBIORIGIN_DATABASE_SETUP.md`](docs/MOBIORIGIN_DATABASE_SETUP.md). Prediction fails closed if any database hash differs.

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

## Visualize predictions

Create summary tables, an editable SVG figure, and a browser-ready HTML dashboard without adding plotting dependencies:

```bash
mobiorigin visualize \
  --predictions-tsv mobiorigin_results/predictions.tsv \
  --output-dir mobiorigin_visualization
```

Open `mobiorigin_visualization/mobiorigin_dashboard.html`. The dashboard reports contig- and base-pair-weighted prediction proportions and length-stratified plasmid calls. It is descriptive and does not calculate accuracy.

## Independent biological annotation

`mobiorigin annotate` adds protein-level biological evidence after
classification. It never changes MobiOrigin labels, probabilities, or the
selective threshold. The ARG profile retains independent CARD, SARG, and
official AMRFinderPlus evidence. The comprehensive profile additionally reports
AMRFinderPlus virulence/stress calls, VFDB core homologs, curated MGE evidence,
BacMet2 biocide/metal-resistance homologs, and MOB-suite replication, relaxase,
and mating-pair-formation markers.

```bash
mobiorigin annotate \
  --input-fasta assembly.fasta \
  --output-dir mobiorigin_annotations \
  --database-dir /path/to/annotation_databases \
  --profile comprehensive \
  --predictions-tsv mobiorigin_predictions/predictions.tsv \
  --amrfinder-mode official \
  --amrfinder-database /path/to/amrfinderplus/database/version \
  --threads 8
```

The integrated table, machine-readable summary, self-contained HTML report,
raw evidence, and every consumed database identity are published atomically.
The A–E evidence-priority tier is a transparent review queue based on ARG and
mobility context; it is not a clinical risk score. Database layout, thresholds,
provenance, licensing boundaries, and output schemas are documented in
[`docs/MOBIORIGIN_ANNOTATION.md`](docs/MOBIORIGIN_ANNOTATION.md).

To add evidence-tier summaries to the visualization:

```bash
mobiorigin visualize \
  --predictions-tsv mobiorigin_results/predictions.tsv \
  --annotated-results-tsv mobiorigin_annotations/mobiorigin_annotated_results.tsv \
  --output-dir mobiorigin_annotated_visualization
```

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

Aggregate validation tables, methods, figure data, and frozen claim boundaries are in [`docs/validation/external_validation`](docs/validation/external_validation/README.md).

### Additional exploratory comparator analysis

The prospective MobiOrigin evaluation used geNomad as the preregistered comparator. After that analysis was complete, predictions from PlasClass, PlasFlow v1, PLASMe, and Platon were frozen on the same 3,000-record external cohort and evaluated under a separate post-hoc exploratory contract.

| Tool | Plasmid F1 | Balanced accuracy | Precision | Sensitivity | Coverage |
|---|---:|---:|---:|---:|---:|
| MobiOrigin | 0.7453 | 0.8085 | 0.7518 | 0.7390 | 0.9737 |
| PlasClass | 0.6070 | 0.6943 | 0.5070 | 0.7560 | 1.0000 |
| PlasFlow v1 | 0.5788 | 0.6835 | 0.5737 | 0.5840 | 0.6987 |
| PLASMe | 0.3635 | 0.6093 | 0.9454 | 0.2250 | 1.0000 |
| Platon | 0.5774 | 0.7023 | 0.9649 | 0.4120 | 1.0000 |

MobiOrigin had the highest F1 and balanced accuracy in this secondary comparison, and all eight paired MobiOrigin-minus-comparator tests were positive after Holm adjustment. These findings are exploratory, not additional preregistered co-primary evidence. PlasClass had slightly higher sensitivity, while PLASMe and Platon had higher precision but substantially lower sensitivity. Full methods, paired intervals, and mandatory claim limitations are included in the validation-evidence directory linked above.

<!-- BEGIN REAL-ASSEMBLY OPERATIONAL VALIDATION -->
## Real-assembly operational validation

MobiOrigin and five comparators were run on two deterministic real-assembly subsets containing 2,488 and 2,445 records (approximately 15.5 Mb each). All 12 dataset–tool routes completed. The publication bundle reports end-to-end runtime, call fraction, coverage, label-free agreement, and biological evidence-priority tiers. These assemblies do not have frozen record-level ground truth, so this analysis does **not** support accuracy or superiority claims. Evidence tiers A–E prioritize follow-up and are not clinical risk scores.

Tables, an editable SVG figure, methods, and mandatory claim boundaries are available in [`docs/validation/operational_validation`](docs/validation/operational_validation).
<!-- END REAL-ASSEMBLY OPERATIONAL VALIDATION -->

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
