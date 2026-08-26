# MobiOrigin installation, prediction, annotation, and visualization tutorial

This tutorial follows one complete analysis from a new environment to an HTML report. Commands are written for macOS, Linux, or WSL2. MobiOrigin currently supports Python 3.10 and 3.11 and performs CPU inference.

## 1. Install MobiOrigin

### Recommended: guided Conda or Mamba installation

This route installs Python, a CPU-only PyTorch build, DIAMOND, Pyrodigal, and
official AMRFinderPlus. It deliberately does **not** install MOB-suite in the
same environment. MOB-suite 3.1.8 requires NumPy below 1.23.5, while MobiOrigin
requires NumPy 1.24 or newer; combining them can produce `numpy.dtype size
changed` failures.

```bash
git clone https://github.com/Raza-pl/MobiOrigin.git
cd MobiOrigin
bash install.sh
conda activate mobiorigin
mobiorigin doctor
```

The installer uses Mamba when available and otherwise Conda. It creates or
updates the `mobiorigin` runtime, builds the marker databases in the isolated
`mobiorigin-db` environment, verifies required software and database identities,
and runs a bundled synthetic example. It checks return codes explicitly and
does not use `set -e`.

The test output is created at `mobiorigin_demo/`. Open
`mobiorigin_demo/visualization/mobiorigin_dashboard.html`; inspect
`mobiorigin_demo/predictions/predictions.tsv` for the per-sequence schema and
`mobiorigin_demo/predictions/provenance.json` for reproducibility metadata. The
synthetic sequence is deliberately shorter than 1,000 bp, so the expected result
is `unclassified` with `unsupported_length`. This fast example confirms package,
model/database verification, provenance, and report generation; it is not a
biological benchmark.

AMRFinderPlus and its executable dependencies are installed in the visible
runtime. Comprehensive annotation additionally uses CARD, SARG, VFDB,
mobileOG-db, optional authorized legacy ISfinder, and BacMet research data. MobiOrigin does not silently accept
licenses, bypass registrations, or redistribute these resources; prepare them
using the exact layout in `docs/MOBIORIGIN_ANNOTATION.md`, then MobiOrigin will
validate every required file before analysis.

Useful choices:

```bash
# Install software now and prepare databases later
bash install.sh --software-only

# Use a custom database location
bash install.sh --database-dir /data/mobiorigin/marker_databases

# Put the test result in a fresh custom location
bash install.sh --demo-dir "$HOME/mobiorigin_installation_test"
```

### Windows

Use Ubuntu under WSL2 and run the Linux commands above inside the WSL terminal. Do not mix Windows Python, Windows Conda, and WSL executables in one environment. A repository cloned under the Linux home directory generally performs better than one under `/mnt/c/`.

### Python virtual environment

Use this route only when a compatible DIAMOND executable is already available. MOB-suite is still used only in the separate database-bootstrap environment described below.

```bash
git clone https://github.com/Raza-pl/MobiOrigin.git
cd MobiOrigin
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
mobiorigin --help
```

The guided database helper expects the recommended named Conda runtime. Virtual-environment users should follow the manual database route in section 2, then reactivate `.venv` before running `mobiorigin setup-databases`.

MobiOrigin is not yet published on PyPI or Bioconda. Do not use `pip install mobiorigin` or `mamba install mobiorigin` until the corresponding release page exists.

## 2. Prepare and verify the marker databases

MobiOrigin does not redistribute the MOB-suite biological sequence databases. The recommended helper keeps MOB-suite's older dependency stack isolated, retrieves its official database, and lets MobiOrigin copy only the three frozen, identity-verified files.

```bash
bash scripts/setup_mobiorigin_databases.sh \
  "${XDG_DATA_HOME:-$HOME/.local/share}/mobiorigin/marker_databases"
```

The helper creates or reuses `mobiorigin-db` for MOB-suite retrieval and a small separate `mobiorigin-marker-build` environment for DIAMOND 2.0.15. Keeping the builder separate avoids retrofitting historical DIAMOND/Boost into MOB-suite's Qt/ICU dependency stack. It runs or reuses `mob_init`, finds MOB-suite's current `databases/` raw-file layout, reconstructs the frozen indexes, then switches back to the `mobiorigin` runtime for hash verification and the final preflight. It is safe to rerun: an existing download or valid output is reused rather than overwritten. The helper also disables user-site Python packages so an unrelated `~/.local` NumPy or pandas cannot contaminate either helper.

On Apple Silicon, Bioconda does not currently provide the older BLAST build required by MOB-suite 3.1.8 as a native arm64 package. The helper therefore creates only the `mobiorigin-db` bootstrap environment as `osx-64` under Rosetta. The MobiOrigin runtime remains native arm64. If Rosetta is absent, the helper stops and prints the one-time installation command rather than changing the system automatically.

You can repeat only the preflight before analyzing real data:

```bash
mobiorigin setup-databases \
  --check \
  --output-dir "${XDG_DATA_HOME:-$HOME/.local/share}/mobiorigin/marker_databases"
```

A successful check prints `"status": "PASS"`, the DIAMOND version, and three verified database identities. The command fails if DIAMOND is missing or any file differs from the frozen hashes.

### Manual two-environment database setup

Use these steps if you prefer not to run the helper:

```bash
mamba env create -f environment.mob-database.yml
mamba env create -f environment.marker-build.yml
conda activate mobiorigin-db
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
mob_init
MOB_DATA_DIR="$(python -c 'import mob_suite, pathlib; print(pathlib.Path(mob_suite.__file__).resolve().parent / "databases")')"

conda activate mobiorigin-marker-build
python src/mobiorigin/marker_database_builder.py \
  --raw-dir "$MOB_DATA_DIR" \
  --output-dir /tmp/mobiorigin_frozen_marker_build \
  --diamond diamond

conda activate mobiorigin
mobiorigin setup-databases \
  --source-dir /tmp/mobiorigin_frozen_marker_build \
  --output-dir "$HOME/mobiorigin_databases"
mobiorigin setup-databases \
  --check \
  --output-dir "$HOME/mobiorigin_databases"
```

On an Apple Silicon Mac, replace the second environment-creation command with:

```bash
mamba env create --platform osx-64 -f environment.mob-database.yml
```

Do not run `pip install mob-suite`, force an old NumPy into `mobiorigin`, or add the historical `ursky` channel. Those workarounds can install obsolete MOB-suite releases and binary-incompatible pandas builds.

## 3. Create an analysis directory

Keep input, prediction, annotation, and visualization outputs separate.

```bash
mkdir -p mobiorigin_analysis/input
cp assembly.fasta mobiorigin_analysis/input/
cd mobiorigin_analysis
```

Input records must have unique FASTA identifiers. Supported sequence lengths are 1,000–500,000 bp. Standard IUPAC DNA ambiguity symbols are accepted.

## 4. Run prediction

```bash
mobiorigin predict \
  --input-fasta input/assembly.fasta \
  --output-dir predictions \
  --threads 8
```

The output directory is created atomically and must not already exist.

`--threads` accepts 1–128 external-search workers. Use no more than the CPUs
allocated to the job. This setting accelerates DIAMOND and related searches;
deterministic neural-network inference remains single-threaded.

Important files:

- `predictions/predictions.tsv`: chromosome, plasmid, phage, or unclassified label; three probabilities; plasmid margin; abstention reason.
- `predictions/provenance.json`: input, model, database, and threshold identities.
- `predictions/SHA256SUMS.txt`: output checksums.

## 5. Visualize predictions

Create publication-oriented TSV summaries, an editable SVG figure, and a browser-ready HTML dashboard:

```bash
mobiorigin visualize \
  --predictions-tsv predictions/predictions.tsv \
  --output-dir visualization
```

Open `visualization/mobiorigin_dashboard.html` in a browser. The figure reports:

1. contig-weighted prediction proportions;
2. base-pair-weighted prediction proportions;
3. plasmid-call fractions across five fixed contig-length bins; and
4. explicit interpretation boundaries.

The SVG can be edited in Inkscape, Illustrator, or PowerPoint:

```text
visualization/mobiorigin_summary.svg
```

The approach mirrors the useful pattern in the MetaPhlAn tutorial: first create a stable tabular result, then generate a named visualization from that table. MobiOrigin keeps the plotting step deterministic and dependency-free.

## 6. Add independent biological annotation

Annotation does not alter the frozen MobiOrigin prediction. The comprehensive profile can summarize CARD, SARG, official AMRFinderPlus, VFDB core, mobileOG-db MGE, BacMet2, replicon, relaxase, and mating-pair-formation evidence.

Download, build, and verify the complete annotation resource set once:

```bash
mobiorigin setup-databases \
  --component annotation \
  --profile comprehensive \
  --accept-third-party-terms
```

MobiOrigin retrieves permitted data from official sources. mobileOG-db is the
default MGE protein-family resource. ISfinder is neither required nor
downloaded. Offline mirrors and optional authorized legacy ISfinder imports are
documented in [`MOBIORIGIN_ANNOTATION.md`](MOBIORIGIN_ANNOTATION.md). Then run:

```bash
mobiorigin annotate \
  --input-fasta input/assembly.fasta \
  --output-dir annotations \
  --profile comprehensive \
  --predictions-tsv predictions/predictions.tsv \
  --amrfinder-mode official \
  --threads 8
```

Open `annotations/mobiorigin_report.html`. The report contains summary cards, transparent A–E evidence-priority definitions, and the highest-priority records. These tiers are a review queue—not a clinical risk score.

## 7. Visualize prediction and annotation together

```bash
mobiorigin visualize \
  --predictions-tsv predictions/predictions.tsv \
  --annotated-results-tsv annotations/mobiorigin_annotated_results.tsv \
  --output-dir annotated_visualization
```

The resulting dashboard adds evidence-tier totals while retaining the prediction, length, and base-pair summaries.

## 8. Recommended interpretation workflow

1. Check `provenance.json` and `SHA256SUMS.txt` before interpreting results.
2. Compare contig-weighted with base-pair-weighted prediction proportions.
3. Inspect length-stratified plasmid-call fractions for assembly effects.
4. Prioritize tier-A/B/C records for biological review.
5. Review ARG source agreement and mobility/MGE context.
6. Treat terminal overlap, assembly multiplicity, and homology as candidate evidence—not proof.
7. Confirm important plasmids with read mapping, an assembly graph, long reads or outward PCR, plasmid extraction, and phenotype testing where appropriate.

## 9. Troubleshooting

### `DIAMOND executable not found`

Activate the Conda environment and rerun the preflight:

```bash
conda activate mobiorigin
which diamond
mobiorigin setup-databases --check --output-dir "$HOME/mobiorigin_databases"
```

### `numpy.dtype size changed`

MOB-suite or pandas was installed into the MobiOrigin runtime environment. Recreate the two environments from their separate files:

```bash
conda deactivate
conda env remove -n mobiorigin
conda env remove -n mobiorigin-db
mamba env create -f environment.yml
bash scripts/setup_mobiorigin_databases.sh "$HOME/mobiorigin_databases_new"
```

Use a fresh database output name while diagnosing the old directory. The helper does not delete data.

### Conda solver selects CUDA packages

The repository environment pins `pytorch=2.5.1=cpu*`. If a solver still proposes CUDA, confirm you are using the current `environment.yml` and that the channels end with `nodefaults`. Do not remove the CPU build constraint.

### WSL command begins with `^[[200~`

That text is a terminal bracketed-paste control sequence, not part of the command. Press `Ctrl+C`, paste one command again into the WSL terminal, and do not include the prompt character or surrounding quotation marks.

### Database hash mismatch

Do not bypass the check. Keep the rejected directory for diagnosis and rerun the database helper with a fresh output directory. A different upstream release or DIAMOND build may not reproduce the frozen byte identities.

### Output directory already exists

MobiOrigin never silently overwrites results. Choose a fresh name, such as `predictions_run2`, or deliberately archive the previous directory first.

### Large assemblies

Prediction is CPU-oriented. Start with eight DIAMOND threads, maintain sufficient temporary disk space, and retain the final provenance and checksums with the result.

## 10. Packaging status

- Source installation, CPU runtime environment, isolated database environment, and guided database helper: available now.
- PyPI: not yet enabled. The current model-inclusive wheel is approximately 112.5 MB, above PyPI's default 100 MB per-file limit. Before publishing, either move the checkpoints to a separately verified download route or obtain a PyPI project limit increase. Trusted publishing should be configured only after that packaging decision is frozen.
- Bioconda: requires a separate recipe pull request after a public source distribution exists.

No documentation should claim PyPI or Bioconda availability before those external release steps pass.
