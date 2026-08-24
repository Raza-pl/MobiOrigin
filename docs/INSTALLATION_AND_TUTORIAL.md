# MobiOrigin installation, prediction, annotation, and visualization tutorial

This tutorial follows one complete analysis from a new environment to an HTML report. Commands are written for macOS or Linux. MobiOrigin currently supports Python 3.10 and 3.11 and performs CPU inference.

## 1. Install MobiOrigin

### Recommended: Conda or Mamba environment

This route installs Python, PyTorch, DIAMOND, Pyrodigal, and MOB-suite together.

```bash
git clone https://github.com/Raza-pl/MobiOrigin.git
cd MobiOrigin
mamba env create -f environment.yml
conda activate mobiorigin
mobiorigin --help
```

Use `conda env create -f environment.yml` if Mamba is unavailable.

### Python virtual environment

Use this route only when DIAMOND and MOB-suite are already installed separately.

```bash
git clone https://github.com/Raza-pl/MobiOrigin.git
cd MobiOrigin
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
mobiorigin --help
```

MobiOrigin is not yet published on PyPI or Bioconda. Do not use `pip install mobiorigin` or `mamba install mobiorigin` until the corresponding release page exists.

## 2. Prepare and verify the marker databases

MobiOrigin does not redistribute the MOB-suite biological sequence databases. Retrieve them through MOB-suite and let MobiOrigin copy only the three frozen, identity-verified marker databases.

```bash
mob_init
MOB_DATA_DIR="$(python -c 'import mob_suite, pathlib; print(pathlib.Path(mob_suite.__file__).parent / "data")')"
mobiorigin setup-databases \
  --source-dir "$MOB_DATA_DIR" \
  --output-dir "$HOME/mobiorigin_databases"
```

Run the preflight check before analyzing real data:

```bash
mobiorigin setup-databases \
  --check \
  --output-dir "$HOME/mobiorigin_databases"
```

A successful check prints `"status": "PASS"`, the DIAMOND version, and three verified database identities. The command fails if DIAMOND is missing or any file differs from the frozen hashes.

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
  --database-dir "$HOME/mobiorigin_databases" \
  --threads 8
```

The output directory is created atomically and must not already exist.

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

Annotation does not alter the frozen MobiOrigin prediction. The comprehensive profile can summarize CARD, SARG, official AMRFinderPlus, VFDB core, curated MGE, BacMet2, replicon, relaxase, and mating-pair-formation evidence.

The annotation database directory must follow [`MOBIORIGIN_ANNOTATION.md`](MOBIORIGIN_ANNOTATION.md). Then run:

```bash
mobiorigin annotate \
  --input-fasta input/assembly.fasta \
  --output-dir annotations \
  --database-dir /path/to/mobiorigin_annotation_databases \
  --profile comprehensive \
  --predictions-tsv predictions/predictions.tsv \
  --amrfinder-mode official \
  --amrfinder-database /path/to/amrfinderplus/database/version \
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

### Database hash mismatch

Do not bypass the check. Remove the incomplete MobiOrigin database output, rerun `mob_init`, and create a fresh output directory with `setup-databases`.

### Output directory already exists

MobiOrigin never silently overwrites results. Choose a fresh name, such as `predictions_run2`, or deliberately archive the previous directory first.

### Large assemblies

Prediction is CPU-oriented. Start with eight DIAMOND threads, maintain sufficient temporary disk space, and retain the final provenance and checksums with the result.

## 10. Packaging status

- Source installation and `environment.yml`: available now.
- PyPI: not yet enabled. The current model-inclusive wheel is approximately 112.5 MB, above PyPI's default 100 MB per-file limit. Before publishing, either move the checkpoints to a separately verified download route or obtain a PyPI project limit increase. Trusted publishing should be configured only after that packaging decision is frozen.
- Bioconda: requires a separate recipe pull request after a public source distribution exists.

No documentation should claim PyPI or Bioconda availability before those external release steps pass.
