# PlasFlow v2 Wiki

Welcome to the PlasFlow v2 documentation. **Current version: 2.0.0-beta**

## Pages

- **[Installation](Installation)** — all four install paths: no-fuss bash, conda, pip, Docker
- **[Output Files](Output-Files)** — complete reference for all output files and TSV columns
- **[AMR Risk Score](AMR-Risk-Score)** — how the 0–10 risk score is calculated
- **[Advanced Usage](Advanced-Usage)** — lenient mode, GTDB taxonomy, geNomad integration, Docker
- **[Retraining](Retraining)** — how to retrain the MLP and XGBoost models on your own data

## Quick start

```bash
git clone https://github.com/Raza-pl/MobiOrigin
cd MobiOrigin
bash install.sh
conda activate plasflow2
plasflow2 run --input assembly.fasta --output results/ --threads 16
```

Open `results/report_plasmid.html` in your browser to view the interactive report.

## No databases yet?

Run without databases for a quick classification (no ARG annotation or hallmark gate):

```bash
plasflow2 classify --input assembly.fasta --output predictions.tsv
```

Or use `--lenient` to accept all MLP plasmid predictions without requiring biological evidence:

```bash
plasflow2 run --input assembly.fasta --output results/ --lenient
```

See [Advanced Usage](Advanced-Usage) for details on lenient mode.
