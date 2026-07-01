# PlasFlow v2 Wiki

Welcome to the PlasFlow v2 documentation.

## Pages

- **[Installation](Installation)** — detailed install guide for all platforms
- **[Output Files](Output-Files)** — complete reference for all output files and TSV columns
- **[AMR Risk Score](AMR-Risk-Score)** — how the 0–10 risk score is calculated
- **[Advanced Usage](Advanced-Usage)** — geNomad integration, two-step workflow, Docker
- **[Retraining](Retraining)** — how to retrain the MLP and XGBoost models on your own data

## Quick start

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0
bash install.sh
conda activate plasflow2
plasflow2 run --input assembly.fasta --output results/ --threads 16
```

Open `results/report_plasmid.html` in your browser to view the interactive report.
