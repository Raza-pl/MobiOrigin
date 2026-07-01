# PlasFlow v2

[![CI](https://github.com/Raza-pl/plasflow2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/Raza-pl/plasflow2.0/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Classifies metagenomic contigs as **plasmid, chromosome, or phage** and annotates each plasmid with antibiotic resistance genes (ARGs), mobility class, and an AMR risk score (0–10). Results are an interactive HTML report plus TSV files.

---

## Install

> Supports: Mac Intel · Mac M1–M5 · Linux · WSL Ubuntu

**Step 1 — get the code:**
```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0
```

**Step 2 — install everything (conda environment + tools + databases):**
```bash
bash install.sh
```

This single command creates a `plasflow2` conda environment with Python 3.10, installs all dependencies (DIAMOND, minimap2, mob-suite, geNomad), and downloads the model weights and annotation databases (~7 GB total). Takes 15–30 min depending on your connection.

**Step 3 — activate:**
```bash
conda activate plasflow2
```

> **Don't have conda?** Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.
> WSL Ubuntu: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh`

---

## Run

```bash
plasflow2 run --input assembly.fasta --output ./results/ --threads 16
```

Outputs land in `./results/`:

| File | What it is |
|---|---|
| `all_predictions.tsv` | Per-contig labels, scores, and all annotations |
| `plasmids.fasta` | Extracted plasmid sequences |
| `report_plasmid.html` | Interactive HTML report — open in browser |

Run `plasflow2 --help` or `plasflow2 run --help` for all options.

### Common variations

```bash
# Adjust AMR risk scoring for clinical samples
plasflow2 run --input assembly.fasta --output results/ --context clinical --threads 16

# Skip taxonomy to save 20–40 min on large datasets
plasflow2 run --input assembly.fasta --output results/ --skip-taxonomy --threads 16

# Quick classify only — no databases required, runs in seconds
plasflow2 classify --input assembly.fasta --output predictions.tsv

# Rebuild the HTML report without re-running the pipeline
plasflow2 report --predictions results/all_predictions.tsv --output results/
```

Context options: `clinical`, `wastewater`, `environmental`, `unspecified` (default).

---

## Performance (June 2026 benchmark)

Benchmark: 60,394 contigs from GTDB r220 genomes + PLSDB/RefSeq plasmids.

| Tool | Plasmid P | Plasmid R | Plasmid F1 |
|---|---|---|---|
| PlasFlow v1 | 0.014 | 0.198 | 0.025 |
| geNomad v1.12 | 0.060 | 0.876 | 0.112 |
| **PlasFlow v2** | **0.871** | **0.783** | **0.825** |

PLSDB-corrected F1 = **0.847**.

---

## Troubleshooting

**"No model weights found"** — The model files are not downloaded yet. Re-run:
```bash
bash scripts/setup_databases.sh --skip-plsdb --skip-card --skip-sarg \
  --skip-amrfinder --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite
```

**mob-suite / pandas / pytz conflict** — Use the conda environment, not pip directly. Run `bash install.sh` which handles this automatically.

**Python version error** — PlasFlow v2 requires Python 3.10. Run `bash install.sh` to create a dedicated conda env with the correct Python version.

**Apple Silicon (M1–M5)** — All conda packages in `environment.yml` have arm64 builds. If mob-suite fails: `pip install mob-suite && mob_init`

**WSL Ubuntu** — If `conda activate` has no effect, run `conda init bash` then restart your terminal.

**Database not found at runtime** — Re-run `bash scripts/setup_databases.sh` to download missing databases. Use `--skip-X` flags to skip ones you already have, or `--plsdb-path` / `--card-path` to point to an existing copy.

---

## Advanced: higher accuracy with geNomad

Running geNomad separately adds 12 gene-signature features to the XGBoost model:

```bash
genomad annotate assembly.fasta genomad_out/ data/databases/genomad_db/ --threads 16

plasflow2 prepare \
  --input assembly.fasta \
  --output annotations.tsv \
  --genomad-genes genomad_out/assembly_annotate/assembly_genes.tsv \
  --threads 16

plasflow2 classify \
  --input assembly.fasta \
  --output predictions.tsv \
  --annotation-tsv annotations.tsv
```

---

## Advanced: skip or reuse existing databases

```bash
# Point to databases you already have
bash scripts/setup_databases.sh \
  --plsdb-path /data/PLSDB.fna \
  --card-path  /data/card/card.dmnd

# Skip individual databases
bash scripts/setup_databases.sh --skip-plsdb --skip-iceberg --threads 16
```

Available skip flags: `--skip-models --skip-card --skip-sarg --skip-amrfinder --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-plsdb --skip-mobsuite`

---

## Docker

```bash
docker build -t plasflow2 .
docker run --rm \
  -v /path/to/data:/data \
  -v /path/to/results:/results \
  plasflow2 run --input /data/assembly.fasta --output /results/ --threads 8
```

---

## How it works

PlasFlow v2 uses a two-stage classifier:

1. **Binary MLP** (k=7 k-mer features) — fast sequence composition classifier. ~15 sec for 60k contigs on CPU.
2. **Marker XGBoost** — refines MLP scores using biological evidence: conjugation proteins, replicon type, geNomad gene signatures, ICE elements, GC content.

Plasmid calls require biological evidence (PLSDB match, relaxase, replicon type, ICE hit, or rep protein). Contigs with no evidence and length < 50 kb are returned as `unclassified`.

---

## Testing

```bash
python -m pytest tests/unit/ -q          # unit tests
python -m pytest tests/integration/ -q   # requires external tools installed
```

---

## Retraining

To retrain on your own data:

- `scripts/train_model.py` — end-to-end MLP retrain
- `scripts/train_marker_model.py` — train XGBoost on an annotation TSV
- Advanced training scripts are in `scripts/dev/`

> Apple Silicon: MPS is disabled by default (PyTorch ≤ 2.3 instability on large float32 ops). Training runs on CPU (~45–60 min for 50 epochs). Set `PLASFLOW_USE_MPS=1` to re-enable.

---

## Citation

> Krawczyk PS, Lipinski L, Dziembowski A. PlasFlow: predicting plasmid sequences in metagenomic data using genome signatures. *Nucleic Acids Research*, 2018, 46(6):e35. https://doi.org/10.1093/nar/gky044

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
