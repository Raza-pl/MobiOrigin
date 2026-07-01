# Installation

PlasFlow v2 supports Mac Intel, Mac M1–M5, Linux x86_64, and WSL Ubuntu.

## Recommended: one-command install

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0
bash install.sh
conda activate plasflow2
```

`bash install.sh` does everything:
- Creates a conda environment (`plasflow2`) with Python 3.10
- Installs all tools: DIAMOND, minimap2, mob-suite, geNomad
- Downloads model weights from GitHub Releases (~84 MB)
- Downloads all annotation databases (~7 GB total)

**Don't have conda?** Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

WSL Ubuntu:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# restart terminal, then run the commands above
```

---

## Manual install (step by step)

### Step 1 — Create the conda environment

```bash
conda env create -f environment.yml
conda activate plasflow2
```

The `environment.yml` pins Python 3.10, installs all external tools, and explicitly pins `pytz` to resolve the mob-suite/pandas compatibility issue.

### Step 2 — Install the PlasFlow v2 package

```bash
pip install -e .
```

### Step 3 — Download databases and model weights

```bash
bash scripts/setup_databases.sh
```

#### Skip flags

Skip any database you don't need or already have:

```bash
bash scripts/setup_databases.sh --skip-plsdb --skip-iceberg
```

Available flags: `--skip-models --skip-card --skip-sarg --skip-amrfinder --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-plsdb --skip-mobsuite`

#### Reuse existing databases

```bash
bash scripts/setup_databases.sh \
  --card-path  /path/to/existing/card.dmnd \
  --plsdb-path /path/to/existing/PLSDB.fna
```

---

## What gets installed

### Tools

| Tool | Purpose | Installed via |
|---|---|---|
| DIAMOND | ARG, VF, MGE, and taxonomy annotation | conda |
| minimap2 | Closest known plasmid match (PLSDB/RefSeq) | conda |
| MOB-suite | Plasmid mobility typing (conjugative/mobilizable) | conda |
| geNomad | Optional — adds 12 gene-signature features to XGBoost | conda |

### Databases

| Database | Size | Purpose |
|---|---|---|
| CARD | ~300 MB | Antibiotic resistance genes (primary source) |
| SARG | ~50 MB | Supplementary ARG database |
| AMRFinderPlus | ~30 MB | NCBI ARG database (third source) |
| VFDB set A | ~10 MB | Experimentally validated virulence factors |
| BacMet2 | ~5 MB | Biocide and metal resistance genes |
| MGE/ISfinder | ~20 MB | IS elements, transposons, integrons |
| ICEberg3 | ~5 MB | Integrative conjugative elements (optional) |
| PLSDB | ~5 GB | Curated plasmid sequences for closest-match lookup |
| MOB-suite DBs | ~500 MB | Replicon and relaxase typing databases |

### Model weights

Downloaded from [GitHub Releases v2.0.0](https://github.com/Raza-pl/plasflow2.0/releases/tag/v2.0.0):

| File | Size | Purpose |
|---|---|---|
| `data/models/mlp_v2.pt` | ~79 MB | MLP binary classifier (k=7 k-mer features) |
| `data/models/marker_xgb.pkl` | ~1 MB | XGBoost stage-2 model (biological markers) |
| `data/models/k6_pca.pkl` | ~4 MB | PCA feature compression |

---

## Troubleshooting

**"No model weights found"**

The model files didn't download. Re-run setup for models only:
```bash
bash scripts/setup_databases.sh \
  --skip-plsdb --skip-card --skip-sarg --skip-amrfinder \
  --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite
```

**mob-suite / pandas / pytz conflict**

Use the conda environment from `environment.yml`, not a raw pip install. Run `bash install.sh` which handles this automatically. The `environment.yml` pins `pytz>=2022.7` explicitly because mob-suite needs it but pandas 2.x no longer depends on it.

**Python version error**

PlasFlow v2 requires Python 3.10. `bash install.sh` creates a conda environment with exactly Python 3.10 regardless of your system Python.

**Apple Silicon (M1–M5)**

All packages in `environment.yml` have native arm64 conda builds. If mob-suite fails via conda:
```bash
pip install mob-suite && mob_init
```

**WSL Ubuntu**

If `conda activate` has no effect:
```bash
conda init bash
# restart your terminal
conda activate plasflow2
```
