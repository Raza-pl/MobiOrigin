# Installation

PlasFlow v2 supports Mac Intel, Mac M1–M5, Linux x86_64, and WSL Ubuntu. Python 3.10 is required.

## Choose your install path

| Option | Best for |
|--------|----------|
| **A — No-fuss** | First-time users. One command does everything. |
| **B — Conda (manual)** | Existing conda users who want control over their environment. |
| **C — Pip** | Environments where conda is not available. |
| **D — Docker** | Reproducible runs, CI, or isolated environments. |

---

## Option A — No-fuss (recommended)

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0
bash install.sh
conda activate plasflow2
```

`bash install.sh` does everything automatically:
- Creates a `plasflow2` conda environment with Python 3.10
- Installs all tools: DIAMOND, minimap2, mob-suite, geNomad
- Downloads model weights from GitHub Releases (~85 MB)
- Downloads all annotation databases (~1 GB)

Takes **15–30 min** depending on your connection.

> **No conda?** Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.
>
> WSL Ubuntu: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh` then restart your terminal.

---

## Option B — Conda (manual)

Use this if you have conda and want to set up the environment yourself.

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0

conda env create -f environment.yml
conda activate plasflow2

pip install -e .
mob_init

bash scripts/setup_databases.sh
```

`environment.yml` pins Python 3.10, installs DIAMOND, minimap2, and geNomad via conda, and pins `pytz` to fix the mob-suite/pandas compatibility issue.

---

## Option C — Pip

Use this if conda is not available. Requires Python 3.10 and system tools installed separately.

**Required system tools:**

| Tool | Install |
|------|---------|
| DIAMOND ≥ 2.1 | `conda install -c bioconda diamond` or `brew install diamond` |
| minimap2 ≥ 2.26 | `conda install -c bioconda minimap2` or `brew install minimap2` |
| mob-suite | `pip install mob-suite==3.1.9` |
| geNomad ≥ 1.7 | `conda install -c bioconda -c conda-forge genomad` |

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0

pip install -e .
mob_init

bash scripts/setup_databases.sh
```

---

## Option D — Docker

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0

docker build -t plasflow2 .

# Set up databases on the host first
bash scripts/setup_databases.sh

# Run with mounted volumes
docker run --rm \
  -v $(pwd)/data:/data \
  -v /path/to/results:/results \
  plasflow2 run \
    --input  /data/assembly.fasta \
    --output /results/ \
    --threads 8
```

The container reads databases from `/data/databases/` and models from `/data/models/`.

---

## Database setup

### Core databases

Downloaded automatically by `install.sh` or `bash scripts/setup_databases.sh`:

| Database | Size | Purpose |
|----------|------|---------|
| Model weights (MLP + XGBoost) | ~85 MB | Classification |
| CARD | ~300 MB | Antibiotic resistance genes (ARGs) |
| SARG | ~50 MB | Supplementary ARG annotation |
| AMRFinderPlus | ~30 MB | NCBI ARG annotation |
| VFDB | ~10 MB | Virulence factors |
| BacMet2 | ~5 MB | Metal and biocide resistance |
| MGE / ISfinder | ~20 MB | Mobile genetic elements |
| ICEberg3 | ~5 MB | Integrative conjugative elements |
| mob-suite | ~500 MB | Mobility typing (via `mob_init`) |

**Total: ~1 GB**

Skip specific databases:

```bash
bash scripts/setup_databases.sh --skip-vfdb --skip-bacmet --skip-mge
```

Available skip flags: `--skip-models --skip-card --skip-sarg --skip-amrfinder --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-plsdb --skip-mobsuite`

### PLSDB — plasmid database (recommended)

PLSDB is a curated database of 45,000+ plasmid sequences used by the hallmark gate to confirm plasmid calls. Without it, hallmark evidence falls back to relaxase, replicon, and ICE hits only.

**Size:** ~1 GB compressed / ~5 GB uncompressed.

```bash
# Download PLSDB only
bash scripts/setup_databases.sh \
  --skip-models --skip-card --skip-sarg --skip-amrfinder \
  --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite
```

If auto-download fails, get it manually from [https://ccb-microbe.cs.uni-saarland.de/plsdb2025/](https://ccb-microbe.cs.uni-saarland.de/plsdb2025/), decompress with `bzip2 -d plsdb.fna.bz2`, and place at `data/databases/plasmids/PLSDB.fna`.

Or symlink it via the setup script:

```bash
bash scripts/setup_databases.sh --plsdb-path /path/to/PLSDB.fna \
  --skip-models --skip-card --skip-sarg --skip-vfdb --skip-mge --skip-mobsuite
```

### GTDB — taxonomy database (optional)

GTDB enables host taxonomy annotation via DIAMOND + LCA. It is large (~8 GB index) and opt-in.

```bash
bash scripts/setup_databases.sh --gtdb --threads 16
```

See [Advanced Usage — Taxonomy](Advanced-Usage#taxonomy-annotation-gtdb) for details.

---

## What gets installed

### Tools

| Tool | Purpose | Installed via |
|------|---------|---------------|
| DIAMOND | ARG, VF, MGE, and taxonomy annotation | conda |
| minimap2 | Plasmid DB matching (PLSDB) | conda |
| mob-suite | Plasmid mobility typing | pip + `mob_init` |
| geNomad | Adds 12 gene-signature features to XGBoost | conda |

### Model weights

Downloaded from [GitHub Releases v2.0.0](https://github.com/Raza-pl/plasflow2.0/releases/tag/v2.0.0):

| File | Size | Purpose |
|------|------|---------|
| `data/models/mlp_v2.pt` | ~79 MB | MLP binary classifier (k=7 k-mer features) |
| `data/models/marker_xgb.pkl` | ~1 MB | XGBoost stage-2 model (biological markers) |
| `data/models/k6_pca.pkl` | ~4 MB | PCA feature compression |

---

## Troubleshooting

**"No model weights found"**

```bash
bash scripts/setup_databases.sh \
  --skip-plsdb --skip-card --skip-sarg --skip-amrfinder \
  --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite
```

**mob-suite / pandas / pytz conflict** — use the conda environment (`bash install.sh` or `conda env create -f environment.yml`). The `environment.yml` pins `pytz>=2022.7` explicitly.

**Python version error** — PlasFlow v2 requires Python 3.10. Run `python --version` to check. Use the conda install path which creates a dedicated 3.10 environment.

**Apple Silicon (M1–M5)** — all conda packages have native arm64 builds. If mob-suite fails:
```bash
pip install mob-suite && mob_init
```

**WSL Ubuntu — `conda activate` has no effect:**
```bash
conda init bash
# restart terminal
conda activate plasflow2
```

**PLSDB download fails** — try again (server occasionally has downtime). The script tries two mirror URLs. If both fail, download manually from [https://ccb-microbe.cs.uni-saarland.de/plsdb2025/](https://ccb-microbe.cs.uni-saarland.de/plsdb2025/).
