# PlasFlow v2

[![CI](https://github.com/Raza-pl/plasflow2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/Raza-pl/plasflow2.0/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Classifies metagenomic contigs as **plasmid, chromosome, or phage** and annotates each plasmid with antibiotic resistance genes (ARGs), mobility class, and an AMR risk score (0–10). Results are an interactive HTML report plus TSV files.

> Supports: **Mac Intel · Mac M1–M5 · Linux · WSL Ubuntu**

---

## Table of Contents

- [How it works](#how-it-works)
- [Choose your install](#choose-your-install)
  - [Option A — No-fuss (recommended)](#option-a--no-fuss-recommended)
  - [Option B — Conda (manual)](#option-b--conda-manual)
  - [Option C — Pip](#option-c--pip)
  - [Option D — Docker](#option-d--docker)
- [Database setup](#database-setup)
  - [Core databases](#core-databases)
  - [PLSDB — plasmid database](#plsdb--plasmid-database)
- [Run](#run)
- [Understanding the output](#understanding-the-output)
- [Lenient mode](#lenient-mode)
- [Performance benchmark](#performance-benchmark)
- [Troubleshooting](#troubleshooting)
- [Advanced](#advanced)
  - [Taxonomy annotation (GTDB)](#taxonomy-annotation-gtdb)
  - [Higher accuracy with geNomad](#higher-accuracy-with-genomad)
  - [Rebuild the HTML report](#rebuild-the-html-report)
  - [Skip or reuse existing databases](#skip-or-reuse-existing-databases)
- [Retraining](#retraining)
- [Citation](#citation)

---

## How it works

PlasFlow v2 uses a two-stage classifier:

1. **Binary MLP** (k=7 k-mer features) — fast sequence-composition classifier. Processes 60,000 contigs in ~15 seconds on CPU.
2. **Marker XGBoost** — refines MLP scores using biological evidence: conjugation proteins, replicon type, geNomad gene signatures, ICE elements, and GC content.

**Hallmark gate** — a plasmid call on any contig shorter than 50 kb must be backed by at least one biological hallmark: a PLSDB match, relaxase, replicon type, ICE hit, or rep protein. Contigs with high MLP scores but no hallmarks are returned as `unclassified`. Use `--lenient` to skip this gate (see [Lenient mode](#lenient-mode)).

---

## Choose your install

| Option | Best for |
|--------|----------|
| **A — No-fuss** | First-time users. One command does everything. |
| **B — Conda (manual)** | Existing conda users who want control over their environment. |
| **C — Pip** | Environments where conda is not available. |
| **D — Docker** | Reproducible runs, CI, or isolated environments. |

---

### Option A — No-fuss (recommended)

Three commands. Everything is handled automatically: conda environment, dependencies, model weights, and annotation databases.

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0
bash install.sh
```

This creates a `plasflow2` conda environment with Python 3.10, installs all required tools (DIAMOND, minimap2, mob-suite, geNomad), downloads model weights, and downloads all annotation databases. Takes **15–30 min** depending on your connection.

After install:

```bash
conda activate plasflow2
plasflow2 --help
```

> **No conda?** Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.
> On WSL Ubuntu: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh`

> **PLSDB** is a 5 GB plasmid database that substantially improves hallmark detection. `install.sh` does **not** download it by default. See [PLSDB setup](#plsdb--plasmid-database) below.

---

### Option B — Conda (manual)

Use this if you already have conda and want to set up the environment yourself, or if `install.sh` fails on your system.

**Step 1 — Clone and create the environment:**

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0

conda env create -f environment.yml   # creates the 'plasflow2' env
conda activate plasflow2
```

The `environment.yml` installs Python 3.10, DIAMOND, minimap2, geNomad, and all Python dependencies.

**Step 2 — Install PlasFlow v2:**

```bash
pip install -e .
```

**Step 3 — Initialise mob-suite:**

mob-suite requires a one-time database download after installation:

```bash
mob_init
```

**Step 4 — Download annotation databases:**

```bash
bash scripts/setup_databases.sh
```

This downloads CARD, SARG, AMRFinderPlus, VFDB, BacMet2, MGE, ICEberg, and model weights (~500 MB total). See [Database setup](#database-setup) for what each database does and how to skip ones you don't need.

**Step 5 — Add PLSDB (recommended):**

See [PLSDB setup](#plsdb--plasmid-database).

---

### Option C — Pip

Use this if conda is not available and you already have Python 3.10 with the required system tools.

**System tools required on PATH before you start:**

| Tool | Install (Mac/Linux) |
|------|---------------------|
| DIAMOND ≥ 2.1 | `conda install -c bioconda diamond` or `brew install diamond` |
| minimap2 ≥ 2.26 | `conda install -c bioconda minimap2` or `brew install minimap2` |
| mob-suite | `pip install mob-suite==3.1.9` |
| geNomad ≥ 1.7 | `conda install -c bioconda -c conda-forge genomad` |

**Step 1 — Clone and install:**

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0

pip install -e .
```

> PlasFlow v2 requires **Python 3.10** exactly (3.11 has compatibility issues with some mob-suite dependencies). Check with `python --version`.

**Step 2 — Initialise mob-suite:**

```bash
mob_init
```

**Step 3 — Download annotation databases:**

```bash
bash scripts/setup_databases.sh
```

**Step 4 — Add PLSDB (recommended):**

See [PLSDB setup](#plsdb--plasmid-database).

---

### Option D — Docker

The Docker image bundles all Python dependencies and DIAMOND. You still need to provide databases via a mounted volume.

**Step 1 — Build the image:**

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0

docker build -t plasflow2 .
```

**Step 2 — Download databases on the host:**

The databases live on your machine and are mounted into the container at runtime. If you haven't run the installer, download them now:

```bash
# Run outside Docker, inside the cloned repo
bash scripts/setup_databases.sh
```

This writes everything to `data/databases/` and `data/models/` inside the repo. PLSDB can be added at any time — see [PLSDB setup](#plsdb--plasmid-database).

**Step 3 — Run:**

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v /path/to/results:/results \
  plasflow2 run \
    --input  /data/assembly.fasta \
    --output /results/ \
    --threads 8
```

The container reads databases from `/data/databases/` and models from `/data/models/`. If your databases are in a different host path, adjust the `-v` mount accordingly.

---

## Database setup

### Core databases

These are downloaded automatically by `install.sh` or `bash scripts/setup_databases.sh`:

| Database | Purpose | Size |
|----------|---------|------|
| Model weights (MLP + XGBoost) | Classification | ~85 MB |
| CARD | Antibiotic resistance genes (ARGs) | ~300 MB |
| SARG | ARG annotation (second opinion) | ~50 MB |
| AMRFinderPlus | NCBI ARG annotation | ~30 MB |
| VFDB | Virulence factors | ~10 MB |
| BacMet2 | Metal and biocide resistance | ~5 MB |
| MGE / ISfinder | Mobile genetic elements | ~20 MB |
| ICEberg3 | Integrative conjugative elements | ~5 MB |
| mob-suite | Mobility typing (via `mob_init`) | ~500 MB |

**Total (core): ~1 GB**

You can skip individual databases:

```bash
bash scripts/setup_databases.sh --skip-vfdb --skip-bacmet --skip-mge
```

Available skip flags: `--skip-models --skip-card --skip-sarg --skip-amrfinder --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-plsdb --skip-mobsuite`

If you have a database elsewhere, point to it instead of downloading:

```bash
bash scripts/setup_databases.sh --card-path /existing/card/card.dmnd --plsdb-path /existing/PLSDB.fna
```

---

### PLSDB — plasmid database

PLSDB is a curated database of 45,000+ complete plasmid sequences. PlasFlow v2 uses it during the hallmark gate to confirm plasmid calls via minimap2 alignment. Without PLSDB, the hallmark gate relies only on relaxase, replicon type, ICE hits, and rep proteins — still functional, but less sensitive.

**Size:** ~1 GB compressed, ~5 GB decompressed.

**Manual download:** If automatic download fails, get it directly from [https://ccb-microbe.cs.uni-saarland.de/plsdb2025/](https://ccb-microbe.cs.uni-saarland.de/plsdb2025/) — download `plsdb.fna.bz2`, decompress with `bzip2 -d plsdb.fna.bz2`, and place the `.fna` file at `data/databases/plasmids/PLSDB.fna`.

#### Download PLSDB automatically

The easiest way is to let `setup_databases.sh` handle it:

```bash
bash scripts/setup_databases.sh --skip-models --skip-card --skip-sarg \
  --skip-amrfinder --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite
```

This downloads and decompresses PLSDB only, saving it to `data/databases/plasmids/PLSDB.fna`. Once it is there, PlasFlow v2 **detects it automatically** — no extra flags needed at runtime.

Expected download time: **20–60 min** depending on connection speed.

#### Use an existing PLSDB file

If you already have PLSDB downloaded elsewhere:

```bash
bash scripts/setup_databases.sh --plsdb-path /path/to/your/PLSDB.fna \
  --skip-models --skip-card --skip-sarg --skip-amrfinder \
  --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite
```

This creates a symlink at `data/databases/plasmids/PLSDB.fna` pointing to your existing file.

#### Pass PLSDB path at runtime

If you don't want to symlink it, pass the path directly when running:

```bash
plasflow2 run --input assembly.fasta --output results/ \
  --plsdb-path /path/to/PLSDB.fna
```

#### Run without PLSDB

PLSDB is optional. Without it, the hallmark gate falls back to relaxase, replicon, ICE, and rep protein evidence. For exploratory analysis or when databases aren't available, use `--lenient` to skip the hallmark gate entirely:

```bash
plasflow2 run --input assembly.fasta --output results/ --lenient
```

---

## Run

**Basic run** (auto-detects databases from `data/databases/`):

```bash
plasflow2 run --input assembly.fasta --output ./results/ --threads 16
```

**With PLSDB from a non-default path:**

```bash
plasflow2 run --input assembly.fasta --output ./results/ \
  --plsdb-path /data/PLSDB.fna --threads 16
```

**Clinical context** — adjusts AMR risk scoring for healthcare samples:

```bash
plasflow2 run --input assembly.fasta --output ./results/ \
  --context clinical --threads 16
```

**Skip taxonomy annotation** — saves 20–40 min on large datasets:

```bash
plasflow2 run --input assembly.fasta --output ./results/ \
  --skip-taxonomy --threads 16
```

**Quick classify only** — no databases, runs in seconds:

```bash
plasflow2 classify --input assembly.fasta --output predictions.tsv
```

Context options: `clinical`, `wastewater`, `environmental`, `unspecified` (default).

Run `plasflow2 run --help` for all options.

---

## Understanding the output

After a run, the `results/` directory contains:

| File | Description |
|------|-------------|
| `all_predictions.tsv` | Every contig: label, confidence score, ARGs, mobility class, risk score |
| `plasmids.fasta` | Extracted plasmid sequences |
| `chromosomes.fasta` | Extracted chromosome sequences |
| `phages.fasta` | Extracted phage sequences |
| `report_plasmid.html` | Interactive plasmid report — open in any browser |
| `report_chromosome.html` | Chromosome report with ARG annotation |
| `report_phage.html` | Phage report |
| `report_unclassified.html` | Contigs with no confident call |

Open `report_plasmid.html` in your browser for:
- Per-plasmid ARG, mobility class, and risk score breakdown
- Drug class distribution chart
- Risk score histogram
- Sortable contig table

---

## Lenient mode

By default PlasFlow v2 requires biological evidence to confirm a plasmid call (`--lenient` off):

- Contigs ≥ 50 kb: MLP score alone is sufficient.
- Contigs < 50 kb: must have a PLSDB match, relaxase, replicon type, ICE hit, or rep protein.

Use `--lenient` when databases are not set up, or for exploratory analysis where sensitivity matters more than precision:

```bash
plasflow2 run --input assembly.fasta --output results/ --lenient
```

`--lenient` does two things:
1. Lowers the MLP plasmid threshold from 0.95 → 0.70 (catches weaker signals).
2. Skips the hallmark gate entirely (no biological evidence required).

Expect **more plasmid calls** and **more false positives** compared to the default mode.

---

## Performance benchmark

Benchmark: 60,394 contigs from GTDB r220 genomes + PLSDB/RefSeq plasmids (June 2026).

| Tool | Plasmid Precision | Plasmid Recall | Plasmid F1 |
|------|:-----------------:|:--------------:|:----------:|
| PlasFlow v1 | 0.014 | 0.198 | 0.025 |
| geNomad v1.12 | 0.060 | 0.876 | 0.112 |
| **PlasFlow v2** | **0.871** | **0.783** | **0.825** |

PLSDB-corrected F1 = **0.847**.

---

## Troubleshooting

**"No model weights found"**

Model files were not downloaded. Run:

```bash
bash scripts/setup_databases.sh --skip-plsdb --skip-card --skip-sarg \
  --skip-amrfinder --skip-vfdb --skip-bacmet --skip-mge --skip-iceberg --skip-mobsuite
```

**mob-suite / pandas / pytz conflict**

Use the conda environment — `install.sh` or `conda env create -f environment.yml` pins the correct versions. Avoid installing mob-suite with `pip` in a bare Python environment.

**Python version error**

PlasFlow v2 requires Python 3.10. Run `python --version` to check. If needed, use the conda install path which creates a dedicated 3.10 environment.

**Apple Silicon (M1–M5)**

All conda packages in `environment.yml` have arm64 builds. If mob-suite fails after conda install, try:

```bash
pip install mob-suite && mob_init
```

**WSL Ubuntu — `conda activate` has no effect**

Run `conda init bash` then restart your terminal (close and reopen the WSL window).

**Database not found at runtime**

Re-run `bash scripts/setup_databases.sh` for the missing database. Use `--skip-X` for databases you already have, or `--plsdb-path` / `--card-path` to point to existing files.

**PLSDB download fails (connection timeout)**

PLSDB servers occasionally have downtime. Try again after a few minutes. The script tries two mirror URLs automatically. If automatic download keeps failing, download manually:

1. Go to **[https://ccb-microbe.cs.uni-saarland.de/plsdb2025/](https://ccb-microbe.cs.uni-saarland.de/plsdb2025/)** and download the FASTA file (`plsdb.fna.bz2`)
2. Decompress: `bzip2 -d plsdb.fna.bz2`
3. Place the file at `data/databases/plasmids/PLSDB.fna` inside the cloned repo, or pass the path directly: `plasflow2 run --plsdb-path /path/to/PLSDB.fna`

**Docker: "database not found" inside container**

The container reads from `/data/databases/`. Make sure you mount your host `data/` directory: `-v $(pwd)/data:/data`. Run `bash scripts/setup_databases.sh` on the host before running the container.

---

## Advanced

### Taxonomy annotation (GTDB)

By default, `plasflow2 run` annotates the host taxonomy of each plasmid contig using **DIAMOND + a Kaiju-style LCA (lowest common ancestor) algorithm**. It runs DIAMOND against a GTDB protein database, collects the top 10 hits per contig, and walks from species up to domain — assigning the deepest taxonomic rank where ≥ 50% of hits agree.

The taxonomy database is **not bundled** and must be built once from GTDB data. It is large (~8 GB DIAMOND index). Skip taxonomy entirely with `--skip-taxonomy` if you don't need it (saves 20–40 min per run).

#### Step 1 — Download GTDB r220 representative proteins

```bash
mkdir -p data/databases/taxonomy
cd data/databases/taxonomy

wget https://data.ace.uq.edu.au/public/gtdb/data/releases/release220/220.0/genomic_files_reps/gtdb_proteins_aa_reps_r220.tar.gz
tar xf gtdb_proteins_aa_reps_r220.tar.gz
```

> **Size:** ~8 GB download, ~30 GB uncompressed. This step takes 30–90 min depending on your connection.

#### Step 2 — Build the DIAMOND database

```bash
# Still inside data/databases/taxonomy/
diamond makedb \
  --in gtdb_prot_reps_r220.faa \
  --db refseq_taxonomy \
  --threads 16
```

This creates `refseq_taxonomy.dmnd` (~8 GB). PlasFlow v2 **auto-detects** it at `data/databases/taxonomy/refseq_taxonomy.dmnd` — no extra flags needed at runtime.

#### Step 3 — Build the taxon map (recommended)

The taxon map improves LCA accuracy by providing explicit accession → lineage lookups rather than relying on DIAMOND header parsing.

Download the GTDB taxonomy metadata:

```bash
wget https://data.ace.uq.edu.au/public/gtdb/data/releases/release220/220.0/bac120_taxonomy_r220.tsv.gz
gunzip bac120_taxonomy_r220.tsv.gz
```

Build the map with PlasFlow v2's built-in helper:

```bash
python -c "
from plasflow2.annotate.taxonomy import build_gtdb_taxon_map
build_gtdb_taxon_map(
    'bac120_taxonomy_r220.tsv',
    'taxon_map.tsv'
)
"
```

This creates `taxon_map.tsv` in `data/databases/taxonomy/`. PlasFlow v2 auto-detects it there.

#### Step 4 — Run with taxonomy

Once the database and map are in place, taxonomy runs automatically:

```bash
plasflow2 run --input assembly.fasta --output results/ --threads 16
```

To use databases at a non-default location:

```bash
plasflow2 run --input assembly.fasta --output results/ \
  --taxonomy-db /path/to/refseq_taxonomy.dmnd \
  --taxon-map   /path/to/taxon_map.tsv \
  --threads 16
```

To skip taxonomy entirely:

```bash
plasflow2 run --input assembly.fasta --output results/ --skip-taxonomy --threads 16
```

> **Note:** PLSDB and GTDB serve different purposes. PLSDB is used by the **hallmark gate** to confirm plasmid calls via minimap2 alignment. GTDB is used for **host taxonomy annotation** via DIAMOND protein search. They are independent — you can use either, both, or neither.

---

### Higher accuracy with geNomad

Running geNomad separately before classification adds 12 gene-signature features to the XGBoost model, which can improve accuracy on novel plasmids:

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

When you use the standard `plasflow2 run` command, geNomad is invoked automatically if it is on your PATH.

### Rebuild the HTML report

Re-generate reports from an existing predictions TSV without re-running the pipeline:

```bash
plasflow2 report --predictions results/all_predictions.tsv --output results/
```

### Skip or reuse existing databases

```bash
# Point to a database you already downloaded
bash scripts/setup_databases.sh \
  --plsdb-path /data/PLSDB.fna \
  --card-path  /data/card/card.dmnd

# Skip specific databases
bash scripts/setup_databases.sh --skip-plsdb --skip-vfdb --skip-bacmet
```

---

## Testing

```bash
# Unit tests (no external tools needed)
python -m pytest tests/unit/ -q

# Integration tests (requires DIAMOND, mob-suite, etc.)
python -m pytest tests/integration/ -q
```

---

## Retraining

To retrain on your own data:

- `scripts/train_model.py` — end-to-end MLP retrain
- `scripts/train_marker_model.py` — train XGBoost on an annotation TSV
- Advanced training and benchmark scripts are in `scripts/dev/` (local-only, not tracked in git)

> **Apple Silicon note:** MPS is disabled by default (PyTorch ≤ 2.3 instability on large float32 ops). Training runs on CPU (~45–60 min for 50 epochs). Set `PLASFLOW_USE_MPS=1` to re-enable.

---

## Citation

If you use PlasFlow v2 in published work, please cite the original PlasFlow paper:

> Krawczyk PS, Lipinski L, Dziembowski A. PlasFlow: predicting plasmid sequences in metagenomic data using genome signatures. *Nucleic Acids Research*, 2018, 46(6):e35. https://doi.org/10.1093/nar/gky044

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
