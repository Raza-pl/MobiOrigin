# PlasFlow v2

[![CI](https://github.com/Raza-pl/plasflow2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/Raza-pl/plasflow2.0/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**PlasFlow v2** classifies metagenomic contigs as **plasmid, chromosome, phage, or archaea** and annotates each plasmid with antibiotic resistance genes (ARGs), virulence factors, mobile genetic elements (MGEs), mobility class, and an AMR risk score (0–10). Results are delivered as an interactive HTML report and structured TSV files.

This is a complete rewrite of [PlasFlow v1](https://github.com/smaegol/PlasFlow) (Krawczyk et al., *Nucleic Acids Research* 2018) on a modern Python/PyTorch stack.

---

## Quick start

```bash
# Install
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0
pip install poetry && poetry install

# Set up databases (one-time, ~15–30 min)
bash scripts/setup_databases.sh

# Run
plasflow2 run --input assembly.fasta --output ./results/ --threads 16
```

Outputs land in `./results/`: classification TSV, FASTA per class, ARG/VF/mobility annotations, and an HTML report.

---

## What it does

PlasFlow v2 uses a **two-stage classifier**:

1. **Binary MLP** (k=7 k-mer features, 9,557 dims) — fast sequence composition classifier, ~15 sec for 60k contigs on CPU.
2. **Marker XGBoost** — refines MLP scores with biological evidence: conjugation proteins, replicon type, geNomad gene signatures, ICE elements, coding density, GC content. Auto-activates when `data/models/marker_xgb.pkl` is present.

Plasmid calls additionally require biological evidence (PLSDB match, relaxase, replicon type, ICE hit, or rep protein). Contigs with no evidence and length < 50 kb are returned as `unclassified` rather than forced into a class.

### Performance (June 2026 benchmark)

Benchmark: 60,394 contigs (394 true plasmids + 60,000 chromosome windows, min 1 kb), built from GTDB r220 genomes + PLSDB/RefSeq plasmids.

| Tool | Plasmid P | Plasmid R | Plasmid F1 |
|---|---|---|---|
| PlasFlow v1 | 0.014 | 0.198 | 0.025 |
| geNomad v1.12 | 0.060 | 0.876 | 0.112 |
| **PlasFlow v2** | **0.871** | **0.783** | **0.825** |

PLSDB-corrected F1 = **0.847** (12/38 reported FPs are benchmark mislabels — sequences present verbatim in PLSDB deposited under chromosome accessions).

---

## Installation

### Requirements

- Python 3.10–3.11
- [conda](https://docs.conda.io/) (recommended for external tools)

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0

# Option A — Poetry (recommended)
pip install poetry && poetry install

# Option B — pip
pip install -e .
```

### External tools

Install via conda (or see `plasflow2 setup` for detailed instructions):

```bash
conda install -c bioconda diamond mob_suite minimap2
```

| Tool | Purpose |
|---|---|
| [DIAMOND](https://github.com/bbuchfink/diamond) | ARG, VF, MGE, and taxonomy annotation |
| [MOB-suite](https://github.com/phac-nml/mob-suite) | Plasmid mobility typing (conjugative / mobilizable) |
| [minimap2](https://github.com/lh3/minimap2) | Closest known plasmid match (PLSDB/RefSeq) |
| [geNomad](https://github.com/apcamargo/genomad) | Optional — adds 12 SPM gene features to stage-2 XGBoost |

> **Apple Silicon note:** If `conda install mob_suite` fails on ARM: `pip install mob-suite && mob_init`

---

## Database setup (one-time)

```bash
bash scripts/setup_databases.sh
```

This downloads and builds all databases at their auto-detected paths under `data/databases/`. PlasFlow v2 finds them automatically — no flags needed after setup.

| Database | Path | Purpose |
|---|---|---|
| CARD | `data/databases/card/` | ARG annotation (primary) |
| SARG | `data/databases/sarg/sarg.dmnd` | ARG annotation (secondary) |
| AMRFinderPlus | `data/databases/amrfinder/amrprot.dmnd` | ARG annotation (tertiary) |
| VFDB set A | `data/databases/vfdb/vfdb.dmnd` | Virulence factors |
| Pärnänen MGE | `data/databases/mge/isfinder.dmnd` | IS elements / transposons |
| BacMet2 | `data/databases/bacmet/bacmet.dmnd` | Biocide & metal resistance |
| ICEberg3 | `data/databases/ice/ice.dmnd` | Integrative conjugative elements |
| PLSDB + RefSeq + COMPASS | `data/databases/plasmids/` | Closest known plasmid match |
| Taxonomy (GTDB/RefSeq) | `data/databases/taxonomy/` | Contig host taxonomy |
| MOB-suite | auto-detected | Replicon/relaxase typing |

### Optional: geNomad database (~3 GB, improves accuracy)

```bash
genomad download-database data/databases/genomad_db/
```

When available, run gene annotation before the pipeline:

```bash
genomad annotate assembly.fasta genomad_out/ data/databases/genomad_db/ --threads 16
```

Then pass the output to `plasflow2 prepare --genomad-genes` (see [Two-step workflow](#two-step-workflow-higher-accuracy)).

---

## Usage

### Simplest run

```bash
plasflow2 run --input assembly.fasta --output ./results/ --threads 16
```

### With sample context (adjusts AMR risk scoring)

```bash
plasflow2 run \
  --input   assembly.fasta \
  --output  ./results/ \
  --context wastewater \
  --threads 16
```

Context options: `clinical`, `wastewater`, `environmental`, `unspecified` (default).

### Fast run — skip taxonomy (saves 20–40 min on large datasets)

```bash
plasflow2 run \
  --input         assembly.fasta \
  --output        ./results/ \
  --skip-taxonomy \
  --threads       16
```

### Classify only (no databases required, seconds)

Useful for a quick look before running the full pipeline:

```bash
plasflow2 classify --input assembly.fasta --output predictions.tsv
```

### Two-step workflow (higher accuracy)

Running MOB-suite annotation separately lets you add geNomad gene features, giving the XGBoost stage-2 model its full 26-feature input:

```bash
# Step 1 — generate annotation TSV (~5–30 min depending on dataset size)
plasflow2 prepare \
  --input        assembly.fasta \
  --output       annotations.tsv \
  --genomad-genes genomad_out/assembly_annotate/assembly_genes.tsv \
  --threads      16

# Step 2 — classify with stage-2 XGBoost
plasflow2 classify \
  --input          assembly.fasta \
  --output         predictions.tsv \
  --annotation-tsv annotations.tsv
```

### Rebuild report from saved predictions

No need to re-run the full pipeline to regenerate the HTML:

```bash
plasflow2 report \
  --predictions results/all_predictions.tsv \
  --output      results/report.html
```

---

## Output files

All outputs are written to the directory you specify with `--output`.

| File | Description |
|---|---|
| `all_predictions.tsv` | Per-contig classification and all annotations (every contig) |
| `annotated_predictions.tsv` | Filtered — only contigs with ARGs, MGEs, VFs, mobility, or pathogen hits |
| `plasmids.fasta` | Classified plasmid sequences |
| `chromosome.fasta` | Classified chromosome sequences |
| `phage.fasta` | Classified phage sequences |
| `archaea.fasta` | Classified archaea sequences |
| `annotations.json` | Full evidence per plasmid contig (ARG + mobility + risk + taxonomy) |
| `report_plasmid.html` | Interactive plasmid report with charts, gene maps, and AMR risk summary |
| `report_chromosome.html` | Chromosome contig report |
| `report_phage.html` | Phage contig report |
| `report_archaea.html` | Archaea contig report |
| `report_unclassified.html` | Unclassified contig report |

### Key columns in `all_predictions.tsv`

| Column | Description |
|---|---|
| `contig_id` | Sequence identifier from input FASTA |
| `predicted` | Classification: `plasmid` / `chromosome` / `phage` / `archaea` / `unclassified` |
| `plasmid_score` | MLP probability score (0–1) |
| `confidence` | Final classification confidence |
| `low_confidence` | `True` if best score < 70% |
| `evidence_type` | What drove the plasmid call (e.g. `xgb_blend`, `conjugative_override`, `replicon_boost`) |
| `is_conjugative` | `1` if conjugation proteins detected |
| `is_mobilizable` | `1` if mobilization proteins detected |
| `replicon_type` | Inc group / replicon type (e.g. IncF, IncP) |
| `mobility_class` | `conjugative` / `mobilizable` / `non-mobilizable` |
| `num_args` | Number of ARGs detected |
| `arg_genes` | ARG names (`;`-separated) |
| `drug_classes` | Drug classes (`;`-separated) |
| `risk_score` | AMR risk score (0–10) |
| `topology` | `circular` / `linear` / `too_short` |
| `taxonomy` | Predicted host organism |
| `plasmid_db_match` | Closest known plasmid (PLSDB/RefSeq/COMPASS) |
| `plasmid_db_ani` | % nucleotide identity to closest known plasmid |

---

## AMR risk score

Each plasmid is scored 0–10 based on mobility, ARG burden, host pathogenicity, and sample context:

| Factor | Points |
|---|---|
| ESKAPE pathogen host (*K. pneumoniae*, *A. baumannii*, *P. aeruginosa*, *S. aureus*, *E. faecium*, *Enterobacter*, *E. coli*) | +3 |
| WHO 2024 critical/high priority pathogen host | +2 |
| Conjugative mobility | +3 |
| Mobilizable | +2 |
| Broad-host-range replicon (IncP / IncQ / IncW) | +2 |
| ≥5 ARGs or ≥3 drug classes | +3 |
| 3–4 ARGs or 2 drug classes | +2 |
| 1–2 ARGs | +1 |
| Context: clinical | +3 |
| Context: wastewater | +2 |
| Context: environmental | +1 |
| **Max (capped at 10)** | |

Risk ≥ 7 = **high** · 4–6 = **medium** · 0–3 = **low**

---

## CLI reference

```
plasflow2 [--verbose] [--version] COMMAND

Commands:
  run       Full pipeline: classify → annotate → risk score → HTML report
  classify  Classify contigs (fast, no databases required)
  prepare   Generate MOB-suite annotation TSV for stage-2 XGBoost
  annotate  Annotate plasmid sequences with ARGs and mobility
  report    Rebuild HTML report from a saved all_predictions.tsv
  setup     Print installation guide for external tools and databases
```

**`plasflow2 run` key options:**

| Option | Default | Description |
|---|---|---|
| `--input` / `-i` | required | Input FASTA (`.fasta`, `.fa`, `.fna`, `.gz`, `.bz2`) |
| `--output` / `-o` | required | Output directory (created if absent) |
| `--threads` | 8 | CPU threads |
| `--context` | unspecified | `clinical` / `wastewater` / `environmental` / `unspecified` |
| `--plasmid-threshold` | 0.95 | Minimum plasmid score to emit a plasmid call |
| `--min-confidence` | — | When set, every contig gets a label (argmax fallback) instead of `unclassified` |
| `--min-length` | 1000 | Minimum contig length in bp |
| `--skip-mobility` | — | Skip MOB-suite (use when `mob_typer` is unavailable) |
| `--skip-taxonomy` | — | Skip taxonomy annotation (saves 20–40 min on large datasets) |
| `--min-identity` | 80.0 | Minimum % identity for DIAMOND ARG hits |

Run `plasflow2 COMMAND --help` for the full option list for any command.

---

## Testing

```bash
python -m pytest tests/unit/ -q         # 192 unit tests
python -m pytest tests/integration/ -q  # requires external tools installed
```

---

## Retraining

To retrain the MLP or XGBoost models on your own data, see the scripts in `scripts/`:

- `scripts/retrain_k7_binary.sh` — end-to-end binary MLP retrain (dataset build → train → benchmark)
- `scripts/retrain_with_genomad.sh` — retrain XGBoost with geNomad SPM features
- `scripts/retrain_hard_neg.sh` — retrain with composition FP hard negatives
- `scripts/build_dataset.py` — build training windows from plasmid + GTDB chromosome FASTAs
- `scripts/train_marker_model.py` — train XGBoost on annotation TSV

> Apple Silicon: MPS is disabled by default due to PyTorch ≤ 2.3 instability on large float32 ops. Training runs on CPU (~45–60 min for 50 epochs). Set `PLASFLOW_USE_MPS=1` to re-enable if your PyTorch version supports it.

---

## Citation

If you use PlasFlow v2, please cite the original PlasFlow paper:

> Krawczyk PS, Lipinski L, Dziembowski A. PlasFlow: predicting plasmid sequences in metagenomic data using genome signatures. *Nucleic Acids Research*, 2018, 46(6):e35. https://doi.org/10.1093/nar/gky044

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
