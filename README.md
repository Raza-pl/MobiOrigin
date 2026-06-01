# PlasFlow v2

[![CI](https://github.com/Raza-pl/plasflow2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/Raza-pl/plasflow2.0/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**PlasFlow v2** classifies metagenomic contigs as plasmid, chromosome, phage, or archaea, then annotates each contig with antibiotic resistance genes (ARGs from CARD + SARG), virulence factors (VFs), mobile genetic elements (MGEs), plasmid mobility class, circular topology detection, and an AMR risk score (0–10). Results are delivered in an interactive 5-page HTML report with plain-English summaries, a gene-level TSV, and a closest-known-plasmid match column.

This is a complete rewrite of [PlasFlow v1](https://github.com/smaegol/PlasFlow) (Krawczyk et al., *Nucleic Acids Research* 2018) on a modern Python/PyTorch stack.

---

## What is new in v2

| Feature | v1 | v2 |
|---|---|---|
| Python | 3.5 / TensorFlow 0.10 | 3.10+ / PyTorch 2.x |
| Classes | plasmid vs chromosome | plasmid · chromosome · **phage** · **archaea** · unclassified |
| Architecture | TF neural net | **4-class MLP** with length feature |
| ARG annotation | ✗ | DIAMOND + **CARD + SARG** (dual-DB, auto-detected) |
| Virulence factors | ✗ | DIAMOND + **VFDB set A** (auto-detected) |
| MGE / IS elements | ✗ | DIAMOND + **Pärnänen MGE database** (auto-detected) |
| Mobility typing | ✗ | **MOB-suite** per-contig (conjugative / mobilizable / non-mobilizable) |
| Contig taxonomy | ✗ | **DIAMOND blastp + GTDB/RefSeq LCA** — reuses ORFs from ARG step |
| Plasmid-DB match | ✗ | **minimap2** vs PLSDB + RefSeq + COMPASS — closest known plasmid + ANI |
| Circular topology | ✗ | **DTR detection** (500 bp terminal window, ≥90 % identity) |
| Confidence flagging | ✗ | `low_confidence` column + ⚠ badge in HTML |
| Gene-level output | ✗ | **genes.tsv** — 169k+ ORFs with coordinates + ARG/VF/MGE flags |
| Compressed input | ✗ | `.gz` and `.bz2` FASTA accepted natively |
| AMR risk score | ✗ | 0–10 with ESKAPE host detection + WHO 2024 pathogens |
| HTML report | ✗ | **5 interactive pages**, plain-English narrative, genome maps, Plotly charts |
| Test suite | ✗ | 192 unit + integration tests |

---

## Benchmarked performance (May 2026)

Tested on GCA_054405655 WWTP metagenome assembly — 24,746 contigs, 177 MB FASTA, Apple Silicon CPU, 16 threads:

| Step | Tool | Time |
|---|---|---|
| MLP classify (24,746 contigs) | PyTorch CPU | ~15 sec |
| ORF prediction | pyrodigal | ~5 min |
| ARG annotation CARD (2.4 MB DB) | DIAMOND blastp | ~14 sec |
| ARG annotation SARG (57 MB DB) | DIAMOND blastp | ~5 min |
| MGE annotation (0.8 MB DB) | DIAMOND blastp | ~9 sec |
| Plasmid-DB match (13 GB combined) | minimap2 split-prefix | ~10 min |
| Mobility typing (2,559 contigs) | MOB-suite per-contig | ~1 min |
| Taxonomy (897 MB DB, blastp reuse) | DIAMOND blastp | ~20–40 min |
| Report generation | Python | ~30 sec |
| **Total (with taxonomy)** | | **~45–65 min** |
| **Total (--skip-taxonomy)** | | **~12–15 min** |

Results on that run: **2,559 plasmid contigs · 73 ARGs (38 CARD + 35 SARG-only) · 250 MGE hits · 169,009 genes in genes.tsv**

---

## Database auto-detection

All optional databases are **auto-detected** from `data/databases/` — no flags needed after a one-time setup:

```
data/databases/
  card/         card.dmnd  aro_index.tsv           ← CARD ARG annotation
  sarg/         sarg.dmnd                           ← SARG ARG annotation (auto)
  vfdb/         vfdb.dmnd                           ← VFDB virulence factors (auto)
  mge/          isfinder.dmnd                       ← MGE / IS elements (auto)
  taxonomy/     refseq_taxonomy.dmnd  taxon_map.tsv ← Contig taxonomy (auto)
  plasmids/     PLSDB.fna  RefSeq.fna  COMPASS.fna  ← Plasmid-DB match (auto)
```

---

## Installation

### Requirements

- Python 3.10+
- Poetry or pip

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0

# Option A — Poetry
pip install poetry && poetry install

# Option B — pip
pip install -e .
```

### External tools

| Tool | Purpose | Install |
|---|---|---|
| [DIAMOND](https://github.com/bbuchfink/diamond) | ARG / VF / MGE / taxonomy | `conda install -c bioconda diamond` |
| [MOB-suite](https://github.com/phac-nml/mob-suite) | Plasmid mobility typing | `conda install -c conda-forge -c bioconda mob_suite` |
| [minimap2](https://github.com/lh3/minimap2) | Plasmid-DB nucleotide match | `conda install -c bioconda minimap2` |

> **Apple Silicon (MOB-suite):** If conda fails on ARM: `pip install mob-suite && mob_init`

### Docker (zero-setup)

```bash
docker build -t plasflow2 .
docker run --rm \
  -v /path/to/databases:/data/databases:ro \
  -v /path/to/input:/data/input:ro \
  -v /path/to/results:/results \
  plasflow2 run --input /data/input/assembly.fasta --output /results/ --threads 16
```

---

## Database setup (one-time, ~15–30 min)

```bash
bash scripts/setup_databases.sh
```

Builds all databases at their auto-detected paths:
1. **CARD** — ARG annotation (`data/databases/card/`)
2. **SARG** — Structured ARG (`data/databases/sarg/sarg.dmnd`)
3. **VFDB set A** — virulence factors (`data/databases/vfdb/vfdb.dmnd`)
4. **Pärnänen MGE database** (`data/databases/mge/isfinder.dmnd`)
5. **Plasmid databases** — PLSDB + RefSeq + COMPASS (`data/databases/plasmids/`)
6. MOB-suite reference data (`mob_init`)

### Optional: taxonomy database (~2 GB)

```bash
python scripts/build_taxonomy_db.py \
    --genomes data/chromosomes/ \
    --out     data/databases/taxonomy/
```

---

## Quickstart

### Simplest run (all databases auto-detected)

```bash
plasflow2 run \
  --input   assembly.fasta \
  --output  ./results/ \
  --context wastewater \
  --threads 16
```

### Fast run — skip taxonomy (saves 20–40 min)

```bash
plasflow2 run \
  --input        assembly.fasta \
  --output       ./results/ \
  --context      wastewater \
  --threads      16 \
  --skip-taxonomy
```

### Full explicit run

```bash
plasflow2 run \
  --input              assembly.fasta \
  --output             ./results/ \
  --card-db            data/databases/card/card.dmnd \
  --aro-index          data/databases/card/aro_index.tsv \
  --sarg-db            data/databases/sarg/sarg.dmnd \
  --vfdb               data/databases/vfdb/vfdb.dmnd \
  --mge-db             data/databases/mge/isfinder.dmnd \
  --taxonomy-db        data/databases/taxonomy/refseq_taxonomy.dmnd \
  --context            wastewater \
  --threads            16 \
  --plasmid-threshold  0.95
```

### Classify only (no databases required, seconds)

```bash
plasflow2 classify --input assembly.fasta --output predictions.tsv
```

### Rebuild HTML report from saved predictions

```bash
plasflow2 report \
  --annotations results/annotations.json \
  --predictions results/predictions.tsv \
  --output      results/report.html
```

---

## Outputs

| File | Description |
|---|---|
| `predictions.tsv` | 42-column per-contig table (all contigs) |
| `genes.tsv` | Per-ORF table — coordinates, ARG/VF/MGE flags, all contigs |
| `plasmid.fasta` | Plasmid sequences |
| `chromosome.fasta` | Chromosome sequences |
| `phage.fasta` | Phage sequences |
| `archaea.fasta` | Archaea sequences |
| `annotations.json` | Full evidence per plasmid contig |
| `report_plasmid.html` | Interactive plasmid report (charts + table + genome maps + narrative) |
| `report_chromosome.html` | Chromosome contig report |
| `report_phage.html` | Phage contig report |
| `report_archaea.html` | Archaea contig report |
| `report_unclassified.html` | Unclassified contig report |

### predictions.tsv schema (42 columns)

**Universal:** `contig_id` · `length` · `label` · `confidence` · `plasmid_score` · `chromosome_score` · `phage_score` · `archaea_score` · `taxonomy` · `taxonomy_rank` · `taxonomy_lineage`

**ARG (CARD + SARG, all contigs):** `num_args` · `arg_genes` · `drug_classes` · `arg_sources`

**Virulence (all contigs):** `num_vf` · `vf_genes`

**MGE (all contigs):** `num_mge` · `mge_genes` · `mge_families`

**Plasmid-specific:** `mobility_class` · `replicon_type` · `relaxase_type` · `mpf_type` · `risk_score` · `mobility_score` · `arg_score` · `replicon_score` · `context_score` · `host_score` · `risk_evidence` · `eskape_host` · `eskape_genus`

**Quality:** `topology` (circular/linear/too_short) · `low_confidence` (True if < 70 %)

**Plasmid-DB match:** `plasmid_db_match` · `plasmid_db_source` · `plasmid_db_ani` · `plasmid_db_cov`

**Pathogen:** `pathogen_species` · `pathogen_threat` · `pathogen_category`

---

## Dual-taxonomy interpretation

PlasFlow v2 reports two complementary signals for plasmid contigs:

| Column | Method | What it tells you |
|---|---|---|
| `taxonomy` | DIAMOND blastp + LCA (RefSeq/GTDB) | Predicted **host organism** from coding genes |
| `plasmid_db_match` + `plasmid_db_ani` | minimap2 vs PLSDB/RefSeq/COMPASS | Closest **known plasmid** + nucleotide identity |

Comparing the two reveals resident plasmids (taxonomy matches DB host), recent horizontal transfer (mismatch), and novel lineages (ANI < 90 %).

---

## DIAMOND performance and faster alternatives

DIAMOND is the default search engine for ARG, VF, MGE, and taxonomy annotation. For the taxonomy step (largest DB at 897 MB), the pipeline optimises throughput by:

- Reusing pre-predicted ORFs (`blastp` mode) instead of translating contigs on-the-fly (`blastx`) — **~6× speedup**
- `--block-size 4.0` — loads large DB chunks into RAM — **~4× speedup**

If DIAMOND taxonomy is still too slow for your use case, faster alternatives exist:

| Tool | Speed vs DIAMOND | Sensitivity | Best for |
|---|---|---|---|
| **Kraken2** | 100–1000× faster | Lower (k-mer exact match) | Quick taxonomy screening, reads/contigs |
| **Kaiju** | 20–50× faster | Good (protein k-mers) | Contig-level taxonomy, no assembly needed |
| **MMseqs2** | 3–10× faster | Similar to DIAMOND | Drop-in DIAMOND replacement, large-scale |
| **AMRFinderPlus** | Similar | Very high (curated) | Clinical ARG annotation (NCBI PGAP pipeline) |
| **ABRicate** | Similar | High | Multi-DB ARG/VF screening |

**Recommended strategy:**
- For **taxonomy**: replace DIAMOND with **Kaiju** (protein k-mer, ~30× faster, good sensitivity for assembled contigs — `conda install -c bioconda kaiju`)
- For **ARG annotation**: add **AMRFinderPlus** as a third database alongside CARD + SARG for clinical-grade precision
- For **very large datasets** (>500k contigs): switch taxonomy to **Kraken2** with a custom protein DB, classify in seconds

> MMseqs2 taxonomy workflow is the highest-quality DIAMOND replacement and fully supports the same `blastp` mode. See [MMseqs2 taxonomy docs](https://github.com/soedinglab/MMseqs2/wiki#taxonomy-assignment).

---

## AMR risk score (0–10)

| Factor | Points |
|---|---|
| ESKAPE host (*K. pneumoniae*, *A. baumannii*, *P. aeruginosa*, *S. aureus*, *E. faecium*, *Enterobacter*, *E. coli*) | +3 |
| WHO 2024 priority pathogen host | +2 |
| Conjugative mobility (MOB-suite) | +3 |
| Mobilizable mobility | +2 |
| ≥5 ARGs or ≥3 drug classes | +3 |
| 3–4 ARGs or 2 drug classes | +2 |
| 1–2 ARGs | +1 |
| Broad-host-range replicon (IncP / IncQ / IncW) | +2 |
| Narrow-host-range replicon | +1 |
| Context: clinical | +3 |
| Context: wastewater or food | +2 |
| Context: environmental | +1 |
| **Max (capped)** | **10** |

Risk ≥ 7 = **high** · 4–6 = **medium** · 0–3 = **low**

---

## PlasFlow v2 vs geNomad

| Capability | geNomad | PlasFlow v2 |
|---|---|---|
| Classification | plasmid / virus / chromosome | plasmid / chromosome / phage / archaea |
| ARG annotation | ✗ | CARD + SARG dual-DB |
| Virulence factors | ✗ | VFDB set A |
| MGE / IS elements | ✗ | Pärnänen database |
| Mobility class | ✗ | MOB-suite per-contig |
| AMR risk score | ✗ | 0–10 with ESKAPE + WHO host |
| Gene-level TSV | ✓ | ✓ (coordinates + ARG/VF/MGE flags) |
| Closest known plasmid | ✗ | ✓ minimap2 vs PLSDB + RefSeq + COMPASS |
| Circular topology | ✓ | ✓ (DTR detection) |
| Contig taxonomy | ✓ | ✓ (DIAMOND/Kaiju + LCA) |
| Confidence flags | ✓ | ✓ (low_confidence column + ⚠ badge) |
| Interactive HTML | ✗ | ✓ (5 pages, charts, genome maps, narrative) |
| Compressed input | ✓ | ✓ (.gz / .bz2) |

---

## CLI reference

```
plasflow2 [--verbose] COMMAND [OPTIONS]

Commands:
  run        Full pipeline: classify → annotate → risk → report
  classify   Classify sequences only (no databases required)
  annotate   Annotate plasmid sequences (ARG, VF, MGE, mobility)
  report     Rebuild HTML report from existing predictions.tsv + annotations.json
  setup      Print installation guide for external dependencies

plasflow2 run key options:
  --input / -i            Input FASTA (.fasta/.fa/.fna/.gz/.bz2)  [required]
  --output / -o           Output directory  [required]
  --threads               CPU threads  [default: 8]
  --context               clinical | wastewater | environmental | unspecified
  --plasmid-threshold     Plasmid confidence threshold  [default: 0.95]
  --min-confidence        Argmax fallback — labels every contig
  --skip-mobility         Skip MOB-suite
  --skip-taxonomy         Skip taxonomy (saves 20–40 min on large datasets)
  --card-db               CARD DIAMOND database  [auto-detected]
  --sarg-db               SARG DIAMOND database  [auto-detected]
  --vfdb                  VFDB DIAMOND database  [auto-detected]
  --mge-db                MGE DIAMOND database  [auto-detected]
  --taxonomy-db           RefSeq/GTDB taxonomy DIAMOND database  [auto-detected]
  --min-identity          Minimum % identity for DIAMOND hits  [default: 80]
  --verbose / -v          Debug logging
```

---

## Retrain the model

The current MLP was trained on 40 chromosome genomes — leading to a high unclassified rate on novel chromosomal contigs. Retrain with the 1,998 diverse genomes in `data/chromosomes/`:

```bash
# Rebuild dataset
python scripts/build_dataset.py \
    --plasmids data/databases/plasmids/ \
    --chroms   data/chromosomes/ \
    --phages   data/databases/inphared/ \
    --archaea  data/archaea/ \
    --out-dir  data/

# Train
python scripts/train_model.py \
    --data   data/features.npy \
    --labels data/labels.npy \
    --mlp --epochs 50 --out data/models
```

> **Apple Silicon:** MPS is disabled by default (PyTorch ≤ 2.3 segfaults on large float32 ops). Training runs on CPU (~15 min). Set `PLASFLOW_USE_MPS=1` to re-enable if your PyTorch version supports it.

---

## Testing

```bash
python -m pytest tests/unit/ --override-ini="addopts=" -q   # 192 tests
python -m pytest tests/integration/ -q                       # requires external tools
```

---

## Citation

If you use PlasFlow v2, please cite:

> Krawczyk PS, Lipinski L, Dziembowski A. PlasFlow: predicting plasmid sequences in metagenomic data using genome signatures. *Nucleic Acids Research*, 2018, 46(6):e35. https://doi.org/10.1093/nar/gky044

---

## License

GPL v3 — see [LICENSE](LICENSE).
