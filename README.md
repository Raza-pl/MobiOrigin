# PlasFlow v2

[![CI](https://github.com/Raza-pl/plasflow2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/Raza-pl/plasflow2.0/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**PlasFlow v2** classifies metagenomic contigs as plasmid, chromosome, phage, or archaea using a **hybrid two-stage classifier** (k-mer MLP + marker XGBoost), then annotates each contig with antibiotic resistance genes (ARGs from CARD + SARG + AMRFinderPlus), virulence factors (VFs), mobile genetic elements (MGEs), plasmid mobility class, circular topology detection, and an AMR risk score (0–10). Results are delivered in an interactive HTML report with plain-English summaries, a gene-level TSV, and a closest-known-plasmid match column.

This is a complete rewrite of [PlasFlow v1](https://github.com/smaegol/PlasFlow) (Krawczyk et al., *Nucleic Acids Research* 2018) on a modern Python/PyTorch stack.

---

## Changelog

### June 2026 (latest)
- **Architecture:** 3-class MLP (plasmid/chromosome/phage, 87.14% val acc) retrained on **900k sequences** from 5,922 GTDB r220 bacterial genomes + PLSDB/RefSeq/COMPASS + INPHARED. Archaea removed from model — detected post-classification via DIAMOND taxonomy ORF voting (archaeal ORF hits > bacterial AND ≥5), following the Chibani et al. method.
- **Hallmark gate (all contigs):** ALL plasmid calls now require biological evidence (PLSDB match, relaxase, replicon type, ICE hit, or rep protein). Contigs <50 kb with no evidence → `unclassified`. Contigs ≥50 kb with no evidence → `low_confidence`. Reduced plasmid over-classification from 14.3% → 1.2% on WWTP metagenomes.
- **Class prior correction:** Bayesian correction applied before thresholding. Context-specific priors: `wastewater` (plasmid=3%, chromosome=93%, phage=4%), `clinical` (5%/90%/5%), `environmental` (2%/95%/3%). Activated via `--context wastewater` on CLI.
- **Rep protein detection:** new `scripts/setup_rep_diamond.sh` translates `mob_suite/rep.dna.fas` → `rep_proteins.dmnd`. DIAMOND run added to pipeline; rep protein hit is hallmark evidence for non-mobile plasmids.
- **Marker XGBoost (stage 2):** 16-feature XGBoost (91.7% val acc) combining MLP scores with biological markers — `is_conjugative`, `is_mobilizable`, `has_rep_protein`, `has_ice`, `n_ice/rep_per_kb`, `coding_density`, `gc_content`. Scores aggregated with MLP via attention-weighting. Auto-activates from `data/models/marker_xgb.pkl`.
- **Kraken2 fallback taxonomy:** `annotate/taxonomy_kraken2.py` + `scripts/setup_kraken2_db.sh`. Classifies the ~56% of contigs DIAMOND misses (no ORFs / short contigs) in ~30 sec using pre-built 8 GB k-mer index. DIAMOND always takes priority.
- **Performance fixes:** O(n²) ORF lookup → O(1) dict; taxonomy TSV parsed once (saves ~3 min per run); DIAMOND mobility tuple unpack fixed (was causing 2-hour mob_typer fallback).
- **Circular topology:** expanded header matching to cover SPAdes (`_circular`), Flye, Unicycler (`circular=true`), Canu (`suggestCircular=yes`), NCBI (`[topology=circular]`), Bandage.
- **Kaiju taxonomy:** Kaiju plasmids FM-index built from NCBI RefSeq plasmid proteins. Fixed `kaiju-makedb` CLI flags.

### June 2026 (earlier)
- **Model:** retrained MLP on **800,000 sequences** (200k per class) using 5,922 GTDB r220 bacterial genomes + 1,032 archaeal genomes + 72,556 PLSDB plasmids. Validation accuracy 86.18% (up from ~82% on old 400k dataset).
- **Fix:** `--plasmid-threshold` is now respected independently of `--min-confidence` — previously `min_confidence` was silently overriding the plasmid threshold, causing plasmid over-classification.
- **Fix:** plasmid DB matcher now auto-detects `plsdb.fasta` in addition to legacy `PLSDB.fna` filename.
- **Fix:** replicon typing now uses `minimap2 -x asm5` (assembled-to-assembled preset). The previous `-x sr` (short-read) preset produced zero hits — IncP, IncQ, IncF types were missing from all output.
- **Fix:** `NonPlasmidContigResult` dataclass now includes the `risk` field — fixes a crash in the pipeline test suite.
- **Fix:** macOS ARM segfault during classification resolved — BLAS thread caps now set at CLI entry point before any numpy/torch import.
- **ARG:** added **AMRFinderPlus DB as a third ARG database** (DIAMOND-based, no CLI dependency). Priority order per ORF: CARD > AMRProt > SARG. Auto-detected from `data/databases/amrfinder/amrprot.dmnd`. Setup: `bash scripts/setup_amrprot_diamond.sh`.
- **Annotation:** added **BacMet2** (biocide & metal resistance) and **ICEberg3** (integrative conjugative elements) annotation via DIAMOND. Both auto-detected from `data/databases/bacmet/` and `data/databases/ice/`. Setup: `bash scripts/setup_bacmet_ice_diamond.sh`.
- **Annotation:** MGE hits now enriched with IS family and class from `mge_database.tsv`. VFG hits enriched with functional category from `vfdb_indx.txt`.
- **Output:** renamed `predictions.tsv` → `all_predictions.tsv`. Added `annotated_predictions.tsv` — a focused 17-column table for contigs with ARGs, MGEs, VFs, BacMet, ICE, mobility, or pathogen hits.
- **Report:** added **circular plasmid SVG maps** — pure SVG genome diagrams for each circular plasmid contig, colour-coded by gene type (ARG/VFG/MGE/BacMet/ICE/mobility). Saved to `report_circular_plasmids.html`, linked from the main plasmid report.
- **Report:** added **Priority Alert** section — surfaces plasmids that are simultaneously mobile, ARG-carrying, and pathogenic-host-matched.
- **Report:** added **Pathogenic Host Summary** table — breakdown by threat level (critical / high / medium) and species.

---

## What is new in v2

| Feature | v1 | v2 |
|---|---|---|
| Python | 3.5 / TensorFlow 0.10 | 3.10+ / PyTorch 2.x |
| Classes | plasmid vs chromosome | plasmid · chromosome · **phage** · **archaea** · unclassified |
| Architecture | TF neural net | **3-class MLP** (k-mer) + **marker XGBoost** (biological features) |
| Archaea detection | ✗ | Post-classification DIAMOND taxonomy ORF voting (archaeal hits > bacterial AND ≥5) |
| Hallmark gate | ✗ | **All** plasmid calls require biological evidence — contigs <50 kb with none → unclassified |
| Class prior correction | ✗ | Bayesian correction by context (wastewater / clinical / environmental) |
| Rep protein detection | ✗ | DIAMOND vs translated `rep.dna.fas` — evidence for non-mobile plasmids |
| Kraken2 fallback taxonomy | ✗ | Fast nucleotide k-mer taxonomy for contigs DIAMOND misses (~30 sec, pre-built 8 GB DB) |
| ARG annotation | ✗ | DIAMOND + **CARD + SARG + AMRFinderPlus DB** (triple-DB, auto-detected) |
| Virulence factors | ✗ | DIAMOND + **VFDB set A** (auto-detected) |
| MGE / IS elements | ✗ | DIAMOND + **Pärnänen MGE database** + IS family/class enrichment (auto-detected) |
| BacMet (biocide/metal) | ✗ | DIAMOND + **BacMet2 experimentally confirmed** (auto-detected) |
| ICE elements | ✗ | DIAMOND + **ICEberg3 experimental** (auto-detected) |
| Mobility typing | ✗ | **MOB-suite + DIAMOND** per-contig (conjugative / mobilizable / non-mobilizable) |
| Contig taxonomy | ✗ | **DIAMOND blastp + GTDB/RefSeq LCA** — reuses ORFs from ARG step |
| Plasmid-DB match | ✗ | **minimap2** vs PLSDB + RefSeq + COMPASS — closest known plasmid + ANI |
| Circular topology | ✗ | **DTR detection** (500 bp terminal window, ≥90 % identity) |
| Confidence flagging | ✗ | `low_confidence` column + ⚠ badge in HTML |
| Gene-level output | ✗ | **genes.tsv** — 169k+ ORFs with coordinates + ARG/VF/MGE flags |
| Compressed input | ✗ | `.gz` and `.bz2` FASTA accepted natively |
| AMR risk score | ✗ | 0–10 with ESKAPE host detection + WHO 2024 pathogens |
| HTML report | ✗ | **6 interactive pages** — plasmid · chromosome · phage · archaea · unclassified · **circular maps** |
| Circular maps | ✗ | **Pure SVG** per-contig genome maps for circular plasmids (ARG/VFG/MGE/BacMet/ICE colour-coded) |
| Test suite | ✗ | 192 unit + integration tests |

---

## Benchmarked performance (June 2026)

### GCA_054405655 — WWTP metagenome (validation dataset)
24,746 contigs · 177 MB FASTA · Apple Silicon CPU · 16 threads

| Step | Tool | Time |
|---|---|---|
| MLP classify (24,746 contigs) | PyTorch CPU | ~15 sec |
| ORF prediction | pyrodigal | ~5 min |
| ARG annotation CARD + SARG + AMRProt | DIAMOND blastp | ~20 sec |
| MGE annotation | DIAMOND blastp | ~9 sec |
| Plasmid-DB match (13 GB) | minimap2 split-prefix | ~10 min |
| Mobility + replicon typing | DIAMOND + minimap2 asm5 | ~1 min |
| Taxonomy (897 MB DB, blastp reuse) | DIAMOND blastp | ~20–40 min |
| Report generation | Python | ~30 sec |
| **Total (with taxonomy)** | | **~45 sec (cached) · ~45–65 min (fresh)** |

**Results (3-class model + priors + hallmark gate, June 2026):**

| Metric | Old 4-class model | New 3-class model |
|---|---|---|
| Plasmid contigs | 24,356 (14.3%) | **1,968 (1.2%)** |
| Chromosome contigs | 136,642 | 166,793 |
| Phage contigs | 5,115 | 1,329 |
| Archaea contigs | 3,994 (k-mer) | **101 (taxonomy-verified)** |
| ARGs (CARD+SARG+AMRProt) | 208 | 208 |
| Rep protein hits | — | 1,134 |
| Pathogenic contigs | 2,041 | 2,041 |

> Plasmid reduction (14.3% → 1.2%) reflects removal of false positives: chromosomal fragments with plasmid-like k-mer composition. All 1,968 retained plasmids have at least one piece of biological evidence (PLSDB match, relaxase, replicon type, ICE hit, or rep protein).

**Model accuracy:** MLP 87.14% val acc · XGBoost 91.7% val acc (16 biological features).

### W1 — Wastewater metagenome assembly (larger dataset)
205,645 contigs · Apple Silicon CPU · 16 threads

| Metric | Old 4-class model | New 3-class model |
|---|---|---|
| Plasmid contigs | 16,229 (7.9%) | **1,156 (0.56%)** |
| Chromosome contigs | 154,393 | 200,278 |
| Phage contigs | 10,477 | 4,211 |
| Archaea contigs | 24,546 (k-mer) | **592 (taxonomy-verified)** |
| ARGs (CARD+SARG+AMRProt) | 185 | 185 |
| BacMet hits | 39 | 39 |
| ICE hits | 1,376 | 1,376 |
| Rep protein hits | — | 1,413 |
| Pathogenic contigs | 589 | 589 |
| Circular contigs detected | 0 | **1** |
| Wall-clock time | 93 min | ~100 min |

> Plasmid reduction (7.9% → 0.56%) and archaea reduction (24,546 k-mer guesses → 592 taxonomy-verified) confirm that previous models were over-calling both classes dramatically on WWTP metagenomes.

---

## Database auto-detection

All optional databases are **auto-detected** from `data/databases/` — no flags needed after a one-time setup:

```
data/databases/
  card/         card.dmnd  aro_index.tsv           ← CARD ARG annotation
  sarg/         sarg.dmnd                           ← SARG ARG annotation (auto)
  amrfinder/    amrprot.dmnd  fam.tab               ← AMRFinderPlus ARG annotation (auto)
  bacmet/       bacmet.dmnd  Bacmet_list.tsv       ← BacMet2 biocide/metal resistance (auto)
  ice/          ice.dmnd  ice_experimental_list.tsv ← ICEberg3 ICE annotation (auto)
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
| [DIAMOND](https://github.com/bbuchfink/diamond) | ARG (CARD+SARG+AMRProt) / VF / MGE / taxonomy | `conda install -c bioconda diamond` |
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
3. **AMRFinderPlus DB** — third ARG database (`data/databases/amrfinder/amrprot.dmnd`). Place `AMRfinder.fasta` in `data/databases/` then run: `bash scripts/setup_amrprot_diamond.sh`
4. **BacMet2** — biocide & metal resistance (`data/databases/bacmet/bacmet.dmnd`). Run: `bash scripts/setup_bacmet_ice_diamond.sh`
5. **ICEberg3** — integrative conjugative elements (`data/databases/ice/ice.dmnd`). Run: `bash scripts/setup_bacmet_ice_diamond.sh`
4. **VFDB set A** — virulence factors (`data/databases/vfdb/vfdb.dmnd`)
5. **Pärnänen MGE database** (`data/databases/mge/isfinder.dmnd`)
6. **Plasmid databases** — PLSDB + RefSeq + COMPASS (`data/databases/plasmids/`)
7. MOB-suite reference data (`mob_init`)

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
| `all_predictions.tsv` | 42-column per-contig table (all contigs, all annotations) |
| `annotated_predictions.tsv` | Focused 12-column table — only contigs with ARGs, MGEs, VFs, mobility, or pathogen hits |
| `genes.tsv` | Per-ORF table — coordinates, ARG/VF/MGE flags, all contigs |
| `plasmid.fasta` | Plasmid sequences |
| `chromosome.fasta` | Chromosome sequences |
| `phage.fasta` | Phage sequences |
| `archaea.fasta` | Archaea sequences |
| `annotations.json` | Full evidence per plasmid contig |
| `report_plasmid.html` | Interactive plasmid report (charts + table + narrative + **priority alert** + pathogen summary) |
| `report_genome_maps.html` | Standalone genome maps — contigs with ≥3 genes or risk > 4 (linked from plasmid report) |
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

### Stage 1 — 3-class MLP (k-mer features)

```bash
# Rebuild dataset (3-class: plasmid / chromosome / phage — no archaea)
python scripts/build_dataset.py \
    --plasmid-dir data/databases/plasmids/ \
    --chrom-dir   data/gtdb_genomes/bacteria/ \
    --data-dir    data/databases/ \
    --max-per-class 300000 \
    --out         data/

# Train MLP
python scripts/train_model.py \
    --data   data/features.npy \
    --labels data/labels.npy \
    --mlp --epochs 50 --out data/models/
```

> **Apple Silicon:** MPS is disabled by default (PyTorch ≤ 2.3 segfaults on large float32 ops). Training runs on CPU (~63 min for 900k × 50 epochs). Set `PLASFLOW_USE_MPS=1` to re-enable if your PyTorch version supports it.

### Stage 2 — Marker XGBoost (16 biological features)

The XGBoost second stage uses 16 biological marker features and auto-activates when `data/models/marker_xgb.pkl` exists. Features include: MLP scores, `is_conjugative`, `is_mobilizable`, `has_rep_protein`, `has_ice`, `n_ice/rep_per_kb`, `coding_density`, `gc_content`, `log10_length`. Build the rep protein database first:

```bash
# One-time: build rep protein DIAMOND DB
bash scripts/setup_rep_diamond.sh

# Build marker features (~25 min — MLP inference + DIAMOND vs mob/mpf/rep/ICE)
python scripts/build_marker_dataset.py \
    --plasmid-dir data/databases/plasmids/ \
    --chrom-dir   data/gtdb_genomes/bacteria/ \
    --model       data/models/mlp_v2.pt \
    --mob-db      data/databases/mob_suite/mob_proteins.dmnd \
    --mpf-db      data/databases/mob_suite/mpf_proteins.dmnd \
    --max-per-class 30000 --threads 16 \
    --out         data/marker_features.npz
# Auto-detects: rep_proteins.dmnd and ice.dmnd

# Train XGBoost (~2 min)
python scripts/train_marker_model.py \
    --features data/marker_features.npz \
    --out      data/models/
```

### Kraken2 fallback taxonomy (optional, recommended)

Provides taxonomy for contigs DIAMOND misses (no ORFs). Pre-built database, no training required.

```bash
bash scripts/setup_kraken2_db.sh   # ~8 GB download, auto-detected at runtime
```

### Archaea detection (no training required)

Archaea are detected post-classification via DIAMOND taxonomy ORF voting. No separate training is needed — uses the existing taxonomy DIAMOND run automatically.

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
