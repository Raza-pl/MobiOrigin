# PlasFlow v2 — Improvement Plan & Testing Strategy

**Date:** June 2026  
**Current status:** Plasmid F1 = 0.392 vs PlasFlow v1 F1 = 0.939 (benchmark) / geNomad ~0.95+

---

## 1. Root Cause Diagnosis

### What's happening right now

| Stage | TP | FP | FN | Plasmid F1 |
|---|---|---|---|---|
| MLP only (k-mer) | 64 | 345 | 330 | 0.159 |
| + XGBoost marker blend | 114 | 96 | 280 | 0.378 |
| + conjugative override | 122 | 106 | 272 | 0.392 |

The benchmark has 394 plasmids against 60,000 chromosomes. The core problems:

**Problem 1 — Marker poverty.** Only 129/394 plasmids (32.7%) carry any MOB-suite-detectable
marker (relaxase, MPF, or rep protein). The other 265 plasmids are non-mobilizable and have
no signal beyond k-mer composition. The marker XGBoost simply has nothing to work with for them.

**Problem 2 — MOB-suite DIAMOND identity.** At 40% amino acid identity, 724/60,000
chromosomes get spurious hits (transposons, prophage-encoded relaxases, integrated conjugative
elements). Hard overrides cause FPs; soft blending dilutes the effect.

**Problem 3 — MLP score distribution overlap.** The MLP gives plasmid score ≥ 0.50 to 13,503
chromosomes. Without a complementary gene-content signal, k-mer alone cannot discriminate at
high precision for 10–20 kb contigs assembled from real metagenomes.

**Problem 4 — Benchmark difficulty vs PlasFlow v1.** PlasFlow v1 reported F1=0.939 on
simulated reads from 40 complete, high-quality reference genomes. Our benchmark uses
metagenome-assembled contigs (real fragmentation, variable coverage, chimeric sequences).
The tasks are not directly comparable — PlasFlow v1 would likely score much lower on our
harder benchmark.

---

## 2. What geNomad Does Differently

### 2.1 Marker database — the key architectural difference

| | MOB-suite (our current) | geNomad |
|---|---|---|
| Database size | ~3 protein DBs (relaxase, MPF, rep) | **227,897 protein family profiles** |
| Search tool | DIAMOND blastp | **MMseqs2** (faster, profile-based) |
| Signal type | Binary hit / no-hit | **Continuous SPM per class** |
| Coverage | ~30–40% of plasmids | Covers all functional gene families |
| Specificity encoding | Categorical (is_conjugative etc.) | **9-class specificity system** (CC, CP, CV, PC, PP, PV, VC, VP, VV) |

Each of geNomad's 227,897 markers carries three **SPM (Specificity Profile Measure)** values
(chromosome, plasmid, virus) ranging 0→1. A plasmid with no relaxase can still produce a
strong signal through partition system genes, toxin-antitoxin systems, plasmid-specific
replication initiators, or any other gene class that is statistically enriched on plasmids.

### 2.2 Marker-based features (25 total in geNomad)

Beyond simple hit counts, geNomad computes:

- **Marker class frequencies**: `cc_marker_freq`, `pp_marker_freq`, `vv_marker_freq`, … (9 features)
- **Aggregate class frequencies**: `c_marker_freq`, `p_marker_freq`, `v_marker_freq` (3 features)
- **Median SPM per class**: `median_c_spm`, `median_p_spm`, `median_v_spm` (3 features)
- **Compound logistic scores**: sigmoid(Σ P_SPM − C_SPM), etc. (3 features)
- **Strand switch rate**: fraction of genes on opposite strand from upstream gene (1 feature)
- **Coding density**: already in our pipeline (1 feature)
- **RBS motif frequencies**: `no_rbs_freq`, `sd_canonical_rbs_freq`, `sd_bacteroidetes_rbs_freq`, `tatata_rbs_freq` (4 features)

We currently use only 16 features in our XGBoost; geNomad uses 25 richer ones. The strand
switch rate and RBS features are directly extractable from pyrodigal output — we already run
pyrodigal, but discard this signal.

### 2.3 Neural network architecture

| | PlasFlow v2 (current) | geNomad |
|---|---|---|
| Input | k-mer frequency histogram (1,493 dims) | 4-mer position matrix (256 × L) |
| Architecture | Fully connected MLP | **IGLOO** (Conv → random patches → self-attention) |
| Sequence context | None (order ignored) | Long-range dependencies captured |
| Training | Tiled windows from complete genomes | Trained on diverse MGE database |

The IGLOO architecture captures positional k-mer relationships — important because plasmids
have characteristic gene order and density patterns that a bag-of-k-mers approach cannot see.

### 2.4 Score aggregation

geNomad fuses the NN score and marker-based score using a learned aggregation, then applies
explicit **score calibration** (Platt scaling per sequence length bin). PlasFlow v2 currently
uses a fixed alpha-weighted blend with hand-tuned thresholds.

---

## 3. Phased Improvement Plan

### Phase 1 — Adopt geNomad marker features (highest impact, 2–3 days)

**Action:** Run `genomad annotate` (already installed) to get per-gene SPM scores on the
training set and benchmark, then replace the MOB-suite binary features with geNomad's
continuous marker features.

```bash
# annotate benchmark with geNomad (already installed)
genomad annotate \
    --threads 8 \
    data/benchmark/benchmark.fna \
    data/benchmark/genomad_ann/ \
    /path/to/genomad_db

# extract features from output: *_genes.tsv contains per-gene marker assignments
```

New XGBoost features to extract from geNomad annotation output:
- `p_marker_freq` — fraction of genes with plasmid-specific markers
- `pp_marker_freq` — fraction with high-plasmid-specificity markers (PP class)
- `median_p_spm` — median plasmid SPM across all genes
- `p_vs_c_logistic` — sigmoid compound score
- `strand_switch_rate` — from gene coordinates in `_genes.tsv`
- `no_rbs_freq`, `canonical_sd_freq` — from pyrodigal RBS predictions

Expected impact: directly addresses the 265 non-mobilizable plasmids, which do carry
plasmid-specific genes (partition, TA, replication) just not MOB-suite mobility genes.

**Keep MOB-suite too** — conjugative/mobilizable detection from MOB-suite is complementary
to geNomad's broader marker space. Use both.

### Phase 2 — Pyrodigal RBS and strand features (low effort, moderate gain, 1 day)

pyrodigal already predicts RBS motifs per gene. We discard this. Add to `annotate_sequences.py`:

```python
# from pyrodigal gene object:
strand_changes = sum(1 for i in range(1, len(genes))
                     if genes[i].strand != genes[i-1].strand)
strand_switch_rate = strand_changes / max(len(genes) - 1, 1)

no_rbs = sum(1 for g in genes if g.rbs_motif is None)
no_rbs_freq = no_rbs / max(len(genes), 1)

canonical_sd = sum(1 for g in genes
                   if g.rbs_motif and g.rbs_motif in CANONICAL_SD_MOTIFS)
canonical_sd_freq = canonical_sd / max(len(genes), 1)
```

These are free signals already available from the ORF prediction we already run.

### Phase 3 — BLASTN replicon typing (already in progress, 1 day)

Add `has_replicon` from BLASTN against `rep.dna.fas` (code already added to
`annotate_sequences.py`). Install BLAST:

```bash
conda install -c bioconda blast -y
```

Then re-run annotation with `--rep-dna data/databases/mob_suite/rep.dna.fas`.
Nucleotide-level replicon typing catches diverged rep genes that are invisible to
protein-level DIAMOND.

### Phase 4 — Better MLP architecture (medium-term, 1–2 weeks)

Replace the k-mer frequency histogram with a position-aware representation:

**Option A (fast):** Add k-mer co-occurrence features — pairs of k-mers within a window.
Captures local sequence grammar without architectural overhaul.

**Option B (recommended):** Replace MLP with a 1D CNN on 4-mer tokens (same input as
geNomad's IGLOO but simpler conv architecture). PyTorch implementation, ~2 weeks.

**Option C (aspirational):** Full IGLOO or transformer on tokenized sequence. Matches
geNomad's architecture exactly but requires significant engineering effort.

### Phase 5 — Training data improvements (ongoing)

Current training:
- Plasmids: PLSDB + COMPASS (complete plasmid sequences, tiled)
- Chromosomes: GTDB bacteria (complete chromosomes, tiled)
- Phages: INPHARED

Gaps:
- **IMG/PR database** (Nayfach et al., NAR 2024): 699,000+ plasmids from metagenomes,
  including many non-mobilizable, cryptic plasmids. Adding even 50,000 of these to
  training would dramatically improve recall on non-mobilizable sequences.
- **Contig-length training samples**: Current training uses fixed-width tiles (2kb, 5kb,
  10kb). Real MAGs have ragged, variable-length contigs. Training on actual assembled
  contigs from known-label datasets would reduce domain shift.

### Phase 6 — Score calibration (after Phase 1–3, 1 day)

After retraining with richer features, add Platt scaling per length bin:

```python
from sklearn.calibration import CalibratedClassifierCV
# or fit sigmoid: P_calibrated = 1 / (1 + exp(-(a * raw_score + b)))
# fit a, b per length tier using ground-truth plasmid/chromosome labels
```

geNomad applies this step explicitly. It converts raw logit-like scores into properly
calibrated probabilities, which makes threshold selection more principled.

---

## 4. Testing on W1.contigs.fa.gz and GCA_054405655

### 4.1 Data summary

| File | Contigs | <2kb | 2–5kb | 5–10kb | 10–20kb | >20kb |
|---|---|---|---|---|---|---|
| W1.contigs.fa.gz | 205,645 | 134,370 (65%) | 51,635 | 11,697 | 4,781 | 3,162 |
| GCA_054405655.1 | 170,118 | 120,499 (71%) | 39,302 | 7,327 | 2,134 | 856 |

Both are heavily short-contig dominated — the hardest regime for any classifier.

### 4.2 Available ground truth (geNomad labels)

**W1** — geNomad has already classified these (in `results/W1/`):
- Plasmid: 6,275 contigs
- Phage: 7,764 contigs
- Chromosome: 191,606 contigs

**GCA_054405655** — geNomad predictions in `data/test/GCA_054405655.predictions.tsv`:
- Chromosome: 151,119 | Phage: 5,883 | Plasmid: 6,860 | Unclassified: 6,245

We can treat geNomad's high-confidence (score ≥ 0.9) predictions as pseudo-ground-truth
to evaluate PlasFlow v2 precision/recall on real metagenome data.

### 4.3 Run commands

```bash
# Step 1: decompress W1
gunzip -k data/test/W1.contigs.fa.gz

# Step 2: annotate both files (after Phase 1–3 improvements)
python scripts/annotate_sequences.py \
    --fasta   data/test/W1.contigs.fa \
    --rep-dna data/databases/mob_suite/rep.dna.fas \
    --out     data/test/W1_annotations.tsv \
    --threads 8

python scripts/annotate_sequences.py \
    --fasta   data/test/GCA_054405655.1_ASM5440565v1_genomic.fna \
    --rep-dna data/databases/mob_suite/rep.dna.fas \
    --out     data/test/GCA_annotations.tsv \
    --threads 8

# Step 3: predict with PlasFlow v2
plasflow2 predict \
    --input          data/test/W1.contigs.fa \
    --model          data/models/mlp_v2.pt \
    --marker-model   data/models/marker_xgb.pkl \
    --annotation-tsv data/test/W1_annotations.tsv \
    --out            results/W1_plasflow2/

plasflow2 predict \
    --input          data/test/GCA_054405655.1_ASM5440565v1_genomic.fna \
    --model          data/models/mlp_v2.pt \
    --marker-model   data/models/marker_xgb.pkl \
    --annotation-tsv data/test/GCA_annotations.tsv \
    --out            results/GCA_plasflow2/

# Step 4: compare against geNomad pseudo-labels
python scripts/compare_with_genomad.py \
    --plasflow2   results/W1_plasflow2/predictions.tsv \
    --genomad     results/W1/annotated_predictions.tsv \
    --min-conf    0.9 \
    --out         results/W1_comparison/
```

### 4.4 Evaluation strategy (no ground truth)

Since we have no wet-lab-confirmed labels for W1 or GCA, we evaluate using:

1. **Agreement rate with geNomad** (high-confidence subset): what fraction of geNomad's
   ≥0.9 confident plasmid calls do we agree with?
2. **Plasmid call rate by length**: both tools should predict more plasmids in the 10–20kb
   range than in the <2kb range
3. **Biological coherence check**: our predicted plasmids — do they carry more MOB-suite
   markers and rep genes than our predicted chromosomes?
4. **Unique calls**: contigs we call plasmid but geNomad calls chromosome (or vice versa) —
   manual inspection of a random 20-contig sample

---

## 5. Priority Order (What to Do First)

1. **Install BLAST, re-annotate with `rep.dna.fas`** — code already done, 30 min install
2. **Add strand_switch_rate + RBS features** to `annotate_sequences.py` — 2 hours, free
   signal from existing pyrodigal runs
3. **Run geNomad annotate on benchmark** — geNomad already installed, extracts 227k-profile
   SPM scores; retrain XGBoost on these richer features — this is the single highest-impact
   change
4. **Run PlasFlow v2 predict on W1 and GCA** — gives real-world validation numbers
5. **IMG/PR training data** — medium-term, adds non-mobilizable plasmid coverage

---

## 6. Expected Outcomes

| Improvement | Expected plasmid F1 gain | Confidence |
|---|---|---|
| Phase 1: geNomad markers | +0.10 – 0.20 | High |
| Phase 2: RBS + strand features | +0.02 – 0.05 | Medium |
| Phase 3: BLASTN rep.dna.fas | +0.01 – 0.03 | Medium |
| Phase 4: CNN architecture | +0.05 – 0.15 | Medium |
| Phase 5: IMG/PR training data | +0.05 – 0.10 | Medium |
| Phase 6: score calibration | +0.02 – 0.04 | High |
| **Combined (Phases 1–3 + 6)** | **~0.55–0.65** | Medium-High |
| **Combined (all phases)** | **~0.70–0.85** | Medium |

geNomad's F1 ≈ 0.95 reflects both better markers AND a purpose-built dataset curated to
include diverse plasmid types. Matching it exactly requires the same training data breadth.
A realistic target for PlasFlow v2 with Phases 1–4 implemented is F1 ≈ 0.75–0.80 on our
harder metagenome benchmark.

---

## 7. Comparison Script to Write

`scripts/compare_with_genomad.py` — takes PlasFlow v2 predictions TSV and geNomad
`annotated_predictions.tsv`, computes agreement statistics, and outputs a comparison table.
This is needed for both W1 and GCA evaluation.
