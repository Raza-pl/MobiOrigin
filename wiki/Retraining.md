# Retraining

PlasFlow v2 has two trainable models: the **MLP binary classifier** and the **XGBoost marker model**. Both can be retrained on your own data.

---

## Model architecture

### Stage 1: MLP binary classifier (`mlp_v2.pt`)

- Input: 9,557-dimensional k-mer frequency vector (k=7, with k=6 PCA transform applied first)
- Architecture: 3-layer MLP with batch normalization
- Output: 4-class probabilities (plasmid / chromosome / phage / archaea)
- Training time: ~45–60 min for 50 epochs on CPU (Apple Silicon M-series)
- GPU supported: CUDA (DataParallel); MPS disabled by default (PyTorch ≤ 2.3 instability)

### Stage 2: XGBoost marker model (`marker_xgb.pkl`)

- Input: 17–26 biological features (conjugation proteins, replicon type, coding density, GC%, ORF density, and optionally 9 geNomad SPM features)
- Output: plasmid / chromosome probability
- Blended with MLP scores at runtime

---

## Retraining the MLP

### Step 1 — Build a training dataset

```bash
# Collect plasmid FASTAs (e.g. from PLSDB) and chromosome FASTAs (e.g. from GTDB r220)
# Then fragment them into training windows:
python scripts/build_dataset.py \
  --plasmid-dir   data/plasmids/ \
  --chromosome-dir data/chromosomes/ \
  --phage-dir     data/phages/ \
  --archaea-dir   data/archaea/ \
  --output-dir    data/ \
  --window        10000 \
  --threads       16
```

This produces `data/features.npy`, `data/labels.npy`, and `data/seq_ids.txt`.

### Step 2 — Train

```bash
bash scripts/retrain_k7_binary.sh
```

Or call the training script directly for more control:

```bash
python scripts/train_model.py \
  --data   data/features.npy \
  --labels data/labels.npy \
  --epochs 50 \
  --output data/models/mlp_v2_custom.pt
```

### Apple Silicon note

MPS (Metal Performance Shaders) is disabled by default due to PyTorch ≤ 2.3 instability on large float32 operations. Training runs on CPU. To re-enable MPS:

```bash
PLASFLOW_USE_MPS=1 python scripts/train_model.py ...
```

---

## Retraining the XGBoost marker model

### Step 1 — Generate annotation TSV for your training contigs

```bash
plasflow2 prepare \
  --input  training_contigs.fasta \
  --output data/training_annotations.tsv \
  --threads 16

# Optional: add geNomad features
genomad annotate training_contigs.fasta genomad_out/ data/databases/genomad_db/ --threads 16
plasflow2 prepare \
  --input training_contigs.fasta \
  --output data/training_annotations.tsv \
  --genomad-genes genomad_out/training_contigs_annotate/training_contigs_genes.tsv
```

### Step 2 — Train XGBoost

```bash
python scripts/train_marker_model.py \
  --annotations data/training_annotations.tsv \
  --labels      data/labels.tsv \
  --output      data/models/marker_xgb_custom.pkl
```

### Step 3 — Use the custom model

```bash
plasflow2 run \
  --input         assembly.fasta \
  --output        results/ \
  --model         data/models/mlp_v2_custom.pt \
  --marker-model  data/models/marker_xgb_custom.pkl
```

---

## Available retrain scripts

| Script | Description |
|---|---|
| `scripts/retrain_k7_binary.sh` | End-to-end MLP retrain: dataset build → train → benchmark |
| `scripts/retrain_with_genomad.sh` | Retrain XGBoost adding geNomad SPM features |
| `scripts/retrain_hard_neg.sh` | Retrain MLP with composition FP hard negatives |
| `scripts/build_dataset.py` | Build training windows from FASTA files |
| `scripts/train_model.py` | Train the MLP from numpy feature arrays |
| `scripts/train_marker_model.py` | Train XGBoost from an annotation TSV |
| `scripts/build_marker_dataset.py` | Build the XGBoost training matrix from annotations |
| `scripts/run_benchmark.sh` | Evaluate a trained model on the benchmark dataset |

---

## Benchmark dataset

The benchmark used for PlasFlow v2 evaluation (June 2026):
- 394 true plasmids from PLSDB + RefSeq
- 60,000 chromosome windows from GTDB r220 representative genomes
- Minimum contig length: 1,000 bp
- Built with `scripts/build_benchmark.py`
