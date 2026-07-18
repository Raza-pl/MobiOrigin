# Retraining

PlasFlow v2 has two trainable models: the **MLP binary classifier** and the **XGBoost marker model**. Both can be retrained on your own data.

---

## Model architecture

### Stage 1: MLP binary classifier (`mlp_v2.pt`)

- Input: 9,557-dimensional k-mer frequency vector (k=7, with k=6 PCA transform applied first)
- Architecture: 3-layer MLP with batch normalization
- Output: 3-class probabilities (plasmid / chromosome / phage)
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
python scripts/dev/build_dataset.py \
  --plasmid-dir    data/plasmids/ \
  --chromosome-dir data/chromosomes/ \
  --phage-dir      data/phages/ \
  --output-dir     data/ \
  --window         10000 \
  --threads        16
```

> `scripts/dev/` contains training and benchmark scripts that are local-only (not committed to git). Contact the developer if you need these files.

This produces `data/features.npy`, `data/labels.npy`, and `data/seq_ids.txt`.

### Step 2 — Train

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

Use `plasflow2 run` on your training data to produce the annotation TSV, then extract it from the output directory:

```bash
plasflow2 run \
  --input   training_contigs.fasta \
  --output  training_run/ \
  --threads 16

# The annotation TSV is written at:
# training_run/work/arg_annotation/  (per-annotation files)
# Copy or use training_run/all_predictions.tsv for features
```

> **Note:** A dedicated `plasflow2 prepare` command for standalone annotation-TSV generation is planned for a future release. For now, use `plasflow2 run` on your training data and extract the annotation columns from `all_predictions.tsv`.

### Step 2 — Train XGBoost

```bash
python scripts/train_marker_model.py \
  --features data/marker_features.npz \
  --out      data/models/
```

The `--features` `.npz` file is produced by `scripts/build_marker_dataset.py`. The trained weights are written to `{--out}/marker_xgb.pkl`.

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

Scripts in `scripts/` are user-facing and tracked in git. Scripts in `scripts/dev/` are local-only training and benchmark tools (not committed to git).

| Script | Location | Description |
|--------|----------|-------------|
| `train_model.py` | `scripts/` | Train the MLP from numpy feature arrays |
| `train_marker_model.py` | `scripts/` | Train XGBoost from an annotation TSV |
| `build_dataset.py` | `scripts/dev/` | Build training windows from FASTA files |
| `build_marker_dataset.py` | `scripts/dev/` | Build the XGBoost training matrix from annotations |
| `retrain_k7_binary.sh` | `scripts/dev/` | End-to-end MLP retrain: dataset build → train → benchmark |
| `retrain_with_genomad.sh` | `scripts/dev/` | Retrain XGBoost adding geNomad SPM features |
| `retrain_hard_neg.sh` | `scripts/dev/` | Retrain MLP with composition FP hard negatives |
| `run_benchmark.sh` | `scripts/dev/` | Evaluate a trained model on the benchmark dataset |

---

## Benchmark dataset

The benchmark used for PlasFlow v2 evaluation (June 2026):
- 394 true plasmids from PLSDB + RefSeq
- 60,000 chromosome windows from GTDB r220 representative genomes
- Minimum contig length: 1,000 bp
- Built with `scripts/build_benchmark.py`
