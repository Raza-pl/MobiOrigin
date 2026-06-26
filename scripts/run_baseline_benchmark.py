#!/usr/bin/env python3
"""
Run the paper baseline model (marker_xgb_binary_backup_20260625_234542.pkl)
on the full benchmark and extract FP IDs for PLSDB validation.

Run from project root:
    python3 scripts/run_baseline_benchmark.py

Then validate FPs against PLSDB:
    bash scripts/run_fp_minimap2.sh \
        results/baseline_plsdb_validation/predictions.tsv \
        data/databases/plasmids/plsdb.fasta \
        data/databases/plasmids/COMPASS.fna
"""
# Cap BLAS/OpenMP threads BEFORE any numpy/torch import — prevents ARM segfault
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, csv, time
sys.path.insert(0, 'src')
import warnings; warnings.filterwarnings('ignore')

from plasflow2.utils.fasta import load_fasta
from plasflow2.classify.predict import predict

BASELINE_MODEL = "data/models/marker_xgb_binary_backup_20260625_234542.pkl"
MLP_MODEL      = "data/models/mlp_v2.pt"
BENCHMARK_FASTA = "data/benchmark/benchmark.fna"
GROUND_TRUTH    = "data/benchmark/ground_truth.tsv"
OUT_DIR         = "results/baseline_plsdb_validation"

gt = {}
with open(GROUND_TRUTH) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        gt[row["contig_id"]] = row["true_label"]

print("Loading benchmark sequences...", flush=True)
records = load_fasta(BENCHMARK_FASTA, min_length=1)
seqs = [str(r.seq) for r in records]
ids  = [r.id      for r in records]
print(f"  {len(seqs)} sequences", flush=True)

print(f"\nRunning baseline model: {BASELINE_MODEL}", flush=True)
t0 = time.time()
preds = predict(
    sequences=seqs,
    sequence_ids=ids,
    model_path=MLP_MODEL,
    threshold=0.70,
    plasmid_threshold=0.95,
    argmax_fallback=False,
    source_context="unspecified",
    apply_prior=False,
    marker_model_path=BASELINE_MODEL,
    annotation_tsv=None,
    use_pyrodigal=True,
    marker_alpha_base=0.3,
)
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)

tp=fp=fn=0
fp_ids=[]
with open(f"{OUT_DIR}/predictions.tsv", "w") as out:
    out.write("contig_id\tpredicted\tconfidence\tplasmid_score\ttrue_label\n")
    for p in preds:
        true = gt.get(p.sequence_id, "unknown")
        out.write(
            f"{p.sequence_id}\t{p.label}\t{p.confidence:.6f}\t"
            f"{p.scores.get('plasmid', 0):.6f}\t{true}\n"
        )
        if p.label not in ("plasmid", "chromosome"):
            continue
        if p.label == "plasmid" and true == "plasmid":   tp += 1
        elif p.label == "plasmid" and true == "chromosome": fp += 1; fp_ids.append(p.sequence_id)
        elif p.label == "chromosome" and true == "plasmid": fn += 1

prec = tp/(tp+fp) if tp+fp else 0
rec  = tp/(tp+fn) if tp+fn else 0
f1   = 2*prec*rec/(prec+rec) if prec+rec else 0

print(f"\n=== Baseline model results (classified-only) ===")
print(f"  TP={tp}  FP={fp}  FN={fn}")
print(f"  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

with open(f"{OUT_DIR}/fp_ids.txt", "w") as f:
    f.write("\n".join(fp_ids))

print(f"\nPredictions → {OUT_DIR}/predictions.tsv")
print(f"FP IDs      → {OUT_DIR}/fp_ids.txt ({len(fp_ids)} FPs)")
print(f"\nNext step — PLSDB validation:")
print(f"  bash scripts/run_fp_minimap2.sh \\")
print(f"      {OUT_DIR}/predictions.tsv \\")
print(f"      data/databases/plasmids/plsdb.fasta \\")
print(f"      data/databases/plasmids/COMPASS.fna")
