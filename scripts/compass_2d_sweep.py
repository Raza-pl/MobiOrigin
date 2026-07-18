#!/usr/bin/env python3
"""
2D sweep over (plasmid_threshold, compass_threshold) using pre-computed COMPASS scores.
"""
import numpy as np
import os

PROJECT = "/sessions/sweet-epic-franklin/mnt/Plasflow"
CAND    = f"{PROJECT}/results/compass_candidate_scores.npz"
SCORES  = f"{PROJECT}/results/tier1_with_compass/scores.npz"
OUTPUT  = f"{PROJECT}/results/compass_2d_sweep.tsv"

# ── Load candidate scores ─────────────────────────────────────────────────────
print("Loading candidate scores...")
cd = np.load(CAND, allow_pickle=True)
cand_labels   = cd['true_labels']       # int64, 0=plasmid
cand_pprobs   = cd['plasmid_probs']     # float32
cand_cscores  = cd['compass_scores']    # float32
cand_lengths  = cd['lengths']           # int64
print(f"  {len(cand_labels):,} candidates")

# ── Load full scores for non-candidate FN counting ───────────────────────────
print("Loading full scores.npz...")
sc = np.load(SCORES, allow_pickle=True)
all_labels  = sc['labels']        # int64
all_probs   = sc['probabilities'] # N×3
all_lengths = sc['lengths']       # int64
all_pprobs  = all_probs[:, 0]

# Non-candidates: plasmid_prob < 0.50 → never predicted as plasmid
noncand_mask       = all_pprobs < 0.50
noncand_labels     = all_labels[noncand_mask]
noncand_pplasmid   = (noncand_labels == 0).sum()   # True plasmids lost as non-candidates
noncand_lengths    = all_lengths[noncand_mask]

# Total true plasmids in the full benchmark
total_true_plasmids = (all_labels == 0).sum()
print(f"  Total true plasmids: {total_true_plasmids:,}")
print(f"  Non-candidate true plasmids (always FN): {noncand_pplasmid:,}")

# ── Sweep parameters ──────────────────────────────────────────────────────────
plasm_thresholds  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.74, 0.75, 0.76,
                     0.78, 0.80, 0.82, 0.84, 0.86, 0.862, 0.88, 0.90]
compass_thresholds = [0.0000, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.010]

results = []
for pt in plasm_thresholds:
    for ct in compass_thresholds:
        # Predict plasmid: prob >= pt AND compass >= ct (for candidates only)
        pred_plasmid = (cand_pprobs >= pt) & (cand_cscores >= ct)

        tp = int(((cand_labels == 0) & pred_plasmid).sum())
        fp = int(((cand_labels != 0) & pred_plasmid).sum())
        # FN = plasmids in candidates not predicted + plasmids in non-candidates
        fn_cand    = int(((cand_labels == 0) & ~pred_plasmid).sum())
        fn_noncand = int(noncand_pplasmid)
        fn = fn_cand + fn_noncand

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        n_pred    = int(pred_plasmid.sum())

        results.append({
            'plasmid_threshold': pt,
            'compass_threshold': ct,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn,
            'n_predicted': n_pred,
        })

# Sort by F1
results.sort(key=lambda x: x['f1'], reverse=True)

# ── Print top 20 ──────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"{'Rank':>4}  {'pt':>6}  {'ct':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'N_pred':>7}")
print(f"{'='*90}")
for i, r in enumerate(results[:20]):
    print(f"{i+1:>4}  {r['plasmid_threshold']:>6.3f}  {r['compass_threshold']:>7.4f}  "
          f"{r['precision']:>7.4f}  {r['recall']:>7.4f}  {r['f1']:>7.4f}  "
          f"{r['tp']:>6}  {r['fp']:>6}  {r['fn']:>6}  {r['n_predicted']:>7}")

best = results[0]
print(f"\nbest combination: plasmid_threshold={best['plasmid_threshold']}, "
      f"compass_threshold={best['compass_threshold']}, F1={best['f1']:.4f}")
print(f"  Precision={best['precision']:.4f}, Recall={best['recall']:.4f}")
print(f"  TP={best['tp']}, FP={best['fp']}, FN={best['fn']}, N_predicted={best['n_predicted']}")

# ── Per-length-tier F1 at best combination ────────────────────────────────────
pt_best = best['plasmid_threshold']
ct_best = best['compass_threshold']

tiers = [
    ('<2 kb',    0,      2000),
    ('2-5 kb',   2000,   5000),
    ('5-10 kb',  5000,   10000),
    ('10-50 kb', 10000,  50000),
    ('>50 kb',   50000,  10**9),
]

print(f"\nPer-length-tier F1 at best combination (pt={pt_best}, ct={ct_best}):")
print(f"  {'Tier':<12}  {'N_true':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")

for tier_name, lo, hi in tiers:
    # Candidates in this tier
    tier_cand = (cand_lengths >= lo) & (cand_lengths < hi)
    tier_pred = (cand_pprobs >= pt_best) & (cand_cscores >= ct_best) & tier_cand
    tp_t = int(((cand_labels == 0) & tier_pred).sum())
    fp_t = int(((cand_labels != 0) & tier_pred).sum())
    fn_cand_t = int(((cand_labels == 0) & tier_cand & ~tier_pred).sum())

    # Non-candidates in this tier
    tier_noncand = (noncand_lengths >= lo) & (noncand_lengths < hi)
    fn_noncand_t = int((noncand_labels[tier_noncand] == 0).sum())
    fn_t = fn_cand_t + fn_noncand_t

    n_true_t = tp_t + fn_t
    p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    f1_t = 2*p_t*r_t/(p_t+r_t) if (p_t+r_t) > 0 else 0.0

    print(f"  {tier_name:<12}  {n_true_t:>7}  {tp_t:>6}  {fp_t:>6}  {fn_t:>6}  {p_t:>7.4f}  {r_t:>7.4f}  {f1_t:>7.4f}")

# ── Save TSV ──────────────────────────────────────────────────────────────────
with open(OUTPUT, 'w') as f:
    f.write("plasmid_threshold\tcompass_threshold\tprecision\trecall\tf1\ttp\tfp\tfn\tn_predicted\n")
    for r in results:
        f.write(f"{r['plasmid_threshold']}\t{r['compass_threshold']}\t"
                f"{r['precision']:.6f}\t{r['recall']:.6f}\t{r['f1']:.6f}\t"
                f"{r['tp']}\t{r['fp']}\t{r['fn']}\t{r['n_predicted']}\n")
print(f"\nSweep results saved to: {OUTPUT}")
