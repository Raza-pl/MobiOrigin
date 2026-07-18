#!/usr/bin/env python3
"""
Compute COMPASS MinHash containment scores for all plasmid candidates.
Runs to completion with periodic checkpoint saves.
"""
import sys, os, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PROJECT = "/sessions/sweet-epic-franklin/mnt/Plasflow"
FASTA   = f"{PROJECT}/data/benchmark/tier1/all_species/input.fasta"
SCORES  = f"{PROJECT}/results/tier1_with_compass/scores.npz"
PREDS   = f"{PROJECT}/results/tier1_with_compass/predictions.tsv"
SKETCH  = f"{PROJECT}/data/databases/sketch_compass_k21_s5m.npy"
IDX_IDS = f"{PROJECT}/data/models/candidates/clean_3class_rev5_hardneg_locked_20260717/evaluation/tier1_candidate/fasta_index_ids.npy"
IDX_OFF = f"{PROJECT}/data/models/candidates/clean_3class_rev5_hardneg_locked_20260717/evaluation/tier1_candidate/fasta_index_offsets.npy"
CKPT    = f"{PROJECT}/results/compass_candidates_checkpoint.npz"
OUTPUT  = f"{PROJECT}/results/compass_candidate_scores.npz"
LOGFILE = f"{PROJECT}/results/compass_scoring.log"

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[logging.FileHandler(LOGFILE), logging.StreamHandler()]
)
log = logging.info

# ── MinHash ─────────────────────────────────────────────────────────────────
K = 21
_BASE = np.zeros(256, dtype=np.uint64)
for _b, _v in zip(b'ACGTacgt', [0, 1, 2, 3, 0, 1, 2, 3]):
    _BASE[_b] = _v
_RC = np.array([3, 2, 1, 0], dtype=np.uint64)
_POW = np.array([4 ** (K - 1 - j) for j in range(K)], dtype=np.uint64)

def _mix64(x):
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))

def _minhash(seq, S=5000):
    arr = _BASE[np.frombuffer(seq.upper().encode('ascii', errors='replace'), dtype=np.uint8)]
    if len(arr) < K:
        return np.array([], dtype=np.uint64)
    w = sliding_window_view(arr, K)
    fwd = (w * _POW).sum(axis=1)
    rc  = (_RC[w[:, ::-1]] * _POW).sum(axis=1)
    h   = _mix64(np.where(fwd < rc, fwd, rc))
    if len(h) <= S:
        return np.sort(h)
    return np.sort(h[np.argpartition(h, S)[:S]])

def _containment(query, db_sketch):
    if len(query) == 0 or len(db_sketch) == 0:
        return 0.0
    return float(np.isin(query, db_sketch, assume_unique=True).mean())

def fetch_seq(cid, id2off, fasta_path):
    off = id2off.get(cid)
    if off is None:
        return ''
    parts = []
    with open(fasta_path, 'rb') as f:
        f.seek(int(off))
        f.readline()
        for line in f:
            if line.startswith(b'>'):
                break
            parts.append(line.rstrip(b'\r\n'))
    return b''.join(parts).decode('ascii', errors='replace')

# ── Load static data ─────────────────────────────────────────────────────────
log("Loading COMPASS sketch...")
db_sketch = np.load(SKETCH)
log(f"  {len(db_sketch):,} hashes")

log("Loading predictions TSV...")
contig_ids_all = []
with open(PREDS) as f:
    f.readline()
    for line in f:
        contig_ids_all.append(line.split('\t')[0])
contig_ids_all = np.array(contig_ids_all)
log(f"  {len(contig_ids_all):,} contigs")

log("Loading scores.npz...")
sc = np.load(SCORES, allow_pickle=True)
probs   = sc['probabilities']
labels  = sc['labels']
lengths = sc['lengths']
plasmid_probs = probs[:, 0]

cand_mask    = plasmid_probs >= 0.50
cand_indices = np.where(cand_mask)[0]
log(f"  {len(cand_indices):,} candidates")

log("Building id→offset map...")
idx_ids = np.load(IDX_IDS, allow_pickle=True)
idx_off = np.load(IDX_OFF, allow_pickle=True)
id2off  = {cid: int(off) for cid, off in zip(idx_ids, idx_off)}
log(f"  {len(id2off):,} entries")

# ── Resume from checkpoint ────────────────────────────────────────────────────
done_cids    = []
done_labels  = []
done_pprobs  = []
done_cscores = []
done_lengths = []
start_pos    = 0

if os.path.exists(CKPT):
    log("Loading checkpoint...")
    ck = np.load(CKPT, allow_pickle=True)
    done_cids    = list(ck['contig_ids'])
    done_labels  = list(ck['true_labels'])
    done_pprobs  = list(ck['plasmid_probs'])
    done_cscores = list(ck['compass_scores'])
    done_lengths = list(ck['lengths'])
    start_pos    = int(ck['next_pos'])
    log(f"  Resumed from pos {start_pos}, {len(done_cids):,} done")

# ── Score all ─────────────────────────────────────────────────────────────────
SAVE_EVERY = 5000
missing = 0
t0 = time.time()
log(f"Starting from pos {start_pos}, total={len(cand_indices)}")

for i, ci in enumerate(cand_indices[start_pos:]):
    global_i = start_pos + i
    cid      = contig_ids_all[ci]
    seq      = fetch_seq(cid, id2off, FASTA)
    if not seq:
        missing += 1
        cs = 0.0
    else:
        h  = _minhash(seq)
        cs = _containment(h, db_sketch)

    done_cids.append(cid)
    done_labels.append(int(labels[ci]))
    done_pprobs.append(float(plasmid_probs[ci]))
    done_cscores.append(cs)
    done_lengths.append(int(lengths[ci]))

    if (global_i + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed
        eta_min = (len(cand_indices) - global_i - 1) / rate / 60
        log(f"  [{global_i+1}/{len(cand_indices)}] {rate:.1f}/s eta={eta_min:.1f}min missing={missing}")

    if (global_i + 1) % SAVE_EVERY == 0:
        np.savez(CKPT,
            contig_ids    = np.array(done_cids),
            true_labels   = np.array(done_labels,  dtype=np.int64),
            plasmid_probs = np.array(done_pprobs,  dtype=np.float32),
            compass_scores= np.array(done_cscores, dtype=np.float32),
            lengths       = np.array(done_lengths, dtype=np.int64),
            next_pos      = np.array(global_i + 1, dtype=np.int64)
        )
        log(f"  Checkpoint saved at pos {global_i+1}")

# ── Save final output ─────────────────────────────────────────────────────────
np.savez(OUTPUT,
    contig_ids    = np.array(done_cids),
    true_labels   = np.array(done_labels,  dtype=np.int64),
    plasmid_probs = np.array(done_pprobs,  dtype=np.float32),
    compass_scores= np.array(done_cscores, dtype=np.float32),
    lengths       = np.array(done_lengths, dtype=np.int64)
)
total_elapsed = time.time() - t0
log(f"\nDONE! {len(done_cids):,} candidates scored in {total_elapsed/60:.1f}min, missing={missing}")
log(f"Output: {OUTPUT}")
