#!/usr/bin/env python3
"""
Compute COMPASS MinHash containment scores for plasmid candidates.
Processes sequences in chunks and saves checkpoints for resumability.
Usage: python compass_score_candidates.py [--chunk-size N] [--start-idx I]
"""
import sys
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time

PROJECT = "/sessions/sweet-epic-franklin/mnt/Plasflow"
FASTA = os.path.join(PROJECT, "data/benchmark/tier1/all_species/input.fasta")
SCORES_NPZ = os.path.join(PROJECT, "results/tier1_with_compass/scores.npz")
PREDS_TSV = os.path.join(PROJECT, "results/tier1_with_compass/predictions.tsv")
COMPASS_NPY = os.path.join(PROJECT, "data/databases/sketch_compass_k21_s5m.npy")
IDX_IDS = os.path.join(PROJECT, "data/models/candidates/clean_3class_rev5_hardneg_locked_20260717/evaluation/tier1_candidate/fasta_index_ids.npy")
IDX_OFFS = os.path.join(PROJECT, "data/models/candidates/clean_3class_rev5_hardneg_locked_20260717/evaluation/tier1_candidate/fasta_index_offsets.npy")
CHECKPOINT = os.path.join(PROJECT, "results/compass_candidates_checkpoint.npz")
OUTPUT = os.path.join(PROJECT, "results/compass_candidate_scores.npz")

# ── MinHash implementation ──────────────────────────────────────────────────
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
    rc = (_RC[w[:, ::-1]] * _POW).sum(axis=1)
    h = _mix64(np.where(fwd < rc, fwd, rc))
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
        f.seek(off)
        f.readline()  # skip header
        for line in f:
            if line.startswith(b'>'):
                break
            parts.append(line.rstrip(b'\r\n'))
    return b''.join(parts).decode('ascii', errors='replace')

# ── Parse args ──────────────────────────────────────────────────────────────
chunk_size = 2000
start_override = None
for i, arg in enumerate(sys.argv[1:]):
    if arg == '--chunk-size' and i+1 < len(sys.argv)-1:
        chunk_size = int(sys.argv[i+2])
    if arg == '--start-idx' and i+1 < len(sys.argv)-1:
        start_override = int(sys.argv[i+2])

# ── Load static data ────────────────────────────────────────────────────────
print("Loading COMPASS sketch...")
db_sketch = np.load(COMPASS_NPY)
print(f"  COMPASS sketch: {len(db_sketch):,} hashes")

print("Loading predictions TSV...")
contig_ids_all = []
with open(PREDS_TSV) as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split('\t')
        contig_ids_all.append(parts[0])
contig_ids_all = np.array(contig_ids_all)
print(f"  {len(contig_ids_all):,} contig ids loaded")

print("Loading scores.npz...")
sc = np.load(SCORES_NPZ, allow_pickle=True)
probs = sc['probabilities']       # N×3: plasmid=0, chr=1, phage=2
labels = sc['labels']             # int64
lengths = sc['lengths']           # int64
plasmid_probs = probs[:, 0]       # float32

print("Identifying candidates (plasmid_prob >= 0.50)...")
cand_mask = plasmid_probs >= 0.50
cand_indices = np.where(cand_mask)[0]
print(f"  {len(cand_indices):,} candidates")

print("Building FASTA id→offset map...")
idx_ids = np.load(IDX_IDS, allow_pickle=True)
idx_offs = np.load(IDX_OFFS, allow_pickle=True)
id2off = {cid: int(off) for cid, off in zip(idx_ids, idx_offs)}
print(f"  {len(id2off):,} entries in index")

# ── Load checkpoint if exists ───────────────────────────────────────────────
done_contig_ids = []
done_true_labels = []
done_plasmid_probs = []
done_compass_scores = []
done_lengths = []

start_pos = 0
if os.path.exists(CHECKPOINT):
    print("Loading checkpoint...")
    ck = np.load(CHECKPOINT, allow_pickle=True)
    done_contig_ids = list(ck['contig_ids'])
    done_true_labels = list(ck['true_labels'])
    done_plasmid_probs = list(ck['plasmid_probs'])
    done_compass_scores = list(ck['compass_scores'])
    done_lengths = list(ck['lengths'])
    start_pos = int(ck['next_pos'])
    print(f"  Resumed from pos {start_pos}, {len(done_contig_ids):,} already done")

if start_override is not None:
    start_pos = start_override
    print(f"  Override start_pos to {start_pos}")

end_pos = min(start_pos + chunk_size, len(cand_indices))
print(f"Processing candidates {start_pos} to {end_pos-1} of {len(cand_indices)-1}")

# ── Score chunk ─────────────────────────────────────────────────────────────
missing = 0
t0 = time.time()
for i, ci in enumerate(cand_indices[start_pos:end_pos]):
    global_i = start_pos + i
    cid = contig_ids_all[ci]
    true_label = int(labels[ci])
    pp = float(plasmid_probs[ci])
    length = int(lengths[ci])

    seq = fetch_seq(cid, id2off, FASTA)
    if not seq:
        missing += 1
        cs = 0.0
    else:
        h = _minhash(seq)
        cs = _containment(h, db_sketch)

    done_contig_ids.append(cid)
    done_true_labels.append(true_label)
    done_plasmid_probs.append(pp)
    done_compass_scores.append(cs)
    done_lengths.append(length)

    if (global_i + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        remaining = (len(cand_indices) - global_i - 1) / rate / 60
        print(f"  [{global_i+1}/{len(cand_indices)}] elapsed={elapsed:.1f}s rate={rate:.1f}/s remaining~{remaining:.1f}min missing={missing}")

# ── Save checkpoint ─────────────────────────────────────────────────────────
next_pos = end_pos
np.savez(CHECKPOINT,
    contig_ids=np.array(done_contig_ids),
    true_labels=np.array(done_true_labels, dtype=np.int64),
    plasmid_probs=np.array(done_plasmid_probs, dtype=np.float32),
    compass_scores=np.array(done_compass_scores, dtype=np.float32),
    lengths=np.array(done_lengths, dtype=np.int64),
    next_pos=np.array(next_pos, dtype=np.int64)
)
print(f"\nCheckpoint saved. next_pos={next_pos}, total_done={len(done_contig_ids)}, missing={missing}")

# If complete, save final output
if next_pos >= len(cand_indices):
    np.savez(OUTPUT,
        contig_ids=np.array(done_contig_ids),
        true_labels=np.array(done_true_labels, dtype=np.int64),
        plasmid_probs=np.array(done_plasmid_probs, dtype=np.float32),
        compass_scores=np.array(done_compass_scores, dtype=np.float32),
        lengths=np.array(done_lengths, dtype=np.int64)
    )
    print(f"\nFINAL OUTPUT saved to {OUTPUT}")
    print(f"Total candidates scored: {len(done_contig_ids)}")
else:
    print(f"\nNot done yet. Run again to continue from pos {next_pos}.")
    print(f"Progress: {next_pos}/{len(cand_indices)} = {100*next_pos/len(cand_indices):.1f}%")
