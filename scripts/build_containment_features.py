"""
Build containment feature matrix for Rev6 training.
Computes COMPASS and chromosome MinHash containment for each training window.
Resumable: saves checkpoint every 5000 sequences.
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path("/sessions/sweet-epic-franklin/mnt/Plasflow")
DATA = ROOT / "data"
EXP  = DATA / "clean_3class_hardneg_experiment"

# ── MinHash parameters ─────────────────────────────────────────────────────
K = 21
_BASE = np.zeros(256, dtype=np.uint64)
for _b, _v in zip(b'ACGTacgt', [0, 1, 2, 3, 0, 1, 2, 3]):
    _BASE[_b] = _v
_RC   = np.array([3, 2, 1, 0], dtype=np.uint64)
_POW  = np.array([4 ** (K - 1 - j) for j in range(K)], dtype=np.uint64)

def _mix64(x):
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))

def _minhash(seq: str, S: int = 5000):
    arr = _BASE[np.frombuffer(seq.upper().encode('ascii', errors='replace'), dtype=np.uint8)]
    if len(arr) < K:
        return np.array([], dtype=np.uint64)
    w   = sliding_window_view(arr, K)
    fwd = (w * _POW).sum(axis=1)
    rc  = (_RC[w[:, ::-1]] * _POW).sum(axis=1)
    h   = _mix64(np.where(fwd < rc, fwd, rc))
    if len(h) <= S:
        return np.sort(h)
    return np.sort(h[np.argpartition(h, S)[:S]])

def containment(query, db_sketch):
    if len(query) == 0 or len(db_sketch) == 0:
        return 0.0
    idx = np.searchsorted(db_sketch, query)
    idx = np.clip(idx, 0, len(db_sketch) - 1)
    return float((db_sketch[idx] == query).mean())

# ── Index building ─────────────────────────────────────────────────────────
def build_index():
    """Return dict: accession_key -> (file_path_str, header_byte_offset)
    
    Keys:
    - For chr/plasmid: first token after '>' (e.g. 'NC_000918.1')
    - For phage: full header text (e.g. 'JF704115_w2000_s11000')
    """
    index = {}
    sources = []
    
    # Chromosome files
    chr_dir = DATA / "chromosomes" / "bacteria"
    for fna in sorted(chr_dir.glob("*.fna")):
        sources.append(("chr", str(fna)))
    
    # Plasmid FASTA
    sources.append(("plasmid", str(DATA / "databases" / "plasmids" / "plsdb.fasta")))
    
    # Phage FASTA
    sources.append(("phage", str(DATA / "marker_work" / "phage_training.fna")))
    
    total_indexed = 0
    for src_type, fpath in sources:
        with open(fpath, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith(b'>'):
                    hdr = line[1:].rstrip(b'\r\n').decode('ascii', errors='replace')
                    # accession = first token (before space)
                    accession = hdr.split()[0]
                    if accession not in index:
                        index[accession] = (fpath, offset)
                        total_indexed += 1
    
    print(f"Index built: {total_indexed} entries from {len(sources)} sources")
    return index

# ── Fetch window ───────────────────────────────────────────────────────────
def fetch_window(accession: str, window_size: int, start: int, index: dict) -> str | None:
    # For chr/plasmid: index key = accession
    # For phage: FASTA header is like 'JF704115_w2000_s11000', so accession == full header
    entry = index.get(accession)
    if entry is None:
        return None
    fpath, hdr_offset = entry
    with open(fpath, 'rb') as f:
        f.seek(hdr_offset)
        f.readline()  # skip header
        seq_parts = []
        while True:
            line = f.readline()
            if not line or line.startswith(b'>'):
                break
            seq_parts.append(line.rstrip(b'\r\n'))
    seq = b''.join(seq_parts).decode('ascii', errors='replace')
    return seq[start : start + window_size]

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-start', type=int, default=0, help='Start index')
    parser.add_argument('--batch-size',  type=int, default=5000)
    parser.add_argument('--build-index-only', action='store_true')
    args = parser.parse_args()

    TOTAL = 358919
    PARTIAL = EXP / "containment_partial.npy"
    OUT     = EXP / "containment_features.npy"

    # Load sketches
    print("Loading sketches...")
    compass_sketch = np.load(DATA / "databases" / "sketch_compass_k21_s5m.npy")
    chr_sketch     = np.load(DATA / "databases" / "sketch_chromosomes_k21_s5m.npy")
    print(f"  compass_sketch: {len(compass_sketch):,} hashes")
    print(f"  chr_sketch:     {len(chr_sketch):,} hashes")

    # Build index
    print("Building accession index...")
    t0 = time.time()
    index = build_index()
    print(f"  Done in {time.time()-t0:.1f}s")

    if args.build_index_only:
        return

    # Load seq_ids
    with open(EXP / "seq_ids.txt") as f:
        seq_ids = [l.strip() for l in f]
    assert len(seq_ids) == TOTAL

    # Determine resume point
    resume_from = 0
    if PARTIAL.exists():
        partial = np.load(PARTIAL)
        resume_from = len(partial)
        print(f"Resuming from seq {resume_from} (partial has {resume_from} rows)")
        buffer = list(partial)
    else:
        buffer = []

    # Determine batch to process
    batch_start = max(args.batch_start, resume_from)
    batch_end   = min(batch_start + args.batch_size, TOTAL)

    if batch_start >= TOTAL:
        print(f"Already complete ({resume_from}/{TOTAL}). Nothing to do.")
        if resume_from == TOTAL and not OUT.exists():
            arr = np.array(buffer, dtype=np.float32)
            np.save(OUT, arr)
            print(f"Saved final: {OUT} shape={arr.shape}")
        return

    print(f"Processing sequences {batch_start} to {batch_end-1} ...")
    
    missing = 0
    t_batch = time.time()
    
    for i in range(batch_start, batch_end):
        sid = seq_ids[i]
        # Parse: ACCESSION_wW_sS
        # Accession may contain dots and underscores, window marker is _w<digits>_s<digits>
        # Split from the right to find _w...
        try:
            # Find last _w<digits>_s<digits> pattern
            import re
            m = re.match(r'^(.+?)_w(\d+)_s(\d+)$', sid)
            if m:
                accession   = m.group(1)
                window_size = int(m.group(2))
                start       = int(m.group(3))
            else:
                # fallback: try splitting on _w
                parts = sid.rsplit('_w', 1)
                accession = parts[0]
                w_s = parts[1].split('_s')
                window_size = int(w_s[0])
                start = int(w_s[1])
        except Exception as e:
            print(f"  Parse error for {sid}: {e}")
            buffer.append([0.0, 0.0])
            missing += 1
            continue

        # For phage sequences stored as windows: FASTA header = 'ACC_wW_sS'
        # So try accession first, then full seq_id
        seq = fetch_window(accession, window_size, start, index)
        
        if seq is None:
            # Try full seq_id as key (for phage)
            full_key = sid  # e.g. 'JF704115_w2000_s11000'
            seq = fetch_window(full_key, 0, 0, index)
            if seq is None:
                # Also try accession with version stripped for phage
                missing += 1
                buffer.append([0.0, 0.0])
                if missing <= 10:
                    print(f"  Missing: {sid}")
                continue

        # Compute MinHash
        q = _minhash(seq, S=5000)
        cc = containment(q, compass_sketch)
        ch = containment(q, chr_sketch)
        buffer.append([cc, ch])

        # Progress report
        if (i + 1) % 5000 == 0:
            arr = np.array(buffer, dtype=np.float32)
            compass_mean = arr[:, 0].mean()
            chr_mean     = arr[:, 1].mean()
            elapsed = time.time() - t_batch
            rate = (i - batch_start + 1) / elapsed
            eta_s = (TOTAL - i - 1) / rate if rate > 0 else 0
            print(f"  Seq {i+1}/{TOTAL}, compass_mean={compass_mean:.4f}, "
                  f"chr_mean={chr_mean:.4f}, missing={missing}, "
                  f"rate={rate:.0f}/s, ETA={eta_s/3600:.1f}h")

    # Save checkpoint
    arr = np.array(buffer, dtype=np.float32)
    np.save(PARTIAL, arr)
    print(f"Checkpoint saved: {PARTIAL} shape={arr.shape}, missing so far={missing}")

    # If done, save final
    if len(buffer) >= TOTAL:
        final = arr[:TOTAL]
        np.save(OUT, final)
        print(f"FINAL SAVED: {OUT} shape={final.shape}")
        print(f"  compass: mean={final[:,0].mean():.4f} max={final[:,0].max():.4f}")
        print(f"  chr:     mean={final[:,1].mean():.4f} max={final[:,1].max():.4f}")
        print(f"  missing: {missing}")

if __name__ == '__main__':
    import re
    main()
