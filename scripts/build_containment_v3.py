"""
Rev6 containment feature builder v3 (final): multiprocessing MinHash + resumable.
Sources:
  Phage:    inphared_phages.fa.gz
  Plasmid:  COMPASS.fna + plsdb.fasta
  Chr:      chromosomes.fna + combined_hard_negatives/*.fna + gtdb_genomes/bacteria/*.fna.gz
"""
from __future__ import annotations
import gzip, os, re, time, pickle, tempfile, shutil
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from multiprocessing import Pool

ROOT = Path("/sessions/sweet-epic-franklin/mnt/Plasflow")
DATA = ROOT / "data"
EXP  = DATA / "clean_3class_hardneg_experiment"

K=21; S=5000
_BASE_INIT = list(zip(b'ACGTacgt', [0,1,2,3,0,1,2,3]))
_RC_LIST   = [3,2,1,0]
_POW_LIST  = [4**(K-1-j) for j in range(K)]

def _init_worker(compass_path, chr_path):
    global _COMPASS_SK, _CHR_SK, _BASE, _RC, _POW
    import numpy as np
    _BASE = np.zeros(256, dtype=np.uint64)
    for b_, v_ in _BASE_INIT: _BASE[b_] = v_
    _RC   = np.array(_RC_LIST,  dtype=np.uint64)
    _POW  = np.array(_POW_LIST, dtype=np.uint64)
    _COMPASS_SK = np.load(compass_path)
    _CHR_SK     = np.load(chr_path)

def _mix64(x):
    x=x^(x>>np.uint64(30)); x=x*np.uint64(0xBF58476D1CE4E5B9)
    x=x^(x>>np.uint64(27)); x=x*np.uint64(0x94D049BB133111EB)
    return x^(x>>np.uint64(31))

def _compute_one(seq):
    arr = _BASE[np.frombuffer(seq.upper().encode('ascii', errors='replace'), dtype=np.uint8)]
    if len(arr) < K: return 0.0, 0.0
    w   = sliding_window_view(arr, K)
    fwd = (w * _POW).sum(axis=1); rc = (_RC[w[:,::-1]] * _POW).sum(axis=1)
    h   = _mix64(np.where(fwd < rc, fwd, rc))
    if len(h) > S: h = h[np.argpartition(h, S)[:S]]
    h = np.sort(h)
    def _c(db):
        if not len(h): return 0.0
        idx = np.clip(np.searchsorted(db, h), 0, len(db)-1)
        return float((db[idx]==h).mean())
    return _c(_COMPASS_SK), _c(_CHR_SK)

_ID_RE = re.compile(r'^(.+?)_w(\d+)_s(\d+)$')

def build_lookup(seq_ids):
    lk = {}
    for i, sid in enumerate(seq_ids):
        m = _ID_RE.match(sid)
        if m:
            lk.setdefault(m.group(1), []).append((i, int(m.group(2)), int(m.group(3))))
    return lk

def open_fasta(fpath):
    return gzip.open(fpath, 'rb') if str(fpath).endswith('.gz') else open(fpath, 'rb')

def stream_source(fpath, label, lookup, results, filled, pool,
                  start_seq=0, time_limit=36.0):
    BATCH = 64
    t0 = time.time()
    print(f"  [{label}] {Path(fpath).name} from seq {start_seq}", flush=True)
    n_seq=0; n_found=0
    batch_wins=[]; batch_rows=[]

    def flush():
        nonlocal n_found
        if not batch_wins: return
        rs = pool.map(_compute_one, batch_wins)
        for (cc, ch), row_idx in zip(rs, batch_rows):
            if not filled[row_idx]:
                results[row_idx, 0] = cc; results[row_idx, 1] = ch
                filled[row_idx] = True; n_found += 1
        batch_wins.clear(); batch_rows.clear()

    with open_fasta(fpath) as f:
        acc=None; parts=[]
        for line in f:
            if line.startswith(b'>'):
                if acc is not None:
                    n_seq += 1
                    if n_seq > start_seq:
                        hits = lookup.get(acc, [])
                        if hits:
                            seq = b''.join(parts).decode('ascii', errors='replace')
                            for (ri, w, s) in hits:
                                if not filled[ri]:
                                    batch_wins.append(seq[s:s+w])
                                    batch_rows.append(ri)
                                    if len(batch_wins) >= BATCH: flush()
                    parts = []
                    if time.time()-t0 > time_limit:
                        flush()
                        print(f"  [{label}] time limit at seq {n_seq}", flush=True)
                        return n_seq, n_found, False
                acc = line[1:].rstrip(b'\r\n').decode('ascii', errors='replace').split()[0]
            else:
                parts.append(line.rstrip(b'\r\n'))
        if acc:
            n_seq += 1
            if n_seq > start_seq:
                hits = lookup.get(acc, [])
                if hits:
                    seq = b''.join(parts).decode('ascii', errors='replace')
                    for (ri, w, s) in hits:
                        if not filled[ri]:
                            batch_wins.append(seq[s:s+w])
                            batch_rows.append(ri)
    flush()
    print(f"  [{label}] DONE: {n_seq} seqs, {n_found} found, {time.time()-t0:.1f}s", flush=True)
    return n_seq, n_found, True

def save_state(results, filled, done_s, partial_s, PARTIAL, STATE):
    """Atomic state save: write to temp then rename."""
    tmp_r = str(PARTIAL).replace(".npy", "_tmp.npy")
    tmp_s = str(STATE) + ".tmp"
    np.save(tmp_r, results)
    with open(tmp_s, 'wb') as f:
        pickle.dump({'done_sources': done_s, 'partial_source': partial_s, 'filled': filled}, f)
    os.replace(tmp_r, str(PARTIAL))
    os.replace(tmp_s, str(STATE))

def main():
    TOTAL=358919
    OUT    =EXP/"containment_features.npy"
    PARTIAL=EXP/"containment_partial.npy"
    STATE  =EXP/"containment_state.pkl"

    if OUT.exists():
        arr=np.load(OUT)
        print(f"Already complete: {arr.shape}  compass={arr[:,0].mean():.4f}  chr={arr[:,1].mean():.4f}")
        return

    with open(EXP/"seq_ids.txt") as f:
        seq_ids=[l.strip() for l in f]
    assert len(seq_ids)==TOTAL
    lookup=build_lookup(seq_ids)
    print(f"Lookup: {len(lookup):,} keys", flush=True)

    # Full source list
    sources = [
        (str(DATA/"databases"/"inphared"/"inphared_phages.fa.gz"), "phage"),
        (str(DATA/"databases"/"plasmids"/"COMPASS.fna"),           "compass"),
        (str(DATA/"databases"/"plasmids"/"plsdb.fasta"),           "plsdb"),
        (str(DATA/"databases"/"chromosomes.fna"),                  "chr_refseq"),
    ]
    for fna in sorted((DATA/"combined_hard_negatives").glob("*.fna")):
        sources.append((str(fna), "chr_hardneg"))
    for gz in sorted((DATA/"gtdb_genomes"/"bacteria").glob("*.fna.gz")):
        sources.append((str(gz), "chr_gtdb"))

    # Load state
    if STATE.exists() and PARTIAL.exists():
        try:
            state=pickle.load(open(STATE,'rb'))
            results=np.load(PARTIAL)
            filled=state['filled']
            done_s=state['done_sources']
            partial_s=state.get('partial_source',None)
            print(f"Resuming: {filled.sum()}/{TOTAL}, {len(done_s)} done", flush=True)
        except Exception as e:
            print(f"State corrupted ({e}), reconstructing from labels...", flush=True)
            labels=np.load(EXP/"labels.npy")
            results=np.load(PARTIAL)
            filled=np.zeros(TOTAL,dtype=bool)
            # plsdb+compass+phage are done, reconstruct chr from partial
            filled[labels==0]=True; filled[labels==2]=True
            filled[labels==1]=(results[labels==1,1]>1e-7)
            done_s={str(DATA/"databases"/"inphared"/"inphared_phages.fa.gz"),
                    str(DATA/"databases"/"plasmids"/"COMPASS.fna"),
                    str(DATA/"databases"/"plasmids"/"plsdb.fasta")}
            partial_s=None
            print(f"Reconstructed: {filled.sum()}/{TOTAL}", flush=True)
    else:
        results=np.zeros((TOTAL,2),dtype=np.float32)
        filled=np.zeros(TOTAL,dtype=bool)
        done_s=set(); partial_s=None
        print("Starting fresh", flush=True)

    compass_path=str(DATA/"databases"/"sketch_compass_k21_s5m.npy")
    chr_path    =str(DATA/"databases"/"sketch_chromosomes_k21_s5m.npy")
    print("Starting pool...", flush=True)
    with Pool(4, initializer=_init_worker, initargs=(compass_path, chr_path)) as pool:
        print(f"Pool ready. Filled: {filled.sum()}/{TOTAL}", flush=True)

        for fpath, label in sources:
            if fpath in done_s: continue
            start=0
            if partial_s and partial_s[0]==fpath:
                start=partial_s[1]

            n_seq, n_found, done = stream_source(
                fpath, label, lookup, results, filled, pool,
                start_seq=start, time_limit=36.0
            )
            if done:
                done_s.add(fpath)
                partial_s=None
            else:
                partial_s=(fpath, n_seq)
                save_state(results, filled, done_s, partial_s, PARTIAL, STATE)
                print(f"Checkpoint: {filled.sum()}/{TOTAL}, {(~filled).sum()} missing", flush=True)
                return

            save_state(results, filled, done_s, None, PARTIAL, STATE)
            print(f"  -> Total filled: {filled.sum()}/{TOTAL}", flush=True)

    # Final
    missing=int((~filled).sum())
    print(f"\n=== COMPLETE === filled={filled.sum()}/{TOTAL} missing={missing}", flush=True)
    print(f"compass: mean={results[:,0].mean():.4f} max={results[:,0].max():.4f}", flush=True)
    print(f"chr:     mean={results[:,1].mean():.4f} max={results[:,1].max():.4f}", flush=True)
    np.save(OUT, results)
    print(f"Saved: {OUT}", flush=True)
    if PARTIAL.exists(): PARTIAL.unlink()
    if STATE.exists():   STATE.unlink()

if __name__=='__main__':
    main()
