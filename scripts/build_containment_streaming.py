"""
Rev6 containment feature builder — streaming approach (v2, correct sources).
Sources:
  - Phage:    data/databases/inphared/inphared_phages.fa.gz  (accession lookup, extract window)
  - Plasmid:  data/databases/plasmids/plsdb.fasta            (accession lookup, extract window)
              data/databases/plasmids/COMPASS.fna             (COMPASS_* prefixed accessions)
  - Chr:      data/chromosomes/bacteria/*.fna                 (accession lookup, extract window)

Streaming approach: read each source file once sequentially;
for each sequence look up required windows; compute MinHash containment.
Saves checkpoint after each source file (resumable).
"""
from __future__ import annotations
import gzip, os, re, sys, time, pickle
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path("/sessions/sweet-epic-franklin/mnt/Plasflow")
DATA = ROOT / "data"
EXP  = DATA / "clean_3class_hardneg_experiment"

K=21; S=5000
_BASE=np.zeros(256,dtype=np.uint64)
for _b,_v in zip(b'ACGTacgt',[0,1,2,3,0,1,2,3]): _BASE[_b]=_v
_RC =np.array([3,2,1,0],dtype=np.uint64)
_POW=np.array([4**(K-1-j) for j in range(K)],dtype=np.uint64)

def _mix64(x):
    x=x^(x>>np.uint64(30)); x=x*np.uint64(0xBF58476D1CE4E5B9)
    x=x^(x>>np.uint64(27)); x=x*np.uint64(0x94D049BB133111EB)
    return x^(x>>np.uint64(31))

def minhash(seq: str) -> np.ndarray:
    arr=_BASE[np.frombuffer(seq.upper().encode('ascii',errors='replace'),dtype=np.uint8)]
    if len(arr)<K: return np.array([],dtype=np.uint64)
    w=sliding_window_view(arr,K); fwd=(w*_POW).sum(axis=1); rc=(_RC[w[:,::-1]]*_POW).sum(axis=1)
    h=_mix64(np.where(fwd<rc,fwd,rc))
    return np.sort(h[np.argpartition(h,S)[:S]]) if len(h)>S else np.sort(h)

def containment(q, db):
    if len(q)==0 or len(db)==0: return 0.0
    idx=np.clip(np.searchsorted(db,q),0,len(db)-1)
    return float((db[idx]==q).mean())

_ID_RE=re.compile(r'^(.+?)_w(\d+)_s(\d+)$')

def build_lookup(seq_ids):
    """accession -> [(row_idx, window_size, start)]"""
    lookup={}
    for i,sid in enumerate(seq_ids):
        m=_ID_RE.match(sid)
        if m:
            acc=m.group(1); w=int(m.group(2)); s=int(m.group(3))
            lookup.setdefault(acc,[]).append((i,w,s))
    return lookup

def open_fasta(fpath):
    """Return file handle (handles .gz)."""
    if str(fpath).endswith('.gz'):
        return gzip.open(fpath,'rb')
    return open(fpath,'rb')

def stream_fasta(fpath, label, lookup, results, filled, compass_sk, chr_sk):
    """Stream a FASTA (plain or gzipped); fill results array in-place."""
    n_seq=0; n_found=0; t0=time.time()
    fsize=os.path.getsize(fpath)
    print(f"  Streaming {label}: {Path(fpath).name} ({fsize//1024//1024}MB)",flush=True)

    def process(acc, parts):
        nonlocal n_found
        hits=lookup.get(acc,[])
        if not hits: return
        seq=b''.join(parts).decode('ascii',errors='replace')
        for (row_idx,w,s) in hits:
            if filled[row_idx]: continue
            window=seq[s:s+w]
            q=minhash(window)
            results[row_idx,0]=containment(q,compass_sk)
            results[row_idx,1]=containment(q,chr_sk)
            filled[row_idx]=True
            n_found+=1

    with open_fasta(fpath) as f:
        acc=None; parts=[]
        for line in f:
            if line.startswith(b'>'):
                if acc is not None:
                    process(acc,parts); n_seq+=1
                    if n_seq%50000==0:
                        print(f"    {label} seq={n_seq:,} found={n_found:,} t={time.time()-t0:.0f}s",flush=True)
                hdr=line[1:].rstrip(b'\r\n').decode('ascii',errors='replace')
                acc=hdr.split()[0]  # first token = accession
                parts=[]
            else:
                parts.append(line.rstrip(b'\r\n'))
        if acc is not None:
            process(acc,parts); n_seq+=1

    elapsed=time.time()-t0
    print(f"  {label} DONE: {n_seq:,} seqs, {n_found:,} rows filled, {elapsed:.1f}s",flush=True)
    return n_found

def main():
    TOTAL=358919
    OUT    =EXP/"containment_features.npy"
    PARTIAL=EXP/"containment_partial.npy"
    STATE  =EXP/"containment_state.pkl"

    if OUT.exists():
        arr=np.load(OUT)
        print(f"Already complete: {arr.shape}  compass_mean={arr[:,0].mean():.4f}  chr_mean={arr[:,1].mean():.4f}")
        return

    print("Loading seq_ids...", flush=True)
    with open(EXP/"seq_ids.txt") as f:
        seq_ids=[l.strip() for l in f]
    assert len(seq_ids)==TOTAL

    print("Loading sketches...", flush=True)
    t0=time.time()
    compass_sk=np.load(DATA/"databases"/"sketch_compass_k21_s5m.npy")
    chr_sk    =np.load(DATA/"databases"/"sketch_chromosomes_k21_s5m.npy")
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    print("Building lookup...", flush=True)
    lookup=build_lookup(seq_ids)
    print(f"  {len(lookup):,} accession keys", flush=True)

    # Source files in order (phage first since small, then chr, then plasmid large)
    sources = [
        (str(DATA/"databases"/"inphared"/"inphared_phages.fa.gz"), "phage"),
        (str(DATA/"databases"/"plasmids"/"COMPASS.fna"),           "compass_plasmid"),
        (str(DATA/"databases"/"plasmids"/"plsdb.fasta"),           "plsdb_plasmid"),
    ]
    # Add 1000 chromosome files
    chr_dir=DATA/"chromosomes"/"bacteria"
    for fna in sorted(chr_dir.glob("*.fna")):
        sources.append((str(fna),"chr"))

    # Load state
    if STATE.exists() and PARTIAL.exists():
        state=pickle.load(open(STATE,'rb'))
        results=np.load(PARTIAL)
        filled=state.get('filled',np.zeros(TOTAL,dtype=bool))
        done_sources=state.get('done_sources',set())
        print(f"Resuming: {filled.sum()}/{TOTAL} rows filled, {len(done_sources)} sources done",flush=True)
    else:
        results=np.zeros((TOTAL,2),dtype=np.float32)
        filled=np.zeros(TOTAL,dtype=bool)
        done_sources=set()
        print("Starting fresh",flush=True)

    for fpath, label in sources:
        if fpath in done_sources:
            # Only skip non-chr sources; for chr files skip after first run
            print(f"  Skipping {label} {Path(fpath).name} (done)",flush=True)
            continue
        n=stream_fasta(fpath,label,lookup,results,filled,compass_sk,chr_sk)
        done_sources.add(fpath)
        np.save(PARTIAL,results)
        pickle.dump({'done_sources':done_sources,'filled':filled},open(STATE,'wb'))
        missing=int((~filled).sum())
        print(f"  Checkpoint: {filled.sum()}/{TOTAL} filled, {missing} missing",flush=True)

    # Final report
    missing_count=int((~filled).sum())
    print(f"\n=== COMPLETE ===",flush=True)
    print(f"Filled: {filled.sum()}/{TOTAL}  Missing: {missing_count}",flush=True)
    print(f"compass: mean={results[:,0].mean():.4f}  max={results[:,0].max():.4f}",flush=True)
    print(f"chr:     mean={results[:,1].mean():.4f}  max={results[:,1].max():.4f}",flush=True)
    np.save(OUT,results)
    print(f"Saved: {OUT}",flush=True)
    if PARTIAL.exists(): PARTIAL.unlink()
    if STATE.exists():   STATE.unlink()
    print("DONE",flush=True)

if __name__=='__main__':
    main()
