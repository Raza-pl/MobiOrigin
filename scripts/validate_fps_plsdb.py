#!/usr/bin/env python3
"""
Validate PlasFlow v2 benchmark false-positives against PLSDB and COMPASS.

For each chromosomal sequence called as 'plasmid' (FP), this script checks
whether the sequence has significant k-mer similarity to known plasmids in
PLSDB or COMPASS using MinHash sketching (Mash-style, k=21, n=500).

Sequences with estimated Jaccard >= 0.10 likely derive from a region with
genuine plasmid-like content, suggesting the benchmark label may be wrong.

Usage:
    python3 scripts/validate_fps_plsdb.py \
        --predictions results/benchmark_no_bleed/plasflow2_predictions.tsv \
        --benchmark  data/benchmark/benchmark.fna \
        --ground-truth data/benchmark/ground_truth.tsv \
        --compass    data/databases/plasmids/COMPASS.fna \
        [--plsdb     data/databases/plasmids/plsdb.fasta] \
        [--out        results/fp_plsdb_hits.tsv]

For faster local execution, install minimap2 and use:
    minimap2 -c -x map-ont <compass.fna> <fps.fna> > fp_compass_hits.paf
"""

import argparse, csv, sys
from pathlib import Path

# ─── MinHash ─────────────────────────────────────────────────────────────────

K = 21
N = 500
JACCARD_THRESHOLD = 0.10

def _kmers(seq, k):
    seq = seq.upper()
    for i in range(len(seq) - k + 1):
        s = seq[i:i+k]
        if 'N' not in s:
            yield s

def sketch(seq, k=K, n=N):
    """Bottom-N MinHash sketch."""
    seen = set()
    for km in _kmers(seq, k):
        seen.add(hash(km))
    return sorted(seen)[:n]

def est_jaccard(sk_a, sk_b):
    if not sk_a or not sk_b:
        return 0.0
    sa, sb = set(sk_a), set(sk_b)
    merged_bottom_n = set(sorted(sa | sb)[:N])
    return len((sa & sb) & merged_bottom_n) / N

# ─── FASTA parser ─────────────────────────────────────────────────────────────

def parse_fasta(path, target_ids=None):
    """Yield (id, seq) pairs. If target_ids set, skip non-targets."""
    current_id, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id:
                    if target_ids is None or current_id in target_ids:
                        yield current_id, "".join(buf)
                current_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if current_id:
        if target_ids is None or current_id in target_ids:
            yield current_id, "".join(buf)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions",  required=True)
    ap.add_argument("--benchmark",    required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--compass",      required=True)
    ap.add_argument("--plsdb",        default=None)
    ap.add_argument("--out",          default="results/fp_plsdb_hits.tsv")
    ap.add_argument("--jaccard-threshold", type=float, default=JACCARD_THRESHOLD)
    args = ap.parse_args()

    # Load ground truth
    gt = {}
    with open(args.ground_truth) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gt[row["contig_id"]] = row["true_label"]

    # Identify FPs
    fp_ids = set()
    with open(args.predictions) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["predicted"] == "plasmid" and gt.get(row["contig_id"]) == "chromosome":
                fp_ids.add(row["contig_id"])

    print(f"FPs to validate: {len(fp_ids)}", flush=True)

    # Extract FP sequences
    print("Extracting FP sequences from benchmark...", flush=True)
    fp_seqs = dict(parse_fasta(args.benchmark, fp_ids))
    print(f"  Extracted: {len(fp_seqs)}", flush=True)

    # Save FP FASTA for external tools
    fp_fasta_out = Path(args.out).parent / "benchmark_fps.fna"
    with open(fp_fasta_out, "w") as fout:
        for sid, seq in fp_seqs.items():
            fout.write(f">{sid}\n{seq}\n")
    print(f"  FP FASTA written: {fp_fasta_out}", flush=True)

    # Build FP sketches
    fp_sketches = {sid: sketch(seq) for sid, seq in fp_seqs.items()}

    # Search each database
    all_results = {}  # fp_id -> {db -> (best_j, best_ref)}
    for db_name, db_path in [("COMPASS", args.compass), ("PLSDB", args.plsdb)]:
        if db_path is None:
            continue
        print(f"\nBuilding {db_name} sketches ({db_path})...", flush=True)
        ref_sketches = {}
        n = 0
        for ref_id, seq in parse_fasta(db_path):
            ref_sketches[ref_id] = sketch(seq)
            n += 1
            if n % 5000 == 0:
                print(f"  {n} sequences sketched...", flush=True)
        print(f"  {n} sketches built. Searching...", flush=True)

        for fp_id, sk_q in fp_sketches.items():
            best_j, best_ref = 0.0, None
            for ref_id, sk_r in ref_sketches.items():
                j = est_jaccard(sk_q, sk_r)
                if j > best_j:
                    best_j, best_ref = j, ref_id
            if fp_id not in all_results:
                all_results[fp_id] = {}
            all_results[fp_id][db_name] = (best_j, best_ref)

    # Write output TSV
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["fp_contig_id", "source_accession",
                  "compass_jaccard", "compass_best_hit",
                  "plsdb_jaccard", "plsdb_best_hit",
                  "is_plsdb_hit"]
    rows = []
    for fp_id in sorted(fp_ids):
        parts = fp_id.split("_")
        # Extract source accession: e.g. NC_017626.1 from GCF_xxx_chr_NC_017626.1_w...
        try:
            chr_idx = parts.index("chr")
            acc = parts[chr_idx + 1] + "." + parts[chr_idx + 2].split("_")[0]
        except Exception:
            acc = "unknown"

        res = all_results.get(fp_id, {})
        c_j, c_ref = res.get("COMPASS", (0.0, None))
        p_j, p_ref = res.get("PLSDB", (0.0, None))
        is_hit = (c_j >= args.jaccard_threshold) or (p_j >= args.jaccard_threshold)
        rows.append({
            "fp_contig_id": fp_id,
            "source_accession": acc,
            "compass_jaccard": f"{c_j:.4f}",
            "compass_best_hit": c_ref or "",
            "plsdb_jaccard": f"{p_j:.4f}",
            "plsdb_best_hit": p_ref or "",
            "is_plsdb_hit": "YES" if is_hit else "no",
        })

    rows.sort(key=lambda r: float(r["compass_jaccard"]), reverse=True)
    with open(args.out, "w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"\nResults written: {args.out}")

    # Print summary
    n_hits = sum(1 for r in rows if r["is_plsdb_hit"] == "YES")
    print(f"\n=== Summary ===")
    print(f"  Total FPs checked:            {len(rows)}")
    print(f"  Match known plasmid (J≥{args.jaccard_threshold:.2f}): {n_hits}")
    print(f"  True novel FPs:               {len(rows) - n_hits}")
    print(f"\nTop hits:")
    for r in rows[:10]:
        marker = " ***" if r["is_plsdb_hit"] == "YES" else ""
        print(f"  {r['fp_contig_id'][-50:]:<52} COMPASS_J={r['compass_jaccard']}{marker}")


if __name__ == "__main__":
    main()
