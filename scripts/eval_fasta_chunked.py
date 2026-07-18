#!/usr/bin/env python3
"""Resumable chunked FASTA evaluation — processes sequences in offset/count windows.

Designed for bash environments with a 45-second per-call timeout.

Usage (three phases):
  # Phase 1 — extract features + predictions for each chunk:
  python3 eval_fasta_chunked.py --fasta FILE --labels FILE --model FILE \\
      --out DIR --offset 0 --count 10000

  # Repeat with --offset 10000, 20000, … until all sequences are done.

  # Phase 2 — merge chunks and compute metrics:
  python3 eval_fasta_chunked.py --finalize --out DIR

  # Phase 3 (optional) — show quick summary:
  python3 eval_fasta_chunked.py --summary --out DIR
"""

from __future__ import annotations

import os as _os
for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_name, "1")

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_labels(labels_path: Path) -> dict[str, dict]:
    with labels_path.open() as fh:
        return {row["contig_id"]: row for row in csv.DictReader(fh, delimiter="\t")}


# ---------------------------------------------------------------------------
# FASTA byte-offset index helpers (O(1) seek per chunk)
# ---------------------------------------------------------------------------

def _build_or_load_fasta_index(
    fasta_path: Path, label_rows: dict, out_dir: Path
) -> tuple[list[str], list[int]]:
    """Return (ordered_ids, byte_offsets) for labeled sequences in FASTA order.

    Index is built on first call and cached to out_dir/fasta_index_{ids,offsets}.npy
    so subsequent chunks skip the scan entirely.
    """
    ids_file    = out_dir / "fasta_index_ids.npy"
    offsets_file = out_dir / "fasta_index_offsets.npy"

    if ids_file.exists() and offsets_file.exists():
        ids     = np.load(ids_file,     allow_pickle=True).tolist()
        offsets = np.load(offsets_file, allow_pickle=False).tolist()
        print(f"[index] loaded {len(ids):,} entries from cache", flush=True)
        return ids, offsets

    print("[index] building byte-offset index (one-time scan)…", flush=True)
    ids: list[str] = []
    offsets: list[int] = []
    with open(fasta_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith(b">"):
                cid = line.decode("ascii", errors="replace").strip()[1:].split()[0]
                if cid in label_rows:
                    ids.append(cid)
                    offsets.append(pos)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(ids_file,     np.array(ids,     dtype=object))
    np.save(offsets_file, np.array(offsets, dtype=np.int64))
    print(f"[index] indexed {len(ids):,} labeled sequences → cached", flush=True)
    return ids, offsets


def _fast_iter_from_byte(fasta_path: Path, start_byte: int):
    """Yield (contig_id, sequence_str) starting from a byte offset in the FASTA."""
    with open(fasta_path, "rb") as f:
        f.seek(start_byte)
        current_id: str | None = None
        seq_parts: list[bytes] = []
        for raw_line in f:
            if raw_line.startswith(b">"):
                if current_id is not None:
                    yield current_id, b"".join(seq_parts).decode("ascii", errors="replace")
                current_id = raw_line.decode("ascii", errors="replace").strip()[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(raw_line.rstrip(b"\r\n"))
        if current_id is not None:
            yield current_id, b"".join(seq_parts).decode("ascii", errors="replace")


# ---------------------------------------------------------------------------

def run_chunk(
    fasta_path: Path,
    labels_path: Path,
    model_path: Path,
    out_dir: Path,
    offset: int,
    count: int,
) -> int:
    """Process sequences [offset, offset+count) from the FASTA. Returns #processed."""

    from plasflow2.classify.features import extract_features
    from plasflow2.classify.model import load_model
    from plasflow2.classify.predict import _assign_label
    from plasflow2.utils.device import CLASS_TO_IDX, IDX_TO_CLASS, get_device

    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = out_dir / f"chunk_{offset:08d}.npz"
    if chunk_file.exists():
        existing = np.load(chunk_file)
        n = int(existing["labels"].shape[0])
        print(f"[chunk {offset}] already done ({n} rows), skipping", flush=True)
        return n

    label_rows = _load_labels(labels_path)
    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(1)
    model = load_model(model_path, device=device)
    model.eval()

    # Build or load byte-offset index for O(1) seek
    all_ids, all_offsets = _build_or_load_fasta_index(fasta_path, label_rows, out_dir)

    if offset >= len(all_ids):
        print(f"[chunk {offset}] offset beyond index ({len(all_ids)} seqs), nothing to do", flush=True)
        return 0

    start_byte = all_offsets[offset]
    remaining  = count   # index only needed for start_byte; scan forward from there

    CLASS_NAMES = ["plasmid", "chromosome", "phage"]
    labels_list, argmax_list, threshold_list, prob_list = [], [], [], []
    ids_list, lengths_list, tiers_list, taxa_list = [], [], [], []

    batch_seqs: list[str] = []
    batch_ids: list[str] = []

    def flush(seqs: list[str], ids: list[str]) -> None:
        if not seqs:
            return
        feat = extract_features(seqs)
        batch = torch.from_numpy(feat).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=-1).cpu().numpy()
        for cid, seq, prob_row in zip(ids, seqs, probs):
            row = label_rows[cid]
            scores = {CLASS_NAMES[j]: float(prob_row[j]) for j in range(3)}
            true_lbl = CLASS_TO_IDX[row["true_label"]]
            argmax_lbl = int(np.argmax(prob_row))
            threshold_name, _ = _assign_label(scores, len(seq), 0.95, 0.70, False)
            threshold_lbl = CLASS_TO_IDX.get(threshold_name, -1)
            labels_list.append(true_lbl)
            argmax_list.append(argmax_lbl)
            threshold_list.append(threshold_lbl)
            prob_list.append(prob_row.astype(np.float32))
            ids_list.append(cid)
            lengths_list.append(int(row["length"]))
            tiers_list.append(row.get("length_tier", ""))
            taxa_list.append(row.get("taxon", ""))

    labeled_seen = 0
    for cid, seq in _fast_iter_from_byte(fasta_path, start_byte):
        if cid not in label_rows:
            continue
        batch_seqs.append(seq)
        batch_ids.append(cid)
        labeled_seen += 1
        if len(batch_seqs) >= 512:
            flush(batch_seqs, batch_ids)
            batch_seqs.clear()
            batch_ids.clear()
        if labeled_seen >= remaining:
            break

    flush(batch_seqs, batch_ids)
    processed = len(labels_list)

    if processed > 0:
        np.savez_compressed(
            chunk_file,
            labels=np.array(labels_list, dtype=np.int64),
            argmax=np.array(argmax_list, dtype=np.int64),
            threshold=np.array(threshold_list, dtype=np.int64),
            probabilities=np.array(prob_list, dtype=np.float32),
            lengths=np.array(lengths_list, dtype=np.int64),
            tiers=np.array(tiers_list, dtype=object),
            taxa=np.array(taxa_list, dtype=object),
            ids=np.array(ids_list, dtype=object),
        )
    print(f"[chunk {offset}] wrote {processed} rows → {chunk_file.name}", flush=True)
    return processed


def finalize(out_dir: Path) -> dict:
    """Merge all chunk_*.npz files and compute metrics.json."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, classification_report,
        confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
    )
    from plasflow2.utils.device import IDX_TO_CLASS, CLASS_TO_IDX

    chunk_files = sorted(out_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.npz files in {out_dir}")

    parts = [np.load(f, allow_pickle=True) for f in chunk_files]
    labels      = np.concatenate([p["labels"] for p in parts])
    argmax      = np.concatenate([p["argmax"] for p in parts])
    threshold   = np.concatenate([p["threshold"] for p in parts])
    probs       = np.concatenate([p["probabilities"] for p in parts])
    lengths     = np.concatenate([p["lengths"] for p in parts])
    tiers       = np.concatenate([p["tiers"] for p in parts])
    taxa        = np.concatenate([p["taxa"] for p in parts])
    ids         = np.concatenate([p["ids"] for p in parts])

    print(f"Merged {len(chunk_files)} chunks → {len(labels):,} rows", flush=True)

    def metrics_for(preds: np.ndarray, grp_labels: np.ndarray, grp_probs: np.ndarray) -> dict:
        class_labels = [0, 1, 2]
        report_labels = class_labels + ([-1] if (preds == -1).any() else [])
        class_names = [IDX_TO_CLASS[i] for i in class_labels]
        report_names = class_names + (["unclassified"] if -1 in report_labels else [])
        observed = sorted(int(l) for l in np.unique(grp_labels))
        obs_recall = [float((preds[grp_labels == l] == l).mean()) for l in observed]
        plas_true = grp_labels == CLASS_TO_IDX["plasmid"]
        plas_scores = grp_probs[:, CLASS_TO_IDX["plasmid"]]
        plas_pred = preds == CLASS_TO_IDX["plasmid"]
        p, r, f, _ = precision_recall_fscore_support(plas_true, plas_pred, average="binary", zero_division=0)
        m: dict = {
            "n_rows": int(len(grp_labels)),
            "accuracy": float(accuracy_score(grp_labels, preds)),
            "balanced_accuracy_observed_classes": float(np.mean(obs_recall)),
            "macro_f1_observed_classes": float(f1_score(grp_labels, preds, labels=observed, average="macro")),
            "plasmid_precision": float(p),
            "plasmid_recall": float(r),
            "plasmid_f1": float(f),
            "confusion_matrix_labels": report_names,
            "confusion_matrix": confusion_matrix(grp_labels, preds, labels=report_labels).tolist(),
            "classification_report": classification_report(
                grp_labels, preds, labels=report_labels, target_names=report_names,
                output_dict=True, zero_division=0,
            ),
        }
        if len(np.unique(plas_true)) == 2:
            m["plasmid_average_precision"] = float(average_precision_score(plas_true, plas_scores))
            m["plasmid_roc_auc"] = float(roc_auc_score(plas_true, plas_scores))
        return m

    def per_group(key_array: np.ndarray, preds: np.ndarray) -> dict:
        result = {}
        for key in sorted(set(key_array)):
            mask = key_array == key
            if mask.sum() < 2:
                continue
            result[str(key)] = metrics_for(preds[mask], labels[mask], probs[mask])
        return result

    metrics = {
        "argmax": metrics_for(argmax, labels, probs),
        "production_thresholds": metrics_for(threshold, labels, probs),
        "per_length_argmax": per_group(tiers, argmax),
        "per_taxon_argmax": per_group(taxa, argmax),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    # Write predictions TSV
    with (out_dir / "predictions.tsv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=[
            "contig_id", "true_label", "argmax_prediction", "threshold_prediction",
            "length", "length_tier", "taxon", "plasmid_score", "chromosome_score", "phage_score",
        ])
        writer.writeheader()
        for i in range(len(labels)):
            writer.writerow({
                "contig_id": ids[i],
                "true_label": IDX_TO_CLASS.get(int(labels[i]), str(labels[i])),
                "argmax_prediction": IDX_TO_CLASS.get(int(argmax[i]), "unclassified"),
                "threshold_prediction": IDX_TO_CLASS.get(int(threshold[i]), "unclassified"),
                "length": int(lengths[i]),
                "length_tier": tiers[i],
                "taxon": taxa[i],
                "plasmid_score": round(float(probs[i, 0]), 5),
                "chromosome_score": round(float(probs[i, 1]), 5),
                "phage_score": round(float(probs[i, 2]), 5),
            })

    print(f"Wrote metrics.json and predictions.tsv to {out_dir}", flush=True)
    return metrics


def print_summary(out_dir: Path) -> None:
    m = json.loads((out_dir / "metrics.json").read_text())
    a = m["argmax"]
    cr = a["classification_report"]
    print(f"\n=== Argmax metrics ({a['n_rows']:,} contigs) ===")
    print(f"  Macro F1 (2-class observed): {a['macro_f1_observed_classes']:.4f}")
    print(f"  Plasmid  — P={a['plasmid_precision']:.4f}  R={a['plasmid_recall']:.4f}  F1={a['plasmid_f1']:.4f}")
    for cls in ("plasmid", "chromosome", "phage"):
        if cls in cr:
            c = cr[cls]
            print(f"  {cls:12s} P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1-score']:.3f} (n={int(c['support'])})")
    cm = a["confusion_matrix"]
    print(f"\n  Confusion matrix ({a['confusion_matrix_labels']}):")
    for row in cm:
        print(f"    {row}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta",   type=Path)
    p.add_argument("--labels",  type=Path)
    p.add_argument("--model",   type=Path)
    p.add_argument("--out",     type=Path, required=True)
    p.add_argument("--offset",  type=int, default=0)
    p.add_argument("--count",   type=int, default=10000)
    p.add_argument("--finalize",  action="store_true")
    p.add_argument("--summary",   action="store_true")
    args = p.parse_args()

    if args.summary:
        print_summary(args.out)
    elif args.finalize:
        m = finalize(args.out)
        a = m["argmax"]
        print(f"\nPlasmid: P={a['plasmid_precision']:.4f}  R={a['plasmid_recall']:.4f}  F1={a['plasmid_f1']:.4f}")
    else:
        if not args.fasta or not args.labels or not args.model:
            p.error("--fasta, --labels, --model required for chunk mode")
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        n = run_chunk(args.fasta, args.labels, args.model, args.out, args.offset, args.count)
        print(f"chunk done: {n} seqs")


if __name__ == "__main__":
    main()
