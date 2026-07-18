#!/usr/bin/env python3
"""Stream a labeled FASTA benchmark through a PlasFlow MLP and report metrics."""

# Imports intentionally follow the thread-pool environment setup below.
# ruff: noqa: E402

from __future__ import annotations

import os as _os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_name, "1")

import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from plasflow2.classify.features import extract_features
from plasflow2.classify.model import load_model
from plasflow2.classify.predict import _assign_label
from plasflow2.utils.device import CLASS_TO_IDX, IDX_TO_CLASS, get_device
from plasflow2.utils.fasta import iter_fasta
from sklearn.metrics import (  # type: ignore[import]
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    class_labels = list(range(probabilities.shape[1]))
    class_names = [IDX_TO_CLASS[label] for label in class_labels]
    report_labels = class_labels + ([-1] if (predictions == -1).any() else [])
    report_names = class_names + (["unclassified"] if -1 in report_labels else [])
    observed = sorted(int(label) for label in np.unique(labels))
    observed_class_recall = [
        float((predictions[labels == label] == label).mean()) for label in observed
    ]
    plasmid_true = labels == CLASS_TO_IDX["plasmid"]
    plasmid_scores = probabilities[:, CLASS_TO_IDX["plasmid"]]
    plasmid_predictions = predictions == CLASS_TO_IDX["plasmid"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        plasmid_true,
        plasmid_predictions,
        average="binary",
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "n_rows": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy_observed_classes": float(np.mean(observed_class_recall)),
        "macro_f1_observed_classes": float(
            f1_score(labels, predictions, labels=observed, average="macro")
        ),
        "plasmid_precision": float(precision),
        "plasmid_recall": float(recall),
        "plasmid_f1": float(f1),
        "confusion_matrix_labels": report_names,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=report_labels).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=report_labels,
            target_names=report_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    if len(np.unique(plasmid_true)) == 2:
        metrics["plasmid_average_precision"] = float(
            average_precision_score(plasmid_true, plasmid_scores)
        )
        metrics["plasmid_roc_auc"] = float(roc_auc_score(plasmid_true, plasmid_scores))
    else:
        metrics["plasmid_average_precision"] = None
        metrics["plasmid_roc_auc"] = None
    return metrics


def _read_processed_ids(predictions_path: Path) -> set[str]:
    """Return contig_ids already written to a partial predictions.tsv."""
    ids: set[str] = set()
    if not predictions_path.exists():
        return ids
    with predictions_path.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ids.add(row["contig_id"])
    return ids


def evaluate_fasta_models(
    fasta_path: Path,
    labels_path: Path,
    model_paths: dict[str, Path],
    out_dirs: dict[str, Path],
    *,
    chunk_size: int = 2000,
    threshold: float = 0.70,
    plasmid_threshold: float = 0.862,
    compass_sketch_path: Path | None = None,
    compass_threshold: float = 0.001,
    chr_sketch_path: Path | None = None,
    resume: bool = False,
    max_seqs: int | None = None,
) -> dict[str, dict[str, object]] | None:
    """Evaluate one or more models while extracting each FASTA feature chunk once.

    When ``resume=True`` the function appends to an existing ``predictions.tsv``
    (skipping already-processed contig_ids) and returns ``None`` if the file is
    still incomplete.  Pass ``max_seqs`` to cap how many **new** sequences are
    processed per call; re-run with ``--resume`` to continue.

    When all sequences have been processed (total rows == len(label_rows))
    metrics are computed and written to ``metrics.json`` and the function
    returns the metrics dict as usual.
    """

    if not model_paths:
        raise ValueError("At least one model is required")
    if set(model_paths) != set(out_dirs):
        raise ValueError("model_paths and out_dirs must have identical keys")
    with labels_path.open() as fh:
        label_rows = {row["contig_id"]: row for row in csv.DictReader(fh, delimiter="\t")}
    label_map = dict(CLASS_TO_IDX)

    # Determine which sequences have already been processed (resume mode)
    # We assume all models stay in sync, so we only check the first model's dir.
    first_out = list(out_dirs.values())[0]
    first_preds = first_out / "predictions.tsv"
    processed_ids: set[str] = set()
    if resume:
        processed_ids = _read_processed_ids(first_preds)
        print(f"Resume: {len(processed_ids):,} sequences already in predictions.tsv", flush=True)

    # Load COMPASS containment filter if requested
    _compass_filter = None
    if compass_sketch_path is not None and compass_sketch_path.exists():
        from plasflow2.classify.containment import CompassFilter
        _compass_filter = CompassFilter.load(compass_sketch_path, threshold=compass_threshold)
        print(f"COMPASS filter loaded: {len(_compass_filter._sketch):,} hashes, "
              f"threshold={compass_threshold}", flush=True)

    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(1)
    models = {
        name: load_model(model_path, device=device)
        for name, model_path in model_paths.items()
    }
    for model in models.values():
        model.eval()

    # Detect Rev6 (9559-feature) models and load sketches if needed
    _compass_sketch_arr = None
    _chr_sketch_arr = None
    any_model = next(iter(models.values()))
    model_input_dim = any_model.net[0].weight.shape[1]
    if model_input_dim == 9559:
        _compass_np = (
            np.load(str(compass_sketch_path))
            if compass_sketch_path and compass_sketch_path.exists()
            else None
        )
        _chr_np = (
            np.load(str(chr_sketch_path))
            if chr_sketch_path and chr_sketch_path.exists()
            else None
        )
        if _compass_np is not None and _chr_np is not None:
            _compass_sketch_arr = _compass_np.astype(np.uint64)
            _chr_sketch_arr = _chr_np.astype(np.uint64)
            print(
                f"Rev6 mode: loaded compass ({len(_compass_sketch_arr):,}) "
                f"and chr ({len(_chr_sketch_arr):,}) sketches",
                flush=True,
            )
        else:
            raise ValueError(
                "Rev6 model (input_dim=9559) requires --compass-sketch and --chr-sketch paths"
            )
    states: dict[str, dict[str, list]] = {
        name: {
            "labels": [],
            "argmax_predictions": [],
            "threshold_predictions": [],
            "probabilities": [],
            "lengths": [],
            "length_tiers": [],
            "taxa": [],
        }
        for name in model_paths
    }
    for out_dir in out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load previously processed rows into states when resuming so that
    # final metrics include the full dataset.
    if resume and processed_ids:
        for name, out_dir in out_dirs.items():
            preds_tsv = out_dir / "predictions.tsv"
            if not preds_tsv.exists():
                continue
            print(f"Loading {len(processed_ids):,} existing predictions from {preds_tsv} ...",
                  flush=True)
            with preds_tsv.open() as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    contig_id = row["contig_id"]
                    if contig_id not in label_rows:
                        continue
                    truth = label_rows[contig_id]
                    state = states[name]
                    state["labels"].append(label_map[row["true_label"]])
                    state["argmax_predictions"].append(
                        label_map.get(row["argmax_prediction"], -1)
                    )
                    state["threshold_predictions"].append(
                        label_map.get(row["threshold_prediction"], -1)
                    )
                    state["probabilities"].append(
                        np.array(
                            [
                                float(row.get("plasmid_score", 0)),
                                float(row.get("chromosome_score", 0)),
                                float(row.get("phage_score", 0)),
                            ],
                            dtype=np.float32,
                        )
                    )
                    state["lengths"].append(int(truth.get("length", 0)))
                    state["length_tiers"].append(truth.get("length_tier", ""))
                    state["taxa"].append(truth.get("taxon", "unknown"))
            print(f"Loaded {len(states[name]['labels']):,} rows.", flush=True)

    with ExitStack() as stack:
        fieldnames = [
            "contig_id",
            "true_label",
            "argmax_prediction",
            "threshold_prediction",
            "length",
            "length_tier",
            "source_accession",
            "taxon",
            "plasmid_score",
            "chromosome_score",
            "phage_score",
        ]
        writers: dict[str, csv.DictWriter] = {}
        for name, out_dir in out_dirs.items():
            _preds_path = out_dir / "predictions.tsv"
            _append = resume and processed_ids and _preds_path.exists()
            predictions_fh = stack.enter_context(
                _preds_path.open("a" if _append else "w", newline="")
            )
            writer = csv.DictWriter(
                predictions_fh,
                fieldnames=fieldnames,
                delimiter="\t",
            )
            if not _append:
                writer.writeheader()
            writers[name] = writer
        chunk_sequences: list[str] = []
        chunk_ids: list[str] = []
        processed_rows = 0

        def flush_chunk() -> None:
            nonlocal processed_rows
            if not chunk_sequences:
                return
            feature_chunk = extract_features(
                chunk_sequences,
                compass_sketch=_compass_sketch_arr,
                chr_sketch=_chr_sketch_arr,
            )
            batch = torch.from_numpy(feature_chunk).to(device)
            for name, model in models.items():
                with torch.no_grad():
                    probabilities = torch.softmax(model(batch), dim=-1).cpu().numpy()
                state = states[name]
                writer = writers[name]
                for contig_id, sequence, probability_row in zip(
                    chunk_ids,
                    chunk_sequences,
                    probabilities,
                ):
                    scores = {
                        IDX_TO_CLASS[j]: float(probability_row[j])
                        for j in range(len(probability_row))
                    }
                    truth = label_rows[contig_id]
                    true_label_name = truth["true_label"]
                    true_label = label_map[true_label_name]
                    argmax_name = max(scores, key=scores.__getitem__)
                    threshold_name, _ = _assign_label(
                        scores,
                        len(sequence),
                        plasmid_threshold,
                        threshold,
                        False,
                    )
                    # Apply COMPASS containment filter to threshold predictions
                    if (
                        _compass_filter is not None
                        and threshold_name == "plasmid"
                        and not _compass_filter.check(sequence)
                    ):
                        threshold_name = "chromosome"
                    argmax_prediction = label_map[argmax_name]
                    threshold_prediction = label_map.get(threshold_name, -1)
                    state["labels"].append(true_label)
                    state["argmax_predictions"].append(argmax_prediction)
                    state["threshold_predictions"].append(threshold_prediction)
                    state["probabilities"].append(
                        np.asarray(probability_row, dtype=np.float32)
                    )
                    state["lengths"].append(int(truth["length"]))
                    state["length_tiers"].append(truth.get("length_tier", ""))
                    state["taxa"].append(truth.get("taxon", "unknown"))
                    writer.writerow(
                        {
                            "contig_id": contig_id,
                            "true_label": true_label_name,
                            "argmax_prediction": argmax_name,
                            "threshold_prediction": threshold_name,
                            "length": truth["length"],
                            "length_tier": truth.get("length_tier", ""),
                            "source_accession": truth.get("source_accession", ""),
                            "taxon": truth.get("taxon", ""),
                            **{
                                f"{class_name}_score": scores.get(class_name, 0.0)
                                for class_name in ("plasmid", "chromosome", "phage")
                            },
                        }
                    )
            processed_rows += len(chunk_sequences)
            if processed_rows % 10_000 == 0 or processed_rows == len(label_rows):
                print(
                    f"FASTA evaluation: {processed_rows:,} / {len(label_rows):,} sequences "
                    f"({', '.join(model_paths)})",
                    flush=True,
                )
            chunk_sequences.clear()
            chunk_ids.clear()

        for record in iter_fasta(fasta_path):
            if record.id not in label_rows:
                continue
            if record.id in processed_ids:
                continue
            chunk_ids.append(record.id)
            chunk_sequences.append(str(record.seq))
            if len(chunk_sequences) >= chunk_size:
                flush_chunk()
            if max_seqs is not None and processed_rows >= max_seqs:
                break
        flush_chunk()

    total_done = len(processed_ids) + processed_rows
    if total_done < len(label_rows):
        print(
            f"Partial run: {total_done:,}/{len(label_rows):,} sequences processed. "
            f"Re-run with --resume to continue.",
            flush=True,
        )
        return None

    scored_rows = len(states[next(iter(states))]["labels"])
    if scored_rows != len(label_rows):
        raise ValueError(
            f"Benchmark FASTA/label mismatch: scored {scored_rows} of {len(label_rows)} rows"
        )

    results: dict[str, dict[str, object]] = {}
    for name, state in states.items():
        labels = np.asarray(state["labels"], dtype=np.int64)
        argmax_predictions = np.asarray(state["argmax_predictions"], dtype=np.int64)
        threshold_predictions = np.asarray(state["threshold_predictions"], dtype=np.int64)
        probability_matrix = np.asarray(state["probabilities"], dtype=np.float32)
        length_tiers = np.asarray(state["length_tiers"])
        taxa = np.asarray(state["taxa"])
        per_length: dict[str, dict[str, object]] = {}
        for tier in sorted(set(length_tiers)):
            mask = length_tiers == tier
            per_length[tier] = _classification_metrics(
                labels[mask],
                argmax_predictions[mask],
                probability_matrix[mask],
            )

        per_taxon: dict[str, dict[str, object]] = {}
        for taxon in sorted(set(taxa)):
            mask = taxa == taxon
            per_taxon[taxon] = _classification_metrics(
                labels[mask],
                argmax_predictions[mask],
                probability_matrix[mask],
            )

        metrics: dict[str, object] = {
            "model": str(model_paths[name]),
            "fasta": str(fasta_path),
            "labels": str(labels_path),
            "argmax": _classification_metrics(labels, argmax_predictions, probability_matrix),
            "production_thresholds": _classification_metrics(
                labels, threshold_predictions, probability_matrix
            ),
            "per_length_argmax": per_length,
            "per_taxon_argmax": per_taxon,
        }
        out_dir = out_dirs[name]
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        np.savez_compressed(
            out_dir / "scores.npz",
            labels=labels,
            argmax_predictions=argmax_predictions,
            threshold_predictions=threshold_predictions,
            probabilities=probability_matrix,
            lengths=np.asarray(state["lengths"], dtype=np.int64),
            taxa=taxa,
        )
        results[name] = metrics
    return results


def evaluate_fasta_model(
    fasta_path: Path,
    labels_path: Path,
    model_path: Path,
    out_dir: Path,
    *,
    chunk_size: int = 2000,
    threshold: float = 0.70,
    plasmid_threshold: float = 0.862,
    compass_sketch_path: Path | None = None,
    compass_threshold: float = 0.001,
    chr_sketch_path: Path | None = None,
    resume: bool = False,
    max_seqs: int | None = None,
) -> dict[str, object] | None:
    """Evaluate one model through the shared streaming implementation."""

    result = evaluate_fasta_models(
        fasta_path,
        labels_path,
        {"model": model_path},
        {"model": out_dir},
        chunk_size=chunk_size,
        threshold=threshold,
        plasmid_threshold=plasmid_threshold,
        compass_sketch_path=compass_sketch_path,
        compass_threshold=compass_threshold,
        chr_sketch_path=chr_sketch_path,
        resume=resume,
        max_seqs=max_seqs,
    )
    if result is None:
        return None
    return result["model"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--plasmid-threshold", type=float, default=0.95)
    parser.add_argument(
        "--compass-sketch",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to COMPASS MinHash sketch (.npy). Enables containment post-filter.",
    )
    parser.add_argument(
        "--compass-threshold",
        type=float,
        default=0.001,
        metavar="FLOAT",
        help="Minimum containment score to retain a plasmid prediction (default 0.001).",
    )
    parser.add_argument(
        "--chr-sketch",
        type=Path,
        default=None,
        metavar="PATH",
        help="Chromosome MinHash sketch (.npy) for Rev6 containment features.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Append to an existing partial predictions.tsv and skip already-processed rows.",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing this many NEW sequences (use with --resume for incremental runs).",
    )
    args = parser.parse_args()

    metrics = evaluate_fasta_model(
        args.fasta,
        args.labels,
        args.model,
        args.out,
        chunk_size=args.chunk_size,
        threshold=args.threshold,
        plasmid_threshold=args.plasmid_threshold,
        compass_sketch_path=args.compass_sketch,
        compass_threshold=args.compass_threshold,
        chr_sketch_path=args.chr_sketch,
        resume=args.resume,
        max_seqs=args.max_seqs,
    )
    if metrics is not None:
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
