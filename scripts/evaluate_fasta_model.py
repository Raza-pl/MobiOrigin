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


def evaluate_fasta_models(
    fasta_path: Path,
    labels_path: Path,
    model_paths: dict[str, Path],
    out_dirs: dict[str, Path],
    *,
    chunk_size: int = 2000,
    threshold: float = 0.70,
    plasmid_threshold: float = 0.862,
) -> dict[str, dict[str, object]]:
    """Evaluate one or more models while extracting each FASTA feature chunk once."""

    if not model_paths:
        raise ValueError("At least one model is required")
    if set(model_paths) != set(out_dirs):
        raise ValueError("model_paths and out_dirs must have identical keys")
    with labels_path.open() as fh:
        label_rows = {row["contig_id"]: row for row in csv.DictReader(fh, delimiter="\t")}
    label_map = dict(CLASS_TO_IDX)
    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(1)
    models = {
        name: load_model(model_path, device=device)
        for name, model_path in model_paths.items()
    }
    for model in models.values():
        model.eval()
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
            predictions_fh = stack.enter_context(
                (out_dir / "predictions.tsv").open("w", newline="")
            )
            writer = csv.DictWriter(
                predictions_fh,
                fieldnames=fieldnames,
                delimiter="\t",
            )
            writer.writeheader()
            writers[name] = writer
        chunk_sequences: list[str] = []
        chunk_ids: list[str] = []
        processed_rows = 0

        def flush_chunk() -> None:
            nonlocal processed_rows
            if not chunk_sequences:
                return
            feature_chunk = extract_features(chunk_sequences)
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
            chunk_ids.append(record.id)
            chunk_sequences.append(str(record.seq))
            if len(chunk_sequences) >= chunk_size:
                flush_chunk()
        flush_chunk()

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
) -> dict[str, object]:
    """Evaluate one model through the shared streaming implementation."""

    return evaluate_fasta_models(
        fasta_path,
        labels_path,
        {"model": model_path},
        {"model": out_dir},
        chunk_size=chunk_size,
        threshold=threshold,
        plasmid_threshold=plasmid_threshold,
    )["model"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--plasmid-threshold", type=float, default=0.95)
    args = parser.parse_args()

    metrics = evaluate_fasta_model(
        args.fasta,
        args.labels,
        args.model,
        args.out,
        chunk_size=args.chunk_size,
        threshold=args.threshold,
        plasmid_threshold=args.plasmid_threshold,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
