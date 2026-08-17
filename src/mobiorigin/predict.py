"""Frozen MobiOrigin dev1 production inference pipeline."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from mobiorigin import __version__
from mobiorigin.fasta import FastaRecord, read_fasta
from mobiorigin.marker_features import extract_marker_features, load_database_manifest
from mobiorigin.model import INPUT_DIM, MobiOriginMLP, load_model
from mobiorigin.provenance import atomic_json, atomic_text, sha256_file
from mobiorigin.sequence_features import extract_sequence_features

CLASS_NAMES = ("chromosome", "plasmid", "phage")
SELECTIVE_THRESHOLD = 0.19835489988327026
MODEL_SHA256 = {
    "seed_20260810.pt": "2ed9a2ae4cbe00213504c27ef705b6af965aae97a8e33259661cb2c630a495c3",
    "seed_20260811.pt": "9270b5d2213ac95cae2821d26d6840974105905eb080ee39a178fe945140037d",
    "seed_20260812.pt": "085608214f4aac424e841cfd57b39c7b968deedebed943a626695fe815fe1c0f",
}
NORMALIZATION_SHA256 = "cb93c881032356f970bc0963969f852f88ab3a9a3a4a3d6c391437e11a4cd8bc"


def configure_runtime() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def verify_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Frozen MobiOrigin artifact identity changed: {path}")


def load_artifacts(model_dir: Path) -> tuple[list[MobiOriginMLP], NDArray[np.float32]]:
    models: list[MobiOriginMLP] = []
    for filename, expected in MODEL_SHA256.items():
        path = model_dir / filename
        verify_hash(path, expected)
        models.append(load_model(path))
    normalization_path = model_dir / "marker_normalization.npy"
    verify_hash(normalization_path, NORMALIZATION_SHA256)
    normalization = np.load(normalization_path, allow_pickle=False)
    if normalization.shape != (2, 17) or normalization.dtype != np.float32:
        raise ValueError("Marker normalization shape or dtype changed")
    if not np.isfinite(normalization).all() or np.any(normalization[1] <= 0):
        raise ValueError("Marker normalization values are invalid")
    return models, normalization


def fuse_features(
    sequence: NDArray[np.float32],
    marker: NDArray[np.float32],
    normalization: NDArray[np.float32],
) -> NDArray[np.float32]:
    if sequence.shape[0] != marker.shape[0] or sequence.shape[1] != 9_557:
        raise ValueError("Sequence and marker feature matrices are incompatible")
    if marker.shape[1] != 17:
        raise ValueError("Marker feature matrix has the wrong width")
    normalized = (marker - normalization[0]) / normalization[1]
    fused = np.concatenate([sequence, normalized], axis=1).astype(np.float32, copy=False)
    if fused.shape[1] != INPUT_DIM or not np.isfinite(fused).all():
        raise ValueError("Fused feature matrix is invalid")
    return fused


def ensemble_probabilities(
    models: Sequence[MobiOriginMLP], values: NDArray[np.float32]
) -> NDArray[np.float32]:
    total = np.zeros((len(values), 3), dtype=np.float64)
    with torch.no_grad():
        tensor = torch.from_numpy(values)
        for model in models:
            model.eval()
            total += torch.softmax(model(tensor), dim=1).cpu().numpy().astype(np.float64)
    probabilities = (total / len(models)).astype(np.float32)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("Ensemble probabilities are invalid")
    return probabilities


def selective_labels(probabilities: NDArray[np.float32]) -> tuple[list[str], NDArray[np.float32]]:
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Probability matrix must have three columns")
    base = np.argmax(probabilities, axis=1)
    competitor = np.maximum(probabilities[:, 0], probabilities[:, 2])
    score = probabilities[:, 1] - competitor
    labels: list[str] = []
    for index, class_index in enumerate(base):
        if class_index == 1 and score[index] < SELECTIVE_THRESHOLD:
            labels.append("unclassified")
        else:
            labels.append(CLASS_NAMES[int(class_index)])
    return labels, score.astype(np.float32)


def _write_predictions(
    path: Path,
    records: Sequence[FastaRecord],
    probabilities: NDArray[np.float32],
    labels: Sequence[str],
    scores: NDArray[np.float32],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "sequence_id",
                "length_bp",
                "prediction",
                "p_chromosome",
                "p_plasmid",
                "p_phage",
                "plasmid_score",
                "abstention_reason",
            ]
        )
        for record, row, label, score in zip(records, probabilities, labels, scores, strict=True):
            if not record.supported:
                reason = "unsupported_length"
            elif label == "unclassified":
                reason = "low_plasmid_score"
            else:
                reason = ""
            writer.writerow(
                [
                    record.identifier,
                    len(record.sequence),
                    label,
                    *(f"{float(value):.9g}" for value in row),
                    f"{float(score):.9g}",
                    reason,
                ]
            )


def predict(
    *,
    input_fasta: Path,
    output_dir: Path,
    database_dir: Path,
    threads: int,
    model_dir: Path | None = None,
) -> None:
    """Run one complete atomic MobiOrigin prediction."""
    if not 1 <= threads <= 8:
        raise ValueError("Threads must be between 1 and 8")
    if output_dir.exists():
        raise FileExistsError("Output directory already exists")
    configure_runtime()
    records = read_fasta(input_fasta)
    supported_indices = [index for index, record in enumerate(records) if record.supported]
    supported = [records[index] for index in supported_indices]
    models_root = model_dir or Path(__file__).parent / "data" / "models" / "dev1"
    models, normalization = load_artifacts(models_root)
    databases = load_database_manifest(database_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        # Unsupported-length records remain explicit abstentions while carrying
        # a neutral, valid probability vector rather than fabricated evidence.
        probabilities = np.full((len(records), 3), 1.0 / 3.0, dtype=np.float32)
        labels = ["unclassified"] * len(records)
        scores = np.zeros(len(records), dtype=np.float32)
        if supported:
            sequence = extract_sequence_features([record.sequence for record in supported])
            marker = extract_marker_features(
                supported,
                databases=databases,
                diamond=Path(os.environ.get("MOBIORIGIN_DIAMOND", "diamond")),
                threads=threads,
                work_dir=temporary / "marker_work",
            )
            supported_probabilities = ensemble_probabilities(
                models, fuse_features(sequence, marker, normalization)
            )
            supported_labels, supported_scores = selective_labels(supported_probabilities)
            for local, global_index in enumerate(supported_indices):
                probabilities[global_index] = supported_probabilities[local]
                labels[global_index] = supported_labels[local]
                scores[global_index] = supported_scores[local]
        predictions = temporary / "predictions.tsv"
        _write_predictions(predictions, records, probabilities, labels, scores)
        provenance: dict[str, Any] = {
            "schema_version": "mobiorigin-prediction-provenance-v1",
            "tool": "MobiOrigin",
            "version": __version__,
            "input_fasta_sha256": sha256_file(input_fasta),
            "input_records": len(records),
            "supported_records": len(supported),
            "unsupported_length_records": len(records) - len(supported),
            "model_sha256": MODEL_SHA256,
            "marker_normalization_sha256": NORMALIZATION_SHA256,
            "database_sha256": {
                family: sha256_file(path) for family, path in sorted(databases.items())
            },
            "selective_threshold": SELECTIVE_THRESHOLD,
            "prediction_sha256": sha256_file(predictions),
            "network_accessed": False,
        }
        atomic_json(temporary / "provenance.json", provenance)
        checksums = "".join(
            f"{sha256_file(temporary / name)}  {name}\n"
            for name in ("predictions.tsv", "provenance.json")
        )
        atomic_text(temporary / "SHA256SUMS.txt", checksums)
        shutil.rmtree(temporary / "marker_work", ignore_errors=True)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
