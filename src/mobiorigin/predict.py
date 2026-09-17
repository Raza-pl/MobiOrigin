"""Frozen MobiOrigin dev1 production inference pipeline."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray

from mobiorigin import __version__
from mobiorigin.fasta import (
    STRONG_AMBIGUITY_WARNING_FRACTION,
    FastaRecord,
    SequenceQuality,
    read_fasta,
    sequence_quality,
)
from mobiorigin.marker_features import extract_marker_features, load_database_manifest
from mobiorigin.model import INPUT_DIM, MobiOriginMLP, load_model
from mobiorigin.model_setup import resolve_model_dir
from mobiorigin.provenance import atomic_json, atomic_text, sha256_file
from mobiorigin.runtime import validate_threads
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


def reverse_orientation_marker_features(
    marker: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Return the analytical reverse-orientation form of frozen marker features.

    Sixteen of the seventeen coding-and-mobility features are orientation
    invariant. The forward-ORF fraction becomes one minus its original value
    when at least one retained ORF is present. This transformation avoids a
    second gene prediction and database search while making the final ensemble
    probability exactly invariant to input strand orientation.
    """
    if marker.ndim != 2 or marker.shape[1] != 17:
        raise ValueError("Marker feature matrix has the wrong width")
    reverse = marker.copy()
    has_orf = marker[:, 1] > 0
    reverse[:, 4] = np.where(has_orf, 1.0 - marker[:, 4], 0.0)
    return reverse


def orientation_invariant_probabilities(
    models: Sequence[MobiOriginMLP],
    sequence: NDArray[np.float32],
    marker: NDArray[np.float32],
    normalization: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Average probabilities from both analytical sequence orientations."""
    forward = ensemble_probabilities(models, fuse_features(sequence, marker, normalization))
    reverse = ensemble_probabilities(
        models,
        fuse_features(sequence, reverse_orientation_marker_features(marker), normalization),
    )
    probabilities = ((forward.astype(np.float64) + reverse.astype(np.float64)) / 2).astype(
        np.float32
    )
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Orientation-averaged probabilities are invalid")
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
    qualities: Sequence[SequenceQuality] | None = None,
) -> None:
    measured = (
        list(qualities)
        if qualities is not None
        else [sequence_quality(record.sequence) for record in records]
    )
    if len(measured) != len(records):
        raise ValueError("Sequence-quality measurements do not match FASTA records")
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
                "non_acgt_bases",
                "non_acgt_fraction",
                "n_bases",
                "n_fraction",
                "normalized_4mer_entropy",
                "input_quality_warnings",
            ]
        )
        for record, row, label, score, quality in zip(
            records, probabilities, labels, scores, measured, strict=True
        ):
            if not record.supported:
                reason = "unsupported_length"
            elif quality.non_acgt_bases:
                reason = "ambiguous_bases"
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
                    quality.non_acgt_bases,
                    f"{quality.non_acgt_fraction:.9g}",
                    quality.n_bases,
                    f"{quality.n_fraction:.9g}",
                    f"{quality.normalized_4mer_entropy:.9g}",
                    ";".join(quality.warnings),
                ]
            )


def predict(
    *,
    input_fasta: Path,
    output_dir: Path,
    database_dir: Path,
    threads: int,
    model_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Run one complete atomic MobiOrigin prediction."""
    notify = progress or (lambda message: None)
    validate_threads(threads)
    if output_dir.exists():
        raise FileExistsError("Output directory already exists")
    configure_runtime()
    notify("Reading and validating the input FASTA")
    records = read_fasta(input_fasta)
    qualities = [sequence_quality(record.sequence) for record in records]
    ambiguous_records = sum(quality.non_acgt_bases > 0 for quality in qualities)
    strong_ambiguity_warnings = sum(
        quality.non_acgt_fraction >= STRONG_AMBIGUITY_WARNING_FRACTION for quality in qualities
    )
    notify(
        "Input quality: "
        f"{ambiguous_records:,}/{len(records):,} records contain non-ACGT bases; "
        f"{strong_ambiguity_warnings:,} meet the >=0.10% ambiguity warning"
    )
    supported_indices = [
        index
        for index, (record, quality) in enumerate(zip(records, qualities, strict=True))
        if record.supported and quality.non_acgt_bases == 0
    ]
    supported = [records[index] for index in supported_indices]
    models_root = resolve_model_dir(model_dir)
    notify("Verifying and loading the frozen model ensemble")
    models, normalization = load_artifacts(models_root)
    notify("Verifying the frozen marker databases")
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
            notify(f"Calculating sequence features for {len(supported):,} supported contigs")
            sequence = extract_sequence_features([record.sequence for record in supported])
            notify("Searching MOB-suite marker proteins with DIAMOND")
            marker = extract_marker_features(
                supported,
                databases=databases,
                diamond=Path(os.environ.get("MOBIORIGIN_DIAMOND", "diamond")),
                threads=threads,
                work_dir=temporary / "marker_work",
            )
            notify("Running the three-network ensemble and selective decision rule")
            supported_probabilities = orientation_invariant_probabilities(
                models, sequence, marker, normalization
            )
            supported_labels, supported_scores = selective_labels(supported_probabilities)
            for local, global_index in enumerate(supported_indices):
                probabilities[global_index] = supported_probabilities[local]
                labels[global_index] = supported_labels[local]
                scores[global_index] = supported_scores[local]
        notify("Writing predictions, provenance, and checksums")
        predictions = temporary / "predictions.tsv"
        _write_predictions(predictions, records, probabilities, labels, scores, qualities)
        provenance: dict[str, Any] = {
            "schema_version": "mobiorigin-prediction-provenance-v2",
            "tool": "MobiOrigin",
            "version": __version__,
            "input_fasta_sha256": sha256_file(input_fasta),
            "input_records": len(records),
            "supported_records": len(supported),
            "unsupported_length_records": sum(not record.supported for record in records),
            "ambiguous_base_abstentions": ambiguous_records,
            "input_quality_control": {
                "records_with_non_acgt_bases": ambiguous_records,
                "records_with_non_acgt_fraction_ge_0.001": strong_ambiguity_warnings,
                "non_acgt_warning_policy": "warn on any non-ACGT content",
                "strong_ambiguity_warning_fraction": STRONG_AMBIGUITY_WARNING_FRACTION,
                "complexity_metric": (
                    "normalized Shannon entropy of observed unambiguous 4-mers; "
                    "descriptive only and not used to alter predictions"
                ),
                "ambiguity_policy": (
                    "records containing an accepted non-ACGT IUPAC symbol are reported as unclassified "
                    "with abstention_reason=ambiguous_bases and are not passed to the model"
                ),
                "prediction_semantics_changed": True,
            },
            "orientation_policy": (
                "ensemble probabilities are averaged across analytical forward and "
                "reverse-orientation coding features"
            ),
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
