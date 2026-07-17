"""Train the PlasFlow v2 classifier (Random Forest + MLP).

Usage:
    python scripts/train_model.py --data data/features.npy --labels data/labels.npy --mlp
    python scripts/train_model.py --data data/features.npy --labels data/labels.npy --rf

Segfault fix for macOS ARM (Apple Silicon)
-------------------------------------------
On macOS ARM, PyTorch + numpy share the same BLAS/Accelerate/OpenBLAS
threading runtime.  When both try to spin up threads simultaneously
(default: one thread per CPU core), the initialisation races → SIGABRT /
segfault.  Setting the thread caps to 1 **before any import** of numpy or
torch is the canonical fix:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

These are set programmatically at the top of this file so you don't need
to set them in your shell.

Memory design for MLP on macOS ARM
------------------------------------
With 400k × 1281 features the full array is ~2 GB.  Copying it into RAM
alongside torch tensors causes macOS memory pressure → segfault.

Solution: MmapDataset reads ONE BATCH at a time directly from the memory-
mapped .npy file.  The full training set is never in RAM simultaneously.

Peak RAM during training:
    Validation set:  40k × 1281 × 4 B  ≈  0.21 GB   (loaded once)
    One batch:      512 × 1281 × 4 B  ≈  0.003 GB  (ephemeral)
    Model weights:                     ≈  0.007 GB
    Total:                             ≈  0.22 GB   ← well within limits
"""

# Imports intentionally follow the thread-pool environment setup below.
# ruff: noqa: E402

from __future__ import annotations

# ── MUST be set before numpy / torch are imported ───────────────────────────
# Root cause of the segfault on macOS ARM: PyTorch and numpy both try to spin
# up one thread per CPU core via the same BLAS runtime (Accelerate / OpenBLAS
# / OpenMP).  The initialisation races → SIGABRT.  Capping all thread pools
# to 1 before importing anything resolves it.  setdefault() leaves any value
# the user explicitly pre-set in their shell intact.
import os as _os

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_v, "1")
# ────────────────────────────────────────────────────────────────────────────

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SEED = 42


# ---------------------------------------------------------------------------
# Memory-mapped dataset — the core of the OOM fix
# ---------------------------------------------------------------------------


def _iter_mmap_batches(
    features: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator | None = None,
):
    """Yield vectorized mmap batches, optionally shuffled by row each epoch.

    Reading 512 rows in one NumPy operation avoids hundreds of Python tensor
    allocations and random file reads per batch. Rows are sorted only for the
    disk read; order inside a gradient batch has no effect on optimization.
    """

    positions = np.arange(len(indices), dtype=np.int64)
    if rng is not None:
        rng.shuffle(positions)
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        batch_indices = indices[batch_positions]
        read_order = np.argsort(batch_indices)
        batch_indices = batch_indices[read_order]
        batch_labels = labels[batch_positions][read_order]
        batch_features = np.ascontiguousarray(features[batch_indices], dtype=np.float32)
        yield torch.from_numpy(batch_features), torch.from_numpy(
            np.ascontiguousarray(batch_labels, dtype=np.int64)
        )


def _exclude_source_groups(
    indices: np.ndarray,
    labels: np.ndarray,
    sequence_ids: list[str],
    excluded_groups: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Remove rows whose biological source group is reserved for evaluation."""

    from plasflow2.classify.splits import source_group_id

    groups = [source_group_id(sequence_id) for sequence_id in sequence_ids]
    keep = np.fromiter(
        (group not in excluded_groups for group in groups),
        dtype=bool,
        count=len(groups),
    )
    return (
        indices[keep],
        labels[keep],
        [sequence_id for sequence_id, retain in zip(sequence_ids, keep) if retain],
        [group for group, retain in zip(groups, keep) if retain],
    )


# ---------------------------------------------------------------------------
# Training loop (standalone — does not call train.py's train_mlp)
# ---------------------------------------------------------------------------


def _train_mlp_mmap(
    data_path: str,
    idx_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 10,
    out_path: Path = Path("data/models/mlp_v2.pt"),
    num_classes: int = 3,
    use_class_weights: bool = True,
    torch_threads: int = 1,
    hidden_dims: tuple[int, int, int] = (2048, 512, 128),
    resume: bool = True,
) -> None:
    from plasflow2.classify.model import PlasFlowMLP, save_model
    from plasflow2.utils.device import get_device
    from sklearn.metrics import accuracy_score, f1_score  # type: ignore[import]

    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(max(1, torch_threads))
        logger.info("PyTorch CPU threads: %d", torch.get_num_threads())

    # Determine input_dim from the mmap without loading it fully
    X_mmap_meta = np.load(data_path, mmap_mode="r")
    input_dim = X_mmap_meta.shape[1]
    del X_mmap_meta

    model = PlasFlowMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
    ).to(device)
    logger.info(
        "Model: input_dim=%d  hidden_dims=%s  num_classes=%d  device=%s",
        input_dim,
        hidden_dims,
        num_classes,
        device,
    )

    X_train_mmap = np.load(data_path, mmap_mode="r")
    n_batches = int(np.ceil(len(idx_tr) / batch_size))
    logger.info("Training batches per epoch: %d", n_batches)
    rng = np.random.default_rng(_SEED)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights: inverse frequency so minority classes get equal gradient.
    # NOTE: use_class_weights=True was tested for the binary model (2.25:1
    # imbalance) but caused FP explosion — chromosomes' plasmid scores inflated
    # above the threshold. Disabled by default; use --no-class-weights flag.
    if use_class_weights:
        class_counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1, class_counts)  # avoid /0
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * num_classes  # normalise to mean=1
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        logger.info(
            "Class weights (inverse-freq, normalised): %s",
            {i: f"{w:.3f}" for i, w in enumerate(class_weights)},
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)
    else:
        logger.info("Class weights: DISABLED (--no-class-weights) — using uniform weights")
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    checkpoint_path = out_path.with_name("training_checkpoint.pt")
    best_val_macro_f1 = float("-inf")
    best_val_accuracy = 0.0
    best_state: dict = {}
    no_improve = 0
    start_epoch = 1
    elapsed_before_resume = 0.0

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_config = {
            "input_dim": input_dim,
            "hidden_dims": tuple(hidden_dims),
            "num_classes": num_classes,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "use_class_weights": use_class_weights,
            "selection_metric": "macro_f1",
        }
        if checkpoint.get("config") != expected_config:
            raise ValueError(
                f"Training checkpoint configuration does not match this run: {checkpoint_path}. "
                "Use --restart-training to discard it."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_macro_f1 = float(checkpoint["best_val_macro_f1"])
        best_val_accuracy = float(checkpoint["best_val_accuracy"])
        best_state = checkpoint["best_state"]
        no_improve = int(checkpoint["no_improve"])
        rng.bit_generator.state = checkpoint["rng_state"]
        start_epoch = int(checkpoint["epoch"]) + 1
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        if no_improve >= patience:
            start_epoch = epochs + 1
        logger.info(
            "Resuming from epoch %d checkpoint "
            "(best_val_macro_f1=%.4f, best_val_accuracy=%.4f, no_improve=%d)",
            checkpoint["epoch"],
            best_val_macro_f1,
            best_val_accuracy,
            no_improve,
        )

    t0 = time.time() - elapsed_before_resume
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in _iter_mmap_batches(
            X_train_mmap,
            idx_tr,
            y_tr,
            batch_size=batch_size,
            rng=rng,
        ):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X_va), batch_size):
                xb = torch.from_numpy(X_va[start : start + batch_size]).to(device)
                val_predictions.append(model(xb).argmax(dim=-1).cpu().numpy())
        preds = np.concatenate(val_predictions)
        val_acc = accuracy_score(y_va, preds)
        val_macro_f1 = f1_score(y_va, preds, average="macro")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_accuracy = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d — loss %.4f  val_acc %.4f  "
                "val_macro_f1 %.4f  best_macro_f1 %.4f  [%.0f s]",
                epoch,
                epochs,
                total_loss / n_batches,
                val_acc,
                val_macro_f1,
                best_val_macro_f1,
                elapsed,
            )

        checkpoint = {
            "epoch": epoch,
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "best_val_accuracy": best_val_accuracy,
            "best_state": best_state,
            "no_improve": no_improve,
            "rng_state": rng.bit_generator.state,
            "elapsed_seconds": time.time() - t0,
            "config": {
                "input_dim": input_dim,
                "hidden_dims": tuple(hidden_dims),
                "num_classes": num_classes,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "use_class_weights": use_class_weights,
                "selection_metric": "macro_f1",
            },
        }
        checkpoint_tmp = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, checkpoint_tmp)
        _os.replace(checkpoint_tmp, checkpoint_path)
        logger.info("Recovery checkpoint saved after epoch %d → %s", epoch, checkpoint_path)

        if no_improve >= patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience
            )
            break

    model.load_state_dict(best_state)
    model.eval()
    del X_train_mmap
    logger.info(
        "Best validation macro-F1: %.4f  (accuracy at best epoch: %.4f)",
        best_val_macro_f1,
        best_val_accuracy,
    )

    save_model(model, out_path)
    logger.info("Model saved → %s", out_path)


def _evaluate_mlp_mmap(
    model_path: Path,
    data_path: str,
    idx_te: np.ndarray,
    y_te: np.ndarray,
    class_names: list[str],
    out_path: Path,
    batch_size: int = 512,
) -> dict[str, object]:
    """Evaluate a saved MLP on the untouched test split without loading all features."""

    from plasflow2.classify.model import load_model
    from plasflow2.utils.device import get_device
    from sklearn.metrics import (  # type: ignore[import]
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    device = get_device()
    model = load_model(model_path, device=device)
    features = np.load(data_path, mmap_mode="r")
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in _iter_mmap_batches(
            features,
            idx_te,
            y_te,
            batch_size=batch_size,
        ):
            predictions.append(model(xb.to(device)).argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(predictions).astype(np.int64, copy=False)
    labels = list(range(len(class_names)))
    report = classification_report(
        y_te,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "n_test": int(len(y_te)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_te, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Held-out test: accuracy=%.4f  balanced_accuracy=%.4f  macro_f1=%.4f",
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
    )
    logger.info("Held-out metrics saved → %s", out_path)
    return metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train PlasFlow v2 models")
    parser.add_argument("--data", required=True, help="Feature matrix (.npy)")
    parser.add_argument("--labels", required=True, help="Labels array (.npy)")
    parser.add_argument(
        "--ids",
        default=None,
        help="Sequence IDs (one per row). Defaults to seq_ids.txt beside --labels. "
        "Required unless --allow-row-split is explicitly used.",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Output TSV describing the grouped split (default: <out>/split_manifest.tsv).",
    )
    parser.add_argument(
        "--allow-row-split",
        action="store_true",
        help="Allow the legacy random row split when sequence IDs are unavailable. "
        "This can leak overlapping windows and is not suitable for reported results.",
    )
    parser.add_argument("--out", default="data/models", help="Output directory")
    parser.add_argument("--rf", action="store_true", help="Train Random Forest")
    parser.add_argument("--mlp", action="store_true", help="Train MLP")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads used by PyTorch (default: 1; >1 may crash on macOS ARM).",
    )
    parser.add_argument(
        "--hidden-dims",
        default="2048,512,128",
        help="Three comma-separated MLP hidden widths (default: 2048,512,128).",
    )
    parser.add_argument(
        "--restart-training",
        action="store_true",
        help="Ignore and replace any recovery checkpoint in the output directory.",
    )
    parser.add_argument(
        "--exclude-groups",
        type=Path,
        default=None,
        help="File containing source-accession groups to reserve and exclude from training.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weighting (use uniform weights)",
    )
    parser.add_argument(
        "--class-filter",
        type=str,
        default=None,
        help="Comma-separated original class indices to keep, e.g. '1,2'. "
        "Filtered classes are renumbered 0,1,... in sorted order. "
        "Useful for Stage 2 cascade training (chr+phage only). "
        "MmapDataset reads only the retained rows from disk.",
    )
    args = parser.parse_args()

    try:
        hidden_dims = tuple(int(value) for value in args.hidden_dims.split(","))
    except ValueError:
        parser.error("--hidden-dims must contain exactly three positive integers")
    if len(hidden_dims) != 3 or any(value <= 0 for value in hidden_dims):
        parser.error("--hidden-dims must contain exactly three positive integers")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Random Forest ────────────────────────────────────────────────────────
    if args.rf:
        from plasflow2.classify.splits import (
            grouped_split_indices,
            load_sequence_ids,
            source_group_id,
            validate_group_labels,
            write_split_manifest,
        )
        from plasflow2.classify.train import evaluate, save_rf, split_data
        from plasflow2.classify.train import train_rf as _train_rf
        from plasflow2.utils.device import IDX_TO_CLASS

        X = np.load(args.data).astype(np.float32)
        y = np.load(args.labels).astype(np.int64)
        logger.info("Loaded X=%s  y=%s", X.shape, y.shape)

        ids_path = Path(args.ids) if args.ids else Path(args.labels).with_name("seq_ids.txt")
        if ids_path.exists():
            sequence_ids = load_sequence_ids(ids_path, expected_count=len(y))
            groups = [source_group_id(sid) for sid in sequence_ids]
            validate_group_labels(y, groups)
            idx_tr, idx_va, idx_te = grouped_split_indices(y, groups, random_state=_SEED)
            X_tr, X_va, X_te = X[idx_tr], X[idx_va], X[idx_te]
            y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
            manifest_path = (
                Path(args.split_manifest) if args.split_manifest else out_dir / "split_manifest.tsv"
            )
            write_split_manifest(manifest_path, sequence_ids, y, groups, idx_tr, idx_va, idx_te)
        elif args.allow_row_split:
            logger.warning("Using leakage-prone legacy row split because %s is absent", ids_path)
            X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, val_size=0.1, test_size=0.1)
        else:
            raise FileNotFoundError(
                f"Sequence IDs not found at {ids_path}. Provide --ids or explicitly use "
                "--allow-row-split for the leakage-prone legacy behavior."
            )
        logger.info("Train=%d  Val=%d  Test=%d", len(X_tr), len(X_va), len(X_te))

        t0 = time.time()
        rf = _train_rf(X_tr, y_tr, cv_folds=0)
        logger.info("RF trained in %.1f s", time.time() - t0)

        class_names = [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS)]
        result = evaluate(y_te, rf.predict(X_te), class_names=class_names)
        logger.info("Test accuracy: %.4f", result["accuracy"])
        logger.info("\n%s", result["report"])
        save_rf(rf, out_dir / "rf_v2.pkl")

    # ── MLP — mmap-based training, never loads full X into RAM ───────────────
    if args.mlp:
        from plasflow2.classify.splits import (
            grouped_split_indices,
            load_sequence_ids,
            source_group_id,
            validate_group_labels,
            write_split_manifest,
        )
        from sklearn.model_selection import train_test_split  # type: ignore[import]

        # Step 1: split INDICES only (labels are 3 MB — trivial)
        logger.info("Loading labels and splitting indices …")
        y_all = np.load(args.labels).astype(np.int64)
        n = len(y_all)
        logger.info("Total samples: %d", n)

        # Optional class filter — keeps only specified classes and renumbers them.
        # MmapDataset uses the original indices into the features file, so no
        # feature copying is needed even when filtering to a subset of classes.
        if args.class_filter:
            keep = sorted(int(c) for c in args.class_filter.split(","))
            mask = np.isin(y_all, keep)
            remap = {orig: new for new, orig in enumerate(keep)}
            # idx_all stores ORIGINAL row indices into features.npy
            idx_all = np.where(mask)[0]
            y_all = np.array([remap[int(y)] for y in y_all[mask]], dtype=np.int64)
            n = len(y_all)
            logger.info(
                "Class filter %s → renumbered 0..%d  (%d samples kept)", keep, len(keep) - 1, n
            )
            original_classes = keep
        else:
            idx_all = np.arange(n)
            original_classes = sorted(int(label) for label in np.unique(y_all))

        ids_path = Path(args.ids) if args.ids else Path(args.labels).with_name("seq_ids.txt")
        if ids_path.exists():
            all_sequence_ids = load_sequence_ids(ids_path, expected_count=len(np.load(args.labels)))
            selected_sequence_ids = [all_sequence_ids[int(i)] for i in idx_all]
            if args.exclude_groups is not None:
                excluded_groups = {
                    line.strip()
                    for line in args.exclude_groups.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                }
                before = len(idx_all)
                idx_all, y_all, selected_sequence_ids, groups = _exclude_source_groups(
                    idx_all,
                    y_all,
                    selected_sequence_ids,
                    excluded_groups,
                )
                logger.info(
                    "Benchmark lockout: excluded %d rows from %d reserved source groups; "
                    "%d samples remain",
                    before - len(idx_all),
                    len(excluded_groups),
                    len(idx_all),
                )
            else:
                groups = [source_group_id(sid) for sid in selected_sequence_ids]
            validate_group_labels(y_all, groups)
            rel_tr, rel_va, rel_te = grouped_split_indices(
                y_all, groups, val_size=0.10, test_size=0.10, random_state=_SEED
            )
            idx_tr, idx_va, idx_te = idx_all[rel_tr], idx_all[rel_va], idx_all[rel_te]
            y_tr, y_va, y_te = y_all[rel_tr], y_all[rel_va], y_all[rel_te]
            manifest_path = (
                Path(args.split_manifest) if args.split_manifest else out_dir / "split_manifest.tsv"
            )
            write_split_manifest(
                manifest_path,
                selected_sequence_ids,
                y_all,
                groups,
                rel_tr,
                rel_va,
                rel_te,
                source_row_indices=idx_all,
            )
            logger.info(
                "Leakage-resistant grouped split: %d train / %d val / %d held-out test rows; "
                "%d source groups; manifest=%s",
                len(idx_tr),
                len(idx_va),
                len(idx_te),
                len(set(groups)),
                manifest_path,
            )
        elif args.allow_row_split:
            if args.exclude_groups is not None:
                raise ValueError("--exclude-groups requires sequence IDs for source matching")
            logger.warning(
                "Sequence IDs not found at %s: using legacy row-level split. "
                "Do not report metrics from this model as leakage-resistant.",
                ids_path,
            )
            idx_trainval, idx_te, y_trainval, y_te = train_test_split(
                idx_all,
                y_all,
                test_size=0.10,
                stratify=y_all,
                random_state=_SEED,
            )
            idx_tr, idx_va, y_tr, y_va = train_test_split(
                idx_trainval,
                y_trainval,
                test_size=0.10 / 0.90,
                stratify=y_trainval,
                random_state=_SEED,
            )
        else:
            raise FileNotFoundError(
                f"Sequence IDs not found at {ids_path}. Provide --ids or explicitly use "
                "--allow-row-split for the leakage-prone legacy behavior."
            )
        num_classes = int(len(np.unique(y_all)))
        logger.info("Unique labels: %s  (num_classes=%d)", np.unique(y_all).tolist(), num_classes)
        del idx_all, y_all
        gc.collect()
        logger.info(
            "Split: Train=%d  Val=%d  Held-out test=%d", len(idx_tr), len(idx_va), len(idx_te)
        )

        # Step 2: load ONLY the validation slice into RAM (~0.21 GB)
        logger.info("Loading validation slice into RAM …")
        X_mmap = np.load(args.data, mmap_mode="r")
        X_va_np = np.ascontiguousarray(X_mmap[idx_va]).astype(np.float32)
        del X_mmap
        gc.collect()
        logger.info("X_va in RAM: %.2f GB  (training data stays on disk)", X_va_np.nbytes / 1e9)

        # Step 3: train from vectorized memory-mapped batches.
        if args.restart_training:
            (out_dir / "training_checkpoint.pt").unlink(missing_ok=True)
        _train_mlp_mmap(
            data_path=args.data,
            idx_tr=idx_tr,
            y_tr=y_tr,
            X_va=X_va_np,
            y_va=y_va,
            epochs=args.epochs,
            batch_size=512,
            lr=1e-3,
            patience=10,
            out_path=out_dir / "mlp_v2.pt",
            num_classes=num_classes,
            use_class_weights=not args.no_class_weights,
            torch_threads=args.torch_threads,
            hidden_dims=hidden_dims,
            resume=not args.restart_training,
        )

        from plasflow2.utils.device import IDX_TO_CLASS

        class_names = [IDX_TO_CLASS.get(label, str(label)) for label in original_classes]
        _evaluate_mlp_mmap(
            model_path=out_dir / "mlp_v2.pt",
            data_path=args.data,
            idx_te=idx_te,
            y_te=y_te,
            class_names=class_names,
            out_path=out_dir / "heldout_test_metrics.json",
        )
        (out_dir / "training_checkpoint.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
