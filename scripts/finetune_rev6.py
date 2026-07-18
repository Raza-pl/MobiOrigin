#!/usr/bin/env python3
"""Fine-tune Rev6 MLP — stratified chunk I/O, time-budgeted, low-RAM.

The 13 GB feature file has class-homogeneous layout:
  chunks 0-6   : 100 % class 0   (chromosome)
  chunks 7-13  : 100 % class 2   (plasmid)
  chunks 14-23 : 100 % class 1   (ambiguous/low-confidence)

Random chunk selection caused wild val-F1 oscillation (0.57 → 0.12) because
some epochs received only one class.  This version uses stratified interleaving:
  1. Group chunks by dominant class label.
  2. Shuffle within each group each epoch.
  3. Round-robin interleave: class0, class2, class1, class0, class2, class1 …
  4. Stop when TRAIN_TIME_BUDGET is exhausted (~33 s → covers ≥ 83 % of data
     with balanced class coverage).

Memory budget:
  val float16 tensor : 647 MB   (float16 → .float() per batch during inference)
  training chunk f32 : 572 MB   (15 K rows × 9559 × 4)
  after del chunk_X  : 458 MB   (training rows only)
  peak               : ~1.8 GB  (well inside 3.8 GB sandbox)
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

torch.set_num_threads(4)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE        = 15_000   # rows per sequential I/O read
TRAIN_TIME_BUDGET = 30.0     # seconds; covers ~83 % of data with balanced classes


def stratified_chunk_order(
    all_starts: list[int],
    is_train: np.ndarray,
    y_all: np.ndarray,
    n_total: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> list[int]:
    """Return chunk indices in a class-stratified interleaved order.

    Groups chunks by dominant training-class label, shuffles within groups,
    then round-robins across groups so that any time-budget prefix covers all
    classes.
    """
    groups: dict[int, list[int]] = {}
    for ci, start in enumerate(all_starts):
        end = min(start + chunk_size, n_total)
        mask = is_train[start:end]
        if not mask.any():
            groups.setdefault(-1, []).append(ci)
            continue
        y_ch = y_all[start:end][mask]
        dominant = int(np.bincount(y_ch.astype(np.intp), minlength=3).argmax())
        groups.setdefault(dominant, []).append(ci)

    # Shuffle within each class group
    for g in groups:
        arr = groups[g]; rng.shuffle(arr); groups[g] = arr

    # Round-robin interleave (sorted keys for deterministic group order)
    keys = sorted(k for k in groups if k >= 0) + ([-1] if -1 in groups else [])
    result: list[int] = []
    max_len = max(len(groups[k]) for k in keys)
    for i in range(max_len):
        for k in keys:
            if i < len(groups[k]):
                result.append(groups[k][i])
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=Path, required=True)
    parser.add_argument("--labels",     type=Path, required=True)
    parser.add_argument("--manifest",   type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--out",        type=Path, required=True)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--max-time",   type=float, default=40.0,
                        help="Wall-clock budget per bash call (s)")
    args = parser.parse_args()

    call_start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt_path        = args.out / "finetune_checkpoint.pt"
    best_path        = args.out / "mlp_best.pt"
    done_path        = args.out / "finetune_done.flag"
    idx_cache_path   = args.out / "idx_cache.npz"
    val_cache_X_path = args.out / "val_cache_X.npy"   # float16
    val_cache_y_path = args.out / "val_cache_y.npy"
    is_train_path    = args.out / "is_train.npy"

    if done_path.exists():
        logger.info("Already complete.")
        return

    # ----------------------------------------------------------------- data
    if idx_cache_path.exists():
        c = np.load(idx_cache_path)
        idx_tr, idx_va = c["idx_tr"], c["idx_va"]
        logger.info("idx cache: %d train, %d val", len(idx_tr), len(idx_va))
    else:
        logger.info("Parsing manifest ...")
        tr_list, va_list = [], []
        with open(args.manifest) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                ri = int(row["row_index"])
                if row["split"] == "train":                tr_list.append(ri)
                elif row["split"] in ("val", "validation"): va_list.append(ri)
        idx_tr = np.array(tr_list, dtype=np.int64)
        idx_va = np.array(va_list, dtype=np.int64)
        np.savez(idx_cache_path, idx_tr=idx_tr, idx_va=idx_va)
        logger.info("Split: %d train, %d val (cached)", len(idx_tr), len(idx_va))

    logger.info("Memory-mapping feature matrix ...")
    X_all   = np.load(args.data, mmap_mode="r")
    y_all   = np.load(args.labels)
    n_total = X_all.shape[0]
    logger.info("Feature matrix: %s  dtype=%s", X_all.shape, X_all.dtype)

    if is_train_path.exists():
        is_train = np.load(is_train_path)
    else:
        is_train = np.zeros(n_total, dtype=bool)
        is_train[idx_tr] = True
        np.save(is_train_path, is_train)

    # ---- val cache: float16 in RAM → convert per batch to avoid peak RAM ----
    expected_f16 = len(idx_va) * X_all.shape[1] * 2
    cache_valid  = (
        val_cache_X_path.exists()
        and val_cache_y_path.exists()
        and val_cache_X_path.stat().st_size >= expected_f16
        and val_cache_y_path.stat().st_size > 0
    )
    if cache_valid:
        logger.info("Loading val cache (float16) ...")
        t0 = time.time()
        X_va_f16 = np.ascontiguousarray(
            np.load(val_cache_X_path, mmap_mode="r"), dtype=np.float16
        )
        y_va = np.load(val_cache_y_path).astype(np.int64)
        logger.info("Val cache %.1fs  shape=%s", time.time() - t0, X_va_f16.shape)
    else:
        logger.info("Building val cache (one-time ~14 s) ...")
        t0 = time.time()
        X_va_f16 = np.ascontiguousarray(X_all[idx_va], dtype=np.float16)
        y_va     = y_all[idx_va].astype(np.int64)
        np.save(val_cache_X_path, X_va_f16)
        np.save(val_cache_y_path, y_va)
        logger.info("Val cache built in %.1fs", time.time() - t0)

    # Keep in RAM as float16; convert to float32 per batch → peak 647 MB
    X_va_t = torch.from_numpy(X_va_f16)           # already contiguous & writable
    del X_va_f16

    classes, counts = np.unique(y_all[idx_tr], return_counts=True)
    class_weights   = torch.tensor(
        len(idx_tr) / (len(classes) * counts), dtype=torch.float32
    )
    logger.info("Class weights: %s",
                dict(zip(classes.tolist(), [f"{w:.3f}" for w in class_weights.tolist()])))

    # ------------------------------------------------ model
    from plasflow2.classify.model import PlasFlowMLP, load_model, save_model

    rev5       = load_model(args.pretrained, device=torch.device("cpu"))
    rev5_state = rev5.state_dict()
    input_dim_r5 = int(rev5_state["net.0.weight"].shape[1])
    input_dim_r6 = int(X_all.shape[1])
    h1 = int(rev5_state["net.0.weight"].shape[0])
    h2 = int(rev5_state["net.4.weight"].shape[0])
    h3 = int(rev5_state["net.8.weight"].shape[0])
    logger.info("Rev5: %d->%d->%d->%d->3  Rev6: %d->%d->%d->%d->3",
                input_dim_r5, h1, h2, h3, input_dim_r6, h1, h2, h3)

    rev6      = PlasFlowMLP(input_dim=input_dim_r6, hidden_dims=(h1, h2, h3), num_classes=3)
    optimizer = torch.optim.AdamW(rev6.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    expected_config = {
        "epochs": args.epochs, "lr": args.lr,
        "input_dim": input_dim_r6, "h1": h1, "h2": h2, "h3": h3,
    }

    start_epoch = 1; best_f1 = float("-inf"); best_state: dict = {}; no_improve = 0

    ckpt_valid = ckpt_path.exists() and ckpt_path.stat().st_size > 0
    if ckpt_valid:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if ckpt.get("config") == expected_config:
                rev6.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                start_epoch = int(ckpt["epoch"]) + 1
                best_f1     = float(ckpt["best_f1"])
                best_state  = ckpt["best_state"]
                no_improve  = int(ckpt["no_improve"])
                logger.info("Resumed epoch %d  best_f1=%.4f  no_improve=%d",
                            ckpt["epoch"], best_f1, no_improve)
            else:
                logger.warning("Config mismatch — fresh Rev5 init")
                ckpt_valid = False
        except Exception as exc:
            logger.warning("Ckpt load error (%s) — fresh init", exc)
            ckpt_valid = False

    if not ckpt_valid:
        rev6_st = rev6.state_dict(); n_cp = 0
        for key, val in rev5_state.items():
            if key not in rev6_st: continue
            if rev6_st[key].shape == val.shape:
                rev6_st[key].copy_(val); n_cp += 1
            elif key == "net.0.weight":
                rev6_st[key][:, :input_dim_r5].copy_(val)
                torch.nn.init.normal_(rev6_st[key][:, input_dim_r5:],
                                      std=float(val.std()) * 0.01)
                n_cp += 1
        rev6.load_state_dict(rev6_st)
        logger.info("Rev6 init from Rev5 (%d tensors)", n_cp)

    if no_improve >= args.patience and start_epoch > 1:
        logger.info("Converged — saving best model")
        if best_state:
            rev6.load_state_dict(best_state); save_model(rev6, best_path); done_path.touch()
        return
    if start_epoch > args.epochs:
        logger.info("All epochs done — saving best model")
        if best_state:
            rev6.load_state_dict(best_state); save_model(rev6, best_path); done_path.touch()
        return

    # ----------------------------------------- stratified chunk order once
    all_chunk_starts = list(range(0, n_total, CHUNK_SIZE))
    epoch            = start_epoch
    epochs_this_call = 0

    while epoch <= args.epochs and no_improve < args.patience:
        elapsed = time.time() - call_start
        if epochs_this_call > 0 and elapsed > args.max_time - 10:
            logger.info("Time budget reached after %d epoch(s) this call.", epochs_this_call)
            break

        logger.info("=== Epoch %d/%d  [elapsed=%.1fs] ===", epoch, args.epochs, elapsed)
        rng = np.random.default_rng(epoch)

        # Stratified interleaved order — guarantees all classes in any prefix
        chunk_order = stratified_chunk_order(
            all_chunk_starts, is_train, y_all, n_total, CHUNK_SIZE, rng
        )

        rev6.train()
        t_train = time.time()
        train_loss, n_batches, chunks_done = 0.0, 0, 0

        for ci in chunk_order:
            if time.time() - t_train > TRAIN_TIME_BUDGET:
                break
            start = all_chunk_starts[ci]
            end   = min(start + CHUNK_SIZE, n_total)

            chunk_X = np.ascontiguousarray(X_all[start:end], dtype=np.float32)
            mask    = is_train[start:end]
            if not mask.any():
                continue

            X_ch = chunk_X[mask]
            y_ch = y_all[start:end][mask].astype(np.int64)
            del chunk_X   # free 572 MB immediately

            perm_c = rng.permutation(len(X_ch))
            X_ch   = X_ch[perm_c];  y_ch = y_ch[perm_c]

            for i in range(0, len(X_ch), args.batch_size):
                xb = torch.from_numpy(X_ch[i : i + args.batch_size])
                yb = torch.from_numpy(y_ch[i : i + args.batch_size])
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(rev6(xb), yb)
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(rev6.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += float(loss)
                n_batches  += 1
            chunks_done += 1

        scheduler.step()
        t_train_s = time.time() - t_train

        # ---- validation (float16 in RAM → float32 per batch) ----
        rev6.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X_va_t), args.batch_size):
                xb = X_va_t[i : i + args.batch_size].float()
                preds.append(rev6(xb).argmax(dim=1).numpy())
        val_f1  = float(f1_score(y_va, np.concatenate(preds), average="macro"))
        t_total = time.time() - t_train

        logger.info(
            "Epoch %d/%d  loss=%.4f  val_f1=%.4f  best=%.4f  no_imp=%d  "
            "chunks=%d/%d  [train=%.0fs val=%.0fs total=%.0fs]",
            epoch, args.epochs,
            train_loss / max(n_batches, 1),
            val_f1, best_f1, no_improve,
            chunks_done, len(all_chunk_starts),
            t_train_s, t_total - t_train_s, t_total,
        )

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in rev6.state_dict().items()}
            no_improve = 0
            logger.info("  ** New best: val_f1=%.4f **", best_f1)
        else:
            no_improve += 1

        ckpt_data = {
            "epoch": epoch, "config": expected_config,
            "model_state": rev6.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_f1": best_f1, "best_state": best_state, "no_improve": no_improve,
        }
        ckpt_tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save(ckpt_data, ckpt_tmp)
        os.replace(ckpt_tmp, ckpt_path)
        logger.info("Checkpoint saved")

        epochs_this_call += 1;  epoch += 1

        if no_improve >= args.patience or (epoch - 1) >= args.epochs:
            logger.info("Converged — saving best model (val_f1=%.4f)", best_f1)
            if best_state:
                rev6.load_state_dict(best_state); save_model(rev6, best_path); done_path.touch()
                logger.info("Best model -> %s", best_path)
            return

    logger.info("Ran %d epoch(s). Next: %d/%d  no_improve=%d",
                epochs_this_call, epoch, args.epochs, no_improve)
    logger.info("Re-run to continue.")


if __name__ == "__main__":
    main()
