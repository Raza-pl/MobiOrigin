"""Pure-Python PlasFlow v1 runner — no R required.

Bypasses the rpy2/R dependency by replicating PlasFlow v1's k-mer feature
computation in numpy, then loading the pre-trained TensorFlow model from the
installed plasflow package.

Requirements (already installed in plasflow1 conda env):
    tensorflow==1.15.0
    numpy

Does NOT require R, rpy2, or Biostrings.

Usage:
    conda activate plasflow1
    python scripts/run_plasflow1_nopy.py \\
        --input  data/test/W1.contigs.fa.gz \\
        --output results/W1_plasflow1.tsv \\
        --threshold 0.7 \\
        --min-length 1000

Output format matches PlasFlow v1:
    contig_id  label  prob_chromosome  prob_plasmid  prob_phage  length
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# K-mer feature computation (replicates R Biostrings oligonucleotideFrequency)
# ---------------------------------------------------------------------------

BASES = "ACGT"
_BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}

# Vectorised base encoding (same approach as features.py)
_ASCII_TO_BASE = np.zeros(256, dtype=np.uint8)
for _ch, _val in [("A",0),("C",1),("G",2),("T",3),("a",0),("c",1),("g",2),("t",3)]:
    _ASCII_TO_BASE[ord(_ch)] = _val

_POWERS = {k: (4 ** np.arange(k - 1, -1, -1, dtype=np.int64)) for k in range(1, 8)}


def _kmer_freq(seq: str, k: int, normalize: bool = False) -> np.ndarray:
    """Compute k-mer count vector for one sequence.

    PlasFlow v1 uses RAW COUNTS (not normalized frequencies).
    R's oligonucleotideFrequency default is as.prob=FALSE (raw counts).
    The TF model was trained on raw integer k-mer counts — normalizing
    to frequencies causes completely wrong output (99.93% plasmid).

    normalize=False (default): raw counts (what PlasFlow v1 expects)
    normalize=True:  divide by total (NOT what PlasFlow v1 expects)
    """
    vocab_size = 4 ** k
    seq_upper = seq.upper()
    if len(seq_upper) < k:
        return np.zeros(vocab_size, dtype=np.float32)

    encoded = _ASCII_TO_BASE[np.frombuffer(seq_upper.encode("ascii"), dtype=np.uint8)]
    windows = np.lib.stride_tricks.sliding_window_view(encoded, k).astype(np.int64)
    kmer_ids = windows @ _POWERS[k]
    counts = np.bincount(kmer_ids, minlength=vocab_size).astype(np.float32)

    if normalize:
        total = counts.sum()
        if total > 0:
            counts /= total
    return counts


def extract_plasflow1_features(seq: str, k_range: range = range(1, 8)) -> np.ndarray:
    """Compute the full PlasFlow v1 feature vector (k=1 through k=7).

    Uses RAW COUNTS per k (not normalized) — matching R's
    oligonucleotideFrequency(..., as.prob=FALSE) default.

    Feature vector layout:
      k=1: 4 dims (indices 0–3)
      k=2: 16 dims (indices 4–19)
      k=3: 64 dims (indices 20–83)
      k=4: 256 dims (indices 84–339)
      k=5: 1024 dims (indices 340–1363)
      k=6: 4096 dims (indices 1364–5459)
      k=7: 16384 dims (indices 5460–21843)
    Total: 21844 features
    """
    parts = [_kmer_freq(seq, k, normalize=False) for k in k_range]
    return np.concatenate(parts)


FEATURE_DIM_V1 = sum(4**k for k in range(1, 8))  # 21844


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def find_plasflow_model() -> Path | None:
    """Locate the PlasFlow v1 TensorFlow model checkpoint."""
    # Standard locations to check
    candidates = []

    if "CONDA_PREFIX" in os.environ:
        prefix = Path(os.environ["CONDA_PREFIX"])
        # Where plasflow.py script lives — models dir is sibling of the script
        candidates.append(prefix / "bin" / "models")
        # Site-packages fallbacks
        candidates.append(prefix / "lib" / "python3.7" / "site-packages" / "plasflow" / "models")

    # GitHub clone locations
    candidates += [
        Path("/tmp/PlasFlow/models"),
        Path.home() / "PlasFlow" / "models",
    ]

    # 2. Current Python's site-packages
    import site
    for sp in site.getsitepackages():
        candidates.append(Path(sp) / "plasflow" / "models")

    # 3. Relative to script location
    candidates.append(Path(__file__).parent.parent / "data" / "models" / "plasflow1")

    for candidate in candidates:
        if candidate.exists():
            # Look for model checkpoint files
            tfa = list(candidate.glob("*.tfa"))
            ckpt = list(candidate.glob("*.index")) + list(candidate.glob("*.ckpt*"))
            if tfa or ckpt:
                logger.info("Found PlasFlow v1 model at: %s", candidate)
                return candidate

    return None


def _rewrite_checkpoint_file(model_dir: Path) -> str | None:
    """Rewrite the 'checkpoint' text file to use the current absolute path.

    TF1 checkpoints embed the original training-machine path in the
    'checkpoint' metadata file.  TF2's validator rejects relative paths
    *and* stale absolute paths, so we patch it to the real current location
    before loading.  Returns the absolute checkpoint prefix, or None.
    """
    import re
    ckpt_text = model_dir / "checkpoint"
    meta_files = sorted(model_dir.glob("*.meta"),
                        key=lambda f: int(f.stem.split("-")[-1]) if f.stem.split("-")[-1].isdigit() else 0)
    if not meta_files:
        return None

    # Use the LAST checkpoint (highest training step = best model)
    stem = meta_files[-1].name[:-5]         # e.g. "model.ckpt-50000"
    logger.info("Using checkpoint: %s (of %d available)", stem, len(meta_files))
    abs_prefix = str((model_dir / stem).resolve())

    # Patch (or create) the checkpoint metadata file so TF2 can find it
    new_content = (
        f'model_checkpoint_path: "{abs_prefix}"\n'
        f'all_model_checkpoint_paths: "{abs_prefix}"\n'
    )
    ckpt_text.write_text(new_content)
    logger.info("Patched checkpoint file → %s", abs_prefix)
    return abs_prefix


def _numpy_forward(features: np.ndarray, weights: dict) -> np.ndarray:
    """Pure-numpy MLP forward pass for PlasFlow v1 (relu + softmax).

    Handles two naming conventions:
      TF-Slim DNN (kmer7 model): hiddenlayer_N/weights, dnn_logits/weights
      TF default:                Variable, Variable_1, Variable_2, ...
    """
    x = features.astype(np.float32)

    # ── PlasFlow v1 TF-Slim naming (kmer7_split_20_20_neurons_relu model) ──
    if "hiddenlayer_0/weights" in weights:
        # Input → hidden layers (relu)
        layer = 0
        while f"hiddenlayer_{layer}/weights" in weights:
            W = weights[f"hiddenlayer_{layer}/weights"].astype(np.float32)
            b = weights[f"hiddenlayer_{layer}/biases"].astype(np.float32)
            pre = x @ W + b
            x = np.maximum(0.0, pre)
            logger.info("  layer hiddenlayer_%d: in=%s W=%s out=%s  "
                        "pre_act min=%.3f max=%.3f mean=%.3f  "
                        "W min=%.4f max=%.4f mean=%.4f  "
                        "b min=%.3f max=%.3f  "
                        "relu_nonzero=%.1f%%",
                        layer, pre.shape[0] if pre.ndim > 0 else 1, W.shape, x.shape,
                        pre.min(), pre.max(), pre.mean(),
                        W.min(), W.max(), W.mean(),
                        b.min(), b.max(),
                        float((x[0] > 0).mean() * 100) if x.ndim > 1 else float(x > 0) * 100)
            layer += 1
        # Output logits (no relu)
        W = weights["dnn_logits/weights"].astype(np.float32)
        b = weights["dnn_logits/biases"].astype(np.float32)
        logits = x @ W + b
        logger.info("  layer dnn_logits: %s → %s  W min=%.4f max=%.4f",
                    W.shape[0], W.shape[1], W.min(), W.max())
        # Centered bias (optional additive bias term from TF-Slim DNN estimator)
        if "centered_bias_weight" in weights:
            logits += weights["centered_bias_weight"].astype(np.float32)
        # Diagnostic: log bias values and sample logits
        cb = weights["centered_bias_weight"].astype(np.float32) if "centered_bias_weight" in weights else np.zeros(logits.shape[1])
        logger.info("dnn_logits/biases + centered_bias (chr classes 0-17, plas classes 18-27):")
        bias_total = weights["dnn_logits/biases"].astype(np.float32) + cb
        logger.info("  chr  bias mean=%.3f  [%s]", bias_total[:18].mean(),
                    " ".join(f"{v:.2f}" for v in bias_total[:18]))
        logger.info("  plas bias mean=%.3f  [%s]", bias_total[18:].mean(),
                    " ".join(f"{v:.2f}" for v in bias_total[18:]))
        logger.info("Sample logits[0]: chr_sum=%.3f plas_sum=%.3f",
                    logits[0, :18].sum(), logits[0, 18:].sum())

        logits -= logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        probs_out = ex / ex.sum(axis=1, keepdims=True)
        logger.info("Sample probs[0]: chr_sum=%.4f plas_sum=%.4f",
                    probs_out[0, :18].sum(), probs_out[0, 18:].sum())
        return probs_out

    # ── TF default Variable_N naming ─────────────────────────────────────────
    Ws, bs = [], []
    idx = 0
    while True:
        wname = "Variable" if idx == 0 else f"Variable_{idx}"
        bname = f"Variable_{idx + 1}"
        if wname not in weights:
            break
        Ws.append(weights[wname].astype(np.float32))
        bs.append(weights[bname].astype(np.float32))
        idx += 2

    if not Ws:
        logger.warning("No recognised weight naming; available keys: %s",
                        sorted(weights.keys())[:20])
        return None

    logger.info("MLP layers: %s → (input dim %d)",
                " → ".join(str(W.shape[1]) for W in Ws), Ws[0].shape[0])
    for i, (W, b) in enumerate(zip(Ws, bs)):
        x = x @ W + b
        if i < len(Ws) - 1:
            x = np.maximum(0.0, x)
    x -= x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def load_and_predict(features: np.ndarray, model_dir: Path, threshold: float = 0.7) -> np.ndarray:
    """Load PlasFlow v1 TF model and return probabilities [chromosome, plasmid, phage].

    Strategy:
    1. Patch the 'checkpoint' metadata file to the current absolute path
       (TF1 checkpoints embed the original training-machine path).
    2. Try Session-based restore (tf.compat.v1) — works on genuine TF2 installs.
    3. If that fails, load raw weight tensors with tf.train.load_checkpoint
       and run a pure-numpy forward pass (no TF graph execution needed).
    """
    import tensorflow as tf

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
        tf1 = tf.compat.v1
        tf1.disable_eager_execution()
    else:
        tf1 = tf

    # Patch checkpoint metadata and get absolute prefix
    abs_prefix = _rewrite_checkpoint_file(model_dir)
    if abs_prefix is None:
        logger.error("No .meta checkpoint found in %s", model_dir)
        return None

    meta_path = abs_prefix + ".meta"
    logger.info("Loading PlasFlow v1 model from %s …", model_dir)

    # ── Attempt 1: Session-based restore (TF1 graph mode) ────────────────
    try:
        with tf1.Session() as sess:
            saver = tf1.train.import_meta_graph(meta_path)
            saver.restore(sess, abs_prefix)

            graph = tf1.get_default_graph()
            x_ph = graph.get_tensor_by_name("Placeholder:0")
            y_op = graph.get_tensor_by_name("Softmax:0")

            batch_size = 512
            probs = []
            for start in range(0, len(features), batch_size):
                prob = sess.run(y_op, feed_dict={x_ph: features[start:start+batch_size]})
                probs.append(prob)
            logger.info("Loaded via TF Session")
            return np.vstack(probs)

    except Exception as e:
        logger.warning("Session restore failed (%s) — falling back to numpy inference", e)

    # ── Attempt 2: Low-level C++ checkpoint reader (bypasses TF2 Python validation) ──
    try:
        from tensorflow.python.training.py_checkpoint_reader import NewCheckpointReader
        reader = NewCheckpointReader(abs_prefix)
        var_map = reader.get_variable_to_shape_map()

        skip = ("Adam", "global_step", "beta1_power", "beta2_power", "ExponentialMovingAverage")
        keep = {k for k in var_map if not any(s in k for s in skip)}
        logger.info("Checkpoint variables (model weights):")
        for k in sorted(keep):
            logger.info("  %-40s  shape=%s", k, var_map[k])

        weights = {k: reader.get_tensor(k) for k in keep}
        logger.info("Loaded %d weight tensors via py_checkpoint_reader", len(weights))

        probs = _numpy_forward(features, weights)
        if probs is not None:
            logger.info("Inference completed via numpy forward pass")
            return probs

    except Exception as e:
        logger.warning("py_checkpoint_reader failed (%s) — trying V1 shard reader", e)

    # ── Attempt 3: SSTable + proto parsing for TF V1 shard files ─────────────
    import struct
    import traceback

    stem = Path(abs_prefix).name
    shard_files = sorted(model_dir.glob(f"{stem}-*-of-*"))
    logger.info("V1 shard files found: %s", [f.name for f in shard_files])

    if not shard_files:
        logger.error("No V1 shard files for %s in %s", stem, model_dir)
        return None

    shard_path = shard_files[0]
    raw_file = shard_path.read_bytes()
    file_size = len(raw_file)
    logger.info("Shard size: %d bytes", file_size)
    logger.info("First 20 bytes (hex): %s", raw_file[:20].hex())
    logger.info("Last   8 bytes (hex): %s", raw_file[-8:].hex())

    SSTABLE_MAGIC = b'\x57\xfb\x80\x8b\x24\x75\x47\xdb'
    is_sstable = (raw_file[-8:] == SSTABLE_MAGIC)
    logger.info("SSTable magic match: %s", is_sstable)

    try:
        from tensorflow.core.util import saved_tensor_slice_pb2
        from tensorflow.core.framework import types_pb2

        dtype_map = {
            types_pb2.DT_FLOAT:  (np.float32, "float_val"),
            types_pb2.DT_DOUBLE: (np.float64, "double_val"),
            types_pb2.DT_INT32:  (np.int32,   "int_val"),
            types_pb2.DT_INT64:  (np.int64,   "int64_val"),
        }

        weights: dict[str, np.ndarray] = {}

        if is_sstable:
            weights = _read_sstable(raw_file, saved_tensor_slice_pb2, dtype_map)
            logger.info("SSTable reader returned %d tensors", len(weights))
        else:
            # Try parsing as raw SavedTensorSlices proto at different offsets
            for skip in (0, 2, 4, 8):
                try:
                    sts = saved_tensor_slice_pb2.SavedTensorSlices()
                    sts.ParseFromString(raw_file[skip:])
                    logger.info("offset=%d: parsed OK, meta.tensor count=%d", skip, len(sts.meta.tensor))
                    if sts.HasField("data") and sts.data.name:
                        t = sts.data.data
                        info = dtype_map.get(t.dtype)
                        if info:
                            np_dtype, rep_field = info
                            shape = [d.size for d in t.tensor_shape.dim]
                            arr = (np.frombuffer(t.tensor_content, dtype=np_dtype).copy()
                                   if t.tensor_content else
                                   np.array(getattr(t, rep_field), dtype=np_dtype))
                            weights[sts.data.name] = arr.reshape(shape) if shape else arr
                    break
                except Exception as ex:
                    logger.info("offset=%d: proto parse failed: %s", skip, ex)

        skip_keys = ("Adagrad", "global_step", "Adam", "beta1_power", "beta2_power")
        weights = {k: v for k, v in weights.items() if not any(s in k for s in skip_keys)}
        logger.info("Model weight tensors after filtering: %d", len(weights))
        for k, v in sorted(weights.items()):
            logger.info("  %-40s  shape=%s", k, list(v.shape))

        if weights:
            # kmer7 model: input is k=7 only (16384 dims = 4^7)
            # Our feature matrix is k=1-7 (21844 dims); k=7 starts at index 5460
            feat = features
            if "hiddenlayer_0/weights" in weights:
                w0 = weights["hiddenlayer_0/weights"]
                if w0.shape[0] == 16384 and features.shape[1] != 16384:
                    K7_START = sum(4**k for k in range(1, 7))  # 5460
                    feat = features[:, K7_START:]
                    logger.info("kmer7 model: sliced features to k=7 only (%d dims)", feat.shape[1])
            probs = _numpy_forward(feat, weights)
            if probs is not None:
                logger.info("Inference completed via V1 shard + numpy forward pass")
                return probs

    except Exception:
        logger.error("V1 shard reader failed:\n%s", traceback.format_exc())

    return None


def _read_sstable(raw: bytes, saved_tensor_slice_pb2, dtype_map: dict) -> dict:
    """Parse a TF1 SSTable checkpoint shard into {var_name: np.ndarray}.

    TF SSTable format (tensorflow/core/lib/io/table/):
      Footer (last 48 bytes): MetaIndex handle | Index handle | padding | 8-byte magic
      Index block: one entry per data block → (last_key, BlockHandle)
      Data blocks: delta-encoded (shared_prefix, key_delta, value) triples
        key   = variable name bytes (empty string "" for metadata entry)
        value = serialized SavedTensorSlices proto
              (TensorSliceWriter.Add() wraps each variable in SavedTensorSlices)
    """
    import struct

    FOOTER_SIZE = 48
    file_size = len(raw)

    def read_varint(buf, pos):
        result, shift = 0, 0
        while True:
            b = buf[pos]; pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result, pos
            shift += 7

    def read_block_handle(buf, pos):
        offset, pos = read_varint(buf, pos)
        size,   pos = read_varint(buf, pos)
        return offset, size, pos

    def parse_block(block_data, label=""):
        """Decode an SSTable data/index block → [(key_bytes, value_bytes)]."""
        blen = len(block_data)
        if blen < 4:
            logger.warning("parse_block(%s): block too small (%d bytes)", label, blen)
            return []
        n_restarts = struct.unpack_from("<I", block_data, blen - 4)[0]
        restarts_end = blen - 4 - 4 * n_restarts
        if restarts_end < 0 or n_restarts > 1_000_000:
            logger.warning("parse_block(%s): bad n_restarts=%d, blen=%d", label, n_restarts, blen)
            return []
        logger.info("parse_block(%s): blen=%d n_restarts=%d restarts_end=%d",
                    label, blen, n_restarts, restarts_end)
        entries, pos, last_key = [], 0, b""
        while pos < restarts_end:
            shared,     pos = read_varint(block_data, pos)
            non_shared, pos = read_varint(block_data, pos)
            value_len,  pos = read_varint(block_data, pos)
            key_delta = bytes(block_data[pos: pos + non_shared]); pos += non_shared
            value     = bytes(block_data[pos: pos + value_len]);  pos += value_len
            last_key  = last_key[:shared] + key_delta
            entries.append((bytes(last_key), value))
        return entries

    # ── Footer → MetaIndex + Index block handles ──────────────────────────────
    footer = raw[file_size - FOOTER_SIZE:]
    logger.info("Footer (hex): %s", footer.hex())
    pos = 0
    mi_off, mi_sz, pos = read_block_handle(footer, pos)
    idx_off, idx_sz, _ = read_block_handle(footer, pos)
    logger.info("MetaIndex block: offset=%d size=%d", mi_off, mi_sz)
    logger.info("Index     block: offset=%d size=%d", idx_off, idx_sz)

    # ── Index block → list of (last_key, BlockHandle) for data blocks ─────────
    idx_block = raw[idx_off: idx_off + idx_sz]
    index_entries = parse_block(idx_block, "index")
    logger.info("Index block entries: %d", len(index_entries))

    # ── Pass 1: collect all (key, val_bytes) from every data block ───────────
    all_entries: list[tuple[bytes, bytes]] = []
    for eidx, (_last_key, handle_bytes) in enumerate(index_entries):
        db_off, db_sz, _ = read_block_handle(handle_bytes, 0)
        logger.info("  data block %d: offset=%d size=%d", eidx, db_off, db_sz)
        block_raw = raw[db_off: db_off + db_sz]
        all_entries.extend(parse_block(block_raw, f"data{eidx}"))

    # ── Pass 2: parse metadata entry (key="") to get per-variable dtypes ────
    # dtype IS in SavedTensorSliceMeta.tensor[i].type, NOT in TensorProto.dtype
    var_meta: dict[str, tuple] = {}   # name → (np_dtype, shape)
    for var_key, val_bytes in all_entries:
        if var_key != b"":
            continue
        try:
            sts_meta = saved_tensor_slice_pb2.SavedTensorSlices()
            sts_meta.ParseFromString(val_bytes)
            for tm in sts_meta.meta.tensor:
                info = dtype_map.get(tm.type)
                np_dtype = info[0] if info else np.float32
                shape = [d.size for d in tm.shape.dim]
                var_meta[tm.name] = (np_dtype, shape)
                logger.info("  meta var: %-35s dtype=%s shape=%s",
                            tm.name, np_dtype.__name__, shape)
        except Exception as ex:
            logger.warning("  metadata parse failed: %s", ex)

    # ── Pass 3: parse data entries using dtype from metadata ─────────────────
    # SSTable key format (old TF1): b"\x00{name}\x00{binary_slice_spec}"
    # The clean variable name is ALSO in SavedTensorSlices.data.name — use that.
    weights: dict[str, np.ndarray] = {}
    for var_key, val_bytes in all_entries:
        if var_key == b"":
            continue  # metadata — already handled

        try:
            sts = saved_tensor_slice_pb2.SavedTensorSlices()
            sts.ParseFromString(val_bytes)
        except Exception as ex:
            logger.warning("  parse failed for key %r: %s", var_key[:20], ex)
            continue

        if not sts.HasField("data"):
            continue

        ss = sts.data    # SavedSlice
        t  = ss.data     # TensorProto — dtype field is often 0 in old TF1 format
        name = ss.name   # clean variable name from proto (no null bytes or slice spec)
        if not name:
            continue

        # dtype and shape from metadata (authoritative; TensorProto.dtype is often 0)
        meta_info = var_meta.get(name)
        if meta_info:
            np_dtype, shape = meta_info
        else:
            info = dtype_map.get(t.dtype)
            if not info:
                logger.warning("  no dtype for %s (t.dtype=%d)", name, t.dtype)
                continue
            np_dtype = info[0]
            shape = [d.size for d in t.tensor_shape.dim]

        if t.tensor_content:
            arr = np.frombuffer(t.tensor_content, dtype=np_dtype).copy()
        else:
            rep_field = {np.float32: "float_val", np.float64: "double_val",
                         np.int32: "int_val", np.int64: "int64_val"}.get(np_dtype, "float_val")
            arr = np.array(getattr(t, rep_field), dtype=np_dtype)

        try:
            weights[name] = arr.reshape(shape) if shape else arr
            logger.info("  loaded: %-35s dtype=%-8s shape=%s",
                        name, np_dtype.__name__, list(weights[name].shape))
        except ValueError as ex:
            logger.warning("  reshape failed for %s: arr.size=%d shape=%s (%s)",
                           name, arr.size, shape, ex)

    return weights


# ---------------------------------------------------------------------------
# FASTA loading
# ---------------------------------------------------------------------------

def load_fasta(path: Path, min_length: int = 1000) -> list[tuple[str, str]]:
    """Load FASTA (plain or .gz) → [(id, seq), ...]."""
    opener = gzip.open if str(path).endswith(".gz") else open
    records: list[tuple[str, str]] = []
    curr_id = curr_seq = None
    with opener(str(path), "rt") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if curr_id and len(curr_seq) >= min_length:
                    records.append((curr_id, curr_seq))
                curr_id = line[1:].split()[0]
                curr_seq = ""
            elif curr_id:
                curr_seq += line
    if curr_id and curr_seq and len(curr_seq) >= min_length:
        records.append((curr_id, curr_seq))
    logger.info("Loaded %d sequences (min_length=%d) from %s",
                len(records), min_length, Path(path).name)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure-Python PlasFlow v1 runner (no R required)"
    )
    parser.add_argument("--input",      type=Path, required=True,
                        help="Input FASTA (plain or .gz)")
    parser.add_argument("--output",     type=Path, required=True,
                        help="Output TSV file")
    parser.add_argument("--threshold",  type=float, default=0.7,
                        help="Minimum probability to call plasmid/chromosome (default: 0.7)")
    parser.add_argument("--min-length", type=int,   default=1000,
                        help="Minimum contig length in bp (default: 1000)")
    parser.add_argument("--model-dir",  type=Path, default=None,
                        help="Path to PlasFlow v1 model directory (auto-detected if omitted)")
    parser.add_argument("--batch-size", type=int,   default=512)
    args = parser.parse_args()

    # Locate model
    model_dir = args.model_dir or find_plasflow_model()
    if model_dir is None:
        logger.error(
            "PlasFlow v1 model not found. Ensure plasflow is pip-installed:\n"
            "  conda activate plasflow1\n"
            "  pip install plasflow\n"
            "Or specify --model-dir /path/to/plasflow/models/"
        )
        sys.exit(1)

    # Load sequences
    records = load_fasta(args.input, min_length=args.min_length)
    if not records:
        logger.error("No sequences loaded from %s", args.input)
        sys.exit(1)

    # Compute features (raw counts, not normalized — PlasFlow v1 was trained this way)
    logger.info("Computing k=1–7 k-mer RAW COUNT features for %d sequences …", len(records))
    features_list = []
    for i, (sid, seq) in enumerate(records):
        features_list.append(extract_plasflow1_features(seq))
        if (i + 1) % 10000 == 0:
            logger.info("  %d / %d sequences processed", i + 1, len(records))
    features = np.array(features_list, dtype=np.float32)
    logger.info("Feature matrix: %s", features.shape)

    # Run model
    probs = load_and_predict(features, model_dir, threshold=args.threshold)
    if probs is None:
        logger.error("Prediction failed — cannot load TF model")
        sys.exit(1)

    # Aggregate multi-class probabilities into chromosome / plasmid / phage.
    # PlasFlow v1 kmer7 model outputs 28 classes:
    #   0-17  = chromosome.* (18 taxonomic classes)
    #   18-27 = plasmid.*    (10 taxonomic classes)
    # No phage class in this model.
    n_classes = probs.shape[1]
    if n_classes == 28:
        prob_chr_agg  = probs[:, :18].sum(axis=1)   # classes 0-17
        prob_plas_agg = probs[:, 18:].sum(axis=1)   # classes 18-27
        prob_phage_agg = np.zeros(len(records))
        agg_probs = np.stack([prob_chr_agg, prob_plas_agg, prob_phage_agg], axis=1)
        logger.info("Aggregated 28-class output → 3-class (chr/plas/phage)")
    elif n_classes == 3:
        agg_probs = probs
    else:
        # Generic: chromosome = first half, plasmid = second half
        mid = n_classes // 2
        agg_probs = np.stack([probs[:, :mid].sum(1), probs[:, mid:].sum(1),
                               np.zeros(len(records))], axis=1)
        logger.warning("Unexpected n_classes=%d; using generic aggregation", n_classes)

    # Write output (PlasFlow v1 format)
    # Columns: id  label  prob_chromosome  prob_plasmid  prob_phage  length
    args.output.parent.mkdir(parents=True, exist_ok=True)
    label_map = {0: "chromosome", 1: "plasmid", 2: "phage"}

    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["id", "label", "prob_chromosome", "prob_plasmid", "prob_phage", "length"])

        n_plasmid = n_chr = n_phage = n_unc = 0
        for i, (sid, seq) in enumerate(records):
            prob_chr  = float(agg_probs[i][0])
            prob_plas = float(agg_probs[i][1])
            prob_phage = float(agg_probs[i][2])
            best_idx  = int(np.argmax(agg_probs[i]))
            best_prob = float(agg_probs[i][best_idx])
            if best_prob >= args.threshold:
                label = label_map[best_idx]
            else:
                label = "unclassified"

            if label == "plasmid":      n_plasmid += 1
            elif label == "chromosome": n_chr += 1
            elif label == "phage":      n_phage += 1
            else:                       n_unc += 1

            writer.writerow([sid, label,
                              f"{prob_chr:.4f}", f"{prob_plas:.4f}", f"{prob_phage:.4f}",
                              len(seq)])

    n_total = len(records)
    logger.info("\n=== PlasFlow v1 (Python) results ===")
    logger.info("  Total          : %d", n_total)
    logger.info("  Plasmid        : %d  (%.2f%%)", n_plasmid, 100*n_plasmid/n_total)
    logger.info("  Chromosome     : %d  (%.2f%%)", n_chr, 100*n_chr/n_total)
    logger.info("  Phage          : %d  (%.2f%%)", n_phage, 100*n_phage/n_total)
    logger.info("  Unclassified   : %d  (%.2f%%)", n_unc, 100*n_unc/n_total)
    logger.info("  Output         → %s", args.output)


if __name__ == "__main__":
    main()
