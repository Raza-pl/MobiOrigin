#!/usr/bin/env python3
"""Minimal PyTorch diagnostic — run this to identify where the crash occurs.

Also validates the BLAS-thread-race fix for macOS ARM segfaults.
"""
import sys, os

# ── Fix for macOS ARM BLAS thread-race segfault ──────────────────────────────
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

print("Python :", sys.version)

import torch
print("PyTorch:", torch.__version__)
print("NumPy  :", np.__version__)

# ── Test 1: simple tensor ────────────────────────────────────────────────────
print("\nTest 1: torch.randn(100, 10) ...", end=" ", flush=True)
x = torch.randn(100, 10)
print("OK")

# ── Test 2: model with BatchNorm (same arch as PlasFlowMLP) ─────────────────
print("Test 2: PlasFlowMLP-like model (input=1281) ...", end=" ", flush=True)
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(1281, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
    nn.Linear(1024,  512), nn.BatchNorm1d( 512), nn.ReLU(),
    nn.Linear( 512,  256), nn.BatchNorm1d( 256), nn.ReLU(),
    nn.Linear( 256,    4),
)
print("OK")

# ── Test 3: forward on 512 samples (one batch) ───────────────────────────────
print("Test 3: forward pass — 512 × 1281 ...", end=" ", flush=True)
xb = torch.from_numpy(np.random.rand(512, 1281).astype("float32"))
out = model(xb)
print("OK", out.shape)

# ── Test 4: backward ──────────────────────────────────────────────────────────
print("Test 4: backward pass ...", end=" ", flush=True)
out.sum().backward()
print("OK")

# ── Test 5: from_numpy on 40k rows (validation tensor) ───────────────────────
print("Test 5: from_numpy — 40k × 1281 (0.21 GB) ...", end=" ", flush=True)
X_va = np.random.rand(40_000, 1281).astype("float32")
X_v  = torch.from_numpy(X_va)
print("OK", X_v.shape)

# ── Test 6: DataLoader with MmapDataset ─────────────────────────────────────
print("Test 6: DataLoader + MmapDataset (1k samples) ...", end=" ", flush=True)
import tempfile, os
from torch.utils.data import DataLoader, Dataset

class TinyMmap(Dataset):
    def __init__(self, path, n):
        self.X = np.load(path, mmap_mode="r")
        self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i):
        return torch.tensor(self.X[i].copy(), dtype=torch.float32), torch.tensor(0)

with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
    np.save(f.name, np.random.rand(1000, 1281).astype("float32"))
    tmp = f.name

ds     = TinyMmap(tmp, 1000)
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
xb2, _ = next(iter(loader))
print("OK", xb2.shape)
os.unlink(tmp)

# ── Test 7: one training step ────────────────────────────────────────────────
print("Test 7: one full optimiser step ...", end=" ", flush=True)
opt  = torch.optim.AdamW(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
model.train()
opt.zero_grad()
loss = crit(model(xb), torch.randint(0, 4, (512,)))
loss.backward()
opt.step()
print("OK  loss=%.4f" % loss.item())

print("\n✅  ALL TESTS PASSED — PyTorch is working correctly on this machine.")
print("The segfault is environment-specific (conda env, library conflict, etc.)")
print("Try: conda install pytorch -c pytorch --force-reinstall")
