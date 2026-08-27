#!/usr/bin/env python3
"""Build the deterministic transport archive for the frozen dev1 artifacts."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from mobiorigin.provenance import sha256_file

FILENAMES = (
    "seed_20260810.pt",
    "seed_20260811.pt",
    "seed_20260812.pt",
    "marker_normalization.npy",
    "model_manifest.json",
)
ARCHIVE_ROOT = "mobiorigin-models-dev1"


def build_bundle(model_dir: Path, output: Path) -> dict[str, object]:
    """Write a byte-reproducible uncompressed tar archive."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    missing = [name for name in FILENAMES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen model artifacts: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w", format=tarfile.USTAR_FORMAT) as archive:
        for filename in FILENAMES:
            source = model_dir / filename
            information = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{filename}")
            information.size = source.stat().st_size
            information.mode = 0o644
            information.mtime = 0
            information.uid = 0
            information.gid = 0
            information.uname = ""
            information.gname = ""
            with source.open("rb") as handle:
                archive.addfile(information, handle)
    return {
        "status": "PASS",
        "archive": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "members": {
            name: {
                "bytes": (model_dir / name).stat().st_size,
                "sha256": sha256_file(model_dir / name),
            }
            for name in FILENAMES
        },
    }


def main() -> int:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--model-dir", type=Path, required=True)
    arguments.add_argument("--output", type=Path, required=True)
    args = arguments.parse_args()
    print(json.dumps(build_bundle(args.model_dir, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
