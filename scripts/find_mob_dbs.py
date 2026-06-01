#!/usr/bin/env python3
"""Find and list all files in mob_suite's database directory."""
import mob_suite, pathlib, os, subprocess, sys

p = pathlib.Path(mob_suite.__file__).parent
candidates = [
    p / 'databases',
    pathlib.Path.home() / '.mob_suite',
    pathlib.Path.home() / '.local/share/mob-suite',
    pathlib.Path('/usr/local/share/mob-suite'),
    pathlib.Path('/opt/conda/share/mob-suite'),
]

db_dir = None
for d in candidates:
    if d.exists() and any(d.iterdir()):
        db_dir = d
        break

if not db_dir:
    print("ERROR: No mob_suite database directory found. Run: mob_init")
    sys.exit(1)

print(f"mob_suite DB dir: {db_dir}")
print(f"\nAll files:")
for f in sorted(db_dir.iterdir()):
    size = f.stat().st_size / 1e6
    print(f"  {f.name:<50} {size:>8.1f} MB")

# Identify key files
print("\nKey files for DIAMOND setup:")
for pattern in ['mob', 'mpf', 'rep', 'orit', 'replicon']:
    matches = [f for f in db_dir.iterdir() if pattern.lower() in f.name.lower()]
    for m in sorted(matches):
        print(f"  [{pattern}] {m.name}")
