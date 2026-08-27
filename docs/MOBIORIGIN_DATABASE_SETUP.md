# MobiOrigin model and marker-database setup

This page covers the frozen model and marker resources used by prediction. The same CLI
also installs the separate downstream annotation resources with
`mobiorigin setup-databases --component annotation`; see
[`MOBIORIGIN_ANNOTATION.md`](MOBIORIGIN_ANNOTATION.md).

MobiOrigin dev1 uses three MOB-suite-derived protein-marker databases: replication (`rep`), relaxase/mobilization (`mob`), and mating-pair formation (`mpf`). These biological records are not bundled with MobiOrigin. Prediction is offline and fails closed unless all three DIAMOND databases exactly match the frozen research identities.

The three unchanged neural-network checkpoints and normalization artifact are
distributed as one versioned GitHub release asset instead of being duplicated
inside every wheel and source archive. Setup requires the archive SHA-256
`10a3e599eae31a72a4d09a4a58685666058f88d6995fbb9ed450e965a6a513cf`
and then verifies the existing per-file identities in `model_manifest.json`.
The destination is published atomically only after all checks pass.

```bash
mobiorigin setup-databases --component models
mobiorigin setup-databases --component models --check
```

The default directory is
`${XDG_DATA_HOME:-$HOME/.local/share}/mobiorigin/models/dev1`. Override it with
`MOBIORIGIN_MODEL_DIR`. An explicit offline archive can be installed with:

```bash
mobiorigin setup-databases \
  --component models \
  --model-archive /path/to/mobiorigin-models-dev1.tar
```

Model download is a setup operation. Prediction itself never accesses the
network and will stop if a required model is absent or changed.

## Provenance and redistribution boundary

- Official MOB-suite source: <https://github.com/phac-nml/mob-suite>
- Audited MOB-suite database archive record: <https://doi.org/10.5281/zenodo.10304948>
- The MOB-suite repository is Apache-2.0 licensed.
- The audited public database record did not expose an explicit license covering redistribution of every biological sequence record. MobiOrigin therefore requires user-side retrieval/preparation and does not bundle the database payloads.

This is a conservative scientific distribution policy, not legal advice.

## Required frozen identities

| Family | Required SHA-256 |
|---|---|
| `rep` | `a70b79237026f1aece9ef70d59fbc37d6f1607d2a0ae53555a2c1dd55c54fbc0` |
| `mob` | `176f3e8be3aab01ddae74f73be8f19ef4f5e419e59bc0299bff54571351aad10` |
| `mpf` | `da7a65ac9fdb8edc80b5fdebf5b0878d97cbeebcf2ede0a7332c2af605192e37` |

The hashes apply to the final `.dmnd` files consumed by MobiOrigin, not merely their source FASTA files. A database made from a different source release or DIAMOND build may not reproduce these byte identities.

## Recommended isolated, identity-verified setup

MobiOrigin and MOB-suite must not be installed into the same Conda environment. Their supported NumPy ranges do not overlap. From a MobiOrigin source checkout, use the guided helper:

```bash
bash scripts/setup_mobiorigin_databases.sh "$HOME/mobiorigin_databases"
```

The helper first retrieves and verifies the model bundle. It then creates a
dedicated `mobiorigin-db` retrieval environment from
`environment.mob-database.yml`, runs MOB-suite 3.1.8's official `mob_init` route
there, and locates the current `mob_suite/databases` raw files. A second small
`mobiorigin-marker-build` helper from `environment.marker-build.yml` contains
only Python and pinned DIAMOND 2.0.15 (database build 153). It verifies the raw
identities, applies the frozen six-frame replication-protein translation, and
reconstructs the three final indexes without changing MOB-suite's older Qt/ICU
dependency stack. MobiOrigin then verifies every frozen SHA-256 identity, writes
the manifest and third-party notice, and atomically publishes the output
directory. It fails without leaving a partial published output if any source or
rebuilt identity differs. Existing valid downloads and helper environments are
reused rather than overwritten or updated in place.

The helper exports `PYTHONNOUSERSITE=1` and clears `PYTHONPATH`/`PYTHONHOME` so packages from `~/.local` cannot contaminate the isolated builder. MOB-suite 3.1.8 names the raw files `rep.dna.fas`, `mob.proteins.faa`, and `mpf.proteins.faa`; `mob_init` does not itself create MobiOrigin's three `.dmnd` filenames.

On Apple Silicon, the helper creates only the database-builder environment as `osx-64` under Rosetta because the compatible historical BLAST build is not available natively for arm64. The MobiOrigin prediction environment stays native arm64. The helper checks for Rosetta and fails with an explicit instruction if it is unavailable.

Verify DIAMOND and all three published database files before prediction:

```bash
mobiorigin setup-databases \
  --check \
  --output-dir "$HOME/mobiorigin_databases"
```

The check prints the DIAMOND version and the three required SHA-256 identities. Missing executables, malformed manifests, missing databases, or changed payload bytes fail closed.

## Manual two-environment route

```bash
mamba env create -f environment.yml
mamba env create -f environment.mob-database.yml
mamba env create -f environment.marker-build.yml

conda activate mobiorigin-db
mob_init
MOB_DATA_DIR="$(python -c 'import mob_suite, pathlib; print(pathlib.Path(mob_suite.__file__).resolve().parent / "databases")')"

conda activate mobiorigin-marker-build
python src/mobiorigin/marker_database_builder.py \
  --raw-dir "$MOB_DATA_DIR" \
  --output-dir /tmp/mobiorigin_frozen_marker_build \
  --diamond diamond

conda activate mobiorigin
mobiorigin setup-databases --component models
mobiorigin setup-databases \
  --source-dir /tmp/mobiorigin_frozen_marker_build \
  --output-dir "$HOME/mobiorigin_databases"
mobiorigin setup-databases \
  --check \
  --output-dir "$HOME/mobiorigin_databases"
```

For Apple Silicon, create the second environment with `mamba env create --platform osx-64 -f environment.mob-database.yml` instead.

The `mobiorigin` environment contains NumPy 1.26.4 and CPU-only PyTorch. The `mobiorigin-db` environment contains MOB-suite's compatible NumPy, pandas, BLAST, and SQLite ranges and is used only for retrieval. The small `mobiorigin-marker-build` environment contains DIAMOND 2.0.15 for deterministic database construction. Neither helper environment is used for prediction. The runtime environment can use a newer DIAMOND to search these database-format-3 indexes.

Do not repair a failed combined environment by downgrading NumPy, forcing pandas, installing the obsolete `mob-suite` PyPI project, or mixing historical channels. Recreate the two isolated environments instead.

For an offline or institutionally mirrored installation, place the three exact `.dmnd` files in one source directory and run the same MobiOrigin command with that directory as `--source-dir`.

```bash
mobiorigin setup-databases \
  --source-dir /path/to/exact_downloaded_assets \
  --output-dir mobiorigin_mob_databases
```

The offline route applies the same identities and produces the same inference-facing filenames and manifest schema.

## Manifest

Place the three databases in one directory and create `mobiorigin_mob_suite_database_manifest.json`:

```json
{
  "schema_version": "mobiorigin-mob-suite-database-manifest-v1",
  "databases": {
    "rep": {
      "path": "rep_proteins.dmnd",
      "sha256": "a70b79237026f1aece9ef70d59fbc37d6f1607d2a0ae53555a2c1dd55c54fbc0"
    },
    "mob": {
      "path": "mob_proteins.dmnd",
      "sha256": "176f3e8be3aab01ddae74f73be8f19ef4f5e419e59bc0299bff54571351aad10"
    },
    "mpf": {
      "path": "mpf_proteins.dmnd",
      "sha256": "da7a65ac9fdb8edc80b5fdebf5b0878d97cbeebcf2ede0a7332c2af605192e37"
    }
  }
}
```

MobiOrigin independently hashes every referenced file before inference. Missing files, schema changes, incorrect paths, or hash mismatches stop prediction. There is no zero-evidence fallback.

## Reproducibility boundary

The setup command reproduces the exact byte identities used for development and prospective validation; it does not silently rebuild different databases from mutable upstream inputs. The biological database records remain third-party data retrieved for local use and are not included in MobiOrigin wheels or source distributions.
