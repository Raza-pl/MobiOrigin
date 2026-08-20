# MobiOrigin marker-database setup

MobiOrigin dev1 uses three MOB-suite-derived protein-marker databases: replication (`rep`), relaxase/mobilization (`mob`), and mating-pair formation (`mpf`). These biological records are not bundled with MobiOrigin. Prediction is offline and fails closed unless all three DIAMOND databases exactly match the frozen research identities.

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

## Automated identity-verified setup

After installing MobiOrigin, run:

```bash
mobiorigin setup-databases --output-dir mobiorigin_mob_databases
```

The command downloads the three database files from the versioned PlasFlow 2.0.0 release assets, verifies the frozen SHA-256 identity of every file, writes the manifest and third-party notice, and atomically publishes the output directory. It fails without leaving a partial output if a download is interrupted or any identity differs. Existing output directories are never overwritten.

For an offline or institutionally mirrored installation, first place the three exact `.dmnd` files in one source directory and run:

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
