# Migrating from PlasFlow2 to MobiOrigin

MobiOrigin is a standalone chromosome/plasmid/phage classifier. It is not a drop-in replacement for every function of the broader PlasFlow2 pipeline.

## Command mapping

| Purpose | Historical interface | MobiOrigin interface |
|---|---|---|
| Core classification | `plasflow2 ...` | `mobiorigin predict --input-fasta ... --output-dir ... --database-dir ...` |
| ARG, MGE, risk, taxonomy, and HTML reporting | PlasFlow2 pipeline | Not provided by standalone MobiOrigin dev1 |
| geNomad-assisted annotation | Optional PlasFlow2 workflow | Not a MobiOrigin model feature or dependency |

## Behavioral differences

- MobiOrigin uses an immutable three-seed sequence-and-MOB-marker ensemble.
- MobiOrigin may abstain from low-margin native plasmid calls.
- MobiOrigin never transfers probability mass or applies hard biological overrides.
- MobiOrigin requires exact identity-verified user-provided MOB marker databases.
- Outputs are a prediction table, provenance JSON, and checksums rather than the full PlasFlow2 annotation/report bundle.

Historical PlasFlow2 documentation is retained in `docs/PLASFLOW2_LEGACY_README.md` for reproducibility. New classifier-focused documentation should use the MobiOrigin name.

The MobiOrigin wheel installs only the `mobiorigin` command. Users who must reproduce the historical multi-stage PlasFlow2 workflow should use its preserved source and documentation rather than expecting it as an entry point in the MobiOrigin distribution.
