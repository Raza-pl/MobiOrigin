# MobiOrigin 0.1.3 release notes

MobiOrigin 0.1.3 prepares secure PyPI distribution without changing the frozen
classifier. The exact dev1 model bytes move from the Python wheel to a versioned
GitHub release asset. Guided setup retrieves the bundle once, verifies the
archive and every contained artifact, and publishes the local model directory
atomically.

## User-visible changes

- Python wheel and source distributions are small enough for the public PyPI
  per-file limit.
- `mobiorigin setup-databases --component models` downloads and verifies the
  exact frozen model bundle.
- `--model-archive` supports offline or institutionally mirrored installation.
- `MOBIORIGIN_MODEL_DIR` selects a custom installed-model location.
- The guided installer prepares models before marker and annotation databases.
- `mobiorigin doctor` verifies models and marker databases together.
- PyPI publication uses GitHub OIDC Trusted Publishing without stored API
  tokens and produces package attestations.

## Frozen model identities

The transport archive contains the same three `.pt` checkpoints, the same
marker normalization array, and the same model manifest distributed with
MobiOrigin 0.1.2. The bundle is published with the version 0.1.3 GitHub release.
Setup verifies both byte counts and SHA-256 identities. Static
validation also confirms that the transported artifacts reproduce byte-identical
normalization and predictions.

The classifier architecture, model tensors, sequence and marker features,
ensemble, selective threshold, labels, probabilities, and external-validation
evidence are unchanged.

## Failure behavior

An incomplete download remains resumable in the user cache. A changed archive,
unexpected archive member, changed artifact, missing model, or pre-existing
output directory stops setup. Prediction never downloads models implicitly and
never substitutes a fallback model. After setup, prediction remains offline.
