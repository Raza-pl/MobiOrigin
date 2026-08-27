# MobiOrigin 0.1.4

MobiOrigin 0.1.4 corrects a comprehensive-annotation compatibility defect
observed on Linux with a rare mobileOG-db DIAMOND result header. The frozen
classifier and its scientific policy are unchanged.

## What changed

- mobileOG identifiers are recovered from the DIAMOND subject field or from a
  supported identifier embedded in the title field.
- Rows that still cannot be resolved are excluded from biological evidence and
  retained verbatim in `annotation_warnings.tsv`.
- New annotation database installations audit all mobileOG FASTA headers and
  record the parser identity and compatibility counts in the frozen manifest.
- Integrated runs retain completed predictions in `<output-dir>.failed` when a
  later annotation stage fails. `ANALYSIS_FAILED.json` identifies the completed
  stages and resumable prediction table.
- The guided installer now performs a bundled comprehensive verification across
  prediction, annotation, visualization, and standard database routes.
- Annotation provenance advances to schema version 3 and records the count and
  policy for excluded mobileOG rows.
- The development dependency group now includes the package build and Twine
  metadata-check tools used by the release gate.

## Compatibility

Existing version 1 or version 2 annotation database manifests remain accepted.
They use the new runtime fail-closed parser policy. Fresh installations create a
version 3 manifest with a source-header audit. No annotation database download
is required solely to use the corrected runtime parser.

## Scientific boundary

This release changes error handling and evidence parsing only. It does not
change model checkpoints, sequence or marker features, normalization, ensemble
weights, the selective threshold, prediction labels, or validation results.
Unresolved external identifiers never contribute inferred biological evidence.
