# MobiOrigin output schema

MobiOrigin writes a new output directory atomically. Existing output directories are rejected.

For the integrated `mobiorigin run` command, a failure after prediction retains
an explicitly incomplete sibling directory named `<output-dir>.failed` (or a
numbered variant if that path exists). `ANALYSIS_FAILED.json` records the error,
completed stages, and any resumable `predictions/predictions.tsv`. This failed
directory is not a completed analysis and must not be used as one.

## `predictions.tsv`

| Column | Meaning |
|---|---|
| `sequence_id` | Unique first-token FASTA identifier, in input order. |
| `length_bp` | Input sequence length in base pairs. |
| `prediction` | `chromosome`, `plasmid`, `phage`, or `unclassified`. |
| `p_chromosome` | Ensemble chromosome probability. |
| `p_plasmid` | Ensemble plasmid probability. |
| `p_phage` | Ensemble phage probability. |
| `plasmid_score` | `p_plasmid - max(p_chromosome, p_phage)`. |
| `abstention_reason` | Empty for classified records; otherwise `low_plasmid_score` or `unsupported_length`. |

The three probabilities remain unchanged by the selective policy and sum to one. If the native argmax is plasmid but `plasmid_score` is below the frozen threshold `0.19835489988327026`, the emitted label is `unclassified`. Native chromosome and phage calls are not changed. Records outside 1,000–500,000 bp are explicitly unclassified with neutral probabilities.

## `provenance.json`

The provenance record contains the MobiOrigin version, input FASTA SHA-256, record accounting, frozen model identities, marker normalization identity, three database identities, frozen selective threshold, prediction-table identity, and network-access status.

## `SHA256SUMS.txt`

This file records the SHA-256 identities of `predictions.tsv` and `provenance.json`.

## Comprehensive annotation tables

`annotation/biological_evidence.tsv` retains one coordinate-aware row per
evidence hit. In addition to the original source-facing fields, it provides the
canonical columns `gene_symbol`, `gene_name`, `gene_family`,
`functional_class`, `functional_subclass`, and `mechanism`. Unsupported or
unreported values are written as `unknown`; they are never inferred from a
different database.

`annotation/mobiorigin_annotated_results.tsv` retains one row per input contig.
The MobiOrigin label and probabilities are copied unchanged. The following
columns summarize the normalized gene annotations on that contig:

| Column | Meaning |
|---|---|
| `annotated_gene_symbols` | Unique retained gene and marker symbols. |
| `annotated_gene_names` | Unique human-readable gene or protein names. |
| `annotated_gene_families` | Unique source-supported families. |
| `annotated_functional_classes` | Unique broad functional classes. |
| `annotated_functional_subclasses` | Unique source-supported subclasses. |
| `annotated_mechanisms` | Unique explicitly reported mechanisms. |
| `annotation_sources` | Databases contributing retained evidence. |

Multiple values use deterministic semicolon separation. The complete
source-specific evidence remains in `biological_evidence.tsv`.

## Interpretation boundary

An `unclassified` result is an explicit abstention, not a chromosome call. Coverage therefore differs from one minus the technical failure rate. Downstream analyses should retain abstentions in recall and accuracy denominators while excluding them from predicted-class precision denominators, matching the frozen evaluation policy.
