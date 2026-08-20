# MobiOrigin output schema

MobiOrigin writes a new output directory atomically. Existing output directories are rejected.

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

## Interpretation boundary

An `unclassified` result is an explicit abstention, not a chromosome call. Coverage therefore differs from one minus the technical failure rate. Downstream analyses should retain abstentions in recall and accuracy denominators while excluding them from predicted-class precision denominators, matching the frozen evaluation policy.
