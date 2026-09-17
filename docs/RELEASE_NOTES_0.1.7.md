# MobiOrigin 0.1.7 release notes

MobiOrigin 0.1.7 corrects two input-level reliability problems without
retraining or replacing the frozen three-member model ensemble.

## Orientation-invariant inference

The nucleotide-composition representation was already reverse-complement
invariant, but one coding-strand feature was orientation dependent. Version
0.1.7 evaluates the frozen ensemble for both analytical coding orientations
and averages their probability vectors. The locked 3,000-record audit produced
zero label changes after reverse complementation under this policy.

## Ambiguity-safe reporting

Earlier versions could implicitly treat non-ACGT symbols as adenine during
sequence-signature extraction. Version 0.1.7 excludes invalid k-mer windows and
fails closed at the record level: any sequence containing an accepted non-ACGT
IUPAC symbol is
reported as `unclassified` with `abstention_reason=ambiguous_bases`. Counts,
fractions and warning codes remain in the prediction table and provenance.

## Compatibility

The model checkpoints, normalization arrays, MOB-suite-derived marker
databases and plasmid-margin threshold are unchanged. Prediction semantics have
changed, so results generated with version 0.1.7 should not be described as
byte-identical to version 0.1.6. Re-run prediction when comparing releases.
