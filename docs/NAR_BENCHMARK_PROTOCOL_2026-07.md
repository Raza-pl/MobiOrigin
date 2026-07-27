# Corrected NAR benchmark protocol

## Frozen PlasFlow2 configuration

- Git commit: `655e315346eec734e4f22f96240c84dff997b4e6`
- Software version: `2.1.2`
- Model ID: `plasflow-rev5-general-production-20260724`
- Model SHA-256: `3564913c14eaad068a217572f9641907fa8c86b1527abbb2f855b0df4bf3cb23`
- Manifest SHA-256: `27c2a329b96a00847bd6d793700735819d3b5e3f6e99a98ba1d2092f6f5405d0`
- Primary profile: `balanced`
- Primary threshold policy: `rev5-balanced-p080-20260724-v1`
- Secondary profile: `evidence-assisted`
- Evidence threshold policy: `rev5-balanced-p080-compass-k21-s5m-20260724-v1`
- COMPASS SHA-256: `06aba56318345b7f38e38aa529389af20665e27b39788db730ea2ee7cb00834f`
- Marker-XGBoost fusion: disabled
- Biological hallmark gate: disabled

The stale installed distribution metadata value `2.0.0a0` is not
an acceptable version identifier. All publication runs must use an isolated
wheel built from the frozen Git commit.

## Benchmark cohorts

### Primary natural-prevalence cohort

This cohort preserves the eligible source population. It supports estimates
of false-positive burden and deployment-relevant precision.

### Secondary balanced diagnostic cohort

This cohort balances plasmid and chromosome examples within predefined length
strata. It supports discrimination and sensitivity comparisons but its raw
precision must not be described as real-world positive predictive value.

### Phage cohort

Phage positives must be combined with independently frozen plasmid and
chromosome negatives. Positive-only phage sets may report sensitivity only.

### Real-world case study

W1 is unlabeled. PlasFlow2/geNomad agreement and biological annotations are
reported as concordance or supporting evidence, never as accuracy.

## Unit of biological independence

- Biological sources or assemblies are the bootstrap unit.
- Windows from one replicon are not independent samples.
- Replicons and assemblies must not cross development and evaluation roles.
- Window contribution per replicon must be capped to prevent large replicons
  from dominating contig-level statistics.
- Source-level macro metrics must accompany pooled window-level metrics.

## Leakage screening

The untouched benchmark must be screened against:

1. all training source identifiers;
2. all calibration and development source identifiers;
3. exact sequence duplicates;
4. near-duplicate sequence similarity;
5. production reference databases used by post-filters.

Known and novel strata should be reported separately. Records are not removed
merely because a comparator database recognizes them unless exclusion was
predeclared. This avoids selectively disadvantaging database-based methods.

## Comparator execution

Each comparator must use:

- an isolated environment;
- a frozen released software version;
- frozen database versions;
- the identical FASTA input;
- documented default or recommended production settings;
- fixed thread and resource accounting;
- no tuning on primary confirmatory labels.

Raw output, standard output, standard error, exit status, wall time and peak
memory must be retained.

## Primary scoring

For plasmid versus non-plasmid:

- true plasmid called plasmid: TP;
- true non-plasmid called plasmid: FP;
- true plasmid unclassified or called another class: FN;
- true non-plasmid not called plasmid: TN.

Tool failures and filtered sequences remain in the denominator as
unclassified. Classified-only scores are secondary.

## Statistics

- Precision, recall, specificity, F1, MCC and balanced accuracy.
- Unclassified fraction and prediction coverage.
- AUROC/AUPRC only when continuous scores are semantically comparable.
- Calibration metrics only for probabilistic outputs.
- At least 2,000 paired bootstrap replicates at biological-source level.
- Ninety-five percent confidence intervals for metrics and paired differences.
- Length, taxon and novelty-stratified results.
- Multiple-testing correction for families of secondary comparisons.
- Wall time, peak RSS and throughput on identical hardware.

## Freeze condition

No model, threshold, profile, post-filter, scoring adapter or inclusion rule
may change after confirmatory inputs and the protocol are committed. Any later
change creates a new development cycle and requires a new untouched benchmark.

---

# Confirmatory cohort design

## Prospective bacterial accrual

Include all current complete bacterial RefSeq assemblies released after
2026-07-24. Stop on the first UTC census date containing at least:

- 100 eligible assemblies;
- 50 plasmid-bearing assemblies;
- 100 plasmid replicons;
- 100 chromosome replicons.

If the target is not met, repeat the same query weekly and extend only
forward in time. The lower release boundary must never be moved backward.

## Prospective phage accrual

Acquire complete RefSeq bacteriophage genomes released after the same
production freeze. At least 100 independent phage sources are required for
primary phage confirmation. Previously inspected phage feature rows remain
secondary evidence only.

## Truth assignment

Bacterial truth derives from NCBI sequence-report molecule-location labels.
Only explicit chromosome and plasmid replicons enter primary scoring.
Ambiguous records remain visible in the exclusion audit.

## Cohorts

1. Full-replicon cohort: every eligible labeled replicon.
2. Source-population fragment cohort: deterministic capped windows from every
   eligible replicon.
3. Balanced diagnostic cohort: class-balanced within each fixed length.
4. Phage three-class cohort: independently frozen phage, plasmid and
   chromosome examples.
5. W1-standardized analysis: W1 supplies length weights but no truth labels.

No synthetic fragment cohort will be described as real-world natural
prevalence. Precision will be tied to its declared sampling design.
Prevalence-standardized PPV and NPV will be reported at plasmid prevalences
of 1%, 5%, 10%, 20% and 50%.

## Fragment generation

Use seed 20260727 and fixed lengths of 1, 2, 5, 10 and 20 kb. Each replicon
contributes at most five deterministic, evenly spaced, non-overlapping windows
per length. Biological source remains the bootstrap unit.

## Leakage

Exact identifiers, exact sequences and >=99% identity with >=90% query
coverage to training data are excluded from primary confirmation and retained
in an audit appendix. Close relatives remain but form a predefined stratum.

Matches to tool reference databases are not selectively excluded. They define
reference-known and reference-novel strata, avoiding unfair removal of cases
recognized by database-oriented methods.

## Blindness

Before predictions, freeze and hash:

- source accessions;
- truth labels;
- all exclusion reasons;
- full-replicon FASTA;
- fragment FASTA files;
- window manifests;
- scoring adapters;
- model/profile contracts;
- comparator versions and databases.

Inspecting predictions before this freeze invalidates the cohort as primary
confirmatory evidence.
