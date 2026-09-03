# MobiOrigin 0.1.6 release notes

MobiOrigin 0.1.6 improves input handling, runtime diagnostics, annotation
reporting, and visualization. The frozen classifier and all scientific decision
rules remain unchanged.

## Easier and safer execution

- FASTA input can be uncompressed or gzip-compressed with `.fa.gz`,
  `.fasta.gz`, `.fna.gz`, or `.fas.gz` extensions.
- Missing-input errors report the resolved path, current working directory, and
  nearby FASTA files instead of returning a Python traceback.
- Expected command-line failures provide short actionable messages. Developers
  can enable tracebacks with `MOBIORIGIN_DEBUG=1`.
- MobiOrigin selects private writable temporary storage for external tools. On
  WSL it prefers Linux-native storage and ignores unsafe inherited temporary
  paths on `/mnt/*`.
- `mobiorigin doctor` reports temporary-storage status, available space, and
  WSL detection.
- AMRFinderPlus resource failures are retried with progressively fewer workers.
  Database, input, and other non-resource failures still fail immediately.

## Visible workflow progress

Prediction, annotation, visualization, and integrated runs now print concise,
flushed progress messages. Individual database searches are identified as they
start. These messages describe activity only and do not change result content
or ordering.

## Expanded annotation reporting

The integrated per-contig table now provides evidence-group-specific genes,
families, classes, subclasses, mechanisms, and source databases. ARG drug
classes are de-duplicated when AMRFinderPlus reports the same term as both class
and subclass. Source-specific terminology remains available in the detailed
biological-evidence table.

The A to E evidence tiers now include stable descriptive labels:

- Tier A: ARG with relaxase and mating-pair-formation markers.
- Tier B: ARG with partial mobility, replication, or MGE context.
- Tier C: ARG without detected mobility context.
- Tier D: non-ARG biological evidence only.
- Tier E: no evidence retained at the configured thresholds.

The tier definitions are unchanged. The descriptions make the existing rules
easier to interpret.

## Visualization

Annotation-enabled visualization now produces:

- evidence-tier and annotation-class summary tables;
- a priority-candidate table for Tier A to C and conjugative candidates;
- an annotation summary figure;
- a priority-candidate figure;
- separate ARG, MGE, virulence-factor, and BacMet resistance-category figures;
- counts split by chromosome, plasmid, phage, and unclassified prediction; and
- a clean searchable HTML annotation table with tier filtering.

BacMet plots use database-supported resistance categories rather than assuming
that a gene-family field is available. No missing family is inferred.

## Verification

- Black, Ruff, and mypy checks passed.
- The production suite passed 77 tests with 84.28% coverage.
- The W1 operational subset completed for 2,445 contigs.
- W1 retained one macrolide ARG and four mercury-resistance genes under the
  frozen evidence rules.
- Cross-platform W1 identifiers, lengths, prediction labels, and abstention
  states matched exactly. Probability differences remained below `1e-6`.

The W1 analysis is an operational reproduction check. It is not an accuracy or
prevalence estimate.

## Scientific boundary

Version 0.1.6 does not change the model checkpoints, sequence features, marker
features, normalization, ensemble weights, plasmid threshold, supported length
range, annotation thresholds, database payloads, origin labels, ARG consensus
rules, or evidence-tier rules. Reporting additions do not establish phenotype,
transferability, pathogenicity, clinical risk, or taxonomic identity.
