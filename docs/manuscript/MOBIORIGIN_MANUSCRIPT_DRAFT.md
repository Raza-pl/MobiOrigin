# MobiOrigin: sequence-and-marker fusion for bacterial replicon classification

Shahbaz Raza

Correspondence: shahbaz.invincible3182@gmail.com
Affiliation: **author confirmation required before submission**

## Abstract

Correctly distinguishing chromosomal, plasmid, and phage-derived sequence fragments is central to studying horizontal gene transfer, microbial ecology, and the dissemination of accessory traits. Existing classifiers differ in their reliance on nucleotide composition, gene-marker evidence, and reference collections, creating length- and class-dependent trade-offs. We developed MobiOrigin, a CPU-oriented classifier that combines 9,557 sequence features with 17 protein-marker features derived from replication, relaxase/mobilization, and mating-pair-formation evidence. Three independently initialized neural networks are combined by an equal-weight softmax mean, followed by a prospectively frozen abstention rule for low-margin native plasmid calls. The qualified candidate was frozen before external evaluation. In a source-disjoint locked development test, MobiOrigin achieved macro-F1 0.8467, balanced accuracy 0.8403, plasmid precision 0.8648, plasmid sensitivity 0.7550, and coverage 0.9833. We then conducted a prospective external comparison on 3,000 fragments from 3,000 distinct versioned accessions, balanced across chromosome, plasmid, phage, and five length bins. Identical class-hidden FASTA bytes were provided to MobiOrigin and geNomad 1.12.0/database 1.9, and both prediction inventories were frozen before label release. MobiOrigin exceeded geNomad on both preregistered co-primary endpoints: three-class macro-F1 (0.7889 versus 0.7574; paired difference 0.0315, 95% CI 0.0135–0.0497; Holm-adjusted *P*=0.00120) and plasmid binary F1 (0.7453 versus 0.6876; difference 0.0578, 95% CI 0.0300–0.0852; Holm-adjusted *P*=0.000400). This improvement reflected higher plasmid sensitivity (0.739 versus 0.559) but lower plasmid precision (0.752 versus 0.893) and modestly lower coverage (0.974 versus 0.999). MobiOrigin provides a reproducible sequence-and-marker classifier whose advantages and trade-offs are explicitly bounded by prospective evaluation.

## Introduction

Plasmids and bacteriophages are major vehicles of horizontal gene transfer, but assembled microbial sequences often lack an immediately observable replicon identity. Classification is especially difficult for short fragments, divergent elements, and integrated or compositionally host-adapted sequences. Alignment-free nucleotide models can generalize beyond known reference entries, whereas gene-marker approaches provide interpretable biological evidence when recognizable proteins are present.

PlasFlow introduced a neural genome-signature approach for plasmid prediction from metagenomic fragments [1]. geNomad subsequently combined sequence and extensive gene-marker evidence for virus and plasmid identification [2]. These methods motivate a compact fusion strategy that retains an alignment-free sequence backbone while incorporating a narrowly scoped, locally executable marker panel. MobiOrigin was developed as a standalone successor classifier rather than as a relabeling of the broader PlasFlow2 annotation workflow.

Our objectives were to: (i) build a balanced, source-cluster-disjoint development corpus with exact and high-similarity firewalls; (ii) prospectively freeze a compact CPU-trainable model and abstention policy; (iii) qualify the final candidate without retrospective locked-test tuning; and (iv) compare the frozen candidate with geNomad on a newly assembled, label-sealed external cohort. We report both the co-primary comparisons and the precision–sensitivity–coverage trade-offs required to interpret them.

## Methods

### Development dataset and firewalls

Candidate chromosome, plasmid, and phage sources were resolved through official metadata and local provenance routes under a prospectively frozen charter. Confirmatory records and labels were sealed before feature analysis. Exact forward/reverse-complement matches and cross-split relationships meeting at least 90% nucleotide identity and 80% symmetric union coverage were excluded at the source-cluster level. Deterministic, same-class, same-split, and same-length-bin replacement rules were frozen before screening results were opened.

The final development dataset contained 66,000 fragments: 22,000 per class, with 60,000 training, 3,000 calibration, and 3,000 locked-test records. Each class contributed 20,000 training and 1,000 records to each held-out split. The five length bins were 1–<2 kb, 2–<5 kb, 5–<10 kb, 10–<50 kb, and 50–500 kb. Source clusters did not cross splits.

### Sequence and marker features

Each fragment was represented by a frozen 9,557-dimensional sequence vector containing normalized nucleotide-composition and canonical *k*-mer features. Protein-coding sequences were predicted in metagenomic mode with Pyrodigal, restricted to translation table 11 and deterministic IUPAC ambiguity masking [3,4]. Predicted proteins were searched with DIAMOND against three MOB-suite-derived protein panels representing replication, relaxase/mobilization, and mating-pair formation [5,6]. Family-specific hit summaries and ORF-composition summaries formed a 17-dimensional marker vector. Marker features were standardized using training rows only and concatenated with the sequence vector. geNomad output, known-plasmid containment, hard class overrides, and probability-mass transfer were prohibited.

### Model training and candidate freeze

The release model was a multilayer perceptron with input dimension 9,574, hidden dimensions 1,024, 256, and 64, GELU activations, batch normalization in the first two layers, and dropout rates 0.30, 0.25, and 0.20. Fresh models were trained for seeds 20260810, 20260811, and 20260812 using AdamW, cross-entropy with 0.05 label smoothing, learning rate 0.001, weight decay 0.0001, batch size 512, cosine annealing, gradient clipping at 1.0, and calibration-only early stopping. The fixed candidate ensemble was the arithmetic mean of the three softmax vectors.

Native three-class argmax labels were retained except for plasmid calls whose score, defined as `p_plasmid - max(p_chromosome, p_phage)`, fell below the automatically selected calibration threshold 0.19835489988327026. Those calls were emitted as unclassified without modifying probabilities or reassigning them to another class. The calibration constraints were plasmid precision at least 0.86, plasmid sensitivity at least 0.75, and coverage at least 0.90. The resulting candidate was frozen before locked-test marker extraction, prediction, and one-time evaluation.

### Locked development-test evaluation

The locked test contained 3,000 fragments from 1,364 source clusters. Uncertainty used 10,000 source-cluster bootstrap replicates. The frozen candidate passed all six prospective minimum targets, including three-class macro-F1, balanced accuracy, coverage, plasmid precision, plasmid sensitivity, and 1–2 kb balanced accuracy. Record-level error mining and further calibration or locked-test tuning were prohibited after evaluation.

### Prospective external cohort

The external cohort was assembled only after the development candidate was frozen. Official NCBI metadata routes were used to form chromosome, plasmid, and phage candidate pools. Exact matches and relationships meeting the frozen 90% identity and 80% symmetric-union-coverage thresholds against protected development/confirmatory material were excluded. Deterministic supplemental rounds resolved screening deficits without changing eligibility thresholds.

The final cohort comprised 3,000 records from 3,000 distinct versioned source accessions, with 1,000 records per class and 200 records per class and length bin. Sequence identifiers were opaque and labels were stored in a permission-restricted sealed map. MobiOrigin and geNomad 1.12.0/database 1.9 received the same class-hidden FASTA bytes in the same order. Both prediction tables and their complete artifact inventories were cryptographically frozen before the label map was authorized for release.

### Endpoints and statistical analysis

The co-primary endpoints were three-class macro-F1 and plasmid-versus-non-plasmid F1. Differences were MobiOrigin minus geNomad. Two-sided percentile 95% intervals and paired empirical *P* values used 10,000 bootstrap replicates over `source_accession` with seed 20260818. Holm adjustment controlled multiplicity across the two co-primary endpoints. Balanced accuracy, precision, sensitivity, coverage, and length-bin analyses were descriptive.

## Results

### Locked-test qualification

On the locked development test, MobiOrigin achieved three-class macro-F1 0.8467 (95% CI 0.8283–0.8643), balanced accuracy 0.8403, MCC 0.7646, and coverage 0.9833 (95% CI 0.9787–0.9877). Plasmid precision was 0.8648 (95% CI 0.8315–0.8955), sensitivity was 0.7550 (95% CI 0.7171–0.7913), and binary F1 was 0.8062. Balanced accuracy in the 1–2 kb group was 0.7367. All prospectively frozen minimum targets were met.

### External comparison

MobiOrigin achieved three-class macro-F1 0.7889 and balanced accuracy 0.7797, compared with 0.7574 and 0.7570 for geNomad. The paired macro-F1 difference was 0.0315 (95% CI 0.0135–0.0497; Holm-adjusted *P*=0.00120), supporting superiority on the first co-primary endpoint.

For plasmid-versus-non-plasmid classification, MobiOrigin achieved F1 0.7453 compared with 0.6876 for geNomad. The paired difference was 0.0578 (95% CI 0.0300–0.0852; Holm-adjusted *P*=0.000400), supporting superiority on the second co-primary endpoint.

The tools showed a clear descriptive trade-off. MobiOrigin plasmid sensitivity was 0.739, 0.180 higher than geNomad, whereas its precision was 0.752, 0.141 lower. MobiOrigin coverage was 0.974 versus 0.999 for geNomad because the frozen selective policy abstained on low-margin plasmid calls. Length-stratified results were heterogeneous: MobiOrigin showed its largest macro-F1 advantage in the 1–<2 kb group, while geNomad was stronger in some intermediate and long-fragment groups. These subgroup results were not co-primary inferential tests.

## Discussion

MobiOrigin supported both preregistered co-primary superiority claims on a prospective, class-balanced external cohort. The design combines alignment-free sequence information with a compact marker panel and avoids using comparator output as a feature or teacher. The improvement in plasmid F1 was driven primarily by sensitivity, including a large descriptive advantage on the shortest fragments.

The results do not establish universal superiority. geNomad retained substantially higher plasmid precision and nearly complete coverage, and it provides broader annotation functions outside the scope of MobiOrigin. Users prioritizing conservative plasmid calls may prefer geNomad or require independent confirmation. MobiOrigin is best interpreted as a replicon-origin classifier with an explicit abstention policy, not as a substitute for complete mobile-element annotation.

Several limitations remain. The external cohort was deliberately balanced rather than prevalence-matched to a particular environment, so predictive values in operational samples will depend on class prevalence. All external sources were drawn through official public sequence routes and do not capture every geographic, ecological, or taxonomic setting. The marker databases are identity-frozen third-party research resources retrieved locally rather than bundled. Finally, the external cohort is permanently closed to retrospective tuning; further improvement requires a newly designed prospective study.

## Availability

Source code, frozen model artifacts, user documentation, and aggregate publication artifacts are maintained in the project repository. The versioned marker-database setup command retrieves and verifies the exact local-use database identities; biological database files are not redistributed in the Python package. Aggregate external tables and an editable vector figure are under `docs/manuscript/mobiorigin_external_validation`. Protected record-level development and evaluation payloads are not publication artifacts.

## Declarations requiring author confirmation

- Affiliation and postal address.
- Funding statement.
- Competing-interests statement.
- Author-contribution taxonomy.
- Data-access wording required by the selected journal.
- Target journal formatting and word limits.

## References

1. Krawczyk PS, Lipinski L, Dziembowski A. PlasFlow: predicting plasmid sequences in metagenomic data using genome signatures. *Nucleic Acids Research*. 2018;46:e35. doi:10.1093/nar/gky044.
2. Camargo AP, Roux S, Schulz F, et al. Identification of mobile genetic elements with geNomad. *Nature Biotechnology*. 2024;42:1303–1312. doi:10.1038/s41587-023-01953-y.
3. Larralde M. Pyrodigal: Python bindings and interface to Prodigal, an efficient method for gene prediction in prokaryotes. *Journal of Open Source Software*. 2022;7:4296. doi:10.21105/joss.04296.
4. Hyatt D, Chen G-L, LoCascio PF, Land ML, Larimer FW, Hauser LJ. Prodigal: prokaryotic gene recognition and translation initiation site identification. *BMC Bioinformatics*. 2010;11:119. doi:10.1186/1471-2105-11-119.
5. Robertson J, Nash JHE. MOB-suite: software tools for clustering, reconstruction and typing of plasmids from draft assemblies. *Microbial Genomics*. 2018;4:e000206. doi:10.1099/mgen.0.000206.
6. Buchfink B, Reuter K, Drost H-G. Sensitive protein alignments at tree-of-life scale using DIAMOND. *Nature Methods*. 2021;18:366–368. doi:10.1038/s41592-021-01101-x.
