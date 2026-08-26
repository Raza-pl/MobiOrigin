# MobiOrigin biological annotation workflow

`mobiorigin annotate` is a downstream biological-evidence workflow. It is
deliberately separate from `mobiorigin predict`: annotation never replaces,
overrides, or recalibrates chromosome, plasmid, phage, or unclassified
predictions. When a matching prediction table is supplied, values are joined
only after exact FASTA identifier, order, and length validation.

## Evidence profiles

The `arg` profile calls translation-table-11 ORFs with Pyrodigal and evaluates
them against three independent resources:

- **CARD** protein homolog models, screened with DIAMOND at at least 80% amino-
  acid identity and 80% query coverage. ARO metadata supplies the gene name,
  family, drug class, and resistance mechanism.
- **SARG**, screened independently with the same 80%/80% rule.
- **AMRFinderPlus**, run through the official protein workflow with
  `--plus --print_node`. Only `Type=AMR` rows enter ARG consensus.

The `comprehensive` profile additionally retains:

- official AMRFinderPlus `VIRULENCE` and `STRESS` rows, including metal,
  biocide, heat, and acid-stress subtypes;
- **VFDB core (set A)** homologs. The core dataset contains representative
  experimentally verified virulence factors, but a homolog alone does not
  establish pathogenic phenotype;
- curated insertion-sequence/mobile-element homologs and metadata;
- **BacMet2 experimental** biocide and metal-resistance homologs;
- **MOB-suite** replication, relaxase, and mating-pair-formation markers.

The [official AMRFinderPlus project](https://github.com/ncbi/amr) describes AMR,
stress, biocide, and virulence detection. The
[official VFDB download page](https://www.mgc.ac.cn/VFs/download.htm) distinguishes
its experimentally verified core dataset from its broader full dataset.

## One-command local installation

MobiOrigin installs a complete, authorized annotation resource directory with:

```bash
mobiorigin setup-databases \
  --component annotation \
  --source-dir /path/to/authorized_annotation_sources \
  --amrfinder-database /path/to/amrfinderplus/data/version \
  --profile comprehensive \
  --accept-third-party-terms
```

The destination defaults to
`${XDG_DATA_HOME:-$HOME/.local/share}/mobiorigin/annotation_databases`, or the
path in `MOBIORIGIN_ANNOTATION_DATABASE_DIR`. Run `amrfinder --update` first
and pass its resolved dated database directory (the directory containing
`version.txt`) to `--amrfinder-database`. MobiOrigin installs that complete
official database as `annotation_databases/amrfinderplus/`, so subsequent
annotation commands need no AMRFinderPlus database argument. Installation is atomic: every
required file is copied independently, its size and SHA-256 identity are
recorded, and an incomplete or changed source leaves no published destination.
The command also writes a third-party notice and links to the upstream terms.

Verify an installed resource set without copying or parsing biological records:

```bash
mobiorigin setup-databases \
  --component annotation \
  --profile comprehensive \
  --check
```

The source directory supplied to the installer must have the following layout.

## Required source layout

Biological database payloads are not bundled with MobiOrigin. Prepare this
layout from appropriately licensed official or institutional copies:

```text
annotation_databases/
├── card/
│   ├── card.dmnd
│   └── aro_index.tsv
├── sarg/
│   └── sarg.dmnd
├── amrfinder/
│   ├── amrprot.dmnd
│   └── fam.tsv
├── vfdb/
│   ├── vfdb_setA.dmnd
│   └── vfdb_indx.txt
├── mge/
│   ├── isfinder.dmnd
│   └── mge_database.tsv
├── bacmet/
│   ├── bacmet.dmnd
│   └── Bacmet_list.tsv
└── mob_suite/
    ├── rep_proteins.dmnd
    ├── mob_proteins.dmnd
    └── mpf_proteins.dmnd
```

The `amrfinder/` subdirectory is needed only for the explicitly supplemental
AMRProt DIAMOND route. Official mode instead requires a complete official
AMRFinderPlus database directory supplied through `--amrfinder-database`.
MobiOrigin hashes every consumed database and executable into provenance.

Third-party database payloads are never bundled. Users must obtain them under
the terms that apply to their intended use. VFDB describes its data as
CC BY-NC 4.0 for non-commercial academic/research use and asks commercial users
to contact VFDB. AMRFinderPlus software and database are U.S. Government works.

## ARG-only workflow

```bash
mobiorigin annotate \
  --input-fasta assembly.fasta \
  --output-dir mobiorigin_arg_annotation \
  --profile arg \
  --amrfinder-mode official \
  --amrfinder-bin amrfinder \
  --threads 8
```

## Comprehensive publication workflow

Run MobiOrigin prediction first, then join its unchanged table to annotation:

```bash
mobiorigin annotate \
  --input-fasta assembly.fasta \
  --output-dir mobiorigin_comprehensive_annotation \
  --profile comprehensive \
  --predictions-tsv mobiorigin_predictions/predictions.tsv \
  --amrfinder-mode official \
  --amrfinder-bin amrfinder \
  --threads 8
```

The output directory must be new. Publication is atomic: failed ORF calling,
database search, parsing, alignment, or hashing leaves no partial published
result.

Comprehensive homology thresholds are frozen in provenance: VFDB core 60%
identity/80% query coverage; curated MGE 70%/80%; BacMet2 80%/80%; and
MOB-suite markers 50%/70%. These calls are homology evidence, not phenotype
confirmation.

## Explicit supplemental AMRProt route

When the official executable and complete database are unavailable, an
existing AMRProt DIAMOND database can be screened with
`--amrfinder-mode amrprot`. It is always labeled `AMRPROT_DIAMOND` and
`official_amrfinderplus_executed=false`. It is not presented as an official
AMRFinderPlus analysis, and official VIRULENCE/STRESS rows are unavailable.

## Outputs

Both profiles write:

- `arg_hits.tsv`: all source-specific ARG evidence;
- `arg_consensus.tsv`: one deterministic ARG call per ORF, using CARD,
  official AMRFinderPlus, supplemental AMRProt, then SARG priority;
- `annotation_summary.tsv`: ordered per-input ARG counts and summaries;
- `predicted_proteins.faa`: deterministic ORFs used for all searches;
- `raw_evidence/`: unchanged search reports;
- `annotation_provenance.json`: input, executable, database, threshold, route,
  output, and prediction-independence identities;
- `SHA256SUMS.txt`: checksums for every published artifact except itself.

The comprehensive profile also writes:

- `biological_evidence.tsv`: normalized, coordinate-aware evidence from every
  source while retaining disagreements;
- `mobiorigin_annotated_results.tsv`: one publication-facing row per input
  sequence, with unchanged MobiOrigin values when supplied;
- `publication_summary.json`: machine-readable aggregate accounting;
- `mobiorigin_report.html`: a self-contained report and the first 100 records
  in deterministic priority order;
- raw VFDB, MGE, BacMet2, and MOB-suite reports.

Empty numeric cells mean that the official source did not report that value;
they do not mean zero.

## Evidence-priority tiers and risk boundary

MobiOrigin deliberately does not emit an unvalidated clinical 0–10 score.
Instead, it emits an auditable research-priority tier:

| Tier | Frozen rule | Interpretation |
|---|---|---|
| A | ARG plus relaxase and mating-pair-formation evidence | Highest dissemination-review priority |
| B | ARG plus partial mobility, replication, or MGE evidence | Elevated dissemination-review priority |
| C | ARG without detected mobility context | Resistance evidence requiring contextual review |
| D | Non-ARG biological evidence only | Biological context, not ARG risk |
| E | No retained evidence | No evidence under the frozen searches |

Virulence co-localization, source context, host taxonomy, and antibiotic class
remain separate columns; they do not silently add points. ARG health-risk
frameworks emphasize human association, mobility, and host pathogenicity, but
an assembled contig alone generally cannot establish all three. Tier A is not
equivalent to clinical danger, in-vivo transmissibility, expression, or
phenotypic resistance. All annotations require appropriate expert and clinical
interpretation and are not diagnoses.
