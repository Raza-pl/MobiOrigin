# Output Files

`plasflow2 run` writes all outputs to the directory you specify with `--output`.

## File overview

| File | Description |
|---|---|
| `all_predictions.tsv` | Every contig: label, scores, ARGs, mobility, risk, taxonomy |
| `annotated_predictions.tsv` | Filtered: only contigs with ARGs, MGEs, VFs, mobility, or pathogen hits |
| `plasmids.fasta` | Classified plasmid sequences |
| `chromosome.fasta` | Classified chromosome sequences |
| `phage.fasta` | Classified phage sequences |
| `archaea.fasta` | Classified archaea sequences |
| `genes.tsv` | Gene-level table: all ORFs with ARG/VF/MGE flags and coordinates |
| `annotations.json` | Full ARG + mobility + risk evidence per plasmid (machine-readable JSON) |
| `report_plasmid.html` | Interactive plasmid report — open in browser |
| `report_chromosome.html` | Chromosome contig report |
| `report_phage.html` | Phage contig report |
| `report_archaea.html` | Archaea contig report |
| `report_unclassified.html` | Unclassified contig report |

---

## `all_predictions.tsv` — column reference

### Classification columns (all contigs)

| Column | Type | Description |
|---|---|---|
| `contig_id` | string | Sequence ID from the input FASTA |
| `length` | int | Contig length in base pairs |
| `label` | string | `plasmid` / `chromosome` / `phage` / `archaea` / `unclassified` |
| `confidence` | float | Final classification confidence (0–1) |
| `plasmid_score` | float | MLP plasmid probability (0–1) |
| `chromosome_score` | float | MLP chromosome probability (0–1) |
| `phage_score` | float | MLP phage probability (0–1) |
| `archaea_score` | float | MLP archaea probability (0–1) |
| `low_confidence` | bool | `True` if confidence < 0.70 — treat these calls with caution |

### XGBoost evidence columns (populated when stage-2 model was used)

| Column | Type | Description |
|---|---|---|
| `mlp_plasmid` | float | Raw MLP plasmid score before XGBoost blending |
| `mlp_chromosome` | float | Raw MLP chromosome score |
| `mlp_phage` | float | Raw MLP phage score |
| `xgb_plasmid` | float | XGBoost stage-2 plasmid score |
| `xgb_chromosome` | float | XGBoost stage-2 chromosome score |
| `is_conjugative` | 0/1 | Conjugation proteins detected by MOB-suite |
| `is_mobilizable` | 0/1 | Mobilization proteins detected |
| `has_replicon` | 0/1 | Replicon type identified |
| `has_ice` | 0/1 | Integrative conjugative element detected |
| `has_rep_protein` | 0/1 | Rep protein detected |
| `n_rep_per_kb` | float | Rep protein density per kb |
| `evidence_type` | string | What drove the call: `mlp_only` / `xgb_blend` / `conjugative_override` / `hallmark_boost` |

### Taxonomy columns (all contigs)

| Column | Type | Description |
|---|---|---|
| `taxonomy` | string | Predicted host organism (e.g. `Klebsiella pneumoniae`) |
| `taxonomy_rank` | string | Taxonomic rank of prediction (`species` / `genus` / `family` / etc.) |
| `taxonomy_lineage` | string | Full GTDB lineage string |

### ARG annotation columns (all contigs)

| Column | Type | Description |
|---|---|---|
| `num_args` | int | Number of ARGs detected |
| `arg_genes` | string | ARG names, semicolon-separated (e.g. `blaNDM-1; sul1; tetA`) |
| `drug_classes` | string | Drug classes, semicolon-separated (e.g. `carbapenem; sulfonamide`) |
| `arg_sources` | string | Database sources (CARD, SARG, AMRFinderPlus) |

### Virulence factor columns (all contigs)

| Column | Type | Description |
|---|---|---|
| `num_vf` | int | Number of virulence factors detected |
| `vf_genes` | string | VF gene names, semicolon-separated |

### MGE columns (all contigs)

| Column | Type | Description |
|---|---|---|
| `num_mge` | int | Number of mobile genetic elements detected |
| `mge_genes` | string | IS element names (e.g. `ISAba1; IS26`) |
| `mge_families` | string | IS families (e.g. `IS4; Tn3`) |

### Mobility columns (plasmid contigs only)

| Column | Type | Description |
|---|---|---|
| `mobility_class` | string | `conjugative` / `mobilizable` / `non-mobilizable` |
| `replicon_type` | string | Inc group (e.g. `IncF`, `IncP`, `IncQ`) |
| `relaxase_type` | string | Relaxase gene family (e.g. `MOBF`, `MOBP`) |
| `mpf_type` | string | Mating pair formation system (e.g. `MPF_T`, `MPF_F`) |

### AMR risk score columns (plasmid contigs only)

| Column | Type | Description |
|---|---|---|
| `risk_score` | int | Total AMR risk score, capped at 10 |
| `mobility_score` | int | Points from mobility class |
| `arg_score` | int | Points from ARG count and drug classes |
| `replicon_score` | int | Points from broad-host-range replicon type |
| `context_score` | int | Points from sample context (clinical/wastewater/environmental) |
| `host_score` | int | Points from ESKAPE/WHO pathogen host |
| `risk_evidence` | string | Semicolon-separated list of risk factors that contributed points |
| `eskape_host` | bool | `True` if host is an ESKAPE pathogen |
| `eskape_genus` | string | ESKAPE genus name if detected |

### Topology and plasmid-DB columns

| Column | Type | Description |
|---|---|---|
| `topology` | string | `circular` / `linear` / `too_short` (< 500 bp for DTR detection) |
| `plasmid_db_match` | string | Closest known plasmid accession from PLSDB/RefSeq/COMPASS |
| `plasmid_db_source` | string | `PLSDB` / `RefSeq` / `COMPASS` |
| `plasmid_db_ani` | float | Approximate nucleotide identity (%) to the DB hit |
| `plasmid_db_cov` | float | Query coverage (%) of the DB hit alignment |

### Pathogen and BacMet columns

| Column | Type | Description |
|---|---|---|
| `pathogen_species` | string | Predicted pathogen species (if host is a known pathogen) |
| `pathogen_threat` | string | WHO threat level (critical / high / medium) |
| `pathogen_category` | string | ESKAPE / WHO2024 / etc. |
| `num_bacmet` | int | Number of biocide/metal resistance genes detected |
| `bacmet_genes` | string | BacMet gene names |
| `bacmet_class` | string | Resistance compound class (biocide / metal) |
| `bacmet_compounds` | string | Specific compounds |
| `num_ice` | int | Number of ICE hits detected |
| `ice_ids` | string | ICEberg3 accessions |
| `ice_functions` | string | ICE gene functions |

---

## `genes.tsv` — gene-level table

One row per predicted ORF across all contigs.

| Column | Description |
|---|---|
| `contig_id` | Parent contig |
| `gene_id` | ORF identifier |
| `start` | Start position (bp, 1-indexed) |
| `end` | End position (bp) |
| `strand` | `+` or `-` |
| `length_bp` | ORF length in base pairs |
| `contig_label` | Classification of the parent contig |
| `arg_flag` | `1` if this ORF is an ARG hit |
| `vf_flag` | `1` if this ORF is a VF hit |
| `mge_flag` | `1` if this ORF is an MGE hit |
| `gene_name` | Gene name (from CARD / SARG / VFDB / ISfinder) |
| `drug_class` | Drug class for ARG hits |
| `amr_family` | AMR gene family |
| `vf_category` | Virulence factor category |
| `is_family` | IS element family |
| `source` | Database source |
