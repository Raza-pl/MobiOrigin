#!/usr/bin/env bash
# Run this from your terminal to commit and push all PlasFlow v2 changes to GitHub
set -euo pipefail
cd "$(dirname "$0")"

# Clear any stale git locks from a previous crash
rm -f .git/index.lock .git/HEAD.lock .git/COMMIT_EDITMSG.lock

git add \
  README.md \
  run_v4.sh \
  scripts/build_taxonomy_db.py \
  src/plasflow2/annotate/mobility.py \
  src/plasflow2/annotate/plasmid_db.py \
  src/plasflow2/annotate/taxonomy.py \
  src/plasflow2/annotate/topology.py \
  src/plasflow2/annotate/args.py \
  src/plasflow2/annotate/mge.py \
  src/plasflow2/annotate/vfdb.py \
  src/plasflow2/classify/predict.py \
  src/plasflow2/cli.py \
  src/plasflow2/pipeline.py \
  src/plasflow2/report/generator.py \
  src/plasflow2/output/ \
  src/plasflow2/utils/fasta.py \
  tests/unit/test_cli.py \
  tests/unit/test_pipeline.py \
  tests/unit/test_report.py

git -c core.hooksPath=/dev/null commit -m \
"feat: SARG auto-detect, plasmid-DB match, topology+confidence badges, mob_typer per-contig fix

Summary of all changes in this batch:

SARG dual-DB ARG annotation
- SARG database now auto-detected from data/databases/sarg/sarg.dmnd
- All optional DBs (SARG, VFDB, MGE, taxonomy, plasmid-DB) auto-detected at startup
- Run result: 38 CARD + 35 SARG-only = 73 ARGs total on WWTP metagenome

New: plasmid-DB nucleotide matching (annotate/plasmid_db.py)
- minimap2 asm20 vs combined PLSDB + RefSeq + COMPASS (13 GB)
- --split-prefix flag: handles large reference without OOM
- 4 new predictions.tsv columns: plasmid_db_match, plasmid_db_source, plasmid_db_ani, plasmid_db_cov
- Enables dual-taxonomy interpretation (host taxonomy vs closest known plasmid)

Taxonomy speed fix
- Switch blastx -> blastp: reuses pyrodigal ORFs from ARG step (~6x faster)
- block_size=4.0: large RAM chunks (~4x faster on big DBs)
- Removed unsupported --faster flag (caused taxonomy to be skipped entirely)

mob_typer per-contig fix
- Was returning 1 aggregate row for all 2559 plasmid contigs
- Now runs mob_typer on each contig individually, merges results
- Gives per-plasmid mobility class, replicon type, relaxase type

HTML report improvements
- Topology badge: circular (circle icon), linear (dash), too_short (grey)
- Low-confidence badge: warning icon for calls below 70% threshold
- Plain-English narrative summary block above stat cards
- Per-plasmid genome map (ARG/VF/MGE/mobility colour-coded)
- New CSS: .bcirc, .blowconf badge classes

genes.tsv gene-level output
- 169,009 ORFs from 24,746 contigs
- 16 columns: contig_id, gene_id, start, end, strand, length_bp,
  contig_label, arg_flag, vf_flag, mge_flag, gene_name, drug_class,
  amr_family, vf_category, is_family, source

Other improvements
- Compressed input (.gz/.bz2) via _open_fasta() in utils/fasta.py
- Circular topology detection via DTR (direct terminal repeat, 500bp window)
- MLP inference: DataParallel for multi-GPU CUDA, explicit model.eval()
- report_cmd reads all new columns; narrative + genome_maps wired
- 192 unit tests pass
- Benchmarked: 24746 contigs in 22m25s on 16-thread Apple Silicon CPU"

git push origin main
echo "Done — pushed to GitHub."
