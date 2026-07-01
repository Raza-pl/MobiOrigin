# Advanced Usage

## Higher accuracy with geNomad

Running geNomad separately before PlasFlow v2 adds 12 SPM (sequence-specific plasmid marker) gene features to the XGBoost stage-2 model, improving precision on borderline contigs.

**Requires:** geNomad installed (`conda install -c bioconda genomad`) and its database.

```bash
# Download geNomad database (~3 GB, one-time)
genomad download-database data/databases/genomad_db/

# Step 1 — annotate with geNomad (~5–30 min)
genomad annotate assembly.fasta genomad_out/ data/databases/genomad_db/ --threads 16

# Step 2 — generate MOB-suite annotation TSV with geNomad features added
plasflow2 prepare \
  --input assembly.fasta \
  --output annotations.tsv \
  --genomad-genes genomad_out/assembly_annotate/assembly_genes.tsv \
  --threads 16

# Step 3 — classify using both k-mer and biological features
plasflow2 classify \
  --input assembly.fasta \
  --output predictions.tsv \
  --annotation-tsv annotations.tsv
```

---

## Two-step workflow (classify + annotate separately)

Useful when you want to classify first to get a quick look, then run the full annotation only on predicted plasmids.

```bash
# Step 1 — fast classification (seconds)
plasflow2 classify --input assembly.fasta --output predictions.tsv

# Step 2 — annotate predicted plasmids
grep "^.*\tplasmid\t" predictions.tsv | cut -f1 > plasmid_ids.txt
# extract plasmid sequences with seqtk or a custom script...

# Step 3 — annotate those plasmids
plasflow2 annotate \
  --input plasmids.fasta \
  --output annotations/ \
  --threads 16
```

---

## Rebuilding the HTML report

You don't need to re-run the full pipeline to regenerate the report. The `report` command reads `all_predictions.tsv` directly:

```bash
plasflow2 report \
  --predictions results/all_predictions.tsv \
  --output      results/ \
  --context     clinical   # re-score as clinical
```

This is useful when you want to change the sample context without waiting for DIAMOND to run again.

---

## Skipping slow steps

```bash
# Skip taxonomy annotation (saves 20–40 min on large datasets)
plasflow2 run --input assembly.fasta --output results/ --skip-taxonomy

# Skip MOB-suite mobility typing (when mob_typer is not installed)
plasflow2 run --input assembly.fasta --output results/ --skip-mobility

# Skip both
plasflow2 run --input assembly.fasta --output results/ \
  --skip-taxonomy --skip-mobility
```

---

## Adjusting classification thresholds

The default thresholds (`--threshold 0.70`, `--plasmid-threshold 0.95`) are tuned for metagenomes where plasmids are rare. Adjust them for unusual datasets:

```bash
# Plasmid-enriched sample — lower plasmid threshold to increase recall
plasflow2 run --input assembly.fasta --output results/ \
  --plasmid-threshold 0.80

# Assign every contig to a class instead of leaving some 'unclassified'
plasflow2 run --input assembly.fasta --output results/ \
  --min-confidence 0.50
```

---

## Using custom database paths

All databases are auto-detected. Override any of them:

```bash
plasflow2 run \
  --input        assembly.fasta \
  --output       results/ \
  --card-db      /custom/card/card.dmnd \
  --aro-index    /custom/card/aro_index.tsv \
  --sarg-db      /custom/sarg/sarg.dmnd \
  --taxonomy-db  /custom/taxonomy/refseq.dmnd \
  --taxon-map    /custom/taxonomy/taxon_map.tsv \
  --threads      16
```

---

## Docker

```bash
# Build the image
docker build -t plasflow2 .

# Run with local data
docker run --rm \
  -v /path/to/data:/data \
  -v /path/to/results:/results \
  plasflow2 run \
    --input   /data/assembly.fasta \
    --output  /results/ \
    --threads 8

# Print setup guide
docker run --rm plasflow2 setup
```

The Docker image includes DIAMOND, minimap2, and mob-suite. Mount your database directory to avoid re-downloading inside the container.

---

## Compressed input

PlasFlow v2 accepts gzip and bzip2 compressed FASTA files directly:

```bash
plasflow2 run --input assembly.fasta.gz  --output results/
plasflow2 run --input assembly.fasta.bz2 --output results/
```
