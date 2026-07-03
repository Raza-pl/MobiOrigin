# Advanced Usage

## Lenient mode

By default PlasFlow v2 requires **biological evidence** to confirm a plasmid call on contigs shorter than 50 kb (PLSDB match, relaxase, replicon type, ICE hit, or rep protein). Contigs with high MLP scores but no hallmark evidence are returned as `unclassified`.

Use `--lenient` to skip this gate — useful when databases aren't set up yet, or for exploratory runs where sensitivity matters more than precision:

```bash
plasflow2 run --input assembly.fasta --output results/ --lenient
```

`--lenient` does two things:
1. Lowers the MLP plasmid threshold from **0.95 → 0.70** (catches weaker signals)
2. Skips the hallmark gate entirely (no biological evidence required)

Expect more plasmid calls and more false positives compared to the default mode.

---

## Taxonomy annotation (GTDB)

PlasFlow v2 annotates host taxonomy using **DIAMOND + a Kaiju-style LCA algorithm** against the GTDB r220 protein database. The taxonomy database is not downloaded by default — it is ~8 GB and opt-in.

### Automated setup

```bash
bash scripts/setup_databases.sh --gtdb --threads 16
```

This downloads GTDB r220 representative proteins (~8 GB), builds the DIAMOND database, downloads the taxonomy TSV, and builds the taxon map. Plan for ~40 GB free disk space and **1–3 hours** total.

If you already have the protein FASTA from a previous install:

```bash
bash scripts/setup_databases.sh --gtdb \
  --gtdb-proteins-path /existing/gtdb_prot_reps_r220.faa
```

### Run with taxonomy

Once built, taxonomy is auto-detected at `data/databases/taxonomy/refseq_taxonomy.dmnd` and runs automatically:

```bash
plasflow2 run --input assembly.fasta --output results/ --threads 16
```

To specify non-default paths:

```bash
plasflow2 run --input assembly.fasta --output results/ \
  --taxonomy-db /path/to/refseq_taxonomy.dmnd \
  --taxon-map   /path/to/taxon_map.tsv
```

To skip taxonomy (saves 20–40 min on large datasets):

```bash
plasflow2 run --input assembly.fasta --output results/ --skip-taxonomy
```

> **Note:** PLSDB and GTDB are independent. PLSDB confirms plasmid calls via minimap2 alignment. GTDB annotates host taxonomy via DIAMOND protein search. You can use either, both, or neither.

---

## Higher accuracy with geNomad

Running geNomad adds 12 SPM (sequence-specific plasmid marker) gene features to the XGBoost stage-2 model, improving precision on borderline contigs.

`plasflow2 run` invokes geNomad **automatically** if it is on your PATH — no extra steps needed. Just make sure geNomad is installed and its database is in `data/databases/genomad_db/`.

```bash
# Install geNomad
conda install -c conda-forge -c bioconda genomad

# Download the geNomad database (one-time, ~3 GB)
genomad download-database data/databases/

# Run — geNomad is called automatically
plasflow2 run --input assembly.fasta --output results/ --threads 16
```

---

## Rebuilding the HTML report

Re-generate reports from an existing TSV without re-running the full pipeline:

```bash
plasflow2 report \
  --predictions results/all_predictions.tsv \
  --output      results/ \
  --context     clinical   # re-score as clinical
```

Useful when you want to change the sample context without waiting for DIAMOND to run again.

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

The default thresholds are tuned for metagenomes where plasmids are rare. Adjust for unusual datasets:

```bash
# Lower plasmid threshold to increase recall
plasflow2 run --input assembly.fasta --output results/ \
  --plasmid-threshold 0.80

# Assign every contig to a class instead of leaving some 'unclassified'
plasflow2 run --input assembly.fasta --output results/ \
  --min-confidence 0.50
```

For maximum sensitivity (no threshold or hallmark gate), use `--lenient` (see above).

---

## Using custom database paths

All databases are auto-detected from `data/databases/`. Override any of them:

```bash
plasflow2 run \
  --input        assembly.fasta \
  --output       results/ \
  --card-db      /custom/card/card.dmnd \
  --aro-index    /custom/card/aro_index.tsv \
  --sarg-db      /custom/sarg/sarg.dmnd \
  --plsdb-path   /custom/plasmids/PLSDB.fna \
  --taxonomy-db  /custom/taxonomy/refseq_taxonomy.dmnd \
  --taxon-map    /custom/taxonomy/taxon_map.tsv \
  --threads      16
```

---

## Docker

```bash
# Build the image
docker build -t plasflow2 .

# Set up databases on the host first (run outside Docker)
bash scripts/setup_databases.sh

# Run with mounted volumes
docker run --rm \
  -v $(pwd)/data:/data \
  -v /path/to/results:/results \
  plasflow2 run \
    --input   /data/assembly.fasta \
    --output  /results/ \
    --threads 8
```

The container reads from `/data/databases/` and `/data/models/`. Run `bash scripts/setup_databases.sh` on the host to populate `data/` before starting the container.

---

## Compressed input

PlasFlow v2 accepts gzip and bzip2 compressed FASTA files directly:

```bash
plasflow2 run --input assembly.fasta.gz  --output results/
plasflow2 run --input assembly.fasta.bz2 --output results/
```
