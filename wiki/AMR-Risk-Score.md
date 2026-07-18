# AMR Risk Score

Each plasmid contig receives an AMR risk score from **0 to 10**. The score combines four independent signals: mobility, ARG burden, host pathogenicity, and sample context. Scores are capped at 10.

## Scoring table

| Factor | Points |
|---|---|
| **Mobility** | |
| Conjugative (self-transmissible) | +3 |
| Mobilizable (transferred by other plasmids) | +2 |
| **Replicon type** | |
| Broad-host-range replicon (IncP / IncQ / IncW) | +2 |
| **ARG burden** | |
| ≥ 5 ARGs or ≥ 3 drug classes | +3 |
| 3–4 ARGs or 2 drug classes | +2 |
| 1–2 ARGs | +1 |
| **Host pathogenicity** | |
| ESKAPE pathogen host (*K. pneumoniae*, *A. baumannii*, *P. aeruginosa*, *S. aureus*, *E. faecium*, *Enterobacter* spp., *E. coli*) | +3 |
| WHO 2024 critical or high priority pathogen | +2 |
| **Sample context** | |
| `--context clinical` | +3 |
| `--context wastewater` | +2 |
| `--context environmental` | +1 |
| `--context unspecified` (default) | +0 |

## Risk categories

| Score | Category |
|---|---|
| 7–10 | **High** — immediate attention recommended |
| 4–6 | **Medium** — warrants monitoring |
| 0–3 | **Low** |

## How to interpret

A high score (≥ 7) means the plasmid is **conjugative or mobilizable**, carries **multiple ARGs or broad-spectrum drug resistance**, and is hosted by a **known ESKAPE pathogen** — the combination most likely to spread clinically relevant resistance.

A low score doesn't mean the plasmid is harmless — it may carry one ARG in a non-pathogenic host. It means the known risk factors associated with rapid spread are not all present.

## TSV columns

The risk breakdown is available in `all_predictions.tsv`:

| Column | Description |
|---|---|
| `risk_score` | Total score (0–10) |
| `mobility_score` | Points contributed by mobility class |
| `arg_score` | Points contributed by ARG count / drug class count |
| `replicon_score` | Points contributed by replicon type |
| `context_score` | Points contributed by `--context` setting |
| `host_score` | Points contributed by host pathogenicity |
| `risk_evidence` | Semicolon-separated list of factors that contributed |
| `eskape_host` | `True` if host is an ESKAPE organism |
| `eskape_genus` | ESKAPE genus name |

## Changing context without re-running

The `report` command re-scores risk based on context when rebuilding the HTML:

```bash
plasflow2 report \
  --predictions results/all_predictions.tsv \
  --output      results/ \
  --context     clinical
```
