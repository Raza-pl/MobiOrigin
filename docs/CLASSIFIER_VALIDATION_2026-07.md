# Classifier validation — July 2026

Balanced plasmid profile: precision 0.9217, recall 0.6755, F1 0.7796 on 7,038 leakage-screened temporal fragments.

Evidence-assisted profile: precision 0.9939, recall 0.5118, F1 0.6757.

Phage thresholds: <=2 kb 0.855; 2-5 kb 0.850; 5-10 kb 0.845; 10-20 kb 0.835; >20 kb 0.750. Final confirmation: estimated precision 0.7281, recall 0.8068, F1 0.7654.

Use `balanced` for general classification. Use `evidence-assisted` when plasmid precision matters more than recall. Main limitations are novel short plasmids and phages <=2 kb.

## Temporal head-to-head against geNomad 1.12.0

The frozen strict low-similarity temporal benchmark contains 7,038 labeled fragments from 35 plasmid and 45 chromosome sources released after the training freeze. Exact identifier overlap with training and production reference sources was zero.

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| PlasFlow balanced | 0.9217 | 0.6755 | 0.7796 |
| geNomad 1.12.0 calibrated | 0.8549 | 0.3549 | 0.5016 |

Per-length F1:

| Length | PlasFlow | geNomad |
|---|---:|---:|
| <=2 kb | 0.7310 | 0.2178 |
| 5-10 kb | 0.8681 | 0.7717 |
| 10-20 kb | 0.8681 | 0.8285 |
| >20 kb | 0.8242 | 0.8276 |

The benchmark contains no sequences from 2,001-4,999 bp because its fixed window sizes are 1, 2, 5, 10 and 20 kb. This comparison evaluates plasmid-versus-chromosome classification and does not evaluate phage sensitivity.

The balanced profile remains recommended for general and novel-plasmid discovery. COMPASS and geNomad evidence should be treated as supporting evidence rather than a hard veto on sequence-model predictions.

