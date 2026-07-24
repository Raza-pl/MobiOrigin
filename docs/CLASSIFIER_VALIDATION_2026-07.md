# Classifier validation — July 2026

Balanced plasmid profile: precision 0.9217, recall 0.6755, F1 0.7796 on 7,038 leakage-screened temporal fragments.

Evidence-assisted profile: precision 0.9939, recall 0.5118, F1 0.6757.

Phage thresholds: <=2 kb 0.855; 2-5 kb 0.850; 5-10 kb 0.845; 10-20 kb 0.835; >20 kb 0.750. Final confirmation: estimated precision 0.7281, recall 0.8068, F1 0.7654.

Use `balanced` for general classification. Use `evidence-assisted` when plasmid precision matters more than recall. Main limitations are novel short plasmids and phages <=2 kb.
