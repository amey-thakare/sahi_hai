# TECHNICAL REPORT TABLE AUDIT

| Table | Source | Status | Comments |
|---|---|---|---|
| Final Canonical Metrics | Phase-12 Results | PASS | Metrics exactly match: Acc 93.89%, F1 90.43%, Stage-1 96.69%, Stage-2 3.31%, FPR 2.53%, FNR 13.27%, ROC-AUC 97.67%. |
| Historical ICP Metrics | Phase-11A | PASS | Metrics cleanly separated. Acc 0.9411, F1 0.9077. Explicitly marked as historically invalid ICP baseline, NOT performance superiority. |
| Model Inventory | Artifact Directory | PASS | `stage1_hxrf.pkl` (Canonical), `stage1_conformal_canonical.pkl` (Canonical), `all-MiniLM-L6-v2` (Canonical), `film_model.pth` (Canonical), `film_xgb.json` (Canonical). Deprecated models correctly flagged. |
| Historical XGBoost Comparisons | Phase 9 | PASS | MiniLM+XGB F1 58.42%, Concat+XGB F1 69.25%, FiLM+XGB F1 71.33%. |
| Dataset Splits | Test Parquet | PASS | 126,569 samples, 84,448 benign, 42,121 malicious. |
| Stage-1 Errors | Phase-12 | PASS | Stage-1 FN: 5043, Stage-1 FP: 1670. |
| Final Errors | Phase-12 | PASS | Final FN: 5591, Final FP: 2137. |
| Statistical Significance | Phase 9 | PASS | McNemar 141 improvements, 410 regressions, chi-square 130.35, p < 0.001. Claim restricted to asymmetric disagreements. |

All numeric variables across the report trace back to the canonical Phase-12 freeze and historical Phase-9/11A frozen audits. No values have been modified, inflated, or hallucinated.
