# Phase 11 — Routing Analysis

## Stage-1 Score Distribution

### P(malicious) Quantiles

| Quantile | True Benign | True Malicious | Stage-1 FN |
|---|---:|---:|---:|
| 5th | 0.0073 | 0.2776 | 0.0772 |
| 10th | 0.0133 | 0.4140 | 0.1062 |
| 25th | 0.0366 | 0.9399 | 0.1863 |
| 50th | 0.0795 | 0.9987 | 0.3034 |
| 75th | 0.1775 | 0.9999 | 0.3788 |
| 90th | 0.3529 | 1.0000 | 0.4174 |
| 95th | 0.4528 | 1.0000 | 0.4290 |

### Stage-1 Decision Counts (Baseline)

- FINAL_BENIGN: 84618
- FINAL_MALICIOUS: 36421
- NEEDS_STAGE2: 5530
- Stage-1 FN: 4753
- Stage-1 FP: 1400
