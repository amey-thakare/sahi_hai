# Phase 11 — End-to-End DSSE Results

## Architecture
Same DSSE architecture as Phase 10, with modified Stage-1 routing policy.

## Baseline Comparison

| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR | S2 Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HXRF only | 0.9277 | 0.9191 | 0.8585 | 0.8877 | 0.9687 | 0.0377 | 0.1415 | N/A |
| FiLM + XGB (all) | 0.9528 | 0.9505 | 0.9051 | 0.9273 | 0.9897 | 0.0235 | 0.0949 | 100% |
| Phase-10 DSSE | 0.9411 | 0.9484 | 0.8703 | 0.9077 | 0.9768 | 0.0236 | 0.1297 | 4.37% |
| **Phase-11 Best** | 0.9441 | 0.9692 | 0.8595 | 0.9110 | 0.9766 | 0.0136 | 0.1405 | 6.57% |

Best policy: **P1: α_b=0.025, α_m=0.05**
