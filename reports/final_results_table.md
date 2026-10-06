# Final Results Table

### Dataset
- **Total:** 126,569
- **Benign:** 84,448
- **Malicious:** 42,121

### Canonical DSSE (Phase 12, alpha=0.05)
- **Stage-1 resolution:** 96.69%
- **Stage-2 escalation:** 3.31%
- **Accuracy:** 0.9389
- **Precision:** 0.9447
- **Recall:** 0.8673
- **F1:** 0.9043
- **ROC-AUC:** 0.9767
- **FPR:** 0.0253
- **FNR:** 0.1327

### Baseline Comparison

| Model | F1 | ROC-AUC | Recall | FPR | FNR | Stage-2 % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| HXRF only (Stage 1) | 0.8872 | 0.9667 | 0.8524 | 0.0326 | 0.1476 | N/A |
| MiniLM + XGB | 0.5842 | 0.7654 | - | - | - | 100% | *(HISTORICAL / NOT DIRECTLY COMPARABLE)*
| Concat + XGB | 0.6925 | 0.8381 | - | - | - | 100% | *(HISTORICAL / NOT DIRECTLY COMPARABLE)*
| FiLM + XGB | 0.7133 | 0.8374 | - | - | - | 100% | *(HISTORICAL / NOT DIRECTLY COMPARABLE)*
| Phase-11A canonical DSSE | 0.9077 | 0.9768 | 0.8703 | 0.0236 | 0.1297 | 4.37% |
| Phase-12 canonical ICP DSSE | 0.9043 | 0.9767 | 0.8673 | 0.0253 | 0.1327 | 3.31% |
