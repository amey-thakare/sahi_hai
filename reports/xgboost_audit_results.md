# Phase 9 Structural-Semantic DSSE XGBoost Classification (AUDIT RUN)

## A. Experimental setup
Phase 9 XGBoost models were trained using the full 215,845 valid training samples (plus validation split), completely separated from test and development stages.

## B. Data Populations
- **Training Population**: 455,647 samples (Full available valid training set).
- **Validation Population**: 50,628 samples.
- **Test Population**: 126,569 samples.
- **NEEDS_STAGE2 Test Population**: 5,473 samples (4.32% of test).

## C. Full-test results
| Model | Population | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM + XGB | Full | 0.8913 | 0.8684 | 0.7936 | 0.8293 | 0.9541 | 0.9400 | 0.0600 | 0.2064 |
| Concat + XGB | Full | 0.9517 | 0.9531 | 0.8991 | 0.9253 | 0.9893 | 0.9779 | 0.0221 | 0.1009 |
| FiLM + XGB | Full | 0.9528 | 0.9505 | 0.9051 | 0.9273 | 0.9897 | 0.9765 | 0.0235 | 0.0949 |

## D. NEEDS_STAGE2 results
| Model | Population | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM + XGB | NEEDS_STAGE2 | 0.6954 | 0.6900 | 0.5065 | 0.5842 | 0.7654 | 0.8336 | 0.1664 | 0.4935 |
| Concat + XGB | NEEDS_STAGE2 | 0.7583 | 0.7484 | 0.6445 | 0.6925 | 0.8381 | 0.8415 | 0.1585 | 0.3555 |
| FiLM + XGB | NEEDS_STAGE2 | 0.7672 | 0.7434 | 0.6856 | 0.7133 | 0.8374 | 0.8270 | 0.1730 | 0.3144 |

## E. Statistical tests
Paired tests conducted on NEEDS_STAGE2 test population.

- McNemar's Test (FiLM vs Concat): p = 0.1141
- McNemar's Test (FiLM vs MiniLM): p = 4.75e-26

## F. Bootstrap confidence intervals
- FiLM - Concat F1 bootstrap 95% CI: [+0.0055, +0.0356]

## G. Latency
Batch-1000 CPU inference benchmark:
- MiniLM encoding: 1217.87 ms / batch 1000
- Structural feature processing: 14.22 ms / batch 1000
- FiLM + XGB: 0.90 ms / batch 1000

The complete batch Stage-2 processing time is approximately:
1217.87 + 14.22 + 0.90 ms = 1232.99 ms per 1000 URLs

## H. Throughput
- Complete Stage-2 Pipeline: ≈ 811 URLs/sec (Batch-1000 CPU inference benchmark)

## I. FiLM stability
- Gamma Mean: 0.7737, Std: 11.0686, Min: -133.2307, Max: 138.6323
- Beta Mean: 0.0091, Std: 0.0760, Min: -1.3812, Max: 1.1168
- Percentage of extreme gamma values (>10 or <-10): 38.9793%
No stabilized FiLM variant was included in the primary Phase-9 experiment.

## J. Limitations
1. FiLM vs concatenation did not achieve statistical significance under the paired McNemar test.
2. FiLM had slightly lower ROC-AUC than concatenation in the NEEDS_STAGE2 population.
3. The latency benchmark was conducted at batch size 1000.
4. MiniLM embedding generation dominates Stage-2 computational cost.
5. Phase 9 used a fixed XGBoost configuration of: n_estimators=100, max_depth=5.
6. The `train_test_split` separating training and validation sets was NOT stratified. However, manual inspection verified that the positive class ratio (dsse_binary_target) was ~0.3331 for train, ~0.3302 for validation, and ~0.3328 for test, indicating no material class-distribution difference was introduced.
7. Phase-9 conclusions are based on the evaluated dataset and should not be interpreted as universal robustness claims.

## K. Conservative conclusions
### Hypothesis Table

| Hypothesis | Comparison | Result | Supported? |
|------------|------------|--------|------------|
| H1 | FiLM vs MiniLM | Supported by observed F1 improvement and McNemar p < 0.05. | Yes |
| H2 | FiLM vs Concatenation | Observed F1 improvement, but McNemar test not significant. | Partially supported / inconclusive |
| H3 | FiLM + XGBoost provides an advantage over the corresponding baselines | Supported descriptively, with the strongest evidence being against the MiniLM-only baseline. | Partially supported |

### Final Claim
The Phase-9 experiment shows that structural-semantic FiLM modulation improves F1 over the MiniLM-only baseline on the ICP-routed NEEDS_STAGE2 population. Compared with naive concatenation, FiLM achieves a modest F1 improvement, although the paired McNemar test does not indicate a statistically significant difference in classification outcomes. Concatenation achieves a marginally higher ROC-AUC, indicating that FiLM's advantage is primarily observed in the selected operating-point F1 rather than ranking performance.
