# DSSE Final Research Freeze

## 1. Project Objective
To construct, evaluate, and thoroughly freeze a Dual-Stage Semantic Ensemble (DSSE) for malicious URL detection, demonstrating how a computationally cheap structural Stage-1 classifier can be wrapped in Inductive Conformal Prediction (ICP) to safely and selectively escalate only mathematically ambiguous URLs to a heavy, transformer-driven Stage-2 semantic classifier.

## 2. Final Architecture
The pipelined architecture cascades from 12 URL character features (Stage-1) to an early-exit conformal prediction set threshold, routing the remaining uncertain distribution subset to a FiLM-fused MiniLM/XGBoost ensemble (Stage-2).

## 3. Dataset
The canonical disjoint validation strategy isolated `test.parquet` (126,569 URLs: 84,448 Benign, 42,121 Malicious) from `proper_train.parquet` and `calibration.parquet`. Identical URLs were purged.

## 4. Stage-1 Features
Frozen to exactly 12 discrete structural features. 

## 5. Stage-1 Model
Frozen: `models/stage1_hxrf.pkl` (25-tree Random Forest).

## 6. Canonical ICP
Frozen: `models/stage1_conformal_canonical.pkl` configured directly from the identically-frozen Stage-1 HXRF baseline across the isolated `calibration.parquet` partition.

## 7. Stage-2 Model
Frozen: MiniLM -> FiLM PyTorch Module -> XGBoost configuration structure.

## 8. Final Evaluation Protocol
Evaluated holistically under a fully deterministic, test-data-clean constraint. The reported operating $\alpha$ of 0.05 is represented strictly as a pre-specified analytical boundary, not an unconditionally optimized test-metric outcome.

## 9. Final Results
**Canonical DSSE** (`alpha=0.05`)
- Stage-1 Resolution: 96.69%
- Stage-2 Escalation: 3.31%
- F1 Score: 0.9043
- ROC-AUC: 0.9767
- Precision: 0.9447
- Recall: 0.8673

## 10. Baseline Comparison
The DSSE mathematically out-competes purely structural Phase-1 baselines (F1: 0.8872 -> 0.9043) by allocating semantic overhead to roughly 3.3% of queries. 

## 11. Computational Trade-off
Calculates roughly a 96% reduction in deep-learning infrastructure cost by circumventing Stage-2 evaluation on confident structural subsets. Latencies indicate >1000 URLs/sec batch evaluation capability.

## 12. Error Analysis
False negatives exclusively in Stage-1 bound the DSSE ceiling (5043 FN observed vs 5591 End-to-End). Conformal escalation successfully offloads ambiguity but structural misclassification (high confidence FN/FP) persists directly through to the output.

## 13. Statistical Analysis
McNemar's statistical testing confirmed statistically valid divergence ($p < 0.001$) across the old methodology against the repaired canonical model prediction behaviors.

## 14. Reproducibility
Single-threaded test matrices (OMP=1) validated strictly 0 prediction disagreement across runs. 74/74 tests pass. 

## 15. Supported Claims
Marginal finite-sample coverage is rigorously bounded over the calibration subset logic, and end-to-end efficiency metrics are solidly measured and statistically validated.

## 16. Limitations
Data shifts naturally degrade static threshold guarantees. False negatives on the underlying structural module are unavoidable. Targeted adversarial string tampering defeats length/entropy measures.

## 17. Final Contributions
1. Lightweight structural Stage-1 gatekeeper using exactly 12 URL-only features.
2. Conformal routing mechanism for selective escalation.
3. MiniLM semantic representation for uncertain URLs.
4. Structural-conditioned FiLM fusion with semantic representation.
5. XGBoost Stage-2 classifier.
6. Measured computation-vs-detection trade-off.

## 18. Final Verdict
RESEARCH PIPELINE FROZEN — PUBLICATION REVIEW READY WITH DOCUMENTED LIMITATIONS
