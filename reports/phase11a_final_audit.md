# Phase 11A — Final Project Audit and Wrap-up

## Overview
Phase 11A serves as the final step for this research project, reconciling inconsistencies in evaluation across Phases 9, 10, and 11, and establishing a single frozen, reproducible baseline. All research and tuning phases are now strictly terminated.

## Key Reconciliations

### 1. Stage-1 Feature Drift
- **Issue:** The original DSSE pipeline extracted features dynamically from URLs at evaluation time. However, due to floating-point representation differences and parsing logic variations, 31,505 rows (24.89%) had differing feature values compared to the canonical Parquet training data.
- **Resolution:** `phase11a_feature_drift.md` isolates this drift. To ensure exact reproduction of the Random Forest training distributions, the Canonical DSSE Evaluation uses the 12 Stage-1 features loaded directly from `data/splits/test.parquet`.

### 2. HXRF Model Mismatch
- **Issue:** `models/stage1_hxrf.pkl` (Phase 10 production) and `models/stage1_hxrf_icp.pkl` (calibration phase) were found to be different Random Forest checkpoints (different number of tree nodes).
- **Resolution:** `stage1_hxrf.pkl` is declared the Canonical Stage-1 model. `stage1_hxrf_icp.pkl` is deprecated as legacy.

### 3. ICP Calibration Drift
- **Issue:** Because `stage1_conformal.pkl` was calibrated on scores from `stage1_hxrf_icp.pkl` but is evaluated against scores from `stage1_hxrf.pkl`, the theoretical marginal coverage guarantee of ICP is technically broken.
- **Resolution:** Despite the theoretical breakage, this exact configuration represents the frozen Phase-10 pipeline. We retain this configuration as the Canonical baseline, acknowledging the limitation. This configuration produces exactly **5,530 escalated URLs**, explaining the discrepancy from Phase 9 (which used the icp checkpoint and escalated 5,473 URLs).

## Canonical Performance Metrics (Frozen Baseline)
- **Dataset:** `data/splits/test.parquet` (126,569 rows)
- **Stage-1 Resolution Rate:** 95.63% (121,039 URLs)
- **Stage-2 Escalation Rate:** 4.37% (5,530 URLs)
- **Overall DSSE Accuracy:** 0.9411
- **Overall DSSE F1:** 0.9077
- **Overall DSSE Recall:** 0.8703
- **Overall DSSE ROC-AUC:** 0.9768

## Determinism
- **Check:** Running the pipeline identically twice produced matching metric values to the 4th decimal place.
- **Mitigation:** Setting `OMP_NUM_THREADS=1` resolved the catastrophic OpenMP threading conflicts observed between PyTorch, XGBoost, and scikit-learn on macOS.

## Conclusion
The Dual-Stage Semantic Ensemble (DSSE) is now fully audited, deterministically reproducible, and frozen. No further phases will be initiated.
