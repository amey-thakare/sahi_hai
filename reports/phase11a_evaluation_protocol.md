# Phase 11A — Evaluation Protocol

## Canonical Evaluation

For all future evaluation and final paper reporting, the DSSE evaluation pipeline must adhere to the following rules:

1. **Test Set Rule:** 
   From this point onward, the canonical test set (`data/splits/test.parquet`) must NEVER be used for:
   - routing threshold selection
   - hyperparameter selection
   - model selection
   - feature selection
   - alpha optimization
   The test set is ONLY for final evaluation. If routing policy tuning is required later, use a validation/calibration split.

2. **Feature Extraction Consistency:** 
   The pipeline must use Canonical Parquet Columns (`data/splits/test.parquet`) corresponding exactly to the 12 Stage-1 features defined in `config/stage1_features.json`. Dynamically extracting features from raw URLs introduces float drift and logic discrepancies, causing variations in downstream evaluation.

3. **Artifact Integrity:** 
   The models (`stage1_hxrf.pkl`, `stage1_conformal.pkl`, `film_xgb.json`, `film_model.pth`) are frozen.

4. **Reproducibility Variables:** 
   All experiments must document random seeds, Python/XGBoost/sklearn/sentence-transformers versions, and threading configurations (`OMP_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=True`).
