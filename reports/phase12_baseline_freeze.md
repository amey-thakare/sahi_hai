# Phase 12 — Baseline Freeze

## Dataset
- **Canonical Test Population**: 126,569
- **Benign**: 84,448
- **Malicious**: 42,121
- **Canonical Test Split**: `data/splits/test.parquet`

## Architecture & Artifacts
- **Canonical Stage-1 Model**: `models/stage1_hxrf.pkl`
- **Canonical Stage-1 Features**: Exactly the 12 features frozen in `config/stage1_features.json` (evaluated via Parquet columns)
- **Canonical Stage-2**: MiniLM -> FiLM (`models/film/film_model.pth`, `film_scaler.pkl`) -> XGBoost (`models/xgboost_audit/film_xgb.json`)

## Baseline Metrics (from Phase 11A)
- **Accuracy**: 0.9411
- **Precision**: 0.9484
- **Recall**: 0.8703
- **F1**: 0.9077
- **ROC-AUC**: 0.9768
- **FPR**: 0.0236
- **FNR**: 0.1297
- **Stage-1 Resolution Rate**: 95.63%
- **Stage-2 Escalation Rate**: 4.37% (5,530 URLs)
