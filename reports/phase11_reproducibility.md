# Phase 11 — Reproducibility

- Python: 3.12.0 (v3.12.0:0fb18b02c8, Oct  2 2023, 09:45:56) [Clang 13.0.0 (clang-1300.0.29.30)]
- PyTorch: 2.13.0
- OS: macOS
- Random seeds: 42
- Test dataset: data/splits/test.parquet (126569 URLs)
- HXRF model: models/stage1_hxrf.pkl
- ICP calibrator: models/stage1_conformal.pkl
- FiLM model: models/film/film_model.pth
- XGBoost model: models/xgboost_audit/film_xgb.json
- FiLM scaler: models/film/film_scaler.pkl
- Features: parquet columns (deterministic)
- Calibrator alpha: 0.05 (baseline)
- Calibrator quantile: 0.557457
- Calibration samples: 101255
