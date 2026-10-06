# Phase 11A — Artifact Inventory

## MODELS

- `models/stage1_hxrf.pkl`: Phase 10 DSSE Random Forest. Canonical model for Stage-1.
- `models/stage1_hxrf_icp.pkl`: Phase 6/9 ICP calibration model. Contains different RF checkpoints. Considered legacy as Phase-10 DSSE overrides it.
- `models/stage1_conformal.pkl`: ICP calibration artifact fit to `stage1_hxrf_icp.pkl` but applied to `stage1_hxrf.pkl` in Phase 10/11. Canonical (frozen) for routing, despite known model mismatch.
- `models/xgboost_audit/film_xgb.json`: Stage-2 XGBoost model trained on FiLM fusion representations (Phase 9). Canonical.
- `models/film/film_model.pth`: FiLM structural-semantic fusion neural network weights (Phase 8). Canonical.
- `models/film/film_scaler.pkl`: StandardScaler for structural features passed to FiLM (Phase 8). Canonical.

## SCRIPTS
- `scripts/phase9_xgboost_audit.py`: Legacy script establishing XGBoost baseline.
- `scripts/phase10_dsse_evaluation.py`: Frozen baseline DSSE pipeline script. Used raw URLs for feature extraction.
- `scripts/phase11_routing_experiment.py`: Experimental script exploring asymmetric routing. Used parquet columns.

## CONFIG
- `config/stage1_features.json`: Defines the canonical 12 Stage-1 features.
- `config/stage1_model_config.json`: Configuration for HXRF training.
- `config/conformal_config.json`: Contains alpha levels and calibration settings.
- `config/minilm_config.json`: Semantic encoder configs.
- `config/film_config.json`: FiLM neural network config.
- `config/dsse_dataset_config.json`: Dataset split configurations.

All artifacts listed above as "Canonical" will be used for Phase 11A baseline evaluation.
