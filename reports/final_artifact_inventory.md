# Final Artifact Inventory

## Stage 1
- **Path:** `models/stage1_hxrf.pkl`
- **Model Type:** Scikit-Learn `RandomForestClassifier` (Historical checkpoint, re-branded as HXRF)
- **Configuration:** 25 Trees, max depth 20
- **Input Dimensions:** 12 features
- **State:** Frozen

## ICP
- **Path:** `models/stage1_conformal_canonical.pkl`
- **Model Type:** Inductive Conformal Prediction Calibrator
- **Configuration:** Splits conformal score definition ($s=1-p(y|x)$), using calibration bounds.
- **State:** Frozen (Fitted during Phase 12 using `models/stage1_hxrf.pkl`)

## Stage 2
### MiniLM
- **Path:** `models/minilm_encoder/` (Artifacts)
- **Model Type:** Transformer encoder (`all-MiniLM-L6-v2`)
- **Input Dimensions:** Tokenized strings (limit 512 tokens)
- **Feature Dimensions:** 384
- **State:** Frozen

### FiLM Scaler
- **Path:** `models/film/film_scaler.pkl`
- **Model Type:** StandardScaler
- **State:** Frozen

### FiLM Artifact
- **Path:** `models/film/film_model.pth`
- **Model Type:** PyTorch Neural Network (FiLMFusionClassifier)
- **Feature Dimensions:** 384 semantic + 12 structural -> 128 hidden -> 128 representations.
- **State:** Frozen

### XGBoost Artifact
- **Path:** `models/xgboost_audit/film_xgb.json`
- **Model Type:** XGBClassifier
- **Input Dimensions:** 128 embedding dimensions (from FiLM)
- **State:** Frozen
