# Conformal Data Integrity Audit

## Data Splits

| Split | Samples | Source | Malicious % |
|-------|---------|--------|-------------|
| Proper-train | 405,020 | 80% of train.parquet | 33.3% |
| Calibration | 101,255 | 20% of train.parquet | 33.3% |
| Test (held-out) | 126,569 | test.parquet | 33.3% |

## Leakage Verification

| Check | Result |
|-------|--------|
| URL overlap: proper-train ∩ calibration | 0 |
| URL overlap: calibration ∩ test | 0 |
| URL overlap: proper-train ∩ test | 0 |
| Calibration data used to train HXRF-ICP? | ❌ No |
| Test data used for conformal calibration? | ❌ No |
| Test data used for α selection? | ❌ No |
| Test data used for threshold selection? | ❌ No |
| Test data used for probability calibration? | ❌ No |
| Test data used ONLY for final evaluation? | ✅ Yes |

## Split Methodology

- Source: `data/splits/train.parquet` (Phase 3 output)
- Split: 80/20 stratified by `dsse_binary_target`
- Random seed: 42
- HXRF-ICP model trained on proper-train **only**
- ICP calibrator fitted on calibration set **only**
- Held-out test set used **only** for final evaluation

## Model Integrity

- Phase 5 model (`models/stage1_hxrf.pkl`): **NOT modified or deleted**
- ICP-specific model (`models/stage1_hxrf_icp.pkl`): Trained on proper-train subset
- Same hyperparameters as Phase 5
- Same 12 features, same random seed
