# Phase 11A — ICP Artifact Reconciliation

## Artifacts Assessed
- `models/stage1_conformal.pkl` (ICP threshold calibration)
- `models/stage1_hxrf_icp.pkl` (Model used for ICP score generation)
- `config/conformal_config.json` (Target alpha levels)

## Findings
- **Underlying Checkpoint Mismatch:** The conformal thresholds were calibrated using scores from `stage1_hxrf_icp.pkl`. However, the production Phase-10 pipeline applies these thresholds to probabilities emitted by `stage1_hxrf.pkl`. As demonstrated in HXRF reconciliation, these models have different random forest checkpoints (different tree nodes).
- **Mathematical Consistency:** The application of conformal routing (using `predict_proba` and checking thresholds against `q_alpha`) is mathematically correct according to the logic implemented, but the marginal coverage guarantee of Inductive Conformal Prediction assumes identical calibration and test distributions (including identical scoring model). By applying calibration derived from Model A onto Model B, the theoretical coverage guarantees are broken.
- **Canonical Decision:** Phase-10 DSSE represents the frozen architecture evaluated for production. We preserve this exact configuration. We acknowledge the broken theoretical guarantee as a documented limitation, but this exact pairing of `stage1_hxrf.pkl` and `stage1_conformal.pkl` defines the canonical Phase 11A evaluation.
