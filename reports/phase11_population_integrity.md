# Phase 11 — Population Integrity Report

## 1. Discrepancy Summary

- Phase-9 NEEDS_STAGE2 count: **5473** (parquet features + ICP-HXRF)
- Phase-10 NEEDS_STAGE2 count: **6171** (URL-extracted features + DSSE-HXRF)
- Difference: **698**

## 2. Root Cause Analysis

Two factors contribute to the discrepancy:

### Factor A: Different HXRF Models

- Phase-9/ICP calibration used `models/stage1_hxrf_icp.pkl` (78 MB)
- Phase-10 DSSE config uses `models/stage1_hxrf.pkl` (84 MB)
- These are different Random Forest checkpoints trained on different data splits.
- The ICP calibrator was fitted against probabilities from `stage1_hxrf_icp.pkl`.
- Using `stage1_hxrf.pkl` with that calibrator introduces a model mismatch.

### Factor B: Feature Extraction Path

- Rows with any feature difference (URL-extracted vs parquet): **31505** / 126569
- Maximum absolute difference per feature:

| Feature | Max Diff |
|---|---:|
| domain_len | 0.000000 |
| path_depth | 2.000000 |
| special_count_v2 | 78.000000 |
| token_count | 148.000000 |
| url_entropy | 1.492800 |
| url_len | 0.000000 |
| = | 0.000000 |
| . | 0.000000 |
| // | 0.000000 |
| abnormal_url | 1.000000 |
| phish_adv_hyphen_count | 24.000000 |
| suspicious_extension | 1.000000 |

### Factorial Analysis

| Features | HXRF Model | NEEDS_STAGE2 |
|---|---|---:|
| Parquet columns | ICP-HXRF | 5473 |
| Parquet columns | DSSE-HXRF | 5530 |
| URL-extracted | ICP-HXRF | 6128 |
| URL-extracted | DSSE-HXRF | 6171 |

## 3. Decision

Phase-10 remains the frozen baseline because it represents the actual deployed DSSE system.
Phase-9 results remain valid for that experiment's scope (same HXRF model, parquet features).

## 4. Phase-11 Methodology

For Phase 11, all routing experiments will use:
- Features: **parquet columns** (deterministic, no floating-point drift)
- HXRF model: **stage1_hxrf.pkl** (the DSSE production model)
- Calibrator: **stage1_conformal.pkl** (unchanged)
- Test set: **data/splits/test.parquet**

This ensures the HXRF probabilities are identical across all Phase-11 experiments.
The baseline NEEDS_STAGE2 count under this configuration is computed fresh.

## 5. Reproducibility Implication

The HXRF model mismatch means ICP conformal coverage guarantees are technically
violated in Phase 10, since the calibrator was fitted on a different model's scores.
This is a known limitation that does not invalidate the routing logic but should
be noted in any publication.
