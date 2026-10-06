# Phase 8 Data Leakage Audit

✓ structural scaler fitted only on training data (train_split = 80% of 50k sample)
✓ FiLM model trained only on training data (train_split)
✓ classifier trained only on training data (part of FiLM model)
✓ test data never used to select the best model (used val_split instead)
✓ ICP routing does not use test labels
✓ NEEDS_STAGE2 membership comes from ICP prediction sets (conformal calibrator routing)
✓ test labels are only used for final evaluation
