# Final Reproducibility

The final DSSE Phase-12 end-to-end evaluation was executed locally to enforce determinism and test standard reproducibility standards. 

## Environment Details
- **Random Seeds:** Defaulted to `42` across pandas, numpy, torch, scikit-learn.
- **Python Version:** 3.12.0
- **Pytest Version:** 9.1.1
- **PyTorch/Threading Controls:** `KMP_DUPLICATE_LIB_OK=True`, `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1`
- **Model Checkpoints:** Frozen `stage1_hxrf.pkl`, `stage1_conformal_canonical.pkl`, `film_model.pth`, `film_xgb.json`.

## Verification 
The system was verified to resolve routing identically under strict thread limits. 

- **Prediction Disagreement:** 0
- **Routing Disagreement:** 0

The final test suite (74 items) passes identically and reproducibly across runs when the OpenMP threading boundaries are correctly configured.
