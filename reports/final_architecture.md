# Final Architecture

## The Frozen Dual-Stage Semantic Ensemble (DSSE) Architecture

The complete system is organized as a pipelined, short-circuiting architecture:

1. **Input URL**
2. **Feature Extraction:** 12 Structural Features Extracted deterministically.
3. **Stage-1 Canonical Gatekeeper:** `models/stage1_hxrf.pkl`
   - Emits $p(y | X_{struct})$.
4. **Canonical Split-Conformal ICP:** `models/stage1_conformal_canonical.pkl`
   - Maps $p(y | X)$ to a conformal prediction set using the finite-sample corrected conformal threshold $q_{0.05} = 0.99341$.
5. **Early Exit / Routing Logic:**
   - If set is `{BENIGN}`, route to `FINAL_BENIGN`.
   - If set is `{MALICIOUS}`, route to `FINAL_MALICIOUS`.
   - If set is `{BENIGN, MALICIOUS}` (Ambiguous) or `EMPTY_SET`, route to `NEEDS_STAGE2`.
6. **Semantic Stage-2 Encoder:** MiniLM (`models/minilm_encoder/`)
   - Computes 384-dimensional dense semantic embedding.
7. **FiLM Structural-Semantic Fusion:** `models/film/film_model.pth`
   - Fuses semantic embeddings via structural conditioning.
8. **Stage-2 Classifier:** XGBoost (`models/xgboost_audit/film_xgb.json`)
   - Emits final malicious/benign probability score over the fused embedding space.
9. **Final Output Prediction**
