# Final Architecture Description

The Dual-Stage Semantic Ensemble (DSSE) for malicious URL detection utilizes a cascaded architecture designed to filter obvious threats rapidly while reserving expensive deep learning resources for ambiguous edge cases.

1. **Input & Extraction:** A raw URL string serves as the sole input. 12 distinct, deterministic structural and character-level features are mathematically extracted.
2. **Structural Evaluation:** The 12 features are passed into a highly optimized Random Forest (`HXRF`), producing a rapid malicious probability estimate $p(y | x)$.
3. **Canonical Conformal Calibration:** A finite-sample, mathematically valid Inductive Conformal Prediction (ICP) module operates on the $p(y | x)$ probability. During offline preparation, it was calibrated on a disjoint partition identically using the $s(x,y) = 1-p(y|x)$ score formulation to bound a distribution threshold $q_\alpha$.
4. **Early Exit / Routing (Online):** 
   - A conformal prediction set is dynamically generated for the incoming URL. 
   - If the set consists precisely of `{BENIGN}` or `{MALICIOUS}`, the pipeline immediately exits with the corresponding confident classification. 
   - If the prediction set is empty or contains both classes, the decision is flagged as ambiguous and escalates to Stage 2.
5. **Semantic Processing:** For escalated URLs, the raw URL string is tokenized and transformed into a 384-dimensional dense semantic representation by a frozen `all-MiniLM-L6-v2` transformer.
6. **Structural-Semantic Fusion:** A Feature-wise Linear Modulation (FiLM) neural network conditions the semantic representation against the 12 scaled structural features, modulating the activation landscape to merge complementary signal domains.
7. **Final Decision:** A final XGBoost model ingests the FiLM-fused 128-dimensional embedding to resolve the uncertainty and emit the terminal classification.
