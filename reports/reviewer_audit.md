# Reviewer Audit

## Reviewer A: Machine Learning Review
**Methodological Correctness:** The cascaded architecture correctly balances structural and semantic evaluation using conformal routing.
**Feature Engineering:** Exactly 12 features are specified. The paper does not imply the use of external features like DNS or WHOIS.
**HXRF Stage 1:** 25 trees and max depth 20 are used.
**Conformal Routing (ICP):** Uses the same HXRF checkpoint for calibration and inference. Relies on the standard exchangeability assumption without claiming distribution-shift robustness. Empty set handles properly by routing to Stage-2. Alpha=0.05.
**Stage 2 (MiniLM + FiLM + XGBoost):** `all-MiniLM-L6-v2` is correctly identified as a pretrained sentence-transformer encoder (not a large language model). Truncated to 256 tokens. FiLM output dimension is 384. FiLM correctly modulates the semantic representation.
**Math/Equations:** 
$s(x,y) = 1 - p(y|x)$
$E_{FiLM} = \gamma(X_{12}) \odot E_{MiniLM} + \beta(X_{12})$ are correct.

## Reviewer B: Cybersecurity Review
**Threat Model:** Explicitly focuses on URL-only inference constraints (no dynamic webpage content, no DNS/WHOIS).
**Security Relevance:** Covers malicious URL identification well (phishing, malware).
**Adversarial Robustness:** Explicitly states that adversarial robustness against explicit evasion attacks was not evaluated. This is correctly preserved as a limitation. No unsupported production-readiness or sub-millisecond latency claims remain.

## Reviewer C: Statistics & Conformal Prediction
**Dataset Split:** The test set was kept separated and not used for alpha tuning.
**Conformal Guarantees:** Wording "Under the standard exchangeability assumption, the split-conformal construction provides the corresponding marginal coverage guarantee" is correct and precise.
**Statistical Tests:** McNemar's test is correctly interpreted as identifying asymmetry in paired classification disagreements, NOT as proving superior predictive performance.
