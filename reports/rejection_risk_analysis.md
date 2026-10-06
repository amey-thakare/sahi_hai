# Rejection-Risk Analysis

This document identifies potential risks for rejection by peer reviewers at a top-tier venue, ranked by severity.

## CRITICAL
- **Citation Gaps in Related Work:** The Related Work section contains placeholder citations `[TODO: AUTHORITATIVE CITATION REQUIRED]`. This will result in an automatic desk reject if submitted as-is.
  - *Required fix:* Fill in authoritative references without inventing citations. (Can be done without changing research).

## HIGH
- **Stage-1 False Negatives Propagating to Final Result:** Reviewers may object that the conformal routing allows 5,043 FN to bypass semantic recovery entirely, meaning the architecture's reliance on Stage-1 structural confidence is a significant weakness.
  - *Required fix:* Ensure the Discussion strongly emphasizes this limitation (already addressed in Section 13). (Can be done without changing research).

## MEDIUM
- **Dataset Age & External Validity (Temporal Drift):** The datasets (ISCX-URL2016, PhishStorm) are relatively old, and malicious URL patterns change rapidly over time (distribution shift).
  - *Required fix:* Explicitly acknowledge temporal distribution shift and the need for continuous recalibration to maintain valid marginal coverage (already addressed in Section 12.2). (Can be done without changing research).
- **Adversarial Robustness:** Reviewers might question how easily an attacker can manipulate the 12 structural features to bypass the Stage-1 gatekeeper.
  - *Required fix:* Keep the explicit limitation that adversarial evasion attacks were not evaluated (already in Section 12.2). (Can be done without changing research).

## LOW
- **Lack of End-to-End Latency Measurements:** Reviewers may ask for precise latency/throughput metrics in milliseconds to prove the computational efficiency claims.
  - *Required fix:* Acknowledge that the computational analysis is currently based on routing volume reduction (avoiding Stage-2) rather than raw benchmarked latency. (Can be done without changing research).
- **Historical vs Final Comparisons:** Reviewers might be confused by Table III containing models evaluated on different subsets.
  - *Required fix:* The explicit table note distinguishes them and prevents confusion (already done). (Can be done without changing research).
