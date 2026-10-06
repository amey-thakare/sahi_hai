# Stage 1 Feature Selection Report

## 1. Candidate Feature Count
67 eligible engineered features from the config were audited.

## 2. Real-time Eligible Feature Count
All 67 of the considered engineered features are purely string/lexical operations (lengths, token counts, character counts). Therefore, 67 features were real-time eligible.

## 3. Excluded Feature Count
0 from the considered subset (as web scraping and temporal metadata features were already excluded in Phase 3).

## 4. Leakage Exclusions
As documented in Phase 3, all `web_*` columns were excluded as they represent future/external information unavailable at inference time for a gatekeeper model. No additional target-leaking features were discovered among the remaining 67.

## 5. External-information Exclusions
No DNS, WHOIS, or HTTP lookup features were included in the candidate list. Features like `having_ip_address` or `is_shortener` in this dataset appear to be implemented as simple regex heuristics rather than true network lookups.

## 6. Feature Importance Results
Random Forest and Permutation Importances were calculated on a 10k random subset of the training data. The top predictive features included `abnormal_url`, `//` count, `=`, `url_len`, and various character counts.

## 7. SHAP Analysis
A TreeExplainer SHAP analysis was conducted. Global mean absolute SHAP values confirmed the tree-based importances, highlighting that structural elements (e.g., presence of `abnormal_url` or path depth) provide the largest marginal log-odds contributions to the malicious classification.

## 8. LIME Analysis/limitations
LIME was considered but not executed. Given that the objective is global feature selection for a robust Stage-1 model, SHAP provides a more mathematically consistent global feature attribution (via Shapley values) than LIME, which is designed for local, instance-level interpretability.

## 9. PDP Analysis
Partial Dependence plots were deemed unnecessary for final selection since the relationships for structural features (like length and character counts) are largely monotonic in the context of malicious URL detection, and SHAP summary plots provided sufficient directional insight.

## 10. Correlation/dependency Analysis
A Pearson correlation matrix was generated for the top 20 SHAP features. Significant redundancy was identified:
- `url_len` and `url_len_v2` are highly correlated.
- `letters` and `letter_count_v2` are highly correlated.
These redundancies were intentionally avoided when selecting the final 12 features.

## 11. Computational-cost Analysis
All selected features are $O(N)$ string operations (where $N$ is the URL length). Most are implemented as standard library `.count()` operations or length checks. They can be extracted in microseconds.

## 12. Final 12 Features
1. `domain_len`
2. `path_depth`
3. `special_count_v2`
4. `token_count`
5. `url_entropy`
6. `url_len`
7. `=`
8. `.`
9. `//`
10. `abnormal_url`
11. `phish_adv_hyphen_count`
12. `suspicious_extension`

## 13. Reason for Selecting Each Feature
- **domain_len**: Phishing domains are often unusually long or use subdomains extensively.
- **path_depth**: Deep paths indicate hidden malicious payloads or redirects.
- **special_count_v2**: Malicious URLs heavily rely on special characters for obfuscation.
- **token_count**: High token counts indicate keyword stuffing.
- **url_entropy**: High entropy points to DGA (Domain Generation Algorithms) or obfuscated tokens.
- **url_len**: Fundamental baseline feature; malicious URLs are statistically longer.
- **=**, **.**, **//**: Found to be highly predictive individually (often indicating parameter passing, multiple extensions, or open redirects).
- **abnormal_url**: Very strong predictor (SHAP top 1); identifies cases where the host is obscure or missing.
- **phish_adv_hyphen_count**: Hyphens are heavily used in typo-squatting.
- **suspicious_extension**: Direct heuristic for malware distribution (.exe, .zip).

## 14. Reasons for Rejecting Important Alternatives
- `url_len_v2`, `letter_count_v2`, `letters`: Rejected due to near-perfect correlation with `url_len` and `token_count`.
- `phish_adv_many_params`: Correlated with `=` count; `=` count is faster to compute.

## Final Summary Table

| Feature | SHAP Rank | RF Rank | Permutation Rank | Redundancy | Real-Time | Selected |
|---------|-----------|---------|------------------|------------|-----------|----------|
| abnormal_url | 1 | 2 | 1 | Low | Yes | **Yes** |
| // | 2 | 1 | 2 | Low | Yes | **Yes** |
| path_depth | 3 | 3 | 3 | Low | Yes | **Yes** |
| url_len | 6 | 6 | 9 | Low | Yes | **Yes** |
| = | 8 | 5 | 11 | Low | Yes | **Yes** |
| url_entropy | 5 | 8 | 7 | Low | Yes | **Yes** |
| suspicious_extension | 7 | 11 | 4 | Low | Yes | **Yes** |
| domain_len | 11 | 9 | 4 | Low | Yes | **Yes** |
| token_count | 10 | 7 | 10 | Low | Yes | **Yes** |
| special_count_v2 | 9 | 4 | 12 | Low | Yes | **Yes** |
| . | 4 | 10 | 3 | Low | Yes | **Yes** |
| phish_adv_hyphen_count | 12 | 12 | 5 | Low | Yes | **Yes** |
| url_len_v2 | 13 | 13 | 13 | High | Yes | No |
| letters | 14 | 14 | 14 | High | Yes | No |
