# Phase 11A — Feature Drift Analysis

Comparison of 12 Stage-1 features: Canonical Parquet vs Dynamically Extracted

| Feature | Exact Match % | Diff Rows | Max Abs Diff | Mean Abs Diff | Note |
|---|---:|---:|---:|---:|---|
| domain_len | 100.00% | 0 | 0.000000 | 0.000000 | Match |
| path_depth | 98.46% | 1947 | 2.000000 | 0.015470 | Logic Diff |
| special_count_v2 | 98.29% | 2160 | 78.000000 | 0.030142 | Logic Diff |
| token_count | 80.80% | 24299 | 148.000000 | 0.627563 | Logic Diff |
| url_entropy | 98.38% | 2046 | 1.492800 | 0.000335 | Logic Diff |
| url_len | 100.00% | 0 | 0.000000 | 0.000000 | Match |
| = | 100.00% | 0 | 0.000000 | 0.000000 | Match |
| . | 100.00% | 0 | 0.000000 | 0.000000 | Match |
| // | 100.00% | 0 | 0.000000 | 0.000000 | Match |
| abnormal_url | 100.00% | 2 | 1.000000 | 0.000016 | Logic Diff |
| phish_adv_hyphen_count | 94.59% | 6844 | 24.000000 | 0.084649 | Logic Diff |
| suspicious_extension | 98.98% | 1296 | 1.000000 | 0.010239 | Logic Diff |

Total rows with at least one difference: 31505 / 126569 (24.89%)
