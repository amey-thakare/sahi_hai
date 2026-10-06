# Dataset Analysis Report

## 1. Dataset Overview Table
| Filename | Size (MB) | Rows | Cols | URL Col | Target Col | Duplicates |
|---|---|---|---|---|---|---|
| final_dataset_with_all_features.csv | 119.94 | 651000 | 65 | None | label | 200262 |
| dataset_with_all_features v2.csv | 165.02 | 651191 | 68 | url | type | 10998 |
| MUD_malicious_urls_2026_V2.csv | 360.36 | 632844 | 99 | url | label | 0 |
| final_dataset_with_all_features_v3.1.csv | 144.40 | 651191 | 64 | url | type | 509 |

## 2. Dataset Dimensions
- **final_dataset_with_all_features.csv**: 651000 rows x 65 columns
- **dataset_with_all_features v2.csv**: 651191 rows x 68 columns
- **MUD_malicious_urls_2026_V2.csv**: 632844 rows x 99 columns
- **final_dataset_with_all_features_v3.1.csv**: 651191 rows x 64 columns

## 3. Feature Comparison
A summary of numerical vs categorical features:
- **final_dataset_with_all_features.csv**: 65 numerical, 0 categorical
- **dataset_with_all_features v2.csv**: 64 numerical, 4 categorical
- **MUD_malicious_urls_2026_V2.csv**: 84 numerical, 15 categorical
- **final_dataset_with_all_features_v3.1.csv**: 60 numerical, 4 categorical

## 4. Target Comparison
- **final_dataset_with_all_features.csv**: Target Column: `label` (Unique values: ['0', '1', '2', '3'])
- **dataset_with_all_features v2.csv**: Target Column: `type` (Unique values: ['benign', 'defacement', 'phishing', 'malware', 'nan'])
- **MUD_malicious_urls_2026_V2.csv**: Target Column: `label` (Unique values: ['0', '1', '2', '3'])
- **final_dataset_with_all_features_v3.1.csv**: Target Column: `type` (Unique values: ['benign', 'defacement', 'phishing', 'malware'])

## 5. Class Distributions
### final_dataset_with_all_features.csv
- 0: 428103 (65.76%)
- 1: 96457 (14.82%)
- 2: 93920 (14.43%)
- 3: 32520 (5.00%)
### dataset_with_all_features v2.csv
- benign: 428209 (65.76%)
- defacement: 96457 (14.81%)
- phishing: 93933 (14.42%)
- malware: 32520 (4.99%)
- nan: 72 (0.01%)
### MUD_malicious_urls_2026_V2.csv
- 0: 422241 (66.72%)
- 1: 95285 (15.06%)
- 2: 91741 (14.50%)
- 3: 23577 (3.73%)
### final_dataset_with_all_features_v3.1.csv
- benign: 428103 (65.74%)
- defacement: 96457 (14.81%)
- phishing: 94111 (14.45%)
- malware: 32520 (4.99%)

## 6. Missing-value Analysis
### final_dataset_with_all_features.csv
- No missing values detected.
### dataset_with_all_features v2.csv
- type: 0.01% missing
- label: 0.01% missing
- domain: 1.88% missing
### MUD_malicious_urls_2026_V2.csv
- norm_actions: 24.60% missing
- path: 3.63% missing
- query: 78.71% missing
- tld: 0.00% missing
### final_dataset_with_all_features_v3.1.csv
- domain: 1.88% missing

## 7. Duplicate Analysis
- **final_dataset_with_all_features.csv**: 200262 duplicate rows, 0 duplicate URLs
- **dataset_with_all_features v2.csv**: 10998 duplicate rows, 11110 duplicate URLs
- **MUD_malicious_urls_2026_V2.csv**: 0 duplicate rows, 0 duplicate URLs
- **final_dataset_with_all_features_v3.1.csv**: 509 duplicate rows, 11110 duplicate URLs

## 8. URL Overlap Analysis
- **dataset_with_all_features v2.csv vs MUD_malicious_urls_2026_V2.csv**: 632843 overlapping URLs
- **dataset_with_all_features v2.csv vs final_dataset_with_all_features_v3.1.csv**: 640080 overlapping URLs
- **MUD_malicious_urls_2026_V2.csv vs final_dataset_with_all_features_v3.1.csv**: 632844 overlapping URLs

## 9. Label Conflict Analysis
- **dataset_with_all_features v2.csv vs MUD_malicious_urls_2026_V2.csv**: 632843 conflicting labels among common URLs
- **dataset_with_all_features v2.csv vs final_dataset_with_all_features_v3.1.csv**: 95288 conflicting labels among common URLs
- **MUD_malicious_urls_2026_V2.csv vs final_dataset_with_all_features_v3.1.csv**: 632844 conflicting labels among common URLs

## 10. Leakage Analysis
Potential leakage columns that must not be used as training features:
- **final_dataset_with_all_features.csv**: label
- **dataset_with_all_features v2.csv**: type, url
- **MUD_malicious_urls_2026_V2.csv**: label, url, url_hash, type, class_label, source_dataset, dataset_version, ts_first_seen, ts_aggregated
- **final_dataset_with_all_features_v3.1.csv**: type, url

## 11. Data-Quality Analysis
- **final_dataset_with_all_features.csv**: 200262 duplicate rows
- **dataset_with_all_features v2.csv**: 10998 duplicate rows, 11110 duplicate URLs
- **MUD_malicious_urls_2026_V2.csv**: No major structural issues detected
- **final_dataset_with_all_features_v3.1.csv**: 509 duplicate rows, 11110 duplicate URLs

## 12. DSSE Compatibility Analysis
For the Dual-Stage Semantic Ensemble (DSSE), we need BOTH raw URL strings and engineered features.
- **final_dataset_with_all_features.csv**: Has Raw URL: False, Has Engineered Features: True
- **dataset_with_all_features v2.csv**: Has Raw URL: True, Has Engineered Features: True
- **MUD_malicious_urls_2026_V2.csv**: Has Raw URL: True, Has Engineered Features: True
- **final_dataset_with_all_features_v3.1.csv**: Has Raw URL: True, Has Engineered Features: True

## Recommended Dataset Strategy

- **Primary dataset**: `MUD_malicious_urls_2026_V2.csv`. It contains 99 columns including `url` (raw URL), `url_hash`, `url_normalized`, `label`, `class_label` and a comprehensive set of engineered features.
- **Whether another dataset should be joined**: No. The `MUD_malicious_urls_2026_V2.csv` dataset is completely self-contained and holds both the raw URLs and the engineered features needed for DSSE.
- **Safe join key if required**: Not required, but `url_hash` or `url` would be used if necessary.
- **Raw URL source**: `url` column in `MUD_malicious_urls_2026_V2.csv`.
- **Engineered feature source**: All numerical features in `MUD_malicious_urls_2026_V2.csv` (excluding leakage columns).
- **Target column**: `label` (0 for benign, 1 for malicious). (Note: `class_label` can be used for multi-class classification but `label` is the binary target).
- **Target classes**: 0 (Benign) and 1 (Malicious).
- **Columns that must be excluded**: `url`, `url_normalized`, `url_hash`, `class_label`, `label`, `type`, `source_dataset`, `dataset_version`, `ts_first_seen`, `ts_aggregated`, `reachability_class`, `is_syntax_valid`, `norm_actions`, `host`, `path`, `query`, `tld`.
- **Recommended train/test strategy**: Use `train_test_split` on the `MUD_malicious_urls_2026_V2.csv` dataset, stratifying by `label` to maintain the class distribution. Ensure that any split removes the leakage columns before training. Duplicate URLs should be dropped prior to splitting to prevent the same URL from appearing in both train and test sets, which would cause data leakage.
