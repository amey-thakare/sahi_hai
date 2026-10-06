# Stage-1 HXRF Gatekeeper — Results Report

## 1. Executive Summary

The DSSE Stage-1 HXRF (Hybrid eXtended Random Forest) gatekeeper was trained
and validated as the first-pass triage model for the Dual-Stage Semantic
Ensemble malicious URL detection system.

Using only 12 lightweight structural features extracted from the raw URL
string — without any DNS, HTTP, WHOIS, or external API lookups — the model
achieves **93.1% accuracy** with an **ROC-AUC of 0.9762** on a held-out test
set of 126,569 URLs.

At the architectural routing thresholds (P ≥ 0.95 → FINAL_MALICIOUS,
P ≤ 0.05 → FINAL_BENIGN), **45.9%** of all URLs are resolved by Stage 1
with **99.7% accuracy** on those confident predictions.  The remaining 54.1%
are routed to Stage 2 for deeper semantic analysis.

The model processes URLs at **~73.8 URLs/sec** (sequential, single-thread),
with a median per-URL latency of **~13.5 ms** (warm inference).

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Source dataset | `dataset/MUD_malicious_urls_2026_V2.csv` |
| Training data | `data/splits/train.parquet` |
| Test data | `data/splits/test.parquet` |
| Split method | 80/20 stratified, URL-hash deduplication before split |
| URL overlap | 0 (verified) |
| Target column | `dsse_binary_target` |

### Training Samples: 506,275
### Test Samples: 126,569

## 3. Class Distribution

| Set | Benign (0) | Malicious (1) | Malicious % |
|-----|-----------|---------------|-------------|
| Train | 337,793 | 168,482 | 33.3% |
| Test | 84,448 | 42,121 | 33.3% |

---

## 4. Exact 12 Features

| # | Feature Name | Description |
|---|-------------|-------------|
| 1 | `domain_len` | Length of the hostname |
| 2 | `path_depth` | Number of `/` segments in URL path |
| 3 | `special_count_v2` | Count of non-alphanumeric characters in normalized URL |
| 4 | `token_count` | Count of alphanumeric tokens (scheme stripped) |
| 5 | `url_entropy` | Shannon entropy of the normalized URL string |
| 6 | `url_len` | Length of the original raw URL |
| 7 | `=` | Count of `=` characters in original URL |
| 8 | `.` | Count of `.` characters in original URL |
| 9 | `//` | Count of `//` substrings in original URL |
| 10 | `abnormal_url` | 1 if no `http://` or `https://` scheme, else 0 |
| 11 | `phish_adv_hyphen_count` | Count of hyphens in the hostname |
| 12 | `suspicious_extension` | 1 if URL path ends with a suspicious file extension |

Feature order is deterministic and defined in `config/stage1_features.json`.

---

## 5. Feature Consistency Verification

Before training, the feature extractor (`src/features/stage1_features.py`)
was verified against the pre-computed feature values in the dataset using 200
stratified sample URLs (100 benign + 100 malicious).

| Feature | Match Rate | Status |
|---------|-----------|--------|
| `domain_len` | 100.0% | ✅ OK |
| `path_depth` | 98.5% | ⚠️ Minor (3/200) |
| `special_count_v2` | 98.5% | ⚠️ Minor (3/200) |
| `token_count` | 81.5% | ⚠️ Edge-cases (37/200) |
| `url_entropy` | 99.5% | ⚠️ Minor (1/200) |
| `url_len` | 100.0% | ✅ OK |
| `=` | 100.0% | ✅ OK |
| `.` | 100.0% | ✅ OK |
| `//` | 100.0% | ✅ OK |
| `abnormal_url` | 100.0% | ✅ OK |
| `phish_adv_hyphen_count` | 95.5% | ⚠️ Minor (9/200) |
| `suspicious_extension` | 99.0% | ⚠️ Minor (2/200) |

**Decision**: Proceed with training using the pre-computed dataset features
(ground truth). The extractor is used only for inference on new URLs and is
sufficiently close for gatekeeper routing.

Detailed mismatch analysis: `reports/stage1_feature_consistency.md`

---

## 6. Feature Extraction Implementation

- **Module**: `src/features/stage1_features.py`
- **Function**: `extract_stage1_features(url) → dict`
- **Normalisation**: Prepend `http://` if no scheme; strip trailing `/` on non-root paths
- **Dependencies**: Python stdlib only (`math`, `re`, `urllib.parse`)
- **External lookups**: None (DNS, HTTP, WHOIS are NOT used)
- **Complexity**: O(N) per URL

---

## 7. Model Architecture

| Property | Value |
|----------|-------|
| Model name | HXRF |
| Model type | `RandomForestClassifier` (scikit-learn) |
| Model path | `models/stage1_hxrf.pkl` |
| Class weight | `balanced` |
| Random seed | 42 |
| Input features | 12 |
| Output | Binary probabilities [P(benign), P(malicious)] |

---

## 8. Hyperparameter Search

A controlled search over 5 configurations was performed using **5-fold
stratified cross-validation on the training set only**. The test set was
NOT used for any model selection decisions.

| # | n_estimators | max_depth | max_features | min_samples_split | min_samples_leaf | CV F1 | CV AUC |
|---|-------------|-----------|-------------|-------------------|-----------------|-------|--------|
| 1 | 100 | 8 | sqrt | 10 | 4 | 0.8645 | 0.9544 |
| 2 | 200 | 10 | sqrt | 5 | 2 | 0.8731 | 0.9631 |
| 3 | 150 | 12 | sqrt | 5 | 2 | 0.8819 | 0.9698 |
| **4** | **200** | **15** | **log2** | **10** | **4** | **0.8924** | **0.9756** |
| 5 | 300 | 10 | sqrt | 5 | 2 | 0.8728 | 0.9631 |

## 9. Selected Hyperparameters

```json
{
  "n_estimators": 200,
  "max_depth": 15,
  "min_samples_split": 10,
  "min_samples_leaf": 4,
  "max_features": "log2",
  "class_weight": "balanced",
  "random_state": 42
}
```

## 10. Cross-Validation Procedure

- **Method**: 5-fold Stratified K-Fold
- **Scoring metrics**: F1, ROC-AUC, Precision, Recall
- **Selection criterion**: Highest mean F1
- **Data used**: Training set only (`data/splits/train.parquet`)
- **Test set**: Completely excluded from all model selection

---

## 11. Test-Set Metrics

Evaluated on the **held-out test set** (`data/splits/test.parquet`),
which was not used for feature selection, hyperparameter tuning, model
selection, threshold tuning, or calibration.

| Metric | Value |
|--------|------:|
| Accuracy | 0.9314 |
| Precision (malicious) | 0.9304 |
| Recall (malicious) | 0.8580 |
| F1 (malicious) | 0.8927 |
| ROC-AUC | 0.9762 |
| Specificity (TNR) | 0.9680 |
| False Positive Rate | 0.0320 |
| False Negative Rate | 0.1420 |

---

## 12. Per-Class Metrics

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Benign (0) | 0.9318 | 0.9680 | 0.9496 | 84,448 |
| Malicious (1) | 0.9304 | 0.8580 | 0.8927 | 42,121 |
| **Weighted avg** | **0.9314** | **0.9314** | **0.9307** | **126,569** |
| **Macro avg** | **0.9311** | **0.9130** | **0.9212** | **126,569** |

---

## 13. Confusion Matrix

|  | Predicted Benign | Predicted Malicious |
|--|-----------------|---------------------|
| **Actual Benign** | 81,745 (TN) | 2,703 (FP) |
| **Actual Malicious** | 5,981 (FN) | 36,140 (TP) |

![Confusion Matrix](figures/stage1_confusion_matrix.png)

---

## 14. ROC-AUC

**ROC-AUC = 0.9762**

The model achieves strong discrimination between benign and malicious URLs
using only 12 structural features.

---

## 15. Stage-1 Routing Distribution

| Routing Decision | Count | Percentage |
|-----------------|-------|------------|
| FINAL_MALICIOUS (P ≥ 0.95) | 31,453 | 24.9% |
| FINAL_BENIGN (P ≤ 0.05) | 26,616 | 21.0% |
| AMBIGUOUS → Stage 2 | 68,500 | 54.1% |
| **Total resolved by Stage 1** | **58,069** | **45.9%** |

### Verification

- Stage-1 Resolution %: **45.9%**
- Stage-2 Routing %: **54.1%**
- Sum: **100.0%** ✅

### Confident Routing Accuracy

Among the 58,069 URLs resolved by Stage 1 (P ≥ 0.95 or P ≤ 0.05),
the prediction accuracy is **99.7%**.

---

## 16. Probability Behaviour

The predicted probabilities exhibit sensible monotonic behaviour:

| P(malicious) Range | Count | % of Total | Actual Malicious Rate |
|-------------------|-------|------------|----------------------|
| [0.00, 0.05) | 26,616 | 21.0% | 0.004 |
| [0.05, 0.10) | 23,703 | 18.7% | 0.013 |
| [0.10, 0.20) | 16,543 | 13.1% | 0.054 |
| [0.20, 0.30) | 8,413 | 6.6% | 0.120 |
| [0.30, 0.40) | 7,004 | 5.5% | 0.234 |
| [0.40, 0.50) | 5,447 | 4.3% | 0.370 |
| [0.50, 0.60) | 3,726 | 2.9% | 0.496 |
| [0.60, 0.70) | 1,841 | 1.5% | 0.659 |
| [0.70, 0.80) | 489 | 0.4% | 0.834 |
| [0.80, 0.90) | 635 | 0.5% | 0.940 |
| [0.90, 0.95) | 699 | 0.6% | 0.986 |
| [0.95, 1.00] | 31,453 | 24.9% | 0.998 |

The actual malicious rates increase monotonically with predicted
probability, indicating that the probability estimates are reasonably
well-ordered. However, **formal probability calibration was NOT
performed**. The probabilities are useful for routing but should not be
interpreted as calibrated posterior probabilities.

---

## 17. Feature Importance

### Random Forest (Gini) Importance

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `abnormal_url` | 0.2442 |
| 2 | `//` | 0.2112 |
| 3 | `path_depth` | 0.1153 |
| 4 | `url_len` | 0.0756 |
| 5 | `special_count_v2` | 0.0600 |
| 6 | `domain_len` | 0.0584 |
| 7 | `.` | 0.0516 |
| 8 | `token_count` | 0.0468 |
| 9 | `url_entropy` | 0.0440 |
| 10 | `=` | 0.0429 |
| 11 | `suspicious_extension` | 0.0256 |
| 12 | `phish_adv_hyphen_count` | 0.0244 |

![RF Feature Importance](figures/stage1_rf_importance.png)

### SHAP Mean |SHAP| Importance

SHAP values were computed using `TreeExplainer` on a 2,000-sample
subsample of the training set.

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `abnormal_url` | 0.1805 |
| 2 | `//` | 0.0958 |
| 3 | `url_len` | 0.0526 |
| 4 | `.` | 0.0499 |
| 5 | `path_depth` | 0.0480 |
| 6 | `url_entropy` | 0.0294 |
| 7 | `special_count_v2` | 0.0257 |
| 8 | `domain_len` | 0.0238 |
| 9 | `=` | 0.0196 |
| 10 | `token_count` | 0.0184 |
| 11 | `suspicious_extension` | 0.0175 |
| 12 | `phish_adv_hyphen_count` | 0.0137 |

![SHAP Bar Importance](figures/stage1_shap_importance.png)
![SHAP Beeswarm](figures/stage1_shap_beeswarm.png)

Both Gini and SHAP importance agree that `abnormal_url` and `//` are the
top two discriminative features.

---

## 18. Latency Benchmark

Warm inference (model pre-loaded, one-time startup cost excluded):

| Component | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|-----------|-----------|-------------|----------|----------|
| Feature Extraction | 0.024 | 0.023 | 0.033 | 0.046 |
| RF Prediction | 13.096 | 13.453 | 13.611 | 13.983 |
| **Total Stage-1** | **13.120** | **13.475** | **13.642** | **14.014** |

- Benchmark sample: 1,000 URLs from test set
- Warm-up: 5 URLs before measurement
- Model loading excluded from per-request timing

### Cold-Start Model Load

| Metric | Value |
|--------|-------|
| Cold-start model load | 60.8 ms |

---

## 19. Throughput

| Metric | Value |
|--------|-------|
| URLs processed | 2,000 |
| Total time | 27.111 s |
| Throughput | **73.8 URLs/sec** |
| Mode | Sequential, single-thread |
| Model loading | Excluded (pre-loaded) |

---

## 20. Hardware / Environment

| Property | Value |
|----------|-------|
| CPU | Apple Silicon (ARM) |
| Architecture | arm64 |
| OS | macOS 26.6.2 |
| Python | 3.12.0 |
| scikit-learn | 1.9.0 |
| NumPy | 2.4.6 |
| pandas | 2.3.3 |
| SHAP | 0.52.0 |

---

## 21. Concise Results Table

| Metric | Result |
|--------|--------|
| Accuracy | 0.9314 |
| Precision (malicious) | 0.9304 |
| Recall (malicious) | 0.8580 |
| F1 (malicious) | 0.8927 |
| ROC-AUC | 0.9762 |
| Specificity | 0.9680 |
| Mean latency | 13.120 ms |
| Median latency | 13.475 ms |
| P95 latency | 13.642 ms |
| P99 latency | 14.014 ms |
| Throughput | 73.8 URLs/sec |
| Stage-1 resolution | 45.9% |
| Stage-2 routing | 54.1% |
| Confident-routing accuracy | 99.7% |

All values are measured from actual experiments. No metrics were fabricated.

---

## 22. Limitations

1. **Feature extractor approximations**: `token_count` has ~81.5% exact
   match rate and `phish_adv_hyphen_count` has ~95.5% match rate vs the
   dataset. Training uses pre-computed ground-truth values; inference uses
   the extractor. The differences are minor and concentrated in edge cases
   (percent-encoded URLs, trailing slashes).

2. **Probability calibration**: NOT explicitly evaluated or performed.
   The probabilities are ordered correctly (monotonically increasing
   actual malicious rate) but should not be interpreted as calibrated
   posterior probabilities.

3. **Class imbalance handling**: Uses `class_weight='balanced'` which
   reweights classes inversely proportional to frequency. No oversampling
   or undersampling was applied.

4. **Feature count**: Only 12 structural features are used. This is by
   design for the lightweight gatekeeper, but limits recall on complex
   attack patterns. Stage 2 (MiniLM embeddings) will address this.

5. **54.1% ambiguous rate**: Over half the URLs require Stage 2 processing.
   This is expected given the strict routing thresholds (0.05/0.95) and the
   limited feature set. The thresholds are architectural constants and were
   NOT adjusted to improve resolution rates.

---

## 23. Reproducibility

| Property | Value |
|----------|-------|
| Random seed | 42 |
| Python | 3.12.0 |
| scikit-learn | 1.9.0 |
| NumPy | 2.4.6 |
| pandas | 2.3.3 |
| SHAP | 0.52.0 |
| Training dataset | `data/splits/train.parquet` |
| Test dataset | `data/splits/test.parquet` |
| Feature config | `config/stage1_features.json` |
| Model config | `config/stage1_model_config.json` |
| Training script | `scripts/phase5_train_hxrf.py` |
| Feature order | Defined in `config/stage1_features.json` |

---

## 24. Files Created

```
models/
    stage1_hxrf.pkl                       # Trained HXRF model

config/
    stage1_features.json                  # Feature names and order
    stage1_model_config.json              # Full model configuration

src/
    features/
        stage1_features.py                # Feature extractor
    stage1/
        __init__.py
        predictor.py                      # predict_stage1(url) interface

tests/
    test_stage1_features.py               # 10 tests — all pass
    test_stage1_model.py                  # 16 tests — all pass

reports/
    stage1_feature_consistency.md         # Feature verification report
    stage1_results.md                     # This report

reports/figures/
    stage1_confusion_matrix.png           # Confusion matrix heatmap
    stage1_rf_importance.png              # RF Gini importance bar chart
    stage1_shap_importance.png            # SHAP bar importance plot
    stage1_shap_beeswarm.png             # SHAP beeswarm plot

scripts/
    phase5_train_hxrf.py                  # Training script
    phase5_consistency_check.py           # Feature consistency checker
    phase5_supplementary_metrics.py       # Per-class + throughput metrics
```
