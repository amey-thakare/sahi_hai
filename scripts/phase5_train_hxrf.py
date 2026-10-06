"""
Phase 5 — Train and validate the DSSE Stage-1 HXRF gatekeeper.

Uses the pre-computed 12 structural features from the training parquet.
Performs hyperparameter search with stratified cross-validation.
Evaluates on the held-out test set.
Generates reports, plots, and model artefacts.
"""
import sys, os, json, time, pickle, platform
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import shap

from src.features.stage1_features import extract_stage1_features

# ── Configuration ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

with open('config/stage1_features.json') as f:
    stage1_cfg = json.load(f)
FEATURES = stage1_cfg['features']
TARGET = 'dsse_binary_target'

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading data ...")
train_df = pd.read_parquet('data/splits/train.parquet')
test_df  = pd.read_parquet('data/splits/test.parquet')

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].copy()
X_test  = test_df[FEATURES].copy()
y_test  = test_df[TARGET].copy()

# Replace inf/NaN with 0 (defensive)
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"  Train: {len(X_train)}  ({y_train.mean():.3f} malicious)")
print(f"  Test:  {len(X_test)}   ({y_test.mean():.3f} malicious)")

# ── 2. Hyperparameter search (stratified 5-fold CV, train only) ───────────
print("\nHyperparameter search (stratified 5-fold CV) ...")
param_grid = [
    {'n_estimators': 100, 'max_depth': 8,  'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt'},
    {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5,  'min_samples_leaf': 2, 'max_features': 'sqrt'},
    {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 5,  'min_samples_leaf': 2, 'max_features': 'sqrt'},
    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2'},
    {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 5,  'min_samples_leaf': 2, 'max_features': 'sqrt'},
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
best_score = -1
best_params = None
cv_results_list = []

for i, params in enumerate(param_grid):
    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **params,
    )
    scores = cross_validate(
        rf, X_train, y_train, cv=cv,
        scoring=['f1', 'roc_auc', 'precision', 'recall'],
        n_jobs=-1, return_train_score=False,
    )
    mean_f1  = scores['test_f1'].mean()
    mean_auc = scores['test_roc_auc'].mean()
    cv_results_list.append({
        'params': params,
        'mean_f1': mean_f1,
        'std_f1':  scores['test_f1'].std(),
        'mean_auc': mean_auc,
        'std_auc':  scores['test_roc_auc'].std(),
        'mean_precision': scores['test_precision'].mean(),
        'mean_recall':    scores['test_recall'].mean(),
    })
    print(f"  [{i+1}/{len(param_grid)}] F1={mean_f1:.4f} ± {scores['test_f1'].std():.4f}  "
          f"AUC={mean_auc:.4f}  P={scores['test_precision'].mean():.4f}  R={scores['test_recall'].mean():.4f}  "
          f"(n={params['n_estimators']}, d={params['max_depth']})")
    if mean_f1 > best_score:
        best_score = mean_f1
        best_params = params

print(f"\n  Best config: {best_params}  (CV F1={best_score:.4f})")

# ── 3. Train final model on full training set ────────────────────────────
print("\nTraining final model ...")
final_rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1,
    **best_params,
)
final_rf.fit(X_train, y_train)

# ── 4. Save model ────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
with open('models/stage1_hxrf.pkl', 'wb') as f:
    pickle.dump(final_rf, f)
print("  Model saved to models/stage1_hxrf.pkl")

model_config = {
    'model_type': 'RandomForestClassifier',
    'model_path': 'models/stage1_hxrf.pkl',
    'features': FEATURES,
    'feature_count': len(FEATURES),
    'target_column': TARGET,
    'class_mapping': {0: 'benign', 1: 'malicious'},
    'random_seed': RANDOM_SEED,
    'hyperparameters': best_params,
    'class_weight': 'balanced',
    'training_dataset': 'data/splits/train.parquet',
    'training_samples': len(X_train),
    'feature_order': FEATURES,
}
with open('config/stage1_model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)
print("  Config saved to config/stage1_model_config.json")

# ── 5. Test-set evaluation ───────────────────────────────────────────────
print("\nTest-set evaluation ...")
y_pred = final_rf.predict(X_test)
y_prob = final_rf.predict_proba(X_test)[:, 1]  # P(malicious)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"  Accuracy:    {acc:.4f}")
print(f"  Precision:   {prec:.4f}")
print(f"  Recall:      {rec:.4f}")
print(f"  F1:          {f1:.4f}")
print(f"  ROC-AUC:     {auc:.4f}")
print(f"  Specificity: {specificity:.4f}")
print(f"  Confusion matrix:\n{cm}")

# ── 6. Routing distribution ──────────────────────────────────────────────
TAU_HIGH = 0.95
TAU_LOW  = 0.05

final_mal  = (y_prob >= TAU_HIGH).sum()
final_ben  = (y_prob <= TAU_LOW).sum()
ambiguous  = len(y_prob) - final_mal - final_ben
resolved   = final_mal + final_ben

print(f"\n  Stage-1 routing (tau_high={TAU_HIGH}, tau_low={TAU_LOW}):")
print(f"    Confident MALICIOUS (P ≥ {TAU_HIGH}):  {final_mal} ({final_mal/len(y_prob)*100:.1f}%)")
print(f"    Confident BENIGN    (P ≤ {TAU_LOW}):   {final_ben} ({final_ben/len(y_prob)*100:.1f}%)")
print(f"    AMBIGUOUS → Stage 2:                 {ambiguous} ({ambiguous/len(y_prob)*100:.1f}%)")
print(f"    Total resolved by Stage 1:           {resolved} ({resolved/len(y_prob)*100:.1f}%)")

# ── 7. Latency benchmark ────────────────────────────────────────────────
print("\nLatency benchmark ...")
sample_urls = test_df['url'].sample(min(1000, len(test_df)), random_state=RANDOM_SEED).tolist()

# Warm up
for u in sample_urls[:5]:
    feats = extract_stage1_features(u)
    final_rf.predict_proba(pd.DataFrame([feats])[FEATURES])

extraction_times = []
prediction_times = []
total_times = []

for u in sample_urls:
    # Feature extraction
    t0 = time.perf_counter()
    feats = extract_stage1_features(u)
    t1 = time.perf_counter()
    extraction_times.append((t1 - t0) * 1000)

    # Prediction
    fv = pd.DataFrame([feats])[FEATURES]
    t2 = time.perf_counter()
    final_rf.predict_proba(fv)
    t3 = time.perf_counter()
    prediction_times.append((t3 - t2) * 1000)

    total_times.append((t1 - t0 + t3 - t2) * 1000)

def pstats(arr, name):
    a = np.array(arr)
    return {
        'mean': np.mean(a), 'median': np.median(a),
        'p95': np.percentile(a, 95), 'p99': np.percentile(a, 99),
    }

ext_stats = pstats(extraction_times, 'extraction')
pred_stats = pstats(prediction_times, 'prediction')
total_stats = pstats(total_times, 'total')

print(f"  Feature extraction: mean={ext_stats['mean']:.3f}ms  med={ext_stats['median']:.3f}ms  "
      f"P95={ext_stats['p95']:.3f}ms  P99={ext_stats['p99']:.3f}ms")
print(f"  RF prediction:      mean={pred_stats['mean']:.3f}ms  med={pred_stats['median']:.3f}ms  "
      f"P95={pred_stats['p95']:.3f}ms  P99={pred_stats['p99']:.3f}ms")
print(f"  Total Stage-1:      mean={total_stats['mean']:.3f}ms  med={total_stats['median']:.3f}ms  "
      f"P95={total_stats['p95']:.3f}ms  P99={total_stats['p99']:.3f}ms")

hw_info = f"{platform.processor()} / {platform.machine()} / Python {platform.python_version()}"
print(f"  Hardware: {hw_info}")

# ── 8. Feature importance / SHAP ─────────────────────────────────────────
print("\nFeature importance & SHAP ...")
os.makedirs('reports/figures', exist_ok=True)

# RF built-in importance
fi = pd.Series(final_rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
fi.plot.barh(ax=ax, color='steelblue')
ax.set_xlabel('Gini Importance')
ax.set_title('Stage-1 HXRF Feature Importance')
ax.invert_yaxis()
plt.tight_layout()
fig.savefig('reports/figures/stage1_rf_importance.png', dpi=150)
plt.close(fig)

# SHAP (on 2k sample for speed)
shap_sample = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_SEED)
explainer = shap.TreeExplainer(final_rf)
shap_values = explainer.shap_values(shap_sample)
if isinstance(shap_values, list):
    shap_mal = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_mal = shap_values[:, :, 1]
else:
    shap_mal = shap_values

fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_mal, shap_sample, plot_type='bar', show=False)
plt.tight_layout()
fig.savefig('reports/figures/stage1_shap_importance.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_mal, shap_sample, show=False)
plt.tight_layout()
fig.savefig('reports/figures/stage1_shap_beeswarm.png', dpi=150)
plt.close(fig)

mean_abs_shap = pd.Series(np.abs(shap_mal).mean(axis=0), index=FEATURES).sort_values(ascending=False)

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'], ax=ax)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title('Stage-1 HXRF Confusion Matrix (Test Set)')
plt.tight_layout()
fig.savefig('reports/figures/stage1_confusion_matrix.png', dpi=150)
plt.close(fig)

# ── 9. Generate report ──────────────────────────────────────────────────
report = f"""# Stage-1 HXRF Gatekeeper — Results Report

## 1. Dataset
- **Training data**: `data/splits/train.parquet`
- **Test data**: `data/splits/test.parquet`
- **Training samples**: {len(X_train):,}
- **Test samples**: {len(X_test):,}

## 2. Class Distribution
| Set | Benign | Malicious | Mal % |
|-----|--------|-----------|-------|
| Train | {(y_train == 0).sum():,} | {(y_train == 1).sum():,} | {y_train.mean()*100:.1f}% |
| Test  | {(y_test == 0).sum():,} | {(y_test == 1).sum():,} | {y_test.mean()*100:.1f}% |

## 3. Feature Consistency
Feature extraction consistency was verified against 200 sample URLs.
- 7/12 features: 100% match
- Remaining 5 features: 81–99.5% match (minor edge-case differences documented in `reports/stage1_feature_consistency.md`)
- **Decision**: Proceed with training using pre-computed dataset features. Inference-time extractor is close enough for gatekeeper routing.

## 4. Exact 12 Features
{chr(10).join(f'{i+1}. `{f}`' for i, f in enumerate(FEATURES))}

## 5. Model Configuration
- **Model**: RandomForestClassifier
- **Class weight**: balanced
- **Random seed**: {RANDOM_SEED}
- **Hyperparameters**: {json.dumps(best_params, indent=2)}

## 6. Cross-Validation Results (5-fold, training set only)

| Config | n_est | max_depth | Mean F1 | Mean AUC | Mean Prec | Mean Recall |
|--------|-------|-----------|---------|----------|-----------|-------------|
"""

for i, res in enumerate(cv_results_list):
    p = res['params']
    marker = ' **← best**' if p == best_params else ''
    report += (f"| {i+1} | {p['n_estimators']} | {p['max_depth']} | "
               f"{res['mean_f1']:.4f} | {res['mean_auc']:.4f} | "
               f"{res['mean_precision']:.4f} | {res['mean_recall']:.4f} |{marker}\n")

report += f"""

## 7. Test-Set Metrics

| Metric | Value |
|--------|------:|
| Accuracy | {acc:.4f} |
| Precision | {prec:.4f} |
| Recall | {rec:.4f} |
| F1 | {f1:.4f} |
| ROC-AUC | {auc:.4f} |
| Specificity | {specificity:.4f} |

## 8. Confusion Matrix

|  | Predicted Benign | Predicted Malicious |
|--|-----------------|---------------------|
| **Actual Benign** | {tn:,} | {fp:,} |
| **Actual Malicious** | {fn:,} | {tp:,} |

![Confusion Matrix](figures/stage1_confusion_matrix.png)

## 9. Feature Importance

### Random Forest (Gini) Importance
{chr(10).join(f'- `{f}`: {v:.4f}' for f, v in fi.items())}

### SHAP Mean Absolute Importance
{chr(10).join(f'- `{f}`: {v:.4f}' for f, v in mean_abs_shap.items())}

![RF Importance](figures/stage1_rf_importance.png)
![SHAP Importance](figures/stage1_shap_importance.png)
![SHAP Beeswarm](figures/stage1_shap_beeswarm.png)

## 10. Stage-1 Routing Distribution

| Routing | Count | Percentage |
|---------|-------|------------|
| FINAL_MALICIOUS (P ≥ {TAU_HIGH}) | {final_mal:,} | {final_mal/len(y_prob)*100:.1f}% |
| FINAL_BENIGN (P ≤ {TAU_LOW}) | {final_ben:,} | {final_ben/len(y_prob)*100:.1f}% |
| AMBIGUOUS → Stage 2 | {ambiguous:,} | {ambiguous/len(y_prob)*100:.1f}% |
| **Total resolved by Stage 1** | **{resolved:,}** | **{resolved/len(y_prob)*100:.1f}%** |

## 11. Latency Benchmark

| Component | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|-----------|-----------|-------------|----------|----------|
| Feature Extraction | {ext_stats['mean']:.3f} | {ext_stats['median']:.3f} | {ext_stats['p95']:.3f} | {ext_stats['p99']:.3f} |
| RF Prediction | {pred_stats['mean']:.3f} | {pred_stats['median']:.3f} | {pred_stats['p95']:.3f} | {pred_stats['p99']:.3f} |
| **Total Stage-1** | **{total_stats['mean']:.3f}** | **{total_stats['median']:.3f}** | **{total_stats['p95']:.3f}** | **{total_stats['p99']:.3f}** |

- Benchmark: {len(sample_urls)} URLs from test set
- Hardware: {hw_info}
- Model loading time NOT included (one-time startup cost)

## 12. Limitations
- Feature extractor has ~81.5% exact match for `token_count` and ~95.5% for
  `phish_adv_hyphen_count` due to edge-case differences in normalization.
  Training uses the pre-computed dataset values; inference uses the extractor.
- Probability calibration was NOT explicitly performed. Probabilities may
  not be well-calibrated despite being useful for routing.
- The model uses `class_weight='balanced'` to handle the benign/malicious
  imbalance, but further threshold tuning may be needed.
"""

with open('reports/stage1_results.md', 'w') as f:
    f.write(report)
print("\nReport saved to reports/stage1_results.md")

print("\n" + "=" * 70)
print("PHASE 5 COMPLETE")
print("=" * 70)
