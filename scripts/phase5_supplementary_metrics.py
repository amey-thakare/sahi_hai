"""
Phase 5 — Supplementary metrics: per-class metrics, throughput, and probability behavior.
Generates data needed for the comprehensive stage1_results.md report.
"""
import sys, os, json, time, pickle
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)
from src.features.stage1_features import extract_stage1_features

# ── Load config, model, data ──────────────────────────────────────────────
with open('config/stage1_model_config.json') as f:
    cfg = json.load(f)
FEATURES = cfg['features']
TARGET = 'dsse_binary_target'
SEED = cfg['random_seed']

with open(cfg['model_path'], 'rb') as f:
    model = pickle.load(f)

test_df = pd.read_parquet('data/splits/test.parquet')
X_test = test_df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
y_test = test_df[TARGET]

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

# ── 1. Per-class metrics ──────────────────────────────────────────────────
print("=" * 70)
print("PER-CLASS CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=['benign', 'malicious'], digits=4))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"Specificity (TNR): {specificity:.4f}")
print(f"False Positive Rate: {fp/(fp+tn):.4f}")
print(f"False Negative Rate: {fn/(fn+tp):.4f}")

# ── 2. Routing distribution ──────────────────────────────────────────────
TAU_HIGH = 0.95
TAU_LOW = 0.05

final_mal = (y_prob >= TAU_HIGH).sum()
final_ben = (y_prob <= TAU_LOW).sum()
ambiguous = len(y_prob) - final_mal - final_ben
resolved = final_mal + final_ben

print(f"\n{'='*70}")
print("ROUTING DISTRIBUTION")
print(f"{'='*70}")
print(f"Total test samples:          {len(y_prob):,}")
print(f"FINAL_MALICIOUS (P≥0.95):    {final_mal:,} ({final_mal/len(y_prob)*100:.1f}%)")
print(f"FINAL_BENIGN (P≤0.05):       {final_ben:,} ({final_ben/len(y_prob)*100:.1f}%)")
print(f"AMBIGUOUS → Stage 2:         {ambiguous:,} ({ambiguous/len(y_prob)*100:.1f}%)")
print(f"Stage-1 resolution:          {resolved:,} ({resolved/len(y_prob)*100:.1f}%)")
print(f"Stage-2 routing:             {ambiguous:,} ({ambiguous/len(y_prob)*100:.1f}%)")
print(f"Sum check:                   {resolved/len(y_prob)*100 + ambiguous/len(y_prob)*100:.1f}%")

# ── 3. Routing accuracy ──────────────────────────────────────────────────
# Among those routed to FINAL by Stage-1, what is the accuracy?
mask_final = (y_prob >= TAU_HIGH) | (y_prob <= TAU_LOW)
if mask_final.sum() > 0:
    y_final_pred = y_pred[mask_final]
    y_final_true = y_test.values[mask_final]
    final_acc = accuracy_score(y_final_true, y_final_pred)
    print(f"\nRouted-confident accuracy:   {final_acc:.4f} ({mask_final.sum():,} samples)")

# ── 4. Probability behavior analysis ─────────────────────────────────────
print(f"\n{'='*70}")
print("PROBABILITY DISTRIBUTION")
print(f"{'='*70}")
bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (y_prob >= lo) & (y_prob < hi) if i < len(bins)-2 else (y_prob >= lo) & (y_prob <= hi)
    count = mask.sum()
    if count > 0:
        mal_rate = y_test.values[mask].mean()
        print(f"  P∈[{lo:.2f}, {hi:.2f}): {count:6,}  ({count/len(y_prob)*100:5.1f}%)  actual_mal_rate={mal_rate:.3f}")

# ── 5. Throughput benchmark ──────────────────────────────────────────────
print(f"\n{'='*70}")
print("THROUGHPUT BENCHMARK")
print(f"{'='*70}")

# Sequential throughput: feature extraction + prediction for each URL
sample_urls = test_df['url'].sample(min(2000, len(test_df)), random_state=SEED).tolist()

# Warm up
for u in sample_urls[:10]:
    feats = extract_stage1_features(u)
    fv = pd.DataFrame([feats])[FEATURES]
    model.predict_proba(fv)

t0 = time.perf_counter()
for u in sample_urls:
    feats = extract_stage1_features(u)
    fv = pd.DataFrame([feats])[FEATURES]
    model.predict_proba(fv)
t1 = time.perf_counter()

elapsed = t1 - t0
throughput = len(sample_urls) / elapsed
print(f"  URLs processed:   {len(sample_urls):,}")
print(f"  Total time:       {elapsed:.3f}s")
print(f"  Throughput:       {throughput:.1f} URLs/sec (sequential, single-thread)")
print(f"  Mode:             sequential, model pre-loaded, no model loading in loop")

# ── 6. Cold-start model load time ─────────────────────────────────────
t_cold0 = time.perf_counter()
with open(cfg['model_path'], 'rb') as f:
    _ = pickle.load(f)
t_cold1 = time.perf_counter()
cold_ms = (t_cold1 - t_cold0) * 1000
print(f"\n  Cold-start model load:  {cold_ms:.1f}ms")

print(f"\n{'='*70}")
print("SUPPLEMENTARY METRICS COMPLETE")
print(f"{'='*70}")
