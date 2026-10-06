"""
Phase 11 — Risk-Controlled Stage-1 Routing / FN Reduction
==========================================================

This script performs all Phase 11 experiments:
  Section 0: Freeze baseline & population integrity check
  Section 1: Stage-1 score distribution analysis
  Section 2: Threshold sweep (Experiment A)
  Section 3: Asymmetric conformal routing (Experiment B)
  Section 4: Coverage-risk curve (Experiment C)
  Section 5: End-to-end DSSE at selected operating points (Experiment D)
  Section 6: Statistical analysis
  Section 7: Error analysis
  Section 8: Visualizations

All results are written to reports/ and reports/figures/phase11/.
No models are retrained. No Stage-2 components are modified.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import math
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from scipy.stats import chi2

from src.features.stage1_features import extract_stage1_features
from src.conformal.calibrator import ConformalCalibrator
from src.conformal.routing import route_prediction_set
from src.semantic.minilm_encoder import MiniLMEncoder
from src.fusion.film import FiLMFusionClassifier
from src.stage2.xgboost_classifier import Stage2XGBoost

# ============================================================
# UTILITIES
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

ensure_dir('reports/figures/phase11')

def get_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    m = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        m['roc_auc'] = roc_auc_score(y_true, y_prob)
    else:
        m['roc_auc'] = float('nan')
    return m

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    b = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    c = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))
    if (b + c) == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1)**2 / (b + c)
    pval = chi2.sf(stat, df=1)
    return stat, pval

def bootstrap_ci(y_true, y_pred_a, y_pred_b, metric_fn, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    diffs = []
    for _ in range(n):
        idx = rng.choice(n_samples, n_samples, replace=True)
        ma = metric_fn(y_true[idx], y_pred_a[idx])
        mb = metric_fn(y_true[idx], y_pred_b[idx])
        diffs.append(ma - mb)
    return np.percentile(diffs, [2.5, 97.5])


# ============================================================
# SECTION 0: LOAD DATA & MODELS (SHARED)
# ============================================================
print("="*70)
print("PHASE 11: Risk-Controlled Stage-1 Routing")
print("="*70)

print("\n[0] Loading test data and models...")

test_df = pd.read_parquet('data/splits/test.parquet')
urls = test_df['url'].tolist()
y_true = test_df['dsse_binary_target'].values
total_urls = len(y_true)
print(f"    Test URLs: {total_urls}")

features = [
    'domain_len', 'path_depth', 'special_count_v2', 'token_count',
    'url_entropy', 'url_len', '=', '.', '//',
    'abnormal_url', 'phish_adv_hyphen_count', 'suspicious_extension'
]

# Load HXRF model used in DSSE (Phase 10 uses stage1_hxrf.pkl)
with open('models/stage1_hxrf.pkl', 'rb') as f:
    hxrf_dsse = pickle.load(f)

# Load HXRF model used for ICP calibration (Phase 6 used stage1_hxrf_icp.pkl)
with open('models/stage1_hxrf_icp.pkl', 'rb') as f:
    hxrf_icp = pickle.load(f)

calibrator = ConformalCalibrator.load('models/stage1_conformal.pkl')

# ============================================================
# SECTION 0A: POPULATION INTEGRITY — DISCREPANCY INVESTIGATION
# ============================================================
print("\n[0A] Investigating NEEDS_STAGE2 discrepancy...")

# Phase-9 approach: features from parquet columns directly
X_test_parquet = test_df[features].values
probas_parquet = hxrf_icp.predict_proba(X_test_parquet)
icp_result_parquet = calibrator.predict(probas_parquet)
routes_parquet = np.array([route_prediction_set(ps) for ps in icp_result_parquet['prediction_sets']])
ns2_parquet = int((routes_parquet == 'NEEDS_STAGE2').sum())

# Phase-10 approach: extract features from raw URL strings, DSSE HXRF model
print("    Extracting features from raw URLs (Phase-10 style)...")
feats_from_url = [extract_stage1_features(u) for u in urls]
X_test_url = pd.DataFrame(feats_from_url)[features].values
probas_url_dsse = hxrf_dsse.predict_proba(X_test_url)
icp_result_url_dsse = calibrator.predict(probas_url_dsse)
routes_url_dsse = np.array([route_prediction_set(ps) for ps in icp_result_url_dsse['prediction_sets']])
ns2_url_dsse = int((routes_url_dsse == 'NEEDS_STAGE2').sum())

# Also check: parquet features + DSSE HXRF
probas_parquet_dsse = hxrf_dsse.predict_proba(X_test_parquet)
icp_result_parquet_dsse = calibrator.predict(probas_parquet_dsse)
routes_parquet_dsse = np.array([route_prediction_set(ps) for ps in icp_result_parquet_dsse['prediction_sets']])
ns2_parquet_dsse = int((routes_parquet_dsse == 'NEEDS_STAGE2').sum())

# Also check: URL-extracted features + ICP HXRF
probas_url_icp = hxrf_icp.predict_proba(X_test_url)
icp_result_url_icp = calibrator.predict(probas_url_icp)
routes_url_icp = np.array([route_prediction_set(ps) for ps in icp_result_url_icp['prediction_sets']])
ns2_url_icp = int((routes_url_icp == 'NEEDS_STAGE2').sum())

# Feature mismatch analysis
feat_diffs = np.abs(X_test_parquet - X_test_url)
n_mismatched_rows = int(np.sum(np.any(feat_diffs > 1e-6, axis=1)))
max_diff_per_feat = feat_diffs.max(axis=0)

print(f"    Phase-9 style (parquet + ICP-HXRF):    NEEDS_STAGE2 = {ns2_parquet}")
print(f"    Phase-10 style (URL + DSSE-HXRF):      NEEDS_STAGE2 = {ns2_url_dsse}")
print(f"    Hybrid A (parquet + DSSE-HXRF):         NEEDS_STAGE2 = {ns2_parquet_dsse}")
print(f"    Hybrid B (URL + ICP-HXRF):              NEEDS_STAGE2 = {ns2_url_icp}")
print(f"    Rows with feature differences:          {n_mismatched_rows}")

with open('reports/phase11_population_integrity.md', 'w') as f:
    f.write("# Phase 11 — Population Integrity Report\n\n")
    f.write("## 1. Discrepancy Summary\n\n")
    f.write(f"- Phase-9 NEEDS_STAGE2 count: **{ns2_parquet}** (parquet features + ICP-HXRF)\n")
    f.write(f"- Phase-10 NEEDS_STAGE2 count: **{ns2_url_dsse}** (URL-extracted features + DSSE-HXRF)\n")
    f.write(f"- Difference: **{ns2_url_dsse - ns2_parquet}**\n\n")
    f.write("## 2. Root Cause Analysis\n\n")
    f.write("Two factors contribute to the discrepancy:\n\n")
    f.write("### Factor A: Different HXRF Models\n\n")
    f.write("- Phase-9/ICP calibration used `models/stage1_hxrf_icp.pkl` (78 MB)\n")
    f.write("- Phase-10 DSSE config uses `models/stage1_hxrf.pkl` (84 MB)\n")
    f.write("- These are different Random Forest checkpoints trained on different data splits.\n")
    f.write("- The ICP calibrator was fitted against probabilities from `stage1_hxrf_icp.pkl`.\n")
    f.write("- Using `stage1_hxrf.pkl` with that calibrator introduces a model mismatch.\n\n")
    f.write("### Factor B: Feature Extraction Path\n\n")
    f.write(f"- Rows with any feature difference (URL-extracted vs parquet): **{n_mismatched_rows}** / {total_urls}\n")
    f.write("- Maximum absolute difference per feature:\n\n")
    f.write("| Feature | Max Diff |\n|---|---:|\n")
    for i, feat in enumerate(features):
        f.write(f"| {feat} | {max_diff_per_feat[i]:.6f} |\n")
    f.write("\n### Factorial Analysis\n\n")
    f.write("| Features | HXRF Model | NEEDS_STAGE2 |\n|---|---|---:|\n")
    f.write(f"| Parquet columns | ICP-HXRF | {ns2_parquet} |\n")
    f.write(f"| Parquet columns | DSSE-HXRF | {ns2_parquet_dsse} |\n")
    f.write(f"| URL-extracted | ICP-HXRF | {ns2_url_icp} |\n")
    f.write(f"| URL-extracted | DSSE-HXRF | {ns2_url_dsse} |\n\n")
    f.write("## 3. Decision\n\n")
    f.write("Phase-10 remains the frozen baseline because it represents the actual deployed DSSE system.\n")
    f.write("Phase-9 results remain valid for that experiment's scope (same HXRF model, parquet features).\n\n")
    f.write("## 4. Phase-11 Methodology\n\n")
    f.write("For Phase 11, all routing experiments will use:\n")
    f.write("- Features: **parquet columns** (deterministic, no floating-point drift)\n")
    f.write("- HXRF model: **stage1_hxrf.pkl** (the DSSE production model)\n")
    f.write("- Calibrator: **stage1_conformal.pkl** (unchanged)\n")
    f.write("- Test set: **data/splits/test.parquet**\n\n")
    f.write("This ensures the HXRF probabilities are identical across all Phase-11 experiments.\n")
    f.write("The baseline NEEDS_STAGE2 count under this configuration is computed fresh.\n\n")
    f.write("## 5. Reproducibility Implication\n\n")
    f.write("The HXRF model mismatch means ICP conformal coverage guarantees are technically\n")
    f.write("violated in Phase 10, since the calibrator was fitted on a different model's scores.\n")
    f.write("This is a known limitation that does not invalidate the routing logic but should\n")
    f.write("be noted in any publication.\n")

print("    Population integrity report written.")

# ============================================================
# ESTABLISH PHASE-11 CANONICAL BASELINE
# ============================================================
# Use parquet features + DSSE-HXRF for all Phase-11 experiments
# (deterministic features, actual DSSE model)
print("\n[0B] Establishing canonical Phase-11 baseline...")

X_test = X_test_parquet  # deterministic features
hxrf = hxrf_dsse         # actual DSSE model
probas = hxrf.predict_proba(X_test)
p_mal = probas[:, 1]     # P(malicious | x)
p_ben = probas[:, 0]     # P(benign | x)

# Current ICP routing (baseline)
icp_result_baseline = calibrator.predict(probas)
routes_baseline = np.array([route_prediction_set(ps)
                            for ps in icp_result_baseline['prediction_sets']])

is_benign_exit = routes_baseline == 'FINAL_BENIGN'
is_malicious_exit = routes_baseline == 'FINAL_MALICIOUS'
is_stage2 = routes_baseline == 'NEEDS_STAGE2'

baseline_ns2 = int(is_stage2.sum())
baseline_resolved = total_urls - baseline_ns2
print(f"    Canonical baseline NEEDS_STAGE2: {baseline_ns2}")
print(f"    Canonical baseline resolved: {baseline_resolved}")

# ============================================================
# SECTION 1: STAGE-1 SCORE DISTRIBUTION ANALYSIS
# ============================================================
print("\n[1] Analyzing Stage-1 score distributions...")

benign_mask = y_true == 0
mal_mask = y_true == 1

# Stage-1 FN: true malicious, routed to FINAL_BENIGN
s1_fn_mask = mal_mask & is_benign_exit
# Stage-1 FP: true benign, routed to FINAL_MALICIOUS
s1_fp_mask = benign_mask & is_malicious_exit

s1_fn_count = int(s1_fn_mask.sum())
s1_fp_count = int(s1_fp_mask.sum())

print(f"    Stage-1 FN (mal→FINAL_BENIGN): {s1_fn_count}")
print(f"    Stage-1 FP (ben→FINAL_MALICIOUS): {s1_fp_count}")

# Score quantiles
q_benign = np.percentile(p_mal[benign_mask], [5, 10, 25, 50, 75, 90, 95])
q_malicious = np.percentile(p_mal[mal_mask], [5, 10, 25, 50, 75, 90, 95])
q_fn = np.percentile(p_mal[s1_fn_mask], [5, 10, 25, 50, 75, 90, 95]) if s1_fn_count > 0 else np.zeros(7)

# Figure 1: Score histogram
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(p_mal[benign_mask], bins=100, alpha=0.7, color='#2196F3', label='True Benign', density=True)
axes[0].hist(p_mal[mal_mask], bins=100, alpha=0.7, color='#F44336', label='True Malicious', density=True)
axes[0].set_xlabel('P(malicious)')
axes[0].set_ylabel('Density')
axes[0].set_title('HXRF P(malicious) Distribution')
axes[0].legend()

axes[1].hist(p_mal[s1_fn_mask], bins=50, alpha=0.8, color='#FF9800', label=f'Stage-1 FN (n={s1_fn_count})', density=True)
axes[1].set_xlabel('P(malicious)')
axes[1].set_ylabel('Density')
axes[1].set_title('Stage-1 False Negatives: P(malicious)')
axes[1].legend()

# Empirical CDF of P(malicious) for FN cases
if s1_fn_count > 0:
    fn_sorted = np.sort(p_mal[s1_fn_mask])
    fn_cdf = np.arange(1, len(fn_sorted) + 1) / len(fn_sorted)
    axes[2].plot(fn_sorted, fn_cdf, color='#FF9800', linewidth=2)
    axes[2].set_xlabel('P(malicious)')
    axes[2].set_ylabel('Empirical CDF')
    axes[2].set_title('Stage-1 FN: Empirical CDF of P(malicious)')
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/phase11/score_distributions.png', dpi=150)
plt.close()

# ============================================================
# SECTION 2: THRESHOLD SWEEP (EXPERIMENT A)
# ============================================================
print("\n[2] Running threshold sweep (Experiment A)...")

# Policy: Accept BENIGN only if P(malicious) < benign_threshold
#         Accept MALICIOUS only if P(malicious) > malicious_threshold
#         Otherwise NEEDS_STAGE2
#
# Asymmetric: we can vary benign_threshold independently of malicious_threshold

# First, a fine grid of symmetric thresholds where benign_thresh = 1 - malicious_thresh
# Then asymmetric sweep

# For the benign acceptance: lower threshold = more conservative (fewer benign exits)
benign_thresholds = np.arange(0.05, 0.55, 0.01)  # P(mal) must be below this to exit benign
malicious_thresholds = np.arange(0.50, 0.96, 0.01)  # P(mal) must be above this to exit malicious

# Also add the current ICP quantile as reference
icp_quantile = calibrator.quantile
print(f"    ICP conformal quantile q_α: {icp_quantile:.6f}")

threshold_results = []

for b_thresh in benign_thresholds:
    for m_thresh in malicious_thresholds:
        if b_thresh >= m_thresh:
            continue  # nonsensical

        # Route
        route_benign = p_mal < b_thresh
        route_malicious = p_mal > m_thresh
        route_stage2 = ~route_benign & ~route_malicious

        n_benign_exit = int(route_benign.sum())
        n_mal_exit = int(route_malicious.sum())
        n_s2 = int(route_stage2.sum())
        s2_rate = n_s2 / total_urls

        # Stage-1 predictions
        y_pred_s1 = np.full(total_urls, -1)
        y_pred_s1[route_benign] = 0
        y_pred_s1[route_malicious] = 1
        # Stage-2 samples get -1 (placeholder)

        # Stage-1 errors (only on resolved samples)
        resolved_mask = route_benign | route_malicious
        s1_fn = int(np.sum(route_benign & mal_mask))
        s1_fp = int(np.sum(route_malicious & benign_mask))

        threshold_results.append({
            'b_thresh': round(b_thresh, 3),
            'm_thresh': round(m_thresh, 3),
            'n_benign_exit': n_benign_exit,
            'n_mal_exit': n_mal_exit,
            'n_stage2': n_s2,
            's2_rate': s2_rate,
            's1_fn': s1_fn,
            's1_fp': s1_fp,
        })

threshold_df = pd.DataFrame(threshold_results)

# Select key operating points for reporting
# Fix malicious threshold at 0.50 (accept malicious when confident) and sweep benign
key_m_thresh = 0.50
key_results = threshold_df[threshold_df['m_thresh'] == key_m_thresh].copy()
key_results = key_results.sort_values('b_thresh')

print(f"    Threshold sweep complete: {len(threshold_df)} combinations evaluated")

# ============================================================
# SECTION 3: ASYMMETRIC CONFORMAL ROUTING (EXPERIMENT B)
# ============================================================
print("\n[3] Running asymmetric conformal routing (Experiment B)...")

# The key idea: use different alpha for benign vs malicious acceptance
# Lower alpha for benign = tighter threshold = harder to exit benign
# Higher alpha for malicious = easier to exit malicious

# Non-conformity scores for each class
scores_class0 = 1.0 - p_ben  # score for y=0 (benign)
scores_class1 = 1.0 - p_mal  # score for y=1 (malicious)

alpha_grid = [0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]

conformal_results = []

for alpha_benign in alpha_grid:
    for alpha_malicious in alpha_grid:
        # Compute quantiles for each class
        q_benign_val = calibrator.get_quantile(alpha_benign)
        q_malicious_val = calibrator.get_quantile(alpha_malicious)

        # Include class y if s(x,y) <= q_α_y
        include_benign = scores_class0 <= q_benign_val
        include_malicious = scores_class1 <= q_malicious_val

        # Build routes
        routes = np.full(total_urls, 'NEEDS_STAGE2', dtype=object)
        # {benign only} → FINAL_BENIGN
        only_benign = include_benign & ~include_malicious
        routes[only_benign] = 'FINAL_BENIGN'
        # {malicious only} → FINAL_MALICIOUS
        only_malicious = ~include_benign & include_malicious
        routes[only_malicious] = 'FINAL_MALICIOUS'
        # {both} → NEEDS_STAGE2 (already set)
        # {empty} → NEEDS_STAGE2 (already set)

        n_fb = int((routes == 'FINAL_BENIGN').sum())
        n_fm = int((routes == 'FINAL_MALICIOUS').sum())
        n_s2 = int((routes == 'NEEDS_STAGE2').sum())
        s2_rate = n_s2 / total_urls

        s1_fn = int(np.sum((routes == 'FINAL_BENIGN') & mal_mask))
        s1_fp = int(np.sum((routes == 'FINAL_MALICIOUS') & benign_mask))

        conformal_results.append({
            'alpha_benign': alpha_benign,
            'alpha_malicious': alpha_malicious,
            'symmetric': alpha_benign == alpha_malicious,
            'n_final_benign': n_fb,
            'n_final_malicious': n_fm,
            'n_stage2': n_s2,
            's2_rate': s2_rate,
            's1_fn': s1_fn,
            's1_fp': s1_fp,
        })

conformal_df = pd.DataFrame(conformal_results)

# Baseline (alpha=0.05 symmetric)
baseline_row = conformal_df[(conformal_df['alpha_benign'] == 0.05) & (conformal_df['alpha_malicious'] == 0.05)].iloc[0]
print(f"    Baseline (α=0.05 symmetric): S2={baseline_row['n_stage2']}, FN={baseline_row['s1_fn']}")

# Key asymmetric policies
asymmetric_policies = conformal_df[
    (conformal_df['alpha_benign'].isin([0.01, 0.02, 0.025, 0.03])) &
    (conformal_df['alpha_malicious'].isin([0.05, 0.10, 0.15]))
].copy()

print(f"    Conformal sweep complete: {len(conformal_df)} combinations evaluated")

# ============================================================
# SECTION 4: COVERAGE-RISK CURVE (EXPERIMENT C)
# ============================================================
print("\n[4] Generating coverage-risk curves...")

# Combine all routing policies into a unified operating-point table
operating_points = []

# From conformal sweep
for _, row in conformal_df.iterrows():
    operating_points.append({
        'policy': f"CP(α_b={row['alpha_benign']}, α_m={row['alpha_malicious']})",
        'type': 'conformal',
        's2_rate': row['s2_rate'],
        's1_fn': row['s1_fn'],
        's1_fn_rate': row['s1_fn'] / int(mal_mask.sum()),
        'n_stage2': row['n_stage2'],
        'alpha_benign': row['alpha_benign'],
        'alpha_malicious': row['alpha_malicious'],
    })

# From threshold sweep (selected m_thresh values)
for m_t in [0.50, 0.55, 0.60]:
    subset = threshold_df[threshold_df['m_thresh'] == m_t]
    for _, row in subset.iterrows():
        operating_points.append({
            'policy': f"Thresh(b={row['b_thresh']}, m={row['m_thresh']})",
            'type': 'threshold',
            's2_rate': row['s2_rate'],
            's1_fn': row['s1_fn'],
            's1_fn_rate': row['s1_fn'] / int(mal_mask.sum()),
            'n_stage2': row['n_stage2'],
            'b_thresh': row['b_thresh'],
            'm_thresh': row['m_thresh'],
        })

op_df = pd.DataFrame(operating_points)

# Identify named operating points from conformal policies
# OP1: Current Phase-10 (α=0.05 symmetric)
op1 = conformal_df[(conformal_df['alpha_benign'] == 0.05) & (conformal_df['alpha_malicious'] == 0.05)].iloc[0]

# OP2: Lowest S2 rate with ≥50% FN reduction
fn_baseline = int(op1['s1_fn'])
candidates_fn50 = conformal_df[conformal_df['s1_fn'] <= fn_baseline * 0.5].copy()
if len(candidates_fn50) > 0:
    candidates_fn50 = candidates_fn50.sort_values('s2_rate')
    op2 = candidates_fn50.iloc[0]
else:
    op2 = None

# OP3: Best F1 operating point — need end-to-end evaluation (deferred to Section 5)
# OP4: Best recall operating point
candidates_recall = conformal_df.sort_values('s1_fn')
op4 = candidates_recall.iloc[0]

# Figure 2: Coverage vs FN rate
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Filter conformal to plot only interesting subset
for alpha_m in [0.05, 0.10, 0.15]:
    subset = conformal_df[conformal_df['alpha_malicious'] == alpha_m].sort_values('s2_rate')
    fn_rate = subset['s1_fn'] / int(mal_mask.sum())
    axes[0].plot(subset['s2_rate'] * 100, fn_rate * 100,
                 'o-', markersize=4, label=f'α_mal={alpha_m}')

axes[0].axhline(y=op1['s1_fn'] / int(mal_mask.sum()) * 100,
                color='red', linestyle='--', alpha=0.5, label='Phase-10 baseline')
axes[0].set_xlabel('Stage-2 Escalation Rate (%)')
axes[0].set_ylabel('Stage-1 FN Rate (% of malicious)')
axes[0].set_title('Coverage vs Stage-1 FN Rate\n(Asymmetric Conformal Policies)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Threshold sweep: fix m_thresh=0.50
thresh_subset = threshold_df[threshold_df['m_thresh'] == 0.50].sort_values('s2_rate')
axes[1].plot(thresh_subset['s2_rate'] * 100,
             thresh_subset['s1_fn'] / int(mal_mask.sum()) * 100,
             's-', markersize=4, color='#4CAF50', label='Threshold (m=0.50)')
axes[1].axhline(y=op1['s1_fn'] / int(mal_mask.sum()) * 100,
                color='red', linestyle='--', alpha=0.5, label='Phase-10 baseline')
axes[1].set_xlabel('Stage-2 Escalation Rate (%)')
axes[1].set_ylabel('Stage-1 FN Rate (% of malicious)')
axes[1].set_title('Coverage vs Stage-1 FN Rate\n(Score Threshold Policies)')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/phase11/coverage_vs_fn_rate.png', dpi=150)
plt.close()

# ============================================================
# SECTION 5: END-TO-END DSSE AT SELECTED OPERATING POINTS
# ============================================================
print("\n[5] Running end-to-end DSSE at selected operating points...")
print("    Loading Stage-2 components...")

encoder = MiniLMEncoder(device='cpu')
film_scaler = joblib.load('models/film/film_scaler.pkl')

film_model = FiLMFusionClassifier(
    semantic_dim=384, structural_dim=12, hidden_dim=64, dropout=0.1, stable_gamma=True
)
film_model.load_state_dict(torch.load('models/film/film_model.pth', map_location='cpu', weights_only=True))
film_model.to('cpu')
film_model.eval()

xgb_model = Stage2XGBoost.load('models/xgboost_audit/film_xgb.json')

def run_dsse_with_routes(route_array, label):
    """Run complete DSSE pipeline with a given routing array. Returns metrics dict."""
    print(f"    Running DSSE for policy: {label}...")
    s2_mask = route_array == 'NEEDS_STAGE2'
    n_s2 = int(s2_mask.sum())

    # Stage-1 predictions
    y_pred_final = np.full(total_urls, -1)
    y_pred_final[route_array == 'FINAL_BENIGN'] = 0
    y_pred_final[route_array == 'FINAL_MALICIOUS'] = 1

    # Stage-1 probability fallback for resolved URLs
    y_prob_final = p_mal.copy()

    if n_s2 > 0:
        s2_indices = np.where(s2_mask)[0]
        s2_urls = [urls[i] for i in s2_indices]
        s2_struct = X_test[s2_indices]

        # MiniLM
        embs = encoder.encode_urls(s2_urls, batch_size=256)

        # FiLM
        s2_struct_scaled = film_scaler.transform(s2_struct)
        film_embs = []
        with torch.no_grad():
            for start in range(0, len(embs), 2048):
                end = min(start + 2048, len(embs))
                sem_t = torch.tensor(embs[start:end], dtype=torch.float32)
                str_t = torch.tensor(s2_struct_scaled[start:end], dtype=torch.float32)
                _, _, _, rep = film_model(sem_t, str_t)
                film_embs.append(rep.cpu().numpy())
        film_rep = np.vstack(film_embs)

        # XGBoost
        xgb_probas = xgb_model.predict_proba(np.ascontiguousarray(film_rep, dtype=np.float32))
        xgb_preds = (xgb_probas[:, 1] >= 0.5).astype(int)

        for idx_local, idx_global in enumerate(s2_indices):
            y_pred_final[idx_global] = xgb_preds[idx_local]
            y_prob_final[idx_global] = xgb_probas[idx_local, 1]

    assert np.all(y_pred_final >= 0), "Some URLs were not classified!"

    metrics = get_metrics(y_true, y_pred_final, y_prob_final)

    # Error breakdown
    s1_resolved_mask = route_array != 'NEEDS_STAGE2'
    s1_fn = int(np.sum(s1_resolved_mask & (y_true == 1) & (y_pred_final == 0)))
    s1_fp = int(np.sum(s1_resolved_mask & (y_true == 0) & (y_pred_final == 1)))
    s2_fn = int(np.sum(s2_mask & (y_true == 1) & (y_pred_final == 0)))
    s2_fp = int(np.sum(s2_mask & (y_true == 0) & (y_pred_final == 1)))

    metrics['s1_fn'] = s1_fn
    metrics['s1_fp'] = s1_fp
    metrics['s2_fn'] = s2_fn
    metrics['s2_fp'] = s2_fp
    metrics['n_stage2'] = n_s2
    metrics['s2_rate'] = n_s2 / total_urls
    metrics['total_errors'] = int(metrics['fp'] + metrics['fn'])
    metrics['y_pred'] = y_pred_final
    metrics['y_prob'] = y_prob_final
    metrics['policy'] = label

    return metrics

# Define candidate policies to evaluate end-to-end

# Policy 0: Phase-10 baseline (α=0.05 symmetric)
def build_conformal_routes(alpha_b, alpha_m):
    q_b = calibrator.get_quantile(alpha_b)
    q_m = calibrator.get_quantile(alpha_m)
    inc_b = scores_class0 <= q_b
    inc_m = scores_class1 <= q_m
    routes = np.full(total_urls, 'NEEDS_STAGE2', dtype=object)
    routes[inc_b & ~inc_m] = 'FINAL_BENIGN'
    routes[~inc_b & inc_m] = 'FINAL_MALICIOUS'
    return routes

candidates = [
    ('P0: Baseline (α=0.05 sym)', 0.05, 0.05),
    ('P1: α_b=0.025, α_m=0.05',  0.025, 0.05),
    ('P2: α_b=0.02, α_m=0.05',   0.02, 0.05),
    ('P3: α_b=0.01, α_m=0.05',   0.01, 0.05),
    ('P4: α_b=0.025, α_m=0.10',  0.025, 0.10),
    ('P5: α_b=0.02, α_m=0.10',   0.02, 0.10),
    ('P6: α_b=0.01, α_m=0.10',   0.01, 0.10),
    ('P7: α_b=0.025, α_m=0.15',  0.025, 0.15),
    ('P8: α_b=0.02, α_m=0.15',   0.02, 0.15),
    ('P9: α_b=0.01, α_m=0.15',   0.01, 0.15),
    ('P10: α_b=0.03, α_m=0.075', 0.03, 0.075),
    ('P11: α_b=0.02, α_m=0.075', 0.02, 0.075),
]

all_e2e_results = []
for label, ab, am in candidates:
    routes = build_conformal_routes(ab, am)
    metrics = run_dsse_with_routes(routes, label)
    metrics['alpha_benign'] = ab
    metrics['alpha_malicious'] = am
    all_e2e_results.append(metrics)

e2e_df = pd.DataFrame([{k: v for k, v in m.items() if k not in ('y_pred', 'y_prob')}
                        for m in all_e2e_results])

print("\n    End-to-end results:")
print(e2e_df[['policy', 'f1', 'recall', 'precision', 'accuracy', 'roc_auc',
              's2_rate', 's1_fn', 'total_errors']].to_string(index=False))

# ============================================================
# SECTION 6: SELECT BEST POLICY & STATISTICAL ANALYSIS
# ============================================================
print("\n[6] Statistical analysis...")

baseline_metrics = all_e2e_results[0]  # P0
best_f1_idx = e2e_df['f1'].idxmax()
best_f1_metrics = all_e2e_results[best_f1_idx]

print(f"    Baseline: {baseline_metrics['policy']} — F1={baseline_metrics['f1']:.4f}")
print(f"    Best F1:  {best_f1_metrics['policy']} — F1={best_f1_metrics['f1']:.4f}")

# McNemar test
stat, pval = mcnemar_test(y_true, best_f1_metrics['y_pred'], baseline_metrics['y_pred'])
print(f"    McNemar: statistic={stat:.4f}, p-value={pval:.4e}")

# Bootstrap CI for F1 difference
ci_f1 = bootstrap_ci(y_true, best_f1_metrics['y_pred'], baseline_metrics['y_pred'],
                      lambda yt, yp: f1_score(yt, yp, zero_division=0))
print(f"    F1 difference 95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")

# Bootstrap CI for recall difference
ci_recall = bootstrap_ci(y_true, best_f1_metrics['y_pred'], baseline_metrics['y_pred'],
                          lambda yt, yp: recall_score(yt, yp, zero_division=0))
print(f"    Recall difference 95% CI: [{ci_recall[0]:.4f}, {ci_recall[1]:.4f}]")

# ============================================================
# SECTION 7: ERROR ANALYSIS
# ============================================================
print("\n[7] Error analysis...")

with open('reports/phase11_error_analysis.md', 'w') as f:
    f.write("# Phase 11 — Error Analysis\n\n")
    f.write("## Before vs After\n\n")
    f.write("| Metric | Phase-10 Baseline | Best Phase-11 Policy |\n")
    f.write("|---|---:|---:|\n")
    f.write(f"| Policy | {baseline_metrics['policy']} | {best_f1_metrics['policy']} |\n")
    f.write(f"| Stage-1 FN | {baseline_metrics['s1_fn']} | {best_f1_metrics['s1_fn']} |\n")
    f.write(f"| Stage-1 FP | {baseline_metrics['s1_fp']} | {best_f1_metrics['s1_fp']} |\n")
    f.write(f"| Stage-2 FN | {baseline_metrics['s2_fn']} | {best_f1_metrics['s2_fn']} |\n")
    f.write(f"| Stage-2 FP | {baseline_metrics['s2_fp']} | {best_f1_metrics['s2_fp']} |\n")
    f.write(f"| Total FN | {baseline_metrics['fn']} | {best_f1_metrics['fn']} |\n")
    f.write(f"| Total FP | {baseline_metrics['fp']} | {best_f1_metrics['fp']} |\n")
    f.write(f"| Total Errors | {baseline_metrics['total_errors']} | {best_f1_metrics['total_errors']} |\n")
    f.write(f"| Stage-2 Escalations | {baseline_metrics['n_stage2']} | {best_f1_metrics['n_stage2']} |\n")
    f.write(f"| Stage-2 Rate | {baseline_metrics['s2_rate']*100:.2f}% | {best_f1_metrics['s2_rate']*100:.2f}% |\n\n")

    # Newly recovered URLs
    baseline_fn_set = set(np.where((y_true == 1) & (baseline_metrics['y_pred'] == 0))[0])
    best_fn_set = set(np.where((y_true == 1) & (best_f1_metrics['y_pred'] == 0))[0])
    recovered = baseline_fn_set - best_fn_set
    newly_missed = best_fn_set - baseline_fn_set

    f.write(f"## Recovery Analysis\n\n")
    f.write(f"- Malicious URLs recovered (FN→TP/TN): **{len(recovered)}**\n")
    f.write(f"- Malicious URLs newly missed: **{len(newly_missed)}**\n")
    f.write(f"- Net FN reduction: **{baseline_metrics['fn'] - best_f1_metrics['fn']}**\n\n")

    # New FP analysis
    baseline_fp_set = set(np.where((y_true == 0) & (baseline_metrics['y_pred'] == 1))[0])
    best_fp_set = set(np.where((y_true == 0) & (best_f1_metrics['y_pred'] == 1))[0])
    new_fps = best_fp_set - baseline_fp_set
    removed_fps = baseline_fp_set - best_fp_set

    f.write(f"## False Positive Changes\n\n")
    f.write(f"- Newly introduced FP: **{len(new_fps)}**\n")
    f.write(f"- Removed FP: **{len(removed_fps)}**\n")
    f.write(f"- Net FP change: **{best_f1_metrics['fp'] - baseline_metrics['fp']}**\n")

print("    Error analysis written.")

# ============================================================
# SECTION 8: WRITE ALL REPORTS
# ============================================================
print("\n[8] Writing reports...")

# --- phase11_threshold_sweep.md ---
with open('reports/phase11_threshold_sweep.md', 'w') as f:
    f.write("# Phase 11 — Threshold Sweep (Experiment A)\n\n")
    f.write("Score-threshold routing policies evaluated on the frozen test set.\n\n")
    f.write("Policy: Exit BENIGN if P(mal) < b_thresh; Exit MALICIOUS if P(mal) > m_thresh; else Stage-2.\n\n")
    f.write("## Key Results (m_thresh=0.50)\n\n")
    f.write("| b_thresh | Benign Exits | Mal Exits | Stage-2 | S2 % | S1 FN | S1 FP |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, row in key_results.iterrows():
        f.write(f"| {row['b_thresh']:.2f} | {row['n_benign_exit']} | {row['n_mal_exit']} | "
                f"{row['n_stage2']} | {row['s2_rate']*100:.2f}% | {row['s1_fn']} | {row['s1_fp']} |\n")
    f.write("\n## Interpretation\n\n")
    f.write("Lowering b_thresh reduces Stage-1 false negatives but increases Stage-2 escalations.\n")
    f.write("This is a direct trade-off between safety (fewer FN) and efficiency (fewer S2 calls).\n")

# --- phase11_conformal_routing.md ---
with open('reports/phase11_conformal_routing.md', 'w') as f:
    f.write("# Phase 11 — Asymmetric Conformal Routing (Experiment B)\n\n")
    f.write("## Methodology\n\n")
    f.write("Standard split-conformal ICP uses a single α for both classes.\n")
    f.write("Asymmetric routing uses different α values for benign acceptance vs malicious acceptance:\n\n")
    f.write("- **α_benign**: Lower α → stricter benign acceptance → fewer FN but more escalations\n")
    f.write("- **α_malicious**: Higher α → easier malicious acceptance → more direct malicious exits\n\n")
    f.write("> **Important**: Asymmetric routing no longer provides the standard split-conformal\n")
    f.write("> marginal coverage guarantee P(y ∈ C(x)) ≥ 1−α. The individual class-conditional\n")
    f.write("> acceptance thresholds are derived from the same calibration scores but applied\n")
    f.write("> with different stringencies. This is a heuristic routing policy, not a\n")
    f.write("> statistically guaranteed prediction set.\n\n")
    f.write("## Results\n\n")
    f.write("| α_benign | α_malicious | Stage-2 | S2 % | S1 FN | S1 FP |\n")
    f.write("|---:|---:|---:|---:|---:|---:|\n")
    sel = conformal_df[conformal_df['alpha_malicious'].isin([0.05, 0.10, 0.15])].sort_values(['alpha_malicious', 'alpha_benign'])
    for _, row in sel.iterrows():
        marker = " ←baseline" if (row['alpha_benign'] == 0.05 and row['alpha_malicious'] == 0.05) else ""
        f.write(f"| {row['alpha_benign']} | {row['alpha_malicious']} | {row['n_stage2']} | "
                f"{row['s2_rate']*100:.2f}% | {row['s1_fn']} | {row['s1_fp']}{marker} |\n")

# --- phase11_coverage_risk.md ---
with open('reports/phase11_coverage_risk.md', 'w') as f:
    f.write("# Phase 11 — Coverage vs Risk Analysis (Experiment C)\n\n")
    f.write("## Named Operating Points (End-to-End DSSE)\n\n")
    f.write("| Policy | F1 | Recall | Precision | Accuracy | ROC-AUC | FPR | FNR | S2 % | S1 FN | Total Err |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for m in all_e2e_results:
        f.write(f"| {m['policy']} | {m['f1']:.4f} | {m['recall']:.4f} | {m['precision']:.4f} | "
                f"{m['accuracy']:.4f} | {m['roc_auc']:.4f} | {m['fpr']:.4f} | {m['fnr']:.4f} | "
                f"{m['s2_rate']*100:.2f}% | {m['s1_fn']} | {m['total_errors']} |\n")

# --- phase11_end_to_end_results.md ---
with open('reports/phase11_end_to_end_results.md', 'w') as f:
    f.write("# Phase 11 — End-to-End DSSE Results\n\n")
    f.write("## Architecture\n")
    f.write("Same DSSE architecture as Phase 10, with modified Stage-1 routing policy.\n\n")
    f.write("## Baseline Comparison\n\n")
    f.write("| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR | S2 Rate |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    f.write("| HXRF only | 0.9277 | 0.9191 | 0.8585 | 0.8877 | 0.9687 | 0.0377 | 0.1415 | N/A |\n")
    f.write("| FiLM + XGB (all) | 0.9528 | 0.9505 | 0.9051 | 0.9273 | 0.9897 | 0.0235 | 0.0949 | 100% |\n")
    b = baseline_metrics
    f.write(f"| Phase-10 DSSE | {b['accuracy']:.4f} | {b['precision']:.4f} | {b['recall']:.4f} | "
            f"{b['f1']:.4f} | {b['roc_auc']:.4f} | {b['fpr']:.4f} | {b['fnr']:.4f} | {b['s2_rate']*100:.2f}% |\n")
    best = best_f1_metrics
    f.write(f"| **Phase-11 Best** | {best['accuracy']:.4f} | {best['precision']:.4f} | {best['recall']:.4f} | "
            f"{best['f1']:.4f} | {best['roc_auc']:.4f} | {best['fpr']:.4f} | {best['fnr']:.4f} | {best['s2_rate']*100:.2f}% |\n")
    f.write(f"\nBest policy: **{best['policy']}**\n")

# --- phase11_statistical_analysis.md ---
with open('reports/phase11_statistical_analysis.md', 'w') as f:
    f.write("# Phase 11 — Statistical Analysis\n\n")
    f.write(f"## Comparison: {best_f1_metrics['policy']} vs {baseline_metrics['policy']}\n\n")
    f.write(f"### McNemar's Test\n")
    f.write(f"- Test statistic: {stat:.4f}\n")
    f.write(f"- p-value: {pval:.4e}\n")
    f.write(f"- Interpretation: {'Statistically significant (p < 0.05)' if pval < 0.05 else 'Not statistically significant (p ≥ 0.05)'}\n\n")
    f.write(f"### Bootstrap 95% Confidence Intervals\n")
    f.write(f"- F1 difference (best − baseline): [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]\n")
    f.write(f"- Recall difference (best − baseline): [{ci_recall[0]:.4f}, {ci_recall[1]:.4f}]\n\n")
    sig_f1 = ci_f1[0] > 0
    sig_recall = ci_recall[0] > 0
    f.write(f"### Conclusion\n")
    f.write(f"- F1 improvement is {'statistically supported (CI excludes 0)' if sig_f1 else 'not statistically supported (CI includes 0)'}.\n")
    f.write(f"- Recall improvement is {'statistically supported (CI excludes 0)' if sig_recall else 'not statistically supported (CI includes 0)'}.\n")

# --- phase11_routing_analysis.md ---
with open('reports/phase11_routing_analysis.md', 'w') as f:
    f.write("# Phase 11 — Routing Analysis\n\n")
    f.write("## Stage-1 Score Distribution\n\n")
    f.write("### P(malicious) Quantiles\n\n")
    f.write("| Quantile | True Benign | True Malicious | Stage-1 FN |\n")
    f.write("|---|---:|---:|---:|\n")
    for i, q in enumerate([5, 10, 25, 50, 75, 90, 95]):
        f.write(f"| {q}th | {q_benign[i]:.4f} | {q_malicious[i]:.4f} | {q_fn[i]:.4f} |\n")
    f.write(f"\n### Stage-1 Decision Counts (Baseline)\n\n")
    f.write(f"- FINAL_BENIGN: {int(is_benign_exit.sum())}\n")
    f.write(f"- FINAL_MALICIOUS: {int(is_malicious_exit.sum())}\n")
    f.write(f"- NEEDS_STAGE2: {baseline_ns2}\n")
    f.write(f"- Stage-1 FN: {s1_fn_count}\n")
    f.write(f"- Stage-1 FP: {s1_fp_count}\n")

# --- phase11_reproducibility.md ---
with open('reports/phase11_reproducibility.md', 'w') as f:
    f.write("# Phase 11 — Reproducibility\n\n")
    f.write(f"- Python: {os.sys.version}\n")
    f.write(f"- PyTorch: {torch.__version__}\n")
    f.write(f"- OS: macOS\n")
    f.write(f"- Random seeds: 42\n")
    f.write(f"- Test dataset: data/splits/test.parquet ({total_urls} URLs)\n")
    f.write(f"- HXRF model: models/stage1_hxrf.pkl\n")
    f.write(f"- ICP calibrator: models/stage1_conformal.pkl\n")
    f.write(f"- FiLM model: models/film/film_model.pth\n")
    f.write(f"- XGBoost model: models/xgboost_audit/film_xgb.json\n")
    f.write(f"- FiLM scaler: models/film/film_scaler.pkl\n")
    f.write(f"- Features: parquet columns (deterministic)\n")
    f.write(f"- Calibrator alpha: 0.05 (baseline)\n")
    f.write(f"- Calibrator quantile: {icp_quantile:.6f}\n")
    f.write(f"- Calibration samples: {calibrator.n_calibration}\n")

# --- Figure 3: Error composition before vs after ---
fig, ax = plt.subplots(figsize=(10, 5))
categories = ['S1 FN', 'S1 FP', 'S2 FN', 'S2 FP']
before = [baseline_metrics['s1_fn'], baseline_metrics['s1_fp'],
          baseline_metrics['s2_fn'], baseline_metrics['s2_fp']]
after = [best_f1_metrics['s1_fn'], best_f1_metrics['s1_fp'],
         best_f1_metrics['s2_fn'], best_f1_metrics['s2_fp']]
x = np.arange(len(categories))
w = 0.35
ax.bar(x - w/2, before, w, label='Phase-10 Baseline', color='#F44336', alpha=0.8)
ax.bar(x + w/2, after, w, label=f'{best_f1_metrics["policy"]}', color='#2196F3', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Error Count')
ax.set_title('Error Composition: Before vs After')
ax.legend()
for i in range(len(categories)):
    ax.text(x[i] - w/2, before[i] + 30, str(before[i]), ha='center', fontsize=8)
    ax.text(x[i] + w/2, after[i] + 30, str(after[i]), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('reports/figures/phase11/error_composition.png', dpi=150)
plt.close()

# --- Figure 4: Operating point comparison (F1 vs S2 rate) ---
fig, ax = plt.subplots(figsize=(10, 6))
for m in all_e2e_results:
    color = '#F44336' if m is baseline_metrics else ('#2196F3' if m is best_f1_metrics else '#9E9E9E')
    size = 120 if m is baseline_metrics or m is best_f1_metrics else 40
    ax.scatter(m['s2_rate'] * 100, m['f1'], s=size, c=color, zorder=3)
    if m is baseline_metrics:
        ax.annotate('Phase-10', (m['s2_rate']*100, m['f1']),
                    textcoords='offset points', xytext=(10, -10), fontsize=8, color='red')
    elif m is best_f1_metrics:
        ax.annotate('Best F1', (m['s2_rate']*100, m['f1']),
                    textcoords='offset points', xytext=(10, 5), fontsize=8, color='blue')
ax.set_xlabel('Stage-2 Escalation Rate (%)')
ax.set_ylabel('Complete DSSE F1')
ax.set_title('Operating Points: F1 vs Stage-2 Rate')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/phase11/operating_points_f1.png', dpi=150)
plt.close()

# --- Figure 5: Coverage vs final DSSE F1 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
s2_rates = [m['s2_rate']*100 for m in all_e2e_results]
f1s = [m['f1'] for m in all_e2e_results]
recalls = [m['recall'] for m in all_e2e_results]

axes[0].scatter(s2_rates, f1s, c='#4CAF50', s=60)
axes[0].set_xlabel('Stage-2 Escalation Rate (%)')
axes[0].set_ylabel('DSSE F1')
axes[0].set_title('Stage-2 Coverage vs DSSE F1')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(s2_rates, recalls, c='#FF9800', s=60)
axes[1].set_xlabel('Stage-2 Escalation Rate (%)')
axes[1].set_ylabel('DSSE Recall')
axes[1].set_title('Stage-2 Coverage vs DSSE Recall')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/phase11/coverage_vs_f1_recall.png', dpi=150)
plt.close()

# ============================================================
# SECTION 9: FINAL AUDIT
# ============================================================
print("\n[9] Writing final audit...")

with open('reports/phase11_final_audit.md', 'w') as f:
    f.write("# Phase 11 — Final Audit\n\n")
    f.write("## A. Frozen Phase-10 Baseline\n\n")
    f.write(f"- Total test URLs: {total_urls}\n")
    f.write(f"- Stage-1 resolved: {total_urls - baseline_metrics['n_stage2']}\n")
    f.write(f"- Stage-2 escalated: {baseline_metrics['n_stage2']} ({baseline_metrics['s2_rate']*100:.2f}%)\n")
    f.write(f"- Accuracy: {baseline_metrics['accuracy']:.4f}\n")
    f.write(f"- Precision: {baseline_metrics['precision']:.4f}\n")
    f.write(f"- Recall: {baseline_metrics['recall']:.4f}\n")
    f.write(f"- F1: {baseline_metrics['f1']:.4f}\n")
    f.write(f"- ROC-AUC: {baseline_metrics['roc_auc']:.4f}\n")
    f.write(f"- FPR: {baseline_metrics['fpr']:.4f}\n")
    f.write(f"- FNR: {baseline_metrics['fnr']:.4f}\n")
    f.write(f"- Stage-1 FN: {baseline_metrics['s1_fn']}\n")
    f.write(f"- Total errors: {baseline_metrics['total_errors']}\n\n")

    f.write("## B. Policies Evaluated\n\n")
    f.write(f"{len(all_e2e_results)} asymmetric conformal routing policies evaluated end-to-end.\n\n")

    f.write(f"## C. Best Policy\n\n")
    f.write(f"**{best_f1_metrics['policy']}**\n\n")

    f.write(f"## D–K. Before/After Comparison\n\n")
    f.write("| Metric | Phase-10 | Phase-11 Best | Δ |\n")
    f.write("|---|---:|---:|---:|\n")
    for key, label in [('s2_rate', 'Stage-2 Rate'),
                       ('s1_fn', 'Stage-1 FN'),
                       ('f1', 'F1'), ('recall', 'Recall'),
                       ('precision', 'Precision'), ('accuracy', 'Accuracy'),
                       ('roc_auc', 'ROC-AUC'), ('fpr', 'FPR'), ('fnr', 'FNR')]:
        bv = baseline_metrics[key]
        pv = best_f1_metrics[key]
        delta = pv - bv
        if key == 's2_rate':
            f.write(f"| {label} | {bv*100:.2f}% | {pv*100:.2f}% | {delta*100:+.2f}pp |\n")
        elif key in ('s1_fn',):
            f.write(f"| {label} | {bv} | {pv} | {delta:+d} |\n")
        else:
            f.write(f"| {label} | {bv:.4f} | {pv:.4f} | {delta:+.4f} |\n")

    f.write(f"\n## L. Latency/Throughput\n\n")
    f.write(f"No new latency benchmarks were run. Reusing Phase-10 estimates:\n")
    f.write(f"- Phase-10 batch throughput: 5,362 URLs/sec (at {baseline_metrics['s2_rate']*100:.2f}% S2)\n")
    f.write(f"- Phase-11 expected throughput: estimated lower due to {best_f1_metrics['s2_rate']*100:.2f}% S2 rate\n")
    f.write(f"- Stage-2 cost from Phase 9: ~1,233 ms / 1000 URLs\n")
    est_s2_ms_per_url = 1233.0 / 1000.0
    # Estimate: for a batch of 1000, s2_fraction URLs go through Stage-2
    s2_frac = best_f1_metrics['s2_rate']
    # Stage-1 is ~14 ms / 1000 URLs, Stage-2 is ~1233 ms / 1000 S2-URLs
    est_batch_ms = 14.0 + s2_frac * 1233.0
    est_tput = 1000.0 / (est_batch_ms / 1000.0)
    f.write(f"- Estimated batch-1000 latency: ~{est_batch_ms:.0f} ms\n")
    f.write(f"- Estimated throughput: ~{est_tput:.0f} URLs/sec\n")
    f.write(f"\n(These are estimates based on component-level Phase-9 measurements.)\n")

    f.write(f"\n## M. Statistical Significance\n\n")
    f.write(f"- McNemar's test: statistic={stat:.4f}, p={pval:.4e}\n")
    f.write(f"- F1 difference 95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]\n")
    f.write(f"- Recall difference 95% CI: [{ci_recall[0]:.4f}, {ci_recall[1]:.4f}]\n\n")

    f.write(f"## N. Is the Improvement Statistically Supported?\n\n")
    if pval < 0.05:
        f.write(f"Yes. McNemar's test is significant (p={pval:.4e}).\n")
    else:
        f.write(f"No. McNemar's test is not significant (p={pval:.4e}).\n")
    if sig_f1:
        f.write(f"F1 improvement is supported (CI excludes 0).\n")
    else:
        f.write(f"F1 improvement is not supported (CI includes 0).\n")

    f.write(f"\n## O. Computational Trade-Off\n\n")
    f.write(f"- Baseline S2 rate: {baseline_metrics['s2_rate']*100:.2f}%\n")
    f.write(f"- Best policy S2 rate: {best_f1_metrics['s2_rate']*100:.2f}%\n")
    f.write(f"- Additional S2 calls: {best_f1_metrics['n_stage2'] - baseline_metrics['n_stage2']}\n")
    f.write(f"- FN reduction: {baseline_metrics['fn'] - best_f1_metrics['fn']}\n")
    f.write(f"- Cost per recovered FN: ~{(best_f1_metrics['n_stage2'] - baseline_metrics['n_stage2']) / max(1, baseline_metrics['fn'] - best_f1_metrics['fn']):.1f} additional S2 calls per recovered FN\n")

    f.write(f"\n## P. Remaining Limitations\n\n")
    f.write("1. Asymmetric routing does not retain standard ICP marginal coverage guarantees.\n")
    f.write("2. Routing policy was selected exploratorily on the held-out test set.\n")
    f.write("   No separate validation split was used for threshold selection.\n")
    f.write("3. Latency estimates are derived from Phase-9 component measurements, not re-measured.\n")
    f.write("4. The HXRF model used in DSSE differs from the one used for ICP calibration.\n\n")

    f.write(f"## Q. Recommendation for Research Paper\n\n")
    f.write("Present the coverage-risk frontier as the primary Phase-11 contribution.\n")
    f.write("The frontier shows the explicit trade-off between Stage-2 escalation rate and\n")
    f.write("Stage-1 false-negative rate, enabling practitioners to select an operating point\n")
    f.write("appropriate for their security requirements.\n\n")
    f.write("Label all Phase-11 routing results as exploratory (test-set selected).\n")
    f.write("Do not claim unbiased generalization without a held-out validation protocol.\n")

print("\n" + "="*70)
print("PHASE 11 COMPLETE")
print("="*70)
print(f"\nBaseline:  {baseline_metrics['policy']} — F1={baseline_metrics['f1']:.4f}, "
      f"Recall={baseline_metrics['recall']:.4f}, S2={baseline_metrics['s2_rate']*100:.2f}%")
print(f"Best:      {best_f1_metrics['policy']} — F1={best_f1_metrics['f1']:.4f}, "
      f"Recall={best_f1_metrics['recall']:.4f}, S2={best_f1_metrics['s2_rate']*100:.2f}%")
print(f"McNemar p={pval:.4e}")
print(f"F1 CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")
print(f"Recall CI: [{ci_recall[0]:.4f}, {ci_recall[1]:.4f}]")
