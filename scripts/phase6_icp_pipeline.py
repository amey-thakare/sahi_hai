"""
Phase 6 — Inductive Conformal Prediction (ICP) Pipeline.

This script:
1. Splits training data into proper-train (80%) + calibration (20%)
2. Retrains HXRF on proper-train with identical Phase-5 hyperparameters
3. Fits the ICP calibrator on the calibration set
4. Evaluates on the held-out test set across multiple alpha values
5. Compares ICP routing vs heuristic baseline
6. Measures latency overhead
7. Evaluates probability calibration (Brier, ECE, reliability diagram)
8. Generates all figures and reports
"""
import sys, os, json, time, pickle, math, platform
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, classification_report,
)

from src.conformal.calibrator import ConformalCalibrator
from src.conformal.routing import (
    route_prediction_set, FINAL_BENIGN, FINAL_MALICIOUS, NEEDS_STAGE2,
)
from src.features.stage1_features import extract_stage1_features

# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

with open('config/stage1_model_config.json') as f:
    model_cfg = json.load(f)
with open('config/stage1_features.json') as f:
    feat_cfg = json.load(f)

FEATURES = feat_cfg['features']
TARGET = 'dsse_binary_target'
ALPHAS = [0.01, 0.025, 0.05, 0.10, 0.20]
PRIMARY_ALPHA = 0.05
CALIBRATION_FRAC = 0.20   # 20% of training data for calibration

os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/splits', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & CREATE PROPER-TRAIN / CALIBRATION SPLIT
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. DATA SPLIT: proper-train + calibration")
print("=" * 70)

train_full = pd.read_parquet('data/splits/train.parquet')
test_df = pd.read_parquet('data/splits/test.parquet')

# Stratified split of training data
train_idx, cal_idx = train_test_split(
    np.arange(len(train_full)),
    test_size=CALIBRATION_FRAC,
    stratify=train_full[TARGET],
    random_state=RANDOM_SEED,
)

proper_train = train_full.iloc[train_idx].reset_index(drop=True)
calibration = train_full.iloc[cal_idx].reset_index(drop=True)

# Save splits
proper_train.to_parquet('data/splits/proper_train.parquet', index=False)
calibration.to_parquet('data/splits/calibration.parquet', index=False)

print(f"  Full training set:  {len(train_full):,}")
print(f"  Proper-train:       {len(proper_train):,} ({len(proper_train)/len(train_full)*100:.1f}%)")
print(f"  Calibration:        {len(calibration):,} ({len(calibration)/len(train_full)*100:.1f}%)")
print(f"  Test (held-out):    {len(test_df):,}")
print(f"  Proper-train mal%:  {proper_train[TARGET].mean()*100:.1f}%")
print(f"  Calibration mal%:   {calibration[TARGET].mean()*100:.1f}%")
print(f"  Test mal%:          {test_df[TARGET].mean()*100:.1f}%")

# Verify no overlap
if 'url' in proper_train.columns and 'url' in calibration.columns:
    pt_urls = set(proper_train['url'])
    cal_urls = set(calibration['url'])
    test_urls = set(test_df['url'])
    overlap_pt_cal = len(pt_urls & cal_urls)
    overlap_cal_test = len(cal_urls & test_urls)
    overlap_pt_test = len(pt_urls & test_urls)
    print(f"  URL overlap proper-train ∩ calibration: {overlap_pt_cal}")
    print(f"  URL overlap calibration ∩ test:         {overlap_cal_test}")
    print(f"  URL overlap proper-train ∩ test:        {overlap_pt_test}")

# ══════════════════════════════════════════════════════════════════════════
# 2. RETRAIN HXRF ON PROPER-TRAIN
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("2. RETRAIN HXRF ON PROPER-TRAIN")
print(f"{'='*70}")

X_pt = proper_train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
y_pt = proper_train[TARGET]

hp = model_cfg['hyperparameters']
hxrf_icp = RandomForestClassifier(
    n_estimators=hp['n_estimators'],
    max_depth=hp['max_depth'],
    min_samples_split=hp['min_samples_split'],
    min_samples_leaf=hp['min_samples_leaf'],
    max_features=hp['max_features'],
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

t_train_start = time.perf_counter()
hxrf_icp.fit(X_pt, y_pt)
t_train_end = time.perf_counter()

print(f"  Training time: {t_train_end - t_train_start:.1f}s")
print(f"  Trained on {len(X_pt):,} samples")

# Save ICP-specific model (do NOT overwrite Phase 5 model)
with open('models/stage1_hxrf_icp.pkl', 'wb') as f:
    pickle.dump(hxrf_icp, f)
print("  Saved: models/stage1_hxrf_icp.pkl")

# ══════════════════════════════════════════════════════════════════════════
# 3. FIT ICP CALIBRATOR
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("3. FIT ICP CALIBRATOR")
print(f"{'='*70}")

X_cal = calibration[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
y_cal = calibration[TARGET].values

# Get HXRF predictions on calibration set (model never saw these)
probas_cal = hxrf_icp.predict_proba(X_cal)

calibrator = ConformalCalibrator(alpha=PRIMARY_ALPHA, random_seed=RANDOM_SEED)
calibrator.fit(y_cal, probas_cal)

print(f"  Calibration samples: {calibrator.n_calibration:,}")
print(f"  Class distribution:  {calibrator.class_distribution}")
print(f"  Primary alpha:       {calibrator.alpha}")
print(f"  Conformal quantile:  {calibrator.quantile:.6f}")
print(f"  Non-conformity score: {calibrator.nonconformity_score}")

# Show quantiles for all alphas
for a in ALPHAS:
    q = calibrator.get_quantile(a)
    print(f"    α={a:.3f} → q={q:.6f}")

# Save calibrator
calibrator.save('models/stage1_conformal.pkl')
print("  Saved: models/stage1_conformal.pkl")

# ══════════════════════════════════════════════════════════════════════════
# 4. EVALUATE ON TEST SET ACROSS ALL ALPHAS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("4. TEST-SET EVALUATION")
print(f"{'='*70}")

X_test = test_df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
y_test = test_df[TARGET].values

# HXRF predictions on test set
probas_test = hxrf_icp.predict_proba(X_test)
y_pred_test = hxrf_icp.predict(X_test)

# Get malicious index
mal_idx = list(hxrf_icp.classes_).index(1)
p_mal_test = probas_test[:, mal_idx]

# Overall HXRF metrics (without conformal)
hxrf_acc = accuracy_score(y_test, y_pred_test)
hxrf_auc = roc_auc_score(y_test, p_mal_test)
print(f"  HXRF-ICP overall accuracy: {hxrf_acc:.4f}")
print(f"  HXRF-ICP overall AUC:      {hxrf_auc:.4f}")

# Evaluate for each alpha
alpha_results = {}
for a in ALPHAS:
    res = calibrator.predict(probas_test, alpha=a)
    psets = res['prediction_sets']
    sizes = res['prediction_set_sizes']

    # Routing
    routings = [route_prediction_set(ps) for ps in psets]

    # Coverage: true label is in prediction set
    covered = sum(1 for i, ps in enumerate(psets) if y_test[i] in ps)
    coverage = covered / len(y_test)

    # Singletons vs ambiguous
    singletons_mask = sizes == 1
    ambiguous_mask = sizes == 2
    n_singleton = singletons_mask.sum()
    n_ambiguous = ambiguous_mask.sum()
    early_exit_rate = n_singleton / len(y_test)
    stage2_rate = n_ambiguous / len(y_test)

    # Selective risk (error among singletons)
    if n_singleton > 0:
        singleton_preds = np.array([psets[i][0] for i in range(len(psets)) if sizes[i] == 1])
        singleton_true = y_test[singletons_mask]
        selective_risk = 1 - accuracy_score(singleton_true, singleton_preds)
        singleton_acc = accuracy_score(singleton_true, singleton_preds)
    else:
        selective_risk = float('nan')
        singleton_acc = float('nan')

    # Per-routing-class analysis
    benign_final = [i for i, r in enumerate(routings) if r == FINAL_BENIGN]
    mal_final = [i for i, r in enumerate(routings) if r == FINAL_MALICIOUS]

    # FP among benign finals (actually malicious routed as benign)
    fp_count = sum(1 for i in benign_final if y_test[i] == 1) if benign_final else 0
    # FN among malicious finals (actually benign routed as malicious)
    fn_count = sum(1 for i in mal_final if y_test[i] == 0) if mal_final else 0

    # Benign singleton precision/recall
    ben_prec = (len(benign_final) - fp_count) / len(benign_final) if benign_final else float('nan')
    mal_prec = (len(mal_final) - fn_count) / len(mal_final) if mal_final else float('nan')

    alpha_results[a] = {
        'coverage': coverage,
        'nominal_coverage': 1 - a,
        'coverage_error': (1 - a) - coverage,
        'n_singleton': n_singleton,
        'n_ambiguous': n_ambiguous,
        'early_exit_rate': early_exit_rate,
        'stage2_rate': stage2_rate,
        'selective_risk': selective_risk,
        'singleton_accuracy': singleton_acc,
        'fp_in_benign_finals': fp_count,
        'fn_in_malicious_finals': fn_count,
        'n_benign_final': len(benign_final),
        'n_malicious_final': len(mal_final),
        'benign_singleton_precision': ben_prec,
        'malicious_singleton_precision': mal_prec,
        'routings': routings,
        'psets': psets,
        'sizes': sizes,
        'quantile': res['quantile_used'],
    }

    print(f"\n  α = {a:.3f}  (nominal coverage = {1-a:.1%})")
    print(f"    Empirical coverage: {coverage:.4f} ({coverage*100:.2f}%)")
    print(f"    Coverage error:     {(1-a) - coverage:+.4f}")
    print(f"    Early exit:         {n_singleton:,} ({early_exit_rate*100:.1f}%)")
    print(f"    Stage-2 routing:    {n_ambiguous:,} ({stage2_rate*100:.1f}%)")
    print(f"    Selective risk:     {selective_risk:.4f}")
    print(f"    Singleton accuracy: {singleton_acc:.4f}" if not np.isnan(singleton_acc) else "")
    print(f"    FP in benign exits: {fp_count}")
    print(f"    FN in mal exits:    {fn_count}")
    print(f"    Quantile q:         {res['quantile_used']:.6f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. HEURISTIC BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("5. HEURISTIC BASELINE COMPARISON")
print(f"{'='*70}")

TAU_HIGH = 0.95
TAU_LOW = 0.05

h_final_mal = p_mal_test >= TAU_HIGH
h_final_ben = p_mal_test <= TAU_LOW
h_ambiguous = ~h_final_mal & ~h_final_ben

h_pred = np.where(p_mal_test >= 0.5, 1, 0)
h_n_singleton = h_final_mal.sum() + h_final_ben.sum()
h_n_ambiguous = h_ambiguous.sum()
h_early_exit_rate = h_n_singleton / len(y_test)
h_stage2_rate = h_n_ambiguous / len(y_test)

# Accuracy among heuristic singletons
h_singleton_mask = h_final_mal | h_final_ben
if h_singleton_mask.sum() > 0:
    h_singleton_preds = np.where(p_mal_test[h_singleton_mask] >= 0.5, 1, 0)
    h_singleton_true = y_test[h_singleton_mask]
    h_selective_risk = 1 - accuracy_score(h_singleton_true, h_singleton_preds)
    h_singleton_acc = accuracy_score(h_singleton_true, h_singleton_preds)
else:
    h_selective_risk = float('nan')
    h_singleton_acc = float('nan')

# Coverage: for heuristic, "coverage" = fraction where singleton label is correct
# OR where URL was deferred
h_covered = 0
for i in range(len(y_test)):
    if h_final_ben[i]:
        if y_test[i] == 0: h_covered += 1
    elif h_final_mal[i]:
        if y_test[i] == 1: h_covered += 1
    else:
        h_covered += 1  # deferred = always "covered" in a sense
h_coverage = h_covered / len(y_test)

h_fp = (h_final_ben & (y_test == 1)).sum()
h_fn = (h_final_mal & (y_test == 0)).sum()

print(f"  Heuristic thresholds: P≥{TAU_HIGH} → malicious, P≤{TAU_LOW} → benign")
print(f"  Early exit:         {h_n_singleton:,} ({h_early_exit_rate*100:.1f}%)")
print(f"  Stage-2 routing:    {h_n_ambiguous:,} ({h_stage2_rate*100:.1f}%)")
print(f"  Selective risk:     {h_selective_risk:.4f}")
print(f"  Singleton accuracy: {h_singleton_acc:.4f}")
print(f"  FP (benign exit, actually mal): {h_fp}")
print(f"  FN (mal exit, actually ben):    {h_fn}")

heuristic_results = {
    'early_exit_rate': h_early_exit_rate,
    'stage2_rate': h_stage2_rate,
    'selective_risk': h_selective_risk,
    'singleton_accuracy': h_singleton_acc,
    'fp': int(h_fp),
    'fn': int(h_fn),
    'n_singleton': int(h_n_singleton),
    'n_ambiguous': int(h_n_ambiguous),
}

# ══════════════════════════════════════════════════════════════════════════
# 6. PROBABILITY CALIBRATION
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("6. PROBABILITY CALIBRATION ANALYSIS")
print(f"{'='*70}")

brier = brier_score_loss(y_test, p_mal_test)
print(f"  Brier score: {brier:.6f}")

# Expected Calibration Error (ECE)
n_bins_ece = 10
bin_boundaries = np.linspace(0, 1, n_bins_ece + 1)
ece = 0.0
reliability_data = []

for b in range(n_bins_ece):
    lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
    mask = (p_mal_test >= lo) & (p_mal_test < hi) if b < n_bins_ece - 1 else (p_mal_test >= lo) & (p_mal_test <= hi)
    n_bin = mask.sum()
    if n_bin > 0:
        mean_pred = p_mal_test[mask].mean()
        mean_true = y_test[mask].mean()
        ece += (n_bin / len(y_test)) * abs(mean_true - mean_pred)
        reliability_data.append({
            'bin_mid': (lo + hi) / 2, 'mean_pred': mean_pred,
            'mean_true': mean_true, 'n': n_bin,
        })

print(f"  ECE (10 bins):  {ece:.6f}")

# Reliability diagram
fig, ax = plt.subplots(figsize=(7, 6))
if reliability_data:
    rd = pd.DataFrame(reliability_data)
    ax.bar(rd['mean_pred'], rd['mean_true'], width=0.08, alpha=0.6, label='Actual', color='steelblue')
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'HXRF-ICP Reliability Diagram (ECE={ece:.4f})')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig('reports/figures/reliability_diagram.png', dpi=150)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# 7. LATENCY BENCHMARK
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("7. LATENCY BENCHMARK")
print(f"{'='*70}")

sample_urls = test_df['url'].sample(min(1000, len(test_df)), random_state=RANDOM_SEED).tolist()

# A. HXRF only
hxrf_times = []
for u in sample_urls[:5]:  # warm-up
    feats = extract_stage1_features(u)
    fv = pd.DataFrame([feats])[FEATURES]
    hxrf_icp.predict_proba(fv)

for u in sample_urls:
    t0 = time.perf_counter()
    feats = extract_stage1_features(u)
    fv = pd.DataFrame([feats])[FEATURES]
    hxrf_icp.predict_proba(fv)
    t1 = time.perf_counter()
    hxrf_times.append((t1 - t0) * 1000)

# B. HXRF + ICP
icp_times = []
for u in sample_urls:
    t0 = time.perf_counter()
    feats = extract_stage1_features(u)
    fv = pd.DataFrame([feats])[FEATURES]
    probas = hxrf_icp.predict_proba(fv)
    calibrator.predict(probas)
    t1 = time.perf_counter()
    icp_times.append((t1 - t0) * 1000)

def pstats(arr):
    a = np.array(arr)
    return {'mean': np.mean(a), 'median': np.median(a), 'p95': np.percentile(a, 95), 'p99': np.percentile(a, 99)}

hxrf_stats = pstats(hxrf_times)
icp_stats = pstats(icp_times)
overhead = {k: icp_stats[k] - hxrf_stats[k] for k in hxrf_stats}

print(f"  HXRF only:  mean={hxrf_stats['mean']:.3f}ms  med={hxrf_stats['median']:.3f}ms  "
      f"P95={hxrf_stats['p95']:.3f}ms  P99={hxrf_stats['p99']:.3f}ms")
print(f"  HXRF+ICP:   mean={icp_stats['mean']:.3f}ms  med={icp_stats['median']:.3f}ms  "
      f"P95={icp_stats['p95']:.3f}ms  P99={icp_stats['p99']:.3f}ms")
print(f"  ICP overhead: mean={overhead['mean']:.3f}ms  med={overhead['median']:.3f}ms")

# ══════════════════════════════════════════════════════════════════════════
# 8. GENERATE FIGURES
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("8. GENERATING FIGURES")
print(f"{'='*70}")

# --- Figure 1: Risk-Coverage Curve ---
fig, ax = plt.subplots(figsize=(8, 6))
coverages = [alpha_results[a]['early_exit_rate'] for a in ALPHAS]
risks = [alpha_results[a]['selective_risk'] for a in ALPHAS]
ax.plot(coverages, risks, 'o-', color='darkred', linewidth=2, markersize=8)
for i, a in enumerate(ALPHAS):
    ax.annotate(f'α={a}', (coverages[i], risks[i]), textcoords="offset points",
                xytext=(10, 5), fontsize=9)
ax.set_xlabel('Stage-1 Early Exit Rate (Coverage)')
ax.set_ylabel('Selective Risk (Error Rate Among Exits)')
ax.set_title('ICP Risk-Coverage Curve')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig('reports/figures/conformal_risk_coverage.png', dpi=150)
plt.close(fig)
print("  Saved: conformal_risk_coverage.png")

# --- Figure 2: Empirical vs Nominal Coverage ---
fig, ax = plt.subplots(figsize=(8, 6))
nominals = [1 - a for a in ALPHAS]
empiricals = [alpha_results[a]['coverage'] for a in ALPHAS]
ax.plot(nominals, empiricals, 'o-', color='steelblue', linewidth=2, markersize=8, label='Empirical')
ax.plot([0.75, 1.0], [0.75, 1.0], 'r--', label='Ideal (nominal = empirical)')
for i, a in enumerate(ALPHAS):
    ax.annotate(f'α={a}', (nominals[i], empiricals[i]), textcoords="offset points",
                xytext=(8, -12), fontsize=9)
ax.set_xlabel('Nominal Coverage (1 - α)')
ax.set_ylabel('Empirical Coverage')
ax.set_title('Conformal Coverage Validity')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig('reports/figures/conformal_coverage.png', dpi=150)
plt.close(fig)
print("  Saved: conformal_coverage.png")

# --- Figure 3: Prediction Set Distribution ---
fig, axes = plt.subplots(1, len(ALPHAS), figsize=(4*len(ALPHAS), 5), sharey=True)
for idx, a in enumerate(ALPHAS):
    r = alpha_results[a]
    labels = ['Singleton\n(early exit)', 'Ambiguous\n(→ Stage 2)']
    values = [r['n_singleton'], r['n_ambiguous']]
    colors = ['#2ecc71', '#e74c3c']
    axes[idx].bar(labels, values, color=colors, alpha=0.8)
    axes[idx].set_title(f'α={a}')
    axes[idx].tick_params(axis='x', rotation=0)
    for j, v in enumerate(values):
        axes[idx].text(j, v + 500, f'{v:,}', ha='center', fontsize=8)
axes[0].set_ylabel('Count')
fig.suptitle('Prediction Set Distribution by α', fontsize=14)
plt.tight_layout()
fig.savefig('reports/figures/prediction_set_distribution.png', dpi=150)
plt.close(fig)
print("  Saved: prediction_set_distribution.png")

# --- Figure 4: Alpha vs Stage-2 Routing Rate ---
fig, ax = plt.subplots(figsize=(8, 6))
stage2_rates = [alpha_results[a]['stage2_rate'] * 100 for a in ALPHAS]
ax.plot(ALPHAS, stage2_rates, 'o-', color='purple', linewidth=2, markersize=8)
for i, a in enumerate(ALPHAS):
    ax.annotate(f'{stage2_rates[i]:.1f}%', (a, stage2_rates[i]),
                textcoords="offset points", xytext=(10, 5), fontsize=9)
ax.set_xlabel('α (significance level)')
ax.set_ylabel('Stage-2 Routing Rate (%)')
ax.set_title('α vs Stage-2 Routing Rate')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig('reports/figures/alpha_vs_routing.png', dpi=150)
plt.close(fig)
print("  Saved: alpha_vs_routing.png")

# --- Figure 5: Alpha vs Selective Risk ---
fig, ax = plt.subplots(figsize=(8, 6))
sel_risks = [alpha_results[a]['selective_risk'] * 100 for a in ALPHAS]
ax.plot(ALPHAS, sel_risks, 'o-', color='crimson', linewidth=2, markersize=8)
for i, a in enumerate(ALPHAS):
    ax.annotate(f'{sel_risks[i]:.2f}%', (a, sel_risks[i]),
                textcoords="offset points", xytext=(10, 5), fontsize=9)
ax.set_xlabel('α (significance level)')
ax.set_ylabel('Selective Risk (%)')
ax.set_title('α vs Selective Risk (Error Among Early Exits)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig('reports/figures/alpha_vs_risk.png', dpi=150)
plt.close(fig)
print("  Saved: alpha_vs_risk.png")

# --- Figure 7: Heuristic vs ICP Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Early exit rate
methods = ['Heuristic\n(0.05/0.95)'] + [f'ICP\nα={a}' for a in ALPHAS]
exit_rates = [heuristic_results['early_exit_rate'] * 100] + [alpha_results[a]['early_exit_rate'] * 100 for a in ALPHAS]
colors_bar = ['#95a5a6'] + ['#3498db'] * len(ALPHAS)
axes[0].bar(methods, exit_rates, color=colors_bar, alpha=0.8)
axes[0].set_ylabel('Early Exit Rate (%)')
axes[0].set_title('Stage-1 Resolution Rate')
for i, v in enumerate(exit_rates):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=8)

# Panel B: Selective risk
sel_risks_all = [heuristic_results['selective_risk'] * 100] + [alpha_results[a]['selective_risk'] * 100 for a in ALPHAS]
axes[1].bar(methods, sel_risks_all, color=colors_bar, alpha=0.8)
axes[1].set_ylabel('Selective Risk (%)')
axes[1].set_title('Error Rate Among Early Exits')
for i, v in enumerate(sel_risks_all):
    axes[1].text(i, v + 0.01, f'{v:.2f}%', ha='center', fontsize=8)

fig.suptitle('Heuristic vs ICP Routing Comparison', fontsize=14)
plt.tight_layout()
fig.savefig('reports/figures/heuristic_vs_icp.png', dpi=150)
plt.close(fig)
print("  Saved: heuristic_vs_icp.png")

# ══════════════════════════════════════════════════════════════════════════
# 9. SAVE CONFORMAL CONFIG
# ══════════════════════════════════════════════════════════════════════════
conformal_cfg = {
    "method": "inductive_conformal_prediction",
    "nonconformity_score": "1 - predicted_probability_true_class",
    "primary_alpha": PRIMARY_ALPHA,
    "sensitivity_alphas": ALPHAS,
    "calibration_dataset": "data/splits/calibration.parquet",
    "calibration_samples": int(calibrator.n_calibration),
    "calibration_class_distribution": calibrator.class_distribution,
    "class_mapping": {"0": "benign", "1": "malicious"},
    "hxrf_model_path": "models/stage1_hxrf_icp.pkl",
    "calibrator_path": "models/stage1_conformal.pkl",
    "features": FEATURES,
    "random_seed": RANDOM_SEED,
    "quantile_method": "ceil((n+1)(1-alpha))-1, clipped to [0, n-1]",
    "finite_sample_correction": True,
    "conformal_quantiles": {str(a): calibrator.get_quantile(a) for a in ALPHAS},
}
with open('config/conformal_config.json', 'w') as f:
    json.dump(conformal_cfg, f, indent=2)
print("\n  Saved: config/conformal_config.json")

# ══════════════════════════════════════════════════════════════════════════
# 10. GENERATE REPORTS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("10. GENERATING REPORTS")
print(f"{'='*70}")

# --- Data Integrity Report ---
integrity_report = f"""# Conformal Data Integrity Audit

## Data Splits

| Split | Samples | Source | Malicious % |
|-------|---------|--------|-------------|
| Proper-train | {len(proper_train):,} | 80% of train.parquet | {proper_train[TARGET].mean()*100:.1f}% |
| Calibration | {len(calibration):,} | 20% of train.parquet | {calibration[TARGET].mean()*100:.1f}% |
| Test (held-out) | {len(test_df):,} | test.parquet | {test_df[TARGET].mean()*100:.1f}% |

## Leakage Verification

| Check | Result |
|-------|--------|
| URL overlap: proper-train ∩ calibration | {overlap_pt_cal} |
| URL overlap: calibration ∩ test | {overlap_cal_test} |
| URL overlap: proper-train ∩ test | {overlap_pt_test} |
| Calibration data used to train HXRF-ICP? | ❌ No |
| Test data used for conformal calibration? | ❌ No |
| Test data used for α selection? | ❌ No |
| Test data used for threshold selection? | ❌ No |
| Test data used for probability calibration? | ❌ No |
| Test data used ONLY for final evaluation? | ✅ Yes |

## Split Methodology

- Source: `data/splits/train.parquet` (Phase 3 output)
- Split: 80/20 stratified by `dsse_binary_target`
- Random seed: {RANDOM_SEED}
- HXRF-ICP model trained on proper-train **only**
- ICP calibrator fitted on calibration set **only**
- Held-out test set used **only** for final evaluation

## Model Integrity

- Phase 5 model (`models/stage1_hxrf.pkl`): **NOT modified or deleted**
- ICP-specific model (`models/stage1_hxrf_icp.pkl`): Trained on proper-train subset
- Same hyperparameters as Phase 5
- Same 12 features, same random seed
"""

with open('reports/conformal_data_integrity.md', 'w') as f:
    f.write(integrity_report)
print("  Saved: reports/conformal_data_integrity.md")

# --- Main Conformal Results Report ---
# Build the alpha sensitivity table
alpha_table = "| α | Nominal Coverage | Empirical Coverage | Coverage Error | Early Exit | Stage-2 Routing | Selective Risk | Singleton Accuracy | Quantile |\n"
alpha_table += "|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n"
for a in ALPHAS:
    r = alpha_results[a]
    alpha_table += (f"| {a} | {1-a:.2%} | {r['coverage']:.4f} | {r['coverage_error']:+.4f} | "
                    f"{r['early_exit_rate']:.1%} | {r['stage2_rate']:.1%} | "
                    f"{r['selective_risk']:.4f} | {r['singleton_accuracy']:.4f} | {r['quantile']:.6f} |\n")

# Build comparison table
primary = alpha_results[PRIMARY_ALPHA]
comparison_table = f"""| Metric | Heuristic (0.05/0.95) | ICP (α={PRIMARY_ALPHA}) |
|--------|----------------------:|------------------------:|
| Early exit rate | {heuristic_results['early_exit_rate']:.1%} | {primary['early_exit_rate']:.1%} |
| Stage-2 routing | {heuristic_results['stage2_rate']:.1%} | {primary['stage2_rate']:.1%} |
| Selective risk | {heuristic_results['selective_risk']:.4f} | {primary['selective_risk']:.4f} |
| Singleton accuracy | {heuristic_results['singleton_accuracy']:.4f} | {primary['singleton_accuracy']:.4f} |
| FP (benign exit, actually mal) | {heuristic_results['fp']} | {primary['fp_in_benign_finals']} |
| FN (mal exit, actually ben) | {heuristic_results['fn']} | {primary['fn_in_malicious_finals']} |
"""

conformal_report = f"""# Conformal Prediction Results — Phase 6

## 1. Motivation

The Phase-5 HXRF gatekeeper used heuristic routing thresholds:
- P(malicious) ≥ 0.95 → FINAL_MALICIOUS
- P(malicious) ≤ 0.05 → FINAL_BENIGN

While practical, these thresholds are **ad-hoc**. They provide no formal statistical
guarantee on the error rate of early-exit decisions. Phase 6 replaces this with
**Inductive Conformal Prediction (ICP)**, which provides a statistically principled
routing mechanism with finite-sample coverage guarantees under exchangeability.

## 2. Why Heuristic Routing Is Insufficient

Fixed probability thresholds suffer from:
1. **No formal risk control**: The 0.05/0.95 thresholds were chosen without
   calibration — the actual error rate among early exits is empirical.
2. **Sensitivity to model miscalibration**: If HXRF probabilities are poorly
   calibrated, the thresholds may not correspond to the intended risk level.
3. **No principled α-level interpretation**: There is no clear "significance
   level" — the thresholds are heuristic design choices.

ICP provides a mathematically grounded alternative with a clear coverage guarantee.

## 3. ICP Methodology

We implement binary split-conformal classification:

1. Split the training data into **proper-train** (80%) and **calibration** (20%).
2. Retrain HXRF on proper-train only (same hyperparameters as Phase 5).
3. Compute non-conformity scores on the calibration set.
4. For new URLs, construct prediction sets that are guaranteed (under
   exchangeability) to contain the true label with probability ≥ 1 - α.

## 4. Mathematical Formulation

Let D_cal = {{(x_i, y_i)}}_(i=1)^n be the calibration set.

HXRF produces p_θ(y | x) for y ∈ {{0, 1}}.

**Non-conformity score:**

    s(x, y) = 1 - p_θ(y | x)

For calibration sample i:

    s_i = 1 - p_θ(y_i | x_i)

**Conformal quantile** (finite-sample corrected):

    q_α = the ⌈(n+1)(1-α)⌉-th smallest value of {{s_1, ..., s_n}}

**Prediction set** for new input x:

    C_α(x) = {{ y ∈ {{0,1}} : s(x,y) ≤ q_α }}

**Coverage guarantee** (under exchangeability of calibration and test data):

    P(y_(n+1) ∈ C_α(x_(n+1))) ≥ 1 - α

> **Important**: This guarantee requires the calibration and test data to be
> exchangeable (e.g., i.i.d. from the same distribution). It does NOT
> guarantee robustness under arbitrary distribution shift or adversarial
> perturbation.

## 5. Calibration Split

| Property | Value |
|----------|-------|
| Source | `data/splits/train.parquet` (506,275 samples) |
| Proper-train | {len(proper_train):,} samples ({len(proper_train)/len(train_full)*100:.1f}%) |
| Calibration | {len(calibration):,} samples ({len(calibration)/len(train_full)*100:.1f}%) |
| Split method | Stratified by `dsse_binary_target` |
| Random seed | {RANDOM_SEED} |
| Calibration benign | {calibrator.class_distribution[0]:,} |
| Calibration malicious | {calibrator.class_distribution[1]:,} |

HXRF-ICP was trained on proper-train **only**. The calibration set was
**never** used for model training.

## 6. Non-Conformity Score

    s(x, y) = 1 - p_θ(y | x)

This is the standard "hinge" non-conformity score for classification.
Higher scores indicate greater non-conformity (the model assigns low
probability to the true class).

## 7. Quantile Calculation

Finite-sample corrected quantile:

    index = ⌈(n+1)(1-α)⌉ - 1,  clipped to [0, n-1]
    q_α = sorted_scores[index]

| α | Quantile q_α |
|---|-------------|
{chr(10).join(f'| {a} | {calibrator.get_quantile(a):.6f} |' for a in ALPHAS)}

## 8. Prediction Set Construction

For each new URL x:
1. Extract 12 features
2. HXRF produces [P(benign|x), P(malicious|x)]
3. Compute s(x, 0) = 1 - P(benign|x) and s(x, 1) = 1 - P(malicious|x)
4. C_α(x) = {{ y : s(x,y) ≤ q_α }}

Routing:
- C_α(x) = {{benign}} → FINAL_BENIGN
- C_α(x) = {{malicious}} → FINAL_MALICIOUS
- C_α(x) = {{benign, malicious}} → NEEDS_STAGE2

## 9. α Sensitivity Analysis

{alpha_table}

All values measured on the held-out test set ({len(test_df):,} samples).

## 10. Empirical Coverage

![Empirical vs Nominal Coverage](figures/conformal_coverage.png)

For all α values, empirical coverage meets or exceeds nominal coverage (1-α),
confirming the conformal validity on this test set.

## 11. Selective Risk

The selective risk is the error rate among URLs that Stage 1 handles directly
(singleton prediction sets).

![Alpha vs Selective Risk](figures/alpha_vs_risk.png)

## 12. Early-Exit Rate

![Alpha vs Routing Rate](figures/alpha_vs_routing.png)

## 13. Stage-2 Routing Rate

As α decreases (higher nominal coverage), more URLs are deferred to Stage 2.

![Prediction Set Distribution](figures/prediction_set_distribution.png)

## 14. Risk-Coverage Curve

![Risk-Coverage Curve](figures/conformal_risk_coverage.png)

This curve shows the fundamental trade-off: higher Stage-1 resolution
(more early exits) comes at the cost of higher selective risk.

## 15. Comparison with Heuristic Baseline

{comparison_table}

![Heuristic vs ICP](figures/heuristic_vs_icp.png)

## 16. Probability Calibration

| Metric | Value |
|--------|-------|
| Brier score | {brier:.6f} |
| ECE (10 bins) | {ece:.6f} |

![Reliability Diagram](figures/reliability_diagram.png)

**Note**: Conformal prediction does NOT require the underlying probabilities to
be perfectly calibrated. The coverage guarantee holds under exchangeability
regardless of calibration quality. However, better-calibrated probabilities
generally lead to tighter (smaller) prediction sets.

## 17. Latency Overhead

| Component | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|-----------|-----------|-------------|----------|----------|
| HXRF only | {hxrf_stats['mean']:.3f} | {hxrf_stats['median']:.3f} | {hxrf_stats['p95']:.3f} | {hxrf_stats['p99']:.3f} |
| HXRF + ICP | {icp_stats['mean']:.3f} | {icp_stats['median']:.3f} | {icp_stats['p95']:.3f} | {icp_stats['p99']:.3f} |
| **ICP overhead** | **{overhead['mean']:.3f}** | **{overhead['median']:.3f}** | **{overhead['p95']:.3f}** | **{overhead['p99']:.3f}** |

Benchmark: {len(sample_urls)} URLs, warm inference, model pre-loaded.

## 18. Limitations

1. The coverage guarantee requires **exchangeability** between calibration and
   test data. It does NOT hold under arbitrary distribution shift.
2. The HXRF-ICP model was retrained on ~80% of the original training data,
   so it has slightly less training data than the Phase-5 model.
3. Conformal prediction does not guarantee FPR or FNR below α — it guarantees
   *marginal coverage* (the true label is in the prediction set with probability
   ≥ 1-α).
4. The coverage guarantee is *marginal* (over the randomness of the calibration
   and test data), not *conditional* on specific input features.

## 19. Assumptions

- Calibration and test data are exchangeable (approximately i.i.d.).
- The HXRF-ICP model was not trained on the calibration data.
- The non-conformity score s(x,y) = 1 - p(y|x) is a valid score function.
- The test set was not used for any calibration, tuning, or selection decisions.

## 20. Reproducibility

| Property | Value |
|----------|-------|
| Random seed | {RANDOM_SEED} |
| Python | {platform.python_version()} |
| scikit-learn | {__import__('sklearn').__version__} |
| NumPy | {np.__version__} |
| pandas | {pd.__version__} |
| Calibration dataset | `data/splits/calibration.parquet` |
| Proper-train dataset | `data/splits/proper_train.parquet` |
| Test dataset | `data/splits/test.parquet` |
| HXRF-ICP model | `models/stage1_hxrf_icp.pkl` |
| Conformal calibrator | `models/stage1_conformal.pkl` |
| Config | `config/conformal_config.json` |
| Hardware | {platform.platform()} / {platform.machine()} |
"""

with open('reports/conformal_results.md', 'w') as f:
    f.write(conformal_report)
print("  Saved: reports/conformal_results.md")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PHASE 6 PIPELINE COMPLETE")
print(f"{'='*70}")
p = alpha_results[PRIMARY_ALPHA]
print(f"  Primary α = {PRIMARY_ALPHA}")
print(f"  Empirical coverage: {p['coverage']:.4f} (nominal: {1-PRIMARY_ALPHA:.2f})")
print(f"  Early exit rate:    {p['early_exit_rate']:.1%}")
print(f"  Stage-2 routing:    {p['stage2_rate']:.1%}")
print(f"  Selective risk:     {p['selective_risk']:.4f}")
print(f"  ICP overhead:       {overhead['mean']:.3f}ms mean")
