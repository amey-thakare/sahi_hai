import pandas as pd
import numpy as np
import json
import os
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

def run_feature_selection():
    print("Loading data...")
    df = pd.read_parquet('data/splits/train.parquet')
    with open('config/dsse_dataset_config.json', 'r') as f:
        config = json.load(f)
    
    features = config['engineered_features']
    target = config['target_column']
    
    # Filter features that are actually in df
    features = [f for f in features if f in df.columns]
    
    print(f"Candidate features: {len(features)}")
    
    # Check for NaN and inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    # Use subset for speed
    print("Sampling for analysis...")
    if len(df) > 10000:
        X_sub = X.sample(10000, random_state=42)
        y_sub = y.loc[X_sub.index]
    else:
        X_sub = X
        y_sub = y

    print("Training RF...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_sub, y_sub)
    
    # 1. RF Importance
    rf_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    
    print("Calculating Permutation Importance...")
    perm_imp_res = permutation_importance(rf, X_sub, y_sub, n_repeats=5, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(perm_imp_res.importances_mean, index=features).sort_values(ascending=False)
    
    print("Calculating SHAP...")
    # Use TreeExplainer for fast SHAP on RF
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sub)
    
    # shap_values might be an Explanation object or an array of shape (N, F, 2)
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = np.array(shap_values)
    
    if len(vals.shape) == 3:
        shap_vals_mal = vals[:, :, 1]
    elif isinstance(shap_values, list):
        shap_vals_mal = shap_values[1]
    else:
        shap_vals_mal = vals
        
    mean_abs_shap = pd.Series(np.abs(shap_vals_mal).mean(axis=0), index=features).sort_values(ascending=False)
    
    print("Saving plots...")
    os.makedirs('reports/figures', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals_mal, X_sub, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary.png')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    corr = X_sub[mean_abs_shap.index[:20]].corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.tight_layout()
    plt.savefig('reports/figures/top20_correlation.png')
    plt.close()
    
    results = pd.DataFrame({
        'RF_Importance': rf_imp,
        'Permutation_Importance': perm_imp,
        'Mean_Abs_SHAP': mean_abs_shap
    })
    results['Combined_Rank'] = results['RF_Importance'].rank(ascending=False) + \
                               results['Permutation_Importance'].rank(ascending=False) + \
                               results['Mean_Abs_SHAP'].rank(ascending=False)
    
    results = results.sort_values('Combined_Rank')
    
    # Save results
    results.to_csv('reports/feature_selection_metrics.csv')
    
    print("Done. Wrote metrics to reports/feature_selection_metrics.csv")

if __name__ == '__main__':
    run_feature_selection()
