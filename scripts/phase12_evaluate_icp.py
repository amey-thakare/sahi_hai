import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import time
import json
import pickle
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.conformal.calibrator import ConformalCalibrator
from src.conformal.routing import route_prediction_set, EMPTY_SET, NEEDS_STAGE2, FINAL_BENIGN, FINAL_MALICIOUS
from src.semantic.minilm_encoder import MiniLMEncoder
from src.fusion.film import FiLMFusionClassifier
from src.stage2.xgboost_classifier import Stage2XGBoost

def calculate_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan')
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'fnr': fnr,
        'fp_count': fp,
        'fn_count': fn
    }

def run_evaluation():
    print("Loading test data...")
    test_df = pd.read_parquet('data/splits/test.parquet')
    urls = test_df['url'].tolist()
    y_true = test_df['dsse_binary_target'].values
    total_urls = len(urls)
    
    with open('config/stage1_features.json', 'r') as f:
        features = json.load(f)['features']
    X_struct = test_df[features].values
    
    # 1. Load canonical models
    with open('models/stage1_hxrf.pkl', 'rb') as f:
        hxrf = pickle.load(f)
        
    calibrator = ConformalCalibrator.load('models/stage1_conformal_canonical.pkl')
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    encoder = MiniLMEncoder(device=device)
    film_scaler = joblib.load('models/film/film_scaler.pkl')
    
    with open('config/film_config.json', 'r') as f:
        film_config = json.load(f)
        
    film_model = FiLMFusionClassifier(
        semantic_dim=film_config['embedding_dim'],
        structural_dim=film_config['structural_dim'],
        hidden_dim=film_config['hidden_dim'],
        dropout=film_config['dropout'],
        stable_gamma=True
    )
    film_model.load_state_dict(torch.load('models/film/film_model.pth', map_location=device, weights_only=True))
    film_model.to(device)
    film_model.eval()
    
    xgb_model = Stage2XGBoost.load('models/xgboost_audit/film_xgb.json')
    
    # Generate HXRF probabilities
    t_start = time.perf_counter()
    stage1_probas = hxrf.predict_proba(test_df[features])
    stage1_time = (time.perf_counter() - t_start) * 1000
    
    alphas = [0.01, 0.025, 0.05, 0.10]
    
    diagnostics_report = ["# Phase 12 — Calibration Diagnostics\n"]
    coverage_risk_report = ["# Phase 12 — Coverage-Risk Analysis\n"]
    old_vs_new_report = ["# Phase 12 — Old vs New Canonical ICP\n"]
    results_report = ["# Phase 12 — End-to-End DSSE Results\n"]
    
    # For McNemar's test, we save the predictions for alpha=0.05
    new_preds_05 = None
    old_preds_05 = None
    
    # We load old Phase 11A canonical baseline metrics to populate old_vs_new
    # From phase 11A: Stage-2 = 5530 (4.37%), F1 = 0.9077, Recall = 0.8703, Accuracy = 0.9411
    
    for alpha in alphas:
        print(f"Evaluating alpha={alpha}")
        t_icp_start = time.perf_counter()
        icp_results = calibrator.predict(stage1_probas, alpha=alpha)
        icp_time = (time.perf_counter() - t_icp_start) * 1000
        
        prediction_sets = icp_results['prediction_sets']
        routes = np.array([route_prediction_set(ps) for ps in prediction_sets])
        
        # Coverage Diagnostics
        covered = np.array([y_true[i] in prediction_sets[i] for i in range(total_urls)])
        empirical_coverage = covered.mean()
        coverage_error = empirical_coverage - (1 - alpha)
        
        avg_set_size = np.mean(icp_results['prediction_set_sizes'])
        singleton_benign = (routes == FINAL_BENIGN).mean()
        singleton_mal = (routes == FINAL_MALICIOUS).mean()
        ambiguous_rate = (routes == NEEDS_STAGE2).mean()
        empty_rate = (routes == EMPTY_SET).mean()
        
        stage2_escalation_mask = (routes == NEEDS_STAGE2) | (routes == EMPTY_SET)
        escalation_rate = stage2_escalation_mask.mean()
        escalated_count = stage2_escalation_mask.sum()
        resolution_rate = 1.0 - escalation_rate
        
        diagnostics_report.append(f"## Alpha = {alpha}")
        diagnostics_report.append(f"- Empirical Coverage: {empirical_coverage:.4f}")
        diagnostics_report.append(f"- Coverage Error: {coverage_error:+.4f}")
        diagnostics_report.append(f"- Average Set Size: {avg_set_size:.3f}")
        diagnostics_report.append(f"- Singleton Benign Rate: {singleton_benign:.4f}")
        diagnostics_report.append(f"- Singleton Malicious Rate: {singleton_mal:.4f}")
        diagnostics_report.append(f"- Ambiguous (NEEDS_STAGE2) Rate: {ambiguous_rate:.4f}")
        diagnostics_report.append(f"- Empty Set Rate: {empty_rate:.4f}")
        diagnostics_report.append(f"- Stage-2 Escalation Rate: {escalation_rate:.4f}\n")
        
        # Stage 1 metrics (before stage 2) - resolving empty sets?
        # If routed to stage 2, it's evaluated by stage 2. What about the rest?
        
        y_pred_dsse = np.zeros(total_urls, dtype=int)
        y_prob_dsse = np.zeros(total_urls, dtype=float)
        
        stage1_benign_mask = (routes == FINAL_BENIGN)
        stage1_mal_mask = (routes == FINAL_MALICIOUS)
        
        y_pred_dsse[stage1_benign_mask] = 0
        y_prob_dsse[stage1_benign_mask] = stage1_probas[stage1_benign_mask, 1]
        
        y_pred_dsse[stage1_mal_mask] = 1
        y_prob_dsse[stage1_mal_mask] = stage1_probas[stage1_mal_mask, 1]
        
        # Calculate Stage 1 FPs and FNs specifically from confident exits
        s1_eval_mask = stage1_benign_mask | stage1_mal_mask
        s1_metrics = calculate_metrics(y_true[s1_eval_mask], y_pred_dsse[s1_eval_mask])
        
        # Stage 2 Execution
        t_s2_start = time.perf_counter()
        if escalated_count > 0:
            needs_stage2_indices = np.where(stage2_escalation_mask)[0]
            needs_stage2_urls = [urls[i] for i in needs_stage2_indices]
            needs_stage2_struct = X_struct[needs_stage2_indices]
            
            embs = encoder.encode_urls(needs_stage2_urls, batch_size=256)
            X_struct_scaled = film_scaler.transform(needs_stage2_struct)
            
            film_embeddings = []
            with torch.no_grad():
                for start_idx in range(0, len(embs), 2048):
                    end_idx = min(start_idx + 2048, len(embs))
                    sem_t = torch.tensor(embs[start_idx:end_idx], dtype=torch.float32).to(device)
                    str_t = torch.tensor(X_struct_scaled[start_idx:end_idx], dtype=torch.float32).to(device)
                    _, _, _, rep = film_model(sem_t, str_t)
                    film_embeddings.append(rep.cpu().numpy())
                    
            film_rep_np = np.vstack(film_embeddings)
            
            xgb_probas = xgb_model.predict_proba(film_rep_np)
            xgb_preds = (xgb_probas[:, 1] >= 0.5).astype(int)
            
            for idx_in_subset, original_idx in enumerate(needs_stage2_indices):
                y_prob_dsse[original_idx] = float(xgb_probas[idx_in_subset][1])
                y_pred_dsse[original_idx] = xgb_preds[idx_in_subset]
                
        stage2_time = (time.perf_counter() - t_s2_start) * 1000
                
        dsse_metrics = calculate_metrics(y_true, y_pred_dsse, y_prob_dsse)
        
        coverage_risk_report.append(f"## Alpha = {alpha}")
        coverage_risk_report.append(f"- Stage-1 Resolution %: {resolution_rate*100:.2f}%")
        coverage_risk_report.append(f"- Stage-2 Escalation %: {escalation_rate*100:.2f}%")
        coverage_risk_report.append(f"- Stage-1 FN: {s1_metrics['fn_count']}")
        coverage_risk_report.append(f"- Stage-1 FP: {s1_metrics['fp_count']}")
        coverage_risk_report.append(f"- Final DSSE FN: {dsse_metrics['fn_count']}")
        coverage_risk_report.append(f"- Final DSSE FP: {dsse_metrics['fp_count']}")
        coverage_risk_report.append(f"- Accuracy: {dsse_metrics['accuracy']:.4f}")
        coverage_risk_report.append(f"- Precision: {dsse_metrics['precision']:.4f}")
        coverage_risk_report.append(f"- Recall: {dsse_metrics['recall']:.4f}")
        coverage_risk_report.append(f"- F1: {dsse_metrics['f1']:.4f}")
        coverage_risk_report.append(f"- ROC-AUC: {dsse_metrics['roc_auc']:.4f}")
        coverage_risk_report.append(f"- FPR: {dsse_metrics['fpr']:.4f}")
        coverage_risk_report.append(f"- FNR: {dsse_metrics['fnr']:.4f}\n")
        
        if alpha == 0.05:
            new_preds_05 = y_pred_dsse.copy()
            # Write Old vs New comparison
            old_vs_new_report.append("| Metric | Old ICP (Phase 11A) | New Canonical ICP |")
            old_vs_new_report.append("|---|---:|---:|")
            old_vs_new_report.append(f"| Stage-2 % | 4.37% | {escalation_rate*100:.2f}% |")
            old_vs_new_report.append(f"| Accuracy | 0.9411 | {dsse_metrics['accuracy']:.4f} |")
            old_vs_new_report.append(f"| Precision | 0.9484 | {dsse_metrics['precision']:.4f} |")
            old_vs_new_report.append(f"| Recall | 0.8703 | {dsse_metrics['recall']:.4f} |")
            old_vs_new_report.append(f"| F1 | 0.9077 | {dsse_metrics['f1']:.4f} |")
            old_vs_new_report.append(f"| ROC-AUC | 0.9768 | {dsse_metrics['roc_auc']:.4f} |")
            old_vs_new_report.append(f"| FPR | 0.0236 | {dsse_metrics['fpr']:.4f} |")
            old_vs_new_report.append(f"| FNR | 0.1297 | {dsse_metrics['fnr']:.4f} |\n")
            
            # Latency logging
            total_time = stage1_time + icp_time + stage2_time
            throughput = total_urls / (total_time / 1000)
            
            lat_report = ["# Phase 12 — Latency"]
            lat_report.append(f"- Total dataset size: {total_urls} URLs")
            lat_report.append(f"- Stage-1 Batch Inference Time: {stage1_time:.2f} ms")
            lat_report.append(f"- ICP Routing Time: {icp_time:.2f} ms")
            lat_report.append(f"- Stage-2 Time (for {escalated_count} URLs): {stage2_time:.2f} ms")
            lat_report.append(f"- End-to-End Latency (Total): {total_time:.2f} ms")
            lat_report.append(f"- Throughput: {throughput:.2f} URLs/sec")
            
            with open('reports/phase12_latency.md', 'w') as f:
                f.write("\n".join(lat_report) + "\n")
                
            results_report.append(f"## Canonical DSSE Performance (alpha=0.05)")
            results_report.append(f"- Accuracy: {dsse_metrics['accuracy']:.4f}")
            results_report.append(f"- Precision: {dsse_metrics['precision']:.4f}")
            results_report.append(f"- Recall: {dsse_metrics['recall']:.4f}")
            results_report.append(f"- F1: {dsse_metrics['f1']:.4f}")
            results_report.append(f"- ROC-AUC: {dsse_metrics['roc_auc']:.4f}")
            results_report.append(f"- FPR: {dsse_metrics['fpr']:.4f}")
            results_report.append(f"- FNR: {dsse_metrics['fnr']:.4f}\n")
    
    with open('reports/phase12_calibration_diagnostics.md', 'w') as f:
        f.write("\n".join(diagnostics_report) + "\n")
        
    with open('reports/phase12_coverage_risk.md', 'w') as f:
        f.write("\n".join(coverage_risk_report) + "\n")
        
    with open('reports/phase12_old_vs_new_icp.md', 'w') as f:
        f.write("\n".join(old_vs_new_report) + "\n")
        
    with open('reports/phase12_end_to_end_results.md', 'w') as f:
        f.write("\n".join(results_report) + "\n")
        
    # Statistical analysis against Phase 11A baseline
    print("Generating statistical analysis...")
    # I need to run Phase 11A baseline using stage1_conformal.pkl to get the old preds array
    # for McNemar test.
    old_calibrator = ConformalCalibrator.load('models/stage1_conformal.pkl')
    old_icp_results = old_calibrator.predict(stage1_probas, alpha=0.05)
    old_prediction_sets = old_icp_results['prediction_sets']
    old_routes = np.array([route_prediction_set(ps) for ps in old_prediction_sets])
    
    old_preds_05 = np.zeros(total_urls, dtype=int)
    old_stage1_benign_mask = (old_routes == FINAL_BENIGN)
    old_stage1_mal_mask = (old_routes == FINAL_MALICIOUS)
    old_preds_05[old_stage1_benign_mask] = 0
    old_preds_05[old_stage1_mal_mask] = 1
    
    old_stage2_mask = (old_routes == NEEDS_STAGE2) | (old_routes == EMPTY_SET)
    old_needs_stage2_indices = np.where(old_stage2_mask)[0]
    
    if len(old_needs_stage2_indices) > 0:
        old_needs_stage2_urls = [urls[i] for i in old_needs_stage2_indices]
        old_needs_stage2_struct = X_struct[old_needs_stage2_indices]
        embs = encoder.encode_urls(old_needs_stage2_urls, batch_size=256)
        X_struct_scaled = film_scaler.transform(old_needs_stage2_struct)
        
        film_embeddings = []
        with torch.no_grad():
            for start_idx in range(0, len(embs), 2048):
                end_idx = min(start_idx + 2048, len(embs))
                sem_t = torch.tensor(embs[start_idx:end_idx], dtype=torch.float32).to(device)
                str_t = torch.tensor(X_struct_scaled[start_idx:end_idx], dtype=torch.float32).to(device)
                _, _, _, rep = film_model(sem_t, str_t)
                film_embeddings.append(rep.cpu().numpy())
                
        film_rep_np = np.vstack(film_embeddings)
        xgb_probas = xgb_model.predict_proba(film_rep_np)
        xgb_preds = (xgb_probas[:, 1] >= 0.5).astype(int)
        
        for idx_in_subset, original_idx in enumerate(old_needs_stage2_indices):
            old_preds_05[original_idx] = xgb_preds[idx_in_subset]
            
    # McNemar Test
    b = np.sum((new_preds_05 == y_true) & (old_preds_05 != y_true))
    c = np.sum((new_preds_05 != y_true) & (old_preds_05 == y_true))
    
    mcnemar_chi2 = ((abs(b - c) - 1)**2) / (b + c) if (b+c) > 0 else 0
    
    from scipy.stats import chi2
    p_value = 1.0 - chi2.cdf(mcnemar_chi2, 1)
    
    stat_report = ["# Phase 12 — Statistical Analysis"]
    stat_report.append("## McNemar's Test")
    stat_report.append(f"- Null Hypothesis: Marginal probabilities for correct classification are equal for Old ICP and New Canonical ICP (alpha=0.05).")
    stat_report.append(f"- b (New correct, Old wrong): {b}")
    stat_report.append(f"- c (Old correct, New wrong): {c}")
    stat_report.append(f"- McNemar chi2: {mcnemar_chi2:.4f}")
    stat_report.append(f"- p-value: {p_value:.4e}")
    if p_value < 0.05:
        stat_report.append("- Conclusion: The difference in performance is statistically significant.")
    else:
        stat_report.append("- Conclusion: The difference in performance is NOT statistically significant.")
        
    with open('reports/phase12_statistical_analysis.md', 'w') as f:
        f.write("\n".join(stat_report) + "\n")
        
    # Reproducibility run to check determinism of new ICP
    print("Running reproducibility check...")
    t_icp_start = time.perf_counter()
    icp_results_re1 = calibrator.predict(stage1_probas, alpha=0.05)
    prediction_sets_re1 = icp_results_re1['prediction_sets']
    routes_re1 = np.array([route_prediction_set(ps) for ps in prediction_sets_re1])
    
    icp_results_re2 = calibrator.predict(stage1_probas, alpha=0.05)
    prediction_sets_re2 = icp_results_re2['prediction_sets']
    routes_re2 = np.array([route_prediction_set(ps) for ps in prediction_sets_re2])
    
    disagreements = np.sum(routes_re1 != routes_re2)
    
    rep_report = ["# Phase 12 — Reproducibility"]
    rep_report.append("- Dataset: data/splits/test.parquet")
    rep_report.append("- Thread configuration: KMP_DUPLICATE_LIB_OK=True, TOKENIZERS_PARALLELISM=false, OMP_NUM_THREADS=1")
    rep_report.append(f"- Routing disagreement between runs: {disagreements}")
    
    with open('reports/phase12_reproducibility.md', 'w') as f:
        f.write("\n".join(rep_report) + "\n")
        
    print("Evaluation complete.")

if __name__ == '__main__':
    run_evaluation()
