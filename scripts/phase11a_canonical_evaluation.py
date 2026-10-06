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
from src.conformal.routing import route_prediction_set
from src.semantic.minilm_encoder import MiniLMEncoder
from src.fusion.film import FiLMFusionClassifier
from src.stage2.xgboost_classifier import Stage2XGBoost

def calculate_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]
                
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan')
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'specificity': specificity,
        'fpr': fpr,
        'fnr': fnr
    }, cm

def run_canonical_evaluation():
    print("Loading test data...")
    test_df = pd.read_parquet('data/splits/test.parquet')
    urls = test_df['url'].tolist()
    y_true = test_df['dsse_binary_target'].values
    total_urls = len(urls)
    
    # 1. Load canonical models
    with open('models/stage1_hxrf.pkl', 'rb') as f:
        hxrf = pickle.load(f)
        
    calibrator = ConformalCalibrator.load('models/stage1_conformal.pkl')
    
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
    
    features = [
        "domain_len", "path_depth", "special_count_v2", "token_count",
        "url_entropy", "url_len", "=", ".", "//",
        "abnormal_url", "phish_adv_hyphen_count", "suspicious_extension"
    ]
    
    # Stage 1 execution (Using Canonical Parquet Features!)
    print(f"Evaluating Canonical DSSE on {total_urls} URLs...")
    t0 = time.perf_counter()
    X_struct_df = test_df[features]
    
    stage1_probas = hxrf.predict_proba(X_struct_df)
    icp_results = calibrator.predict(stage1_probas)
    prediction_sets = icp_results['prediction_sets']
    routes = np.array([route_prediction_set(ps) for ps in prediction_sets])
    
    y_pred_dsse = np.zeros(total_urls, dtype=int)
    y_prob_dsse = np.zeros(total_urls, dtype=float)
    stage_used = np.ones(total_urls, dtype=int)
    
    stage1_benign_mask = (routes == 'FINAL_BENIGN')
    stage1_mal_mask = (routes == 'FINAL_MALICIOUS')
    needs_stage2_mask = (routes == 'NEEDS_STAGE2')
    
    # Populate Stage 1 exits
    y_pred_dsse[stage1_benign_mask] = 0
    y_prob_dsse[stage1_benign_mask] = stage1_probas[stage1_benign_mask, 1]
    
    y_pred_dsse[stage1_mal_mask] = 1
    y_prob_dsse[stage1_mal_mask] = stage1_probas[stage1_mal_mask, 1]
    
    # Stage 2 Execution
    if needs_stage2_mask.sum() > 0:
        stage_used[needs_stage2_mask] = 2
        
        needs_stage2_indices = np.where(needs_stage2_mask)[0]
        needs_stage2_urls = [urls[i] for i in needs_stage2_indices]
        needs_stage2_struct = X_struct_df.iloc[needs_stage2_indices].values
        
        print(f"Executing Stage 2 on {len(needs_stage2_urls)} URLs...")
        
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
            
    print("Execution complete.")
    
    # Compute Metrics
    dsse_metrics, _ = calculate_metrics(y_true, y_pred_dsse, y_prob_dsse)
    
    stage1_mask = stage_used == 1
    stage2_mask = stage_used == 2
    
    resolved_count = stage1_mask.sum()
    escalated_count = stage2_mask.sum()
    resolution_rate = (resolved_count / total_urls) * 100
    escalation_rate = (escalated_count / total_urls) * 100
    
    # Save Canonical Results
    with open('reports/phase11a_canonical_results.md', 'w') as f:
        f.write("# Phase 11A — Canonical Results\n\n")
        f.write("## 1. Architecture\nCanonical DSSE (HXRF -> ICP -> FiLM/XGBoost)\n\n")
        f.write("## 2. Dataset\nTest set only (`data/splits/test.parquet`). Using Parquet Columns directly, no dynamic extraction float drift.\n\n")
        f.write("## 3. Routing Statistics\n")
        f.write(f"- Total test URLs: {total_urls}\n")
        f.write(f"- Stage-1 resolved count: {resolved_count}\n")
        f.write(f"- Stage-2 escalation count: {escalated_count}\n")
        f.write(f"- Stage-1 resolution rate: {resolution_rate:.2f}%\n")
        f.write(f"- Stage-2 escalation rate: {escalation_rate:.2f}%\n\n")
        f.write("## 4. Classification Results\n")
        f.write("| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(f"| Canonical DSSE | {dsse_metrics['accuracy']:.4f} | {dsse_metrics['precision']:.4f} | {dsse_metrics['recall']:.4f} | {dsse_metrics['f1']:.4f} | {dsse_metrics['roc_auc']:.4f} | {dsse_metrics['fpr']:.4f} | {dsse_metrics['fnr']:.4f} |\n")
        
    print("Wrote reports/phase11a_canonical_results.md")

if __name__ == '__main__':
    run_canonical_evaluation()
