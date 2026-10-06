import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import time
import pickle
import joblib
import pandas as pd
import numpy as np
import torch
from typing import List, Union

from .types import DSSEResult
from .config import DSSEConfig
from src.features.stage1_features import extract_stage1_features
from src.conformal.calibrator import ConformalCalibrator
from src.conformal.routing import route_prediction_set
from src.semantic.minilm_encoder import MiniLMEncoder
from src.fusion.film import FiLMFusionClassifier
from src.stage2.xgboost_classifier import Stage2XGBoost

class DSSEPredictor:
    def __init__(self, config_path: str):
        self.config = DSSEConfig(config_path)
        
        # Determine device
        if self.config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = self.config.device
            
        # Load Stage-1 components
        with open(self.config.stage1_model_path, 'rb') as f:
            self.hxrf = pickle.load(f)
            
        self.calibrator = ConformalCalibrator.load(self.config.conformal_calibrator_path)
        
        # Stage-2 components (lazy loaded on first use if preferred, but loading here for latency stability)
        self.encoder = MiniLMEncoder(device=self.device)
        self.film_scaler = joblib.load(self.config.film_scaler_path)
        
        self.film_model = FiLMFusionClassifier(
            semantic_dim=self.config.semantic_dim,
            structural_dim=self.config.structural_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            stable_gamma=True
        )
        self.film_model.load_state_dict(torch.load(self.config.film_model_path, map_location=self.device, weights_only=True))
        self.film_model.to(self.device)
        self.film_model.eval()
        
        self.xgb_model = Stage2XGBoost.load(self.config.xgboost_model_path)
        
        # Thread isolation to prevent macOS KMP crash when XGBoost meets PyTorch
        torch.set_num_threads(1)

    def extract_stage1_features(self, urls: List[str]) -> pd.DataFrame:
        features_list = [extract_stage1_features(url) for url in urls]
        df = pd.DataFrame(features_list)
        # Ensure ordering matches expected HXRF features
        return df[self.config.structural_features]
        
    def predict_url(self, url: str) -> DSSEResult:
        return self.predict_urls([url])[0]
        
    def predict_urls(self, urls: List[str]) -> List[DSSEResult]:
        if not urls:
            return []
            
        t0 = time.perf_counter()
        
        # 1. Structural features extraction
        struct_df = self.extract_stage1_features(urls)
        
        # 2. HXRF prediction
        stage1_probas = self.hxrf.predict_proba(struct_df)
        
        # 3. ICP routing
        icp_results = self.calibrator.predict(stage1_probas)
        prediction_sets = icp_results['prediction_sets']
        routes = [route_prediction_set(ps) for ps in prediction_sets]
        
        # 4. Identify NEEDS_STAGE2 URLs
        results = []
        needs_stage2_indices = []
        needs_stage2_urls = []
        needs_stage2_struct = []
        
        for i, (url, ps, route) in enumerate(zip(urls, prediction_sets, routes)):
            if route == 'FINAL_BENIGN':
                results.append({
                    "url": url,
                    "stage1_prediction_set": ps,
                    "stage1_route": route,
                    "final_prediction": "BENIGN",
                    "stage_used": 1,
                    "stage2_used": False,
                    "malicious_probability": stage1_probas[i][1] # Fallback probability from Stage 1
                })
            elif route == 'FINAL_MALICIOUS':
                results.append({
                    "url": url,
                    "stage1_prediction_set": ps,
                    "stage1_route": route,
                    "final_prediction": "MALICIOUS",
                    "stage_used": 1,
                    "stage2_used": False,
                    "malicious_probability": stage1_probas[i][1]
                })
            else: # NEEDS_STAGE2
                # Placeholder, will be replaced
                results.append(None)
                needs_stage2_indices.append(i)
                needs_stage2_urls.append(url)
                needs_stage2_struct.append(struct_df.iloc[i].values)

        # 5. Stage-2 execution (Lazy - only if needed)
        if needs_stage2_urls:
            # A. MiniLM encoding
            # Use fixed batch_size to prevent OOM
            embs = self.encoder.encode_urls(needs_stage2_urls, batch_size=256)
            
            # B. FiLM representation extraction
            X_struct_np = np.array(needs_stage2_struct)
            X_struct_scaled = self.film_scaler.transform(X_struct_np)
            
            # Batch extraction to prevent OOM/crash
            film_embeddings = []
            with torch.no_grad():
                for start_idx in range(0, len(embs), 2048):
                    end_idx = min(start_idx + 2048, len(embs))
                    sem_t = torch.tensor(embs[start_idx:end_idx], dtype=torch.float32).to(self.device)
                    str_t = torch.tensor(X_struct_scaled[start_idx:end_idx], dtype=torch.float32).to(self.device)
                    _, _, _, rep = self.film_model(sem_t, str_t)
                    film_embeddings.append(rep.cpu().numpy())
                    
            film_rep_np = np.vstack(film_embeddings)
            
            # C. XGBoost prediction
            xgb_probas = self.xgb_model.predict_proba(film_rep_np)
            xgb_preds = (xgb_probas[:, 1] >= self.config.threshold).astype(int)
            
            # Merge back
            for idx_in_subset, original_idx in enumerate(needs_stage2_indices):
                prob = float(xgb_probas[idx_in_subset][1])
                final_class = "MALICIOUS" if xgb_preds[idx_in_subset] == 1 else "BENIGN"
                
                results[original_idx] = {
                    "url": urls[original_idx],
                    "stage1_prediction_set": prediction_sets[original_idx],
                    "stage1_route": routes[original_idx],
                    "final_prediction": final_class,
                    "stage_used": 2,
                    "stage2_used": True,
                    "malicious_probability": prob
                }
                
        total_time_ms = (time.perf_counter() - t0) * 1000
        # Average latency per URL in this batch
        per_url_latency = total_time_ms / len(urls) if urls else 0.0
        
        # Convert to dataclasses
        final_results = []
        for r in results:
            r['latency_ms'] = per_url_latency
            final_results.append(DSSEResult(**r))
            
        return final_results
