import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import time
import json
import torch
import numpy as np
import pandas as pd
from src.semantic.minilm_encoder import MiniLMEncoder
from src.features.stage1_features import extract_stage1_features
from src.fusion.film import FiLMFusionClassifier
import xgboost as xgb

def measure_latencies():
    print("Measuring latencies...")
    # Load sample URLs
    df = pd.read_parquet("data/splits/test.parquet")
    sample_urls = df['url'].iloc[:1000].tolist()
    
    # 1. MiniLM Latency
    encoder = MiniLMEncoder(device='cpu')
    for _ in range(3):
        encoder.encode_urls(sample_urls[:10])
    times_minilm = []
    for _ in range(5):
        t0 = time.perf_counter()
        embs_batch = encoder.encode_urls(sample_urls)
        times_minilm.append(time.perf_counter() - t0)
    minilm_lat = np.mean(times_minilm) * 1000
    
    # 2. Structural feature processing latency
    import joblib
    with open('models/xgboost_audit/structural_scaler.pkl', 'rb') as f:
        scaler = joblib.load(f)
    with open('config/film_config.json', 'r') as f:
        film_config = json.load(f)
    features = film_config['structural_features']
    
    times_struct = []
    for _ in range(5):
        t0 = time.perf_counter()
        st_feats = [extract_stage1_features(u) for u in sample_urls]
        st_df = pd.DataFrame(st_feats)[features].fillna(0)
        st_scaled = scaler.transform(st_df)
        times_struct.append(time.perf_counter() - t0)
    struct_lat = np.mean(times_struct) * 1000
    
    # 3. FiLM Latency
    film_model = FiLMFusionClassifier(
        semantic_dim=film_config['embedding_dim'], 
        structural_dim=film_config['structural_dim'],
        hidden_dim=film_config['hidden_dim'],
        dropout=film_config['dropout'],
        stable_gamma=True
    )
    film_model.load_state_dict(torch.load('models/film/film_model.pth', map_location='cpu'))
    film_model.eval()
    
    times_film = []
    sem_t = torch.tensor(embs_batch, dtype=torch.float32).to('cpu')
    struct_t = torch.tensor(st_scaled, dtype=torch.float32).to('cpu')
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            _, _, _, f_emb = film_model(sem_t, struct_t)
        times_film.append(time.perf_counter() - t0)
    film_lat = np.mean(times_film) * 1000
    
    with open('reports/latencies.json', 'w') as f:
        json.dump({
            'minilm_lat': minilm_lat,
            'struct_lat': struct_lat,
            'film_lat': film_lat,
        }, f)
        
if __name__ == '__main__':
    measure_latencies()
