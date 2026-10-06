import time
import json
import numpy as np
import xgboost as xgb
from src.stage2.xgboost_classifier import Stage2XGBoost

with open('config/xgboost_config.json', 'r') as f:
    xgb_config = json.load(f)

# Mock inputs for batch of 1000
concat_input = np.random.rand(1000, 396).astype(np.float32)
film_input = np.random.rand(1000, 384).astype(np.float32)

model_b = Stage2XGBoost.load('models/xgboost_audit/concat_xgb.json', xgb_config)
model_c = Stage2XGBoost.load('models/xgboost_audit/film_xgb.json', xgb_config)

times_concat = []
for _ in range(10):
    t0 = time.perf_counter()
    model_b.predict_proba(concat_input)
    times_concat.append(time.perf_counter() - t0)
    
times_film = []
for _ in range(10):
    t0 = time.perf_counter()
    model_c.predict_proba(film_input)
    times_film.append(time.perf_counter() - t0)

print(f"concat_xgb_lat = {np.mean(times_concat) * 1000}")
print(f"film_xgb_lat = {np.mean(times_film) * 1000}")
