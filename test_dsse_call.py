import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
torch.set_num_threads(1)
import joblib
import numpy as np
from src.semantic.minilm_encoder import MiniLMEncoder
from src.fusion.film import FiLMFusionClassifier
from src.stage2.xgboost_classifier import Stage2XGBoost

print("Loading...")
encoder = MiniLMEncoder(device='cpu')
film_scaler = joblib.load('models/film/film_scaler.pkl')
film_model = FiLMFusionClassifier(semantic_dim=384, structural_dim=12, hidden_dim=64, dropout=0.1, stable_gamma=True)
film_model.load_state_dict(torch.load('models/film/film_model.pth', map_location='cpu', weights_only=True))
film_model.eval()
xgb_model = Stage2XGBoost.load('models/xgboost_audit/film_xgb.json')
# Force n_jobs=1
xgb_model.model.set_params(n_jobs=1)

print("Encoding...")
urls = ["http://example.com/bad", "http://google.com"]
embs = encoder.encode_urls(urls, batch_size=2)
struct = np.zeros((2, 12))
struct_scaled = film_scaler.transform(struct)

with torch.no_grad():
    sem_t = torch.tensor(embs, dtype=torch.float32)
    str_t = torch.tensor(struct_scaled, dtype=torch.float32)
    _, _, _, rep = film_model(sem_t, str_t)
    film_rep = rep.cpu().numpy()

print("XGB...")
xgb_probas = xgb_model.predict_proba(np.ascontiguousarray(film_rep, dtype=np.float32))
print("Done:", xgb_probas)
