import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
torch.set_num_threads(1)
print("1")
from src.stage2.xgboost_classifier import Stage2XGBoost
xgb_model = Stage2XGBoost.load('models/xgboost_audit/film_xgb.json')
xgb_model.model.set_params(n_jobs=1)
print("2")
from src.semantic.minilm_encoder import MiniLMEncoder
encoder = MiniLMEncoder(device='cpu')
print("3")
