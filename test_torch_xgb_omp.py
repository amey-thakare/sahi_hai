import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
print("importing xgboost")
import xgboost as xgb
print("loading model")
model = xgb.XGBClassifier()
model.load_model('models/xgboost_audit/film_xgb.json')
print("done")
