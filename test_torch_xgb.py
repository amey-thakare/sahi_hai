import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
print("importing xgboost")
import xgboost as xgb
print("loading model")
model = xgb.XGBClassifier()
model.load_model('models/xgboost_audit/film_xgb.json')
print("done")
