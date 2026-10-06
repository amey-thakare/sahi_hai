import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import xgboost as xgb
import time
import json
import numpy as np
import pandas as pd
import pickle
import joblib

def run_safe_xgb():
    print("Loading data...")
    try:
        embs_train = np.load('embs_train.npy')
        y_train = np.load('y_train.npy')
    except:
        print("Data not found on disk, running PyTorch to generate it...")
        import torch
        from src.semantic.minilm_encoder import MiniLMEncoder
        from sklearn.model_selection import train_test_split
        
        with open('config/xgboost_config.json', 'r') as f:
            xgb_config = json.load(f)
            
        train_df = pd.read_parquet(xgb_config['data']['train_dataset'])
        train_sample = train_df.sample(min(50000, len(train_df)), random_state=xgb_config['random_seed']).copy()
        train_split, val_split = train_test_split(train_sample, test_size=0.2, random_state=xgb_config['random_seed'])
        
        y_train = train_split['dsse_binary_target'].values
        
        encoder = MiniLMEncoder(device='cpu')
        embs_train = encoder.encode_urls(train_split['url'].tolist(), batch_size=128)
        
        np.save('embs_train.npy', embs_train)
        np.save('y_train.npy', y_train)
        print("Data generated. Please run the script again so PyTorch is not in memory!")
        return
        
    print(f"X_train shape: {embs_train.shape}, y_train shape: {y_train.shape}")
    
    print("Training XGBoost...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, tree_method='hist', n_jobs=4)
    t0 = time.time()
    model.fit(embs_train, y_train)
    print(f"Fit complete in {time.time()-t0:.2f}s!")
    
if __name__ == "__main__":
    run_safe_xgb()
