import json
import pandas as pd
from sklearn.model_selection import train_test_split

with open('config/xgboost_config.json', 'r') as f:
    xgb_config = json.load(f)

train_df = pd.read_parquet(xgb_config['data']['train_dataset'])
test_df = pd.read_parquet(xgb_config['data']['test_dataset'])

train_split, val_split = train_test_split(train_df, test_size=0.1, random_state=xgb_config['random_seed'])

print("Train pos ratio:", train_split['dsse_binary_target'].mean())
print("Val pos ratio:", val_split['dsse_binary_target'].mean())
print("Test pos ratio:", test_df['dsse_binary_target'].mean())
