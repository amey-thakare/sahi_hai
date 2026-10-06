import pandas as pd
import json
import numpy as np

def run_stats():
    df = pd.read_parquet('data/splits/train.parquet')
    with open('config/dsse_dataset_config.json', 'r') as f:
        config = json.load(f)
    
    features = config['engineered_features']
    target = config['target_column']
    
    stats = []
    for f in features:
        if f not in df.columns:
            continue
        col_data = df[f]
        missing_rate = col_data.isna().mean()
        nunique = col_data.nunique()
        variance = col_data.var() if pd.api.types.is_numeric_dtype(col_data) else 0
        corr = df[f].corr(df[target]) if pd.api.types.is_numeric_dtype(col_data) else 0
        dtype = str(col_data.dtype)
        
        # Real-time eligibility rules
        is_real_time = True
        cost = "Low"
        external_info = False
        
        stats.append({
            "feature": f,
            "dtype": dtype,
            "missing_rate": missing_rate,
            "nunique": nunique,
            "variance": variance,
            "correlation": corr,
            "real_time_eligible": is_real_time
        })
        
    with open('reports/temp_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    run_stats()
