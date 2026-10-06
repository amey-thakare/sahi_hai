import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def prepare_dataset():
    # 1. Load config
    with open('config/dsse_dataset_config.json', 'r') as f:
        config = json.load(f)
        
    # 2. Load dataset
    df = pd.read_csv('dataset/MUD_malicious_urls_2026_V2.csv')
    total_original = len(df)
    
    # 3. Target Definition
    # label == 0 is benign, > 0 is malicious
    df['dsse_binary_target'] = (df['label'] > 0).astype(int)
    
    # 4. Drop Leakage
    df = df.drop(columns=config['drop_columns'], errors='ignore')
    
    # 5. Duplicate Handling
    # Sort by target descending so 1 (malicious) comes first
    df = df.sort_values('dsse_binary_target', ascending=False)
    
    # Drop duplicates by 'url'
    df_clean = df.drop_duplicates(subset=[config['raw_url_column']], keep='first')
    total_clean = len(df_clean)
    duplicates_dropped = total_original - total_clean
    
    # 6. Split
    train_df, test_df = train_test_split(
        df_clean, 
        test_size=0.2, 
        stratify=df_clean['dsse_binary_target'], 
        random_state=42
    )
    
    # 7. Export
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/splits', exist_ok=True)
    df_clean.to_parquet('data/processed/dsse_dataset_clean.parquet', index=False)
    train_df.to_parquet('data/splits/train.parquet', index=False)
    test_df.to_parquet('data/splits/test.parquet', index=False)
    
    # Validation metrics
    train_dist = train_df['dsse_binary_target'].value_counts(normalize=True).to_dict()
    test_dist = test_df['dsse_binary_target'].value_counts(normalize=True).to_dict()
    
    train_urls = set(train_df[config['raw_url_column']])
    test_urls = set(test_df[config['raw_url_column']])
    overlap = len(train_urls.intersection(test_urls))
    
    # 8. Report Generation
    report_md = f"""# Data Split Validation Report

## Duplicate Handling
- Total rows before duplicate removal: {total_original}
- Total rows after duplicate removal: {total_clean}
- Number of duplicates dropped: {duplicates_dropped}

## Split Sizes
- Final train size: {len(train_df)}
- Final test size: {len(test_df)}

## Class Distributions

### Train Set
- Class 0 (Benign): {train_dist.get(0, 0):.4f}
- Class 1 (Malicious): {train_dist.get(1, 0):.4f}

### Test Set
- Class 0 (Benign): {test_dist.get(0, 0):.4f}
- Class 1 (Malicious): {test_dist.get(1, 0):.4f}

## Leakage / Overlap Validation
- URL overlap between Train and Test sets: {overlap}
- Boolean confirmation that NO URLs exist in both sets: {overlap == 0}
"""
    os.makedirs('reports', exist_ok=True)
    with open('reports/data_split_validation.md', 'w') as f:
        f.write(report_md)
        
    print("Dataset prepared and saved to data/. Report saved to reports/data_split_validation.md")

if __name__ == '__main__':
    prepare_dataset()
