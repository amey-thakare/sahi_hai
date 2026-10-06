import pandas as pd
import os
import json
import glob
import numpy as np

# Set the dataset directory
DATASET_DIR = 'dataset'

# The exact files available
# I will use glob to handle any minor name differences like spaces
files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
print("Files found:", files)

results = {}

datasets = {}

for f in files:
    filename = os.path.basename(f)
    print(f"Loading {filename}...")
    # Get file size
    size_mb = os.path.getsize(f) / (1024 * 1024)
    
    # Load dataset
    df = pd.read_csv(f, low_memory=False)
    datasets[filename] = df
    
    # Basic info
    num_rows, num_cols = df.shape
    columns = list(df.columns)
    dtypes = {k: str(v) for k, v in df.dtypes.items()}
    
    # Missing values
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / num_rows * 100).to_dict()
    missing_counts = missing_counts.to_dict()
    
    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    
    # Detect URL column, label column, etc.
    # Usually names like 'url', 'URL', 'raw_url'
    url_col = None
    for c in columns:
        if c.lower() == 'url':
            url_col = c
            break
            
    # Target column (often 'label', 'target', 'status', 'type', 'result')
    target_col = None
    for c in columns:
        if c.lower() in ['label', 'target', 'status', 'type', 'result', 'class']:
            target_col = c
            break
            
    # Hash column
    hash_col = None
    for c in columns:
        if 'hash' in c.lower():
            hash_col = c
            break
            
    duplicate_urls = 0
    if url_col:
        duplicate_urls = int(df.duplicated(subset=[url_col]).sum())
        
    unique_targets = []
    class_dist = {}
    class_pct = {}
    if target_col:
        vc = df[target_col].value_counts(dropna=False)
        for k, v in vc.items():
            k_str = str(k)
            class_dist[k_str] = int(v)
            class_pct[k_str] = float((v / num_rows) * 100)
        unique_targets = list(class_dist.keys())
        
    num_features = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    
    results[filename] = {
        'filename': filename,
        'size_mb': size_mb,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'columns': columns,
        'dtypes': dtypes,
        'missing_percentages': missing_percentages,
        'duplicate_rows': duplicate_rows,
        'duplicate_urls': duplicate_urls,
        'url_column': url_col,
        'hash_column': hash_col,
        'target_column': target_col,
        'unique_targets': unique_targets,
        'class_dist': class_dist,
        'class_pct': class_pct,
        'num_features': len(num_features),
        'cat_features': len(cat_features),
        'numerical_features': num_features,
        'categorical_features': cat_features
    }

print("Cross-dataset comparison...")
overlap_results = {}
file_keys = list(datasets.keys())
for i in range(len(file_keys)):
    for j in range(i+1, len(file_keys)):
        f1 = file_keys[i]
        f2 = file_keys[j]
        df1 = datasets[f1]
        df2 = datasets[f2]
        
        c1 = set(df1.columns)
        c2 = set(df2.columns)
        
        overlap_results[f"{f1} vs {f2}"] = {
            'common_columns': list(c1.intersection(c2)),
            'f1_only_cols': list(c1 - c2),
            'f2_only_cols': list(c2 - c1)
        }
        
        u1_col = results[f1]['url_column']
        u2_col = results[f2]['url_column']
        
        if u1_col and u2_col:
            urls1 = set(df1[u1_col].astype(str).unique())
            urls2 = set(df2[u2_col].astype(str).unique())
            common_urls = urls1.intersection(urls2)
            overlap_results[f"{f1} vs {f2}"]['common_urls'] = len(common_urls)
            
            # Label conflict check on common URLs
            t1 = results[f1]['target_column']
            t2 = results[f2]['target_column']
            
            if t1 and t2 and len(common_urls) > 0:
                # Merge on url to compare targets
                merged = pd.merge(df1[[u1_col, t1]].drop_duplicates(subset=[u1_col]), 
                                  df2[[u2_col, t2]].drop_duplicates(subset=[u2_col]), 
                                  left_on=u1_col, right_on=u2_col, how='inner', suffixes=('_f1', '_f2'))
                
                t1_merged = t1 + '_f1' if t1 == t2 else t1
                t2_merged = t2 + '_f2' if t1 == t2 else t2
                
                # If target types are different, just convert to string
                merged[t1_merged] = merged[t1_merged].astype(str).str.lower().str.strip()
                merged[t2_merged] = merged[t2_merged].astype(str).str.lower().str.strip()
                
                # Check conflicts
                conflicts = (merged[t1_merged] != merged[t2_merged]).sum()
                overlap_results[f"{f1} vs {f2}"]['conflicting_labels'] = int(conflicts)

# Save to JSON
with open('reports/dataset_analysis.json', 'w') as f:
    json.dump({'datasets': results, 'comparisons': overlap_results}, f, indent=2)

print("Analysis complete. Saved to reports/dataset_analysis.json")
