import pandas as pd
import numpy as np
from src.features.stage1_features import extract_stage1_features
from tqdm import tqdm

def main():
    print("Loading test data...")
    df = pd.read_parquet('data/splits/test.parquet')
    
    # 12 selected features (as per stage1 config)
    features = [
        "domain_len", "path_depth", "special_count_v2", "token_count",
        "url_entropy", "url_len", "=", ".", "//",
        "abnormal_url", "phish_adv_hyphen_count", "suspicious_extension"
    ]
    
    print("Extracting features from raw URLs...")
    extracted_features = []
    for url in tqdm(df['url']):
        extracted_features.append(extract_stage1_features(url))
        
    df_extracted = pd.DataFrame(extracted_features)
    
    report = ["# Phase 11A — Feature Drift Analysis\n\n"]
    report.append("Comparison of 12 Stage-1 features: Canonical Parquet vs Dynamically Extracted\n\n")
    
    total_rows = len(df)
    any_diff_count = 0
    
    report.append("| Feature | Exact Match % | Diff Rows | Max Abs Diff | Mean Abs Diff | Note |\n")
    report.append("|---|---:|---:|---:|---:|---|\n")
    
    diff_mask = np.zeros(total_rows, dtype=bool)
    
    for f in features:
        if f not in df_extracted.columns:
            # Let's map names if needed
            # e.g., 'phish_adv_hyphen_count' might just be extracted as something else?
            # Wait, extract_stage1_features produces exactly these keys or superset.
            pass
            
        col_parquet = df[f].values if f in df else np.zeros(total_rows)
        col_extracted = df_extracted[f].values if f in df_extracted else np.zeros(total_rows)
        
        diff = np.abs(col_parquet - col_extracted)
        is_diff = diff > 0
        diff_mask = diff_mask | is_diff
        
        num_diff = is_diff.sum()
        match_pct = 100.0 * (total_rows - num_diff) / total_rows
        max_diff = diff.max() if num_diff > 0 else 0
        mean_diff = diff.mean()
        
        note = "Float Precision" if (max_diff > 0 and max_diff < 1e-4) else "Logic Diff" if max_diff > 0 else "Match"
        if f not in df_extracted.columns: note = "Missing in Extracted"
        
        report.append(f"| {f} | {match_pct:.2f}% | {num_diff} | {max_diff:.6f} | {mean_diff:.6f} | {note} |\n")
        
    any_diff_count = diff_mask.sum()
    report.append(f"\nTotal rows with at least one difference: {any_diff_count} / {total_rows} ({(any_diff_count/total_rows)*100:.2f}%)\n")
    
    # Save the report
    with open('reports/phase11a_feature_drift.md', 'w') as f:
        f.writelines(report)
        
    print(f"Feature drift report written to reports/phase11a_feature_drift.md")

if __name__ == '__main__':
    main()
