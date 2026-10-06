import json
import pickle
import numpy as np
import pandas as pd
from src.conformal.calibrator import ConformalCalibrator

def run_calibration():
    print("Loading calibration data...")
    calib_df = pd.read_parquet('data/splits/calibration.parquet')
    
    with open('config/stage1_features.json', 'r') as f:
        features = json.load(f)['features']
        
    X_calib = calib_df[features]
    y_calib = calib_df['dsse_binary_target'].values
    
    print("Loading Canonical HXRF model...")
    with open('models/stage1_hxrf.pkl', 'rb') as f:
        hxrf = pickle.load(f)
        
    # Generate calibration probabilities
    print("Generating probabilities...")
    probas_calib = hxrf.predict_proba(X_calib)
    
    # Initialize Conformal Calibrator
    calibrator = ConformalCalibrator(alpha=0.05)
    calibrator.fit(y_calib, probas_calib)
    
    # Save the canonical conformal model
    print("Saving calibrator...")
    calibrator.save('models/stage1_conformal_canonical.pkl')
    
    # Generate config
    config = {
        "method": "inductive_conformal_prediction",
        "nonconformity_score": "1 - predicted_probability_true_class",
        "primary_alpha": 0.05,
        "sensitivity_alphas": [0.01, 0.025, 0.05, 0.1, 0.2],
        "calibration_dataset": "data/splits/calibration.parquet",
        "calibration_samples": len(y_calib),
        "calibration_class_distribution": calibrator.class_distribution,
        "class_mapping": {"0": "benign", "1": "malicious"},
        "hxrf_model_path": "models/stage1_hxrf.pkl",
        "calibrator_path": "models/stage1_conformal_canonical.pkl",
        "features": features,
        "random_seed": 42,
        "quantile_method": "ceil((n+1)(1-alpha))-1, clipped to [0, n-1]",
        "finite_sample_correction": True,
        "conformal_quantiles": {
            str(alpha): calibrator.get_quantile(alpha)
            for alpha in [0.01, 0.025, 0.05, 0.1, 0.2]
        }
    }
    
    with open('config/conformal_canonical_config.json', 'w') as f:
        json.dump(config, f, indent=2)
        
    # Generate Diagnostics
    report = []
    report.append("# Phase 12 — Calibration Diagnostics\n")
    report.append(f"- Calibration Size: {len(y_calib)}")
    report.append(f"- Benign Calibration Count: {calibrator.class_distribution[0]}")
    report.append(f"- Malicious Calibration Count: {calibrator.class_distribution[1]}\n")
    
    report.append("## Conformal Quantiles")
    for alpha in [0.01, 0.025, 0.05, 0.1, 0.2]:
        report.append(f"- alpha={alpha}: q={calibrator.get_quantile(alpha):.5f}")
        
    report.append("\n## Score Distributions")
    scores = calibrator.calibration_scores
    report.append(f"- Min Score: {np.min(scores):.5f}")
    report.append(f"- Max Score: {np.max(scores):.5f}")
    report.append(f"- Median Score: {np.median(scores):.5f}")
    report.append(f"- Mean Score: {np.mean(scores):.5f}")
    
    with open('reports/phase12_calibration_diagnostics.md', 'w') as f:
        f.write("\n".join(report) + "\n")
        
    print("Calibration complete.")

if __name__ == '__main__':
    run_calibration()
