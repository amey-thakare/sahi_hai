"""
Conformal Predictor — end-to-end ICP prediction for raw URLs.

Provides ``predict_conformal(url)`` which:
1. Extracts the 12 structural features.
2. Runs the HXRF model (ICP variant trained on proper-train).
3. Applies the conformal calibrator.
4. Returns prediction set, routing, p-values, and diagnostics.
"""
import json
import os
import pickle
import time

import pandas as pd

from src.features.stage1_features import extract_stage1_features
from src.conformal.calibrator import ConformalCalibrator
from src.conformal.routing import route_prediction_set

# ── Lazy-loaded singletons ────────────────────────────────────────────────
_model = None
_calibrator = None
_config = None
_conformal_config = None


def _load():
    """Load the ICP-specific HXRF model, conformal calibrator, and configs."""
    global _model, _calibrator, _config, _conformal_config

    if _model is not None:
        return

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load conformal config
    cc_path = os.path.join(base, "config", "conformal_config.json")
    with open(cc_path) as f:
        _conformal_config = json.load(f)

    # Load model config (for feature list)
    mc_path = os.path.join(base, "config", "stage1_model_config.json")
    with open(mc_path) as f:
        _config = json.load(f)

    # Load ICP-specific HXRF model
    model_path = os.path.join(base, _conformal_config["hxrf_model_path"])
    with open(model_path, "rb") as f:
        _model = pickle.load(f)

    # Load conformal calibrator
    cal_path = os.path.join(base, _conformal_config["calibrator_path"])
    _calibrator = ConformalCalibrator.load(cal_path)


def predict_conformal(url: str, alpha: float = None) -> dict:
    """Run ICP-augmented Stage-1 prediction on a raw URL.

    Parameters
    ----------
    url : str
        Raw URL string (with or without scheme).
    alpha : float, optional
        Override the configured alpha for sensitivity analysis.

    Returns
    -------
    dict
        {
            "url": str,
            "prediction_set": list[str],       # ["benign"], ["malicious"], or ["benign","malicious"]
            "prediction_set_size": int,         # 1 or 2
            "routing": str,                     # FINAL_BENIGN / FINAL_MALICIOUS / NEEDS_STAGE2
            "malicious_probability": float,     # HXRF P(malicious)
            "conformal_pvalue_benign": float,   # conformal p-value for benign
            "conformal_pvalue_malicious": float, # conformal p-value for malicious
            "alpha": float,
            "conformal_quantile": float,
            "latency_ms": float,
        }
    """
    _load()

    features = _config["features"]
    class_map = {0: "benign", 1: "malicious"}

    t0 = time.perf_counter()

    # 1. Feature extraction
    feat_dict = extract_stage1_features(url)

    # 2. Build feature vector
    import numpy as np
    fv = pd.DataFrame([feat_dict])[features]

    # 3. HXRF prediction
    probas = _model.predict_proba(fv)  # shape (1, 2)

    # 4. Conformal prediction
    result = _calibrator.predict(probas, alpha=alpha)

    t1 = time.perf_counter()

    # Extract results for this single URL
    pset_indices = result["prediction_sets"][0]
    pset_labels = [class_map[c] for c in pset_indices]
    routing = route_prediction_set(pset_indices)

    # Get malicious probability
    mal_idx = list(_model.classes_).index(1)
    p_mal = float(probas[0, mal_idx])

    return {
        "url": url,
        "prediction_set": pset_labels,
        "prediction_set_size": len(pset_labels),
        "routing": routing,
        "malicious_probability": round(p_mal, 6),
        "conformal_pvalue_benign": round(float(result["pvalues"][0, 0]), 6),
        "conformal_pvalue_malicious": round(float(result["pvalues"][0, 1]), 6),
        "alpha": result["alpha_used"],
        "conformal_quantile": round(result["quantile_used"], 6),
        "latency_ms": round((t1 - t0) * 1000, 3),
    }
