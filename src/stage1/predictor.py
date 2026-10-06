"""
Stage-1 HXRF Predictor — DSSE Gatekeeper.

Provides a single ``predict_stage1(url)`` entry-point that:

1. Receives a raw URL string.
2. Extracts the 12 structural features.
3. Runs the HXRF Random Forest.
4. Returns prediction, probability, confidence, routing decision, and latency.
"""
import json
import os
import pickle
import time

import pandas as pd

from src.features.stage1_features import extract_stage1_features

# ── Routing thresholds ────────────────────────────────────────────────────
TAU_HIGH = 0.95   # P(malicious) ≥ this → FINAL_MALICIOUS
TAU_LOW  = 0.05   # P(malicious) ≤ this → FINAL_BENIGN

# ── Lazy-loaded singletons ────────────────────────────────────────────────
_model = None
_config = None


def _load_model():
    """Load the HXRF model and config once."""
    global _model, _config
    if _model is not None:
        return

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    cfg_path = os.path.join(base, 'config', 'stage1_model_config.json')
    with open(cfg_path) as f:
        _config = json.load(f)

    model_path = os.path.join(base, _config['model_path'])
    with open(model_path, 'rb') as f:
        _model = pickle.load(f)


def predict_stage1(url: str) -> dict:
    """
    Run the Stage-1 HXRF gatekeeper on a raw URL.

    Parameters
    ----------
    url : str
        The raw URL (with or without scheme).

    Returns
    -------
    dict
        {
            "url": str,
            "prediction": "benign" | "malicious",
            "class_id": 0 | 1,
            "malicious_probability": float,  # 0-1
            "confidence": float,             # 0-1
            "routing_decision": "final_stage1" | "needs_stage2",
            "latency_ms": float,
        }
    """
    _load_model()
    assert _model is not None and _config is not None

    features = _config['features']
    class_map = _config['class_mapping']

    t0 = time.perf_counter()

    # 1. Feature extraction
    feat_dict = extract_stage1_features(url)

    # 2. Build feature vector in the correct order
    fv = pd.DataFrame([feat_dict])[features]

    # 3. Predict
    prob = _model.predict_proba(fv)[0]   # [P(benign), P(malicious)]

    t1 = time.perf_counter()

    # The model's classes_ attribute tells us which index is which class.
    # class_map is {"0": "benign", "1": "malicious"}.
    mal_idx = list(_model.classes_).index(1)
    ben_idx = list(_model.classes_).index(0)
    p_mal = float(prob[mal_idx])

    predicted_class = 1 if p_mal >= 0.5 else 0
    prediction_label = class_map[str(predicted_class)]

    # Confidence = how far the probability is from the decision boundary (0.5)
    confidence = abs(p_mal - 0.5) * 2  # maps [0.5, 1.0] → [0.0, 1.0]

    # 4. Routing
    if p_mal >= TAU_HIGH or p_mal <= TAU_LOW:
        routing = 'final_stage1'
    else:
        routing = 'needs_stage2'

    return {
        'url': url,
        'prediction': prediction_label,
        'class_id': predicted_class,
        'malicious_probability': round(p_mal, 6),
        'confidence': round(confidence, 6),
        'routing_decision': routing,
        'latency_ms': round((t1 - t0) * 1000, 3),
    }
