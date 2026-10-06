import pytest
import numpy as np
import os
import json
from src.stage2.xgboost_classifier import Stage2XGBoost

def test_xgboost_classifier_basics():
    # 4. Correct input dimensions.
    # 5. Predictions are valid.
    # 6. Malicious probabilities in [0,1].
    # 7. No NaN. 8. No Inf.
    X_train = np.random.randn(100, 384)
    y_train = np.random.randint(0, 2, size=100)
    
    config = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_seed': 42
    }
    
    model = Stage2XGBoost(config)
    model.fit(X_train, y_train)
    
    # 10. Single inference works
    x_single = np.random.randn(1, 384)
    prob_single = model.predict_proba(x_single)
    
    assert prob_single.shape == (1, 2)
    assert 0 <= prob_single[0, 1] <= 1
    assert not np.isnan(prob_single).any()
    assert not np.isinf(prob_single).any()
    
    # 9. Batch inference works
    x_batch = np.random.randn(10, 384)
    prob_batch = model.predict_proba(x_batch)
    
    assert prob_batch.shape == (10, 2)
    assert np.all((prob_batch >= 0) & (prob_batch <= 1))
    
    preds = model.predict(x_batch)
    assert preds.shape == (10,)
    assert set(preds).issubset({0, 1})
    
    # 11. Models can be saved and loaded.
    model.save('test_xgb.json')
    loaded_model = Stage2XGBoost.load('test_xgb.json')
    
    prob_loaded = loaded_model.predict_proba(x_batch)
    np.testing.assert_array_almost_equal(prob_batch, prob_loaded)
    
    os.remove('test_xgb.json')

def test_reproducible_results():
    # 15. Same model/configuration produces reproducible results
    X_train = np.random.randn(100, 384)
    y_train = np.random.randint(0, 2, size=100)
    
    X_test = np.random.randn(50, 384)
    
    model1 = Stage2XGBoost({'random_seed': 42})
    model1.fit(X_train, y_train)
    prob1 = model1.predict_proba(X_test)
    
    model2 = Stage2XGBoost({'random_seed': 42})
    model2.fit(X_train, y_train)
    prob2 = model2.predict_proba(X_test)
    
    np.testing.assert_array_equal(prob1, prob2)
