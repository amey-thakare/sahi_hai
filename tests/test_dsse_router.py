import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
torch.set_num_threads(1)
import pytest
from src.dsse.predictor import DSSEPredictor

@pytest.fixture(scope="module")
def predictor():
    # Only load once for all tests to speed up
    return DSSEPredictor('config/dsse_config.json')

def test_single_url_works(predictor):
    res = predictor.predict_url("http://google.com")
    assert res.final_prediction in ["BENIGN", "MALICIOUS"]
    assert 0.0 <= res.malicious_probability <= 1.0

def test_batch_prediction_works_and_preserves_order(predictor):
    urls = ["http://google.com", "http://bad-malware-site.com/login.php", "http://github.com"]
    results = predictor.predict_urls(urls)
    assert len(results) == 3
    assert results[0].url == urls[0]
    assert results[1].url == urls[1]
    assert results[2].url == urls[2]

def test_empty_input(predictor):
    assert predictor.predict_urls([]) == []

def test_stage_logic(predictor):
    urls = ["http://google.com", "http://bad-malware-site.com/login.php"]
    results = predictor.predict_urls(urls)
    
    for r in results:
        assert r.stage1_route in ["FINAL_BENIGN", "FINAL_MALICIOUS", "NEEDS_STAGE2"]
        
        if r.stage1_route in ["FINAL_BENIGN", "FINAL_MALICIOUS"]:
            assert r.stage_used == 1
            assert r.stage2_used is False
        else:
            assert r.stage_used == 2
            assert r.stage2_used is True
            
def test_valid_probabilities(predictor):
    urls = ["http://google.com", "http://bad-malware-site.com/login.php"]
    results = predictor.predict_urls(urls)
    for r in results:
        assert not isinstance(r.malicious_probability, complex)
        assert r.malicious_probability >= 0.0
        assert r.malicious_probability <= 1.0
        import math
        assert not math.isnan(r.malicious_probability)
        assert not math.isinf(r.malicious_probability)
