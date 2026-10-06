import numpy as np
import pytest
from src.semantic import MiniLMEncoder

@pytest.fixture(scope="module")
def encoder():
    # Use auto device to allow mps or cpu fallback depending on machine
    return MiniLMEncoder(device="auto", normalize_embeddings=False)

def test_model_loads(encoder):
    assert encoder.model is not None
    assert encoder.get_embedding_dimension() == 384

def test_encode_single_url(encoder):
    url = "https://example.com/login"
    emb = encoder.encode_url(url)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (384,)
    assert not np.isnan(emb).any()
    assert not np.isinf(emb).any()

def test_encode_multiple_urls(encoder):
    urls = ["https://example.com", "http://malicious.com/path?q=1"]
    embs = encoder.encode_urls(urls, batch_size=2)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, 384)
    assert not np.isnan(embs).any()
    assert not np.isinf(embs).any()

def test_consistency(encoder):
    url = "https://test.com"
    emb_single = encoder.encode_url(url)
    emb_batch = encoder.encode_urls([url, "https://other.com"])[0]
    np.testing.assert_allclose(emb_single, emb_batch, rtol=1e-5, atol=1e-5)

def test_deterministic(encoder):
    url = "https://deterministic.com"
    emb1 = encoder.encode_url(url)
    emb2 = encoder.encode_url(url)
    np.testing.assert_array_equal(emb1, emb2)

def test_long_unicode_urls(encoder):
    url = "https://عربي.com/path?val=" + "a" * 1000
    emb = encoder.encode_url(url)
    assert emb.shape == (384,)

def test_device_selection():
    cpu_encoder = MiniLMEncoder(device="cpu")
    assert cpu_encoder.get_device() == "cpu"
    # Even if we pass cuda/mps, it will fallback to CPU if not available.
    # The initialization shouldn't crash.
    encoder = MiniLMEncoder(device="cuda")
    assert encoder.get_device() in ["cuda", "cpu"]
