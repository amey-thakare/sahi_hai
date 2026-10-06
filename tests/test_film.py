import pytest
import torch
import numpy as np
from src.fusion.film import FiLMNetwork, FiLMFusionClassifier, MiniLMOnlyClassifier, ConcatenationFusionClassifier

def test_film_network_shapes():
    # 1. X12 input shape = 12
    # 3. Gamma shape = 384
    # 4. Beta shape = 384
    batch_size = 32
    input_dim = 12
    output_dim = 384
    
    net = FiLMNetwork(input_dim=input_dim, hidden_dim=64, output_dim=output_dim, stable_gamma=True)
    x = torch.randn(batch_size, input_dim)
    
    gamma, beta = net(x)
    
    assert gamma.shape == (batch_size, output_dim)
    assert beta.shape == (batch_size, output_dim)
    
    # 6. No NaN, 7. No Inf
    assert not torch.isnan(gamma).any()
    assert not torch.isinf(gamma).any()
    assert not torch.isnan(beta).any()
    assert not torch.isinf(beta).any()

def test_film_fusion_classifier():
    # 2. MiniLM embedding shape = 384
    # 5. FiLM output shape = 384
    batch_size = 16
    semantic_dim = 384
    structural_dim = 12
    
    model = FiLMFusionClassifier(semantic_dim=semantic_dim, structural_dim=structural_dim)
    
    semantic_emb = torch.randn(batch_size, semantic_dim)
    structural_feat = torch.randn(batch_size, structural_dim)
    
    logits, gamma, beta, modulated_emb = model(semantic_emb, structural_feat)
    
    # 5. Output shape = 384
    assert modulated_emb.shape == (batch_size, semantic_dim)
    assert logits.shape == (batch_size, 1)
    
    # 8. Batch processing works
    assert gamma.shape == (batch_size, semantic_dim)
    assert beta.shape == (batch_size, semantic_dim)

def test_gradients_propagate():
    # 9. Gradients propagate
    model = FiLMFusionClassifier()
    semantic_emb = torch.randn(4, 384, requires_grad=True)
    structural_feat = torch.randn(4, 12, requires_grad=True)
    
    logits, _, _, _ = model(semantic_emb, structural_feat)
    loss = logits.sum()
    loss.backward()
    
    # Check that parameters in gamma and beta nets have gradients
    for param in model.film.gamma_net.parameters():
        assert param.grad is not None
        
    for param in model.film.beta_net.parameters():
        assert param.grad is not None

def test_reproducible_inference():
    # 12. Same input/configuration gives reproducible inference
    torch.manual_seed(42)
    model1 = FiLMFusionClassifier()
    
    torch.manual_seed(42)
    model2 = FiLMFusionClassifier()
    
    semantic_emb = torch.randn(4, 384)
    structural_feat = torch.randn(4, 12)
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        out1, _, _, _ = model1(semantic_emb, structural_feat)
        out2, _, _, _ = model2(semantic_emb, structural_feat)
    
    assert torch.allclose(out1, out2)
