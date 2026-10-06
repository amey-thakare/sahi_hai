import torch
import torch.nn as nn
import json

class FiLMNetwork(nn.Module):
    """
    Computes Gamma and Beta from structural features.
    """
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=384, dropout=0.1, stable_gamma=True):
        super().__init__()
        self.stable_gamma = stable_gamma
        
        self.gamma_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        gamma = self.gamma_net(x)
        beta = self.beta_net(x)
        
        if self.stable_gamma:
            # stable formulation: E_FiLM = (1 + gamma) * E + beta
            gamma = gamma + 1.0
            
        return gamma, beta

class FiLMFusionClassifier(nn.Module):
    """
    Conditioning the semantic representation using structural URL features through FiLM.
    """
    def __init__(self, semantic_dim=384, structural_dim=12, hidden_dim=64, dropout=0.1, stable_gamma=True):
        super().__init__()
        self.film = FiLMNetwork(structural_dim, hidden_dim, semantic_dim, dropout, stable_gamma)
        
        # Classifier family: Logistic Regression over the modulated representation
        self.classifier = nn.Linear(semantic_dim, 1)
        
    def forward(self, semantic_emb, structural_feat):
        gamma, beta = self.film(structural_feat)
        
        # E_FiLM = gamma * E + beta
        modulated_emb = gamma * semantic_emb + beta
        
        logits = self.classifier(modulated_emb)
        return logits, gamma, beta, modulated_emb

class MiniLMOnlyClassifier(nn.Module):
    """
    Baseline A: MiniLM semantic embedding only.
    """
    def __init__(self, semantic_dim=384):
        super().__init__()
        self.classifier = nn.Linear(semantic_dim, 1)
        
    def forward(self, semantic_emb, structural_feat=None):
        logits = self.classifier(semantic_emb)
        return logits, None, None, semantic_emb

class ConcatenationFusionClassifier(nn.Module):
    """
    Baseline B: Naive concatenation of MiniLM and structural features.
    """
    def __init__(self, semantic_dim=384, structural_dim=12):
        super().__init__()
        self.classifier = nn.Linear(semantic_dim + structural_dim, 1)
        
    def forward(self, semantic_emb, structural_feat):
        # E_Concat = [E, X]
        concat_emb = torch.cat([semantic_emb, structural_feat], dim=1)
        logits = self.classifier(concat_emb)
        return logits, None, None, concat_emb
