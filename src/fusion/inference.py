import torch
import json
import joblib
from src.fusion.film import FiLMFusionClassifier

class FiLMInference:
    def __init__(self, model_path, config_path, scaler_path, device='cpu'):
        self.device = device
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.model = FiLMFusionClassifier(
            semantic_dim=self.config.get('embedding_dim', 384),
            structural_dim=self.config.get('structural_dim', 12),
            hidden_dim=self.config.get('hidden_dim', 64),
            dropout=self.config.get('dropout', 0.0),
            stable_gamma=True
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.scaler = joblib.load(scaler_path)
        
    def predict(self, semantic_embs, structural_feats):
        # Scale structural features
        structural_feats_scaled = self.scaler.transform(structural_feats)
        
        sem_tensor = torch.tensor(semantic_embs, dtype=torch.float32).to(self.device)
        struct_tensor = torch.tensor(structural_feats_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits, gamma, beta, modulated_emb = self.model(sem_tensor, struct_tensor)
            probs = torch.sigmoid(logits)
            
        return probs.cpu().numpy(), gamma.cpu().numpy(), beta.cpu().numpy(), modulated_emb.cpu().numpy()
