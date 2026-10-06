import json

class DSSEConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.stage1_model_path = self.config['stage1']['hxrf_model_path']
        self.structural_features = self.config['stage1']['structural_features']
        self.conformal_calibrator_path = self.config['conformal']['calibrator_path']
        
        self.minilm_model = self.config['stage2']['minilm_model']
        self.film_model_path = self.config['stage2']['film_model_path']
        self.film_scaler_path = self.config['stage2']['film_scaler_path']
        self.xgboost_model_path = self.config['stage2']['xgboost_model_path']
        
        self.semantic_dim = self.config['stage2']['semantic_dim']
        self.structural_dim = self.config['stage2']['structural_dim']
        self.hidden_dim = self.config['stage2']['hidden_dim']
        self.dropout = self.config['stage2']['dropout']
        self.threshold = self.config['stage2']['threshold']
        self.batch_size = self.config['stage2']['batch_size']
        
        self.device = self.config['device']
        self.target_mapping = self.config['target_mapping']
