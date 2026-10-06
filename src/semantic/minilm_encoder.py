import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

from .config import load_config

class MiniLMEncoder:
    def __init__(self, device: str = None, normalize_embeddings: bool = None):
        self.config = load_config()
        
        self.device = device if device else self.config.get('device', 'cpu')
        
        # Check if MPS (Apple Silicon GPU) or CUDA is available if requested, or auto-detect
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        elif self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
        elif self.device == 'mps' and not torch.backends.mps.is_available():
            self.device = 'cpu'

        self.normalize_embeddings = normalize_embeddings if normalize_embeddings is not None else self.config.get('normalization', False)
        self.model_name = self.config['model_name']
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.max_seq_length = self.config.get('max_seq_length', 256)
        # Ensure we don't accidentally fetch anything
        # The model should be cached from HuggingFace after the first download

    def encode_url(self, url: str) -> np.ndarray:
        return self.encode_urls([url])[0]

    def encode_urls(self, urls: List[str], batch_size: int = None) -> np.ndarray:
        if batch_size is None:
            batch_size = self.config.get('batch_size', 32)
        
        embeddings = self.model.encode(
            urls,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

    def get_embedding_dimension(self) -> int:
        if hasattr(self.model, 'get_embedding_dimension'):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def get_device(self) -> str:
        return self.device
