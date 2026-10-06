import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
torch.set_num_threads(1)
from src.semantic.minilm_encoder import MiniLMEncoder

encoder = MiniLMEncoder(device='cpu')
print("Encoding...")
embs = encoder.encode_urls(["http://example.com"], batch_size=256)
print("Done")
