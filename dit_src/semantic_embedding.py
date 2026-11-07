import torch
import torch.nn as nn
from config import Config

class Semantic_embedding(nn.Module):
    def __init__(self,vocab_size:int,embeddings_dim:int):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embeddings_dim,padding_idx=Config.PAD_TOKEN)

    def forward(self,semantic_ids):
        semantic_embed=self.embedding(semantic_ids)
        return semantic_embed
