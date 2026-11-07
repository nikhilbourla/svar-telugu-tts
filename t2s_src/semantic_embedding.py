import torch.nn as nn


class Semantic_embedding(nn.Module):

    def __init__(self, vocab_size: int, embeddings_dim: int, padding_idx: int = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embeddings_dim, padding_idx=padding_idx)

    def forward(self, semantic_ids):
        return self.embedding(semantic_ids)
