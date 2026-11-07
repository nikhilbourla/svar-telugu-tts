import torch.nn as nn


class TextEmbedding(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, text_ids):
        return self.embedding(text_ids)
