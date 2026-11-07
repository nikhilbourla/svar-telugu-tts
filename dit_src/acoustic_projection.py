import torch
import torch.nn as nn


class AcousticProjection(nn.Module):

    def __init__(self, latent_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(latent_dim, embed_dim)

    def forward(self, x):
        # x: (B, T, latent_dim) -> (B, T, embed_dim)
        return self.proj(x)