import torch
import torch.nn as nn
from dit_block import DitBlock


class Dit(nn.Module):
    def __init__(self, time_embed_dim, embed_dim, num_heads, num_layers, latent_dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_dim)

        self.out = nn.Linear(embed_dim, latent_dim)

        self.blocks = nn.ModuleList(
            [DitBlock(time_embed_dim, embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, semantic, timestep, semantic_mask, acoustic_mask):
        for block in self.blocks:
            x = block(x, semantic, timestep, semantic_mask, acoustic_mask)
        x = self.layernorm(x)
        x = self.out(x)
        return x