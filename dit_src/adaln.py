import torch
import torch.nn as nn


class Adaln(nn.Module):

    def __init__(self, x_dim, cond_dim=None):
        super().__init__()
        cond_dim = cond_dim if cond_dim is not None else x_dim

        self.norm = nn.LayerNorm(x_dim)
        self.modulation = nn.Linear(cond_dim, x_dim * 2)

    def forward(self, x, condition):
        gamma, beta = self.modulation(condition).chunk(2, dim=-1)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        x = self.norm(x)
        x = x * (1 + gamma) + beta

        return x
