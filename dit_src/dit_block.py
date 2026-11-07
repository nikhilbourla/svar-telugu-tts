import torch
import torch.nn as nn
from mhca import CrossAttention
from mhsa import MultiHeadSelfAttention
from adaln import Adaln
from ffn import FFN


class DitBlock(nn.Module):
    def __init__(self, cond_dim, embed_dim, num_heads):
        super().__init__()
        
        self.adaln1 = Adaln(embed_dim, cond_dim)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads)
        self.adaln2 = Adaln(embed_dim, cond_dim)
        self.mhca = CrossAttention(embed_dim, num_heads)
        self.adaln3 = Adaln(embed_dim, cond_dim)
        self.ffn = FFN(embed_dim)

    def forward(self, x_aco, x_sem, condition, semantic_mask, acoustic_mask):
        # self attention
        residual = x_aco
        x1 = self.adaln1(x_aco, condition)
        att = self.mhsa(x1, acoustic_mask)
        x = residual + att

        # cross attention
        residual = x
        x2 = self.adaln2(x, condition)
        cross = self.mhca(x2, x_sem, semantic_mask)
        x = residual + cross

        # ffn
        residual = x
        x3 = self.adaln3(x, condition)
        xffn = self.ffn(x3)
        x = residual + xffn
        return x
