import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, ffn_mult, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, text_pad_mask):
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, key_pad_mask=text_pad_mask))   
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class Encoder(nn.Module):

    def __init__(self, n_layers, d_model, n_heads, ffn_mult, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, text_pad_mask):
        for layer in self.layers:
            x = layer(x, text_pad_mask)
        return self.final_norm(x)
