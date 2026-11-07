import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q_in, kv_in, key_pad_mask=None, causal=False):
        B, Lq, _ = q_in.shape
        Lk = kv_in.shape[1]

        q = self.q(q_in).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(kv_in).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(kv_in).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        allowed = None
        if key_pad_mask is not None:
            allowed = ~key_pad_mask[:, None, None, :]                       
        if causal:
            causal_ok = torch.ones(Lq, Lk, dtype=torch.bool, device=q_in.device).tril()   
            allowed = causal_ok if allowed is None else (allowed & causal_ok)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, Lq, -1)                         
        return self.out(out)
