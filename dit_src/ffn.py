import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.embed_dim=embed_dim
        self.linear1=nn.Linear(embed_dim,embed_dim*4)
        self.linear2=nn.Linear(embed_dim*4,embed_dim)
        self.gelu = nn.GELU()

    def forward(self,x):
        out=self.linear1(x)
        out=self.gelu(out)
        out=self.linear2(out)

        return out