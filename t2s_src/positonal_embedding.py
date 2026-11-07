import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.embed_dim=embed_dim
    def forward(self,x):
        t=x.shape[1]
        pos=torch.arange(t,device=x.device)
        pos=pos.float()
        half_embed=self.embed_dim//2
        indeces=torch.arange(half_embed,device=x.device)
        freq=torch.exp(-math.log(10000) * indeces/half_embed)
        pos=pos.unsqueeze(1)
        args=pos*freq
        sin_part=torch.sin(args)
        cos_part=torch.cos(args)
        pos_embedding=torch.concat([sin_part,cos_part],dim=-1)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

        
