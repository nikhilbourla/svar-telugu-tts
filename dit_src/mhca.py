import torch
import torch.nn as nn
import torch.nn.functional as f
import math


class CrossAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.d_head= embed_dim // num_heads

        self.q=nn.Linear(embed_dim,embed_dim)
        self.k=nn.Linear(embed_dim,embed_dim)
        self.v=nn.Linear(embed_dim,embed_dim)

        self.out=nn.Linear(embed_dim,embed_dim)
    def forward(self,aco_x,sem_x,semantic_mask=None):
        b,aco_t,_=aco_x.shape
        sem_t=sem_x.shape[1]
        q=self.q(aco_x)
        k=self.k(sem_x)
        v=self.v(sem_x)
        wq=q.reshape(b,aco_t,self.num_heads,self.d_head).transpose(1,2)
        wk=k.reshape(b,sem_t,self.num_heads,self.d_head).transpose(1,2)
        wv=v.reshape(b,sem_t,self.num_heads,self.d_head).transpose(1,2)

        scores= wq @ wk.transpose(-2,-1)
        scores= scores/ math.sqrt(self.d_head)
        if semantic_mask is not None:

            mask = semantic_mask.unsqueeze(1).unsqueeze(2)

            scores = scores.masked_fill(mask, float("-inf"))
        weights= f.softmax(scores,dim=-1)
        outputs= weights @ wv
        outputs = outputs.transpose(1, 2).reshape(b, aco_t, self.embed_dim)
        outputs = self.out(outputs)
        return outputs