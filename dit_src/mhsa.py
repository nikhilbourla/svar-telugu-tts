# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F

# class DitBlock(nn.Module):
#     def __init__(self,embed_dim,num_head,):
#         super().__init__()
#         self.embed_dim=embed_dim
#         self.num_heads=num_head
#         self.d_head=embed_dim//num_head
#         self.q=nn.Linear(embed_dim,embed_dim)
#         self.k=nn.Linear(embed_dim,embed_dim)
#         self.v=nn.Linear(embed_dim,embed_dim)

#         self.out=nn.Linear(embed_dim,embed_dim)
#     def forward(self,x):
#         b=x.shape(0)
#         t=x.shape(1)
#         q=self.q(x)
#         k=self.k(x)
#         v=self.v(x)
#         q=q.reshape(b,t,self.num_heads,self.d_head).transpose(self.num_heads,t)
#         k=k.reshape(b,t,self.num_heads,self.d_head).transpose(self.num_heads,t)
#         v=v.reshape(b,t,self.num_heads,self.d_head).transpose(self.num_heads,t)
#         # Step 3: Attention scores
#         scores = q @ k.transpose(-2, -1)

#         # Step 4: Scale
#         scores = scores / math.sqrt(self.head_dim)

#         # Step 5: Softmax
#         weights = F.softmax(scores, dim=-1)

#         # Step 6: Weighted sum of Values
#         output = weights @ V

#         # Step 7: Merge heads
#         output = output.transpose(1, 2).reshape(B, T, self.embed_dim)

#         # Step 8: Output projection
#         output = self.out_proj(output)

#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x,acoustic_mask=None):

        b, t, _ = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.reshape(b, t, self.num_heads, self.d_head).transpose(1, 2)
        k = k.reshape(b, t, self.num_heads, self.d_head).transpose(1, 2)
        v = v.reshape(b, t, self.num_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)

        scores = scores / math.sqrt(self.d_head)
        if acoustic_mask is not None:

            
            mask = acoustic_mask.unsqueeze(1).unsqueeze(2)

            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        output = weights @ v

        output = output.transpose(1, 2).reshape(b, t, self.embed_dim)

        output = self.out(output)

        return output

