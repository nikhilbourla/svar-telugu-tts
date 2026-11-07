import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, d_model, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)
