import torch
import torch.nn as nn


class FlowMatching(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x0):

        t = torch.rand(x0.shape[0], device=x0.device)

        noise = torch.randn_like(x0)

        t_expand = t.view(x0.shape[0], 1, 1)

        xt = (1 - t_expand) * x0 + t_expand * noise

        target_velocity = noise - x0

        return xt, t, target_velocity