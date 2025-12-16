import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # [half_dim]
        emb = t[:, None] * emb[None, :]  # [B, 1] * [1, half_dim] → [B, half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [B, dim]
        return emb