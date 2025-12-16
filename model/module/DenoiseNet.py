import torch
import torch.nn as nn
from model.module.SinusoidalPosEmb import SinusoidalPosEmb

class DenoiseNet(nn.Module):
    def __init__(self, embed_dim, time_dim=64, hidden_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2 + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x_t, c_hat, t):
        # x_t: [B, D], t: [B], c_hat: [B, D]
        t_emb = self.time_mlp(t)  # [B, time_dim]
        inp = torch.cat([x_t, c_hat, t_emb], dim=-1)  # [B, 2D + time_dim]
        x0_pred = self.net(inp)   # [B, D]
        return x0_pred