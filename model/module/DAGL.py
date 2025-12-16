import torch
import torch.nn as nn

class DAGL(nn.Module):
    def __init__(self, embed_dim, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mlp_mu = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, c, sample=True):
        mu = self.mlp_mu(c)          # [B, latent_dim]
        logvar = self.mlp_logvar(c)  # [B, latent_dim]
        
        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu  # 用于确定性推理（可选）
            
        c_hat = self.decoder(z)      # [B, embed_dim]
        return c_hat, mu, logvar