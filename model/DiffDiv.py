import torch
import torch.nn as nn
from model.module.DAGL import DAGL
from model.module.DenoiseNet import DenoiseNet
from model.module.SeqEncoder import SeqEncoder
import torch.nn.functional as F

def compute_alpha_bars(T: int, beta_min: float, beta_max: float) -> torch.Tensor:
    """
    预计算 DDPM 中的 alpha_bar_t = ∏_{s=1}^t (1 - beta_s)
    
    Args:
        T: 扩散总步数（如 50）
        beta_min: 最小 beta（如 1e-4）
        beta_max: 最大 beta（如 0.02）
    
    Returns:
        alpha_bars: [T] tensor, alpha_bars[t-1] = \bar{α}_t
    """
    # 线性调度 beta
    betas = torch.linspace(beta_min, beta_max, T)  # [T]
    alphas = 1.0 - betas                           # [T]
    alpha_bars = torch.cumprod(alphas, dim=0)      # [T], cumulative product
    return alpha_bars

class DiffDiv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = SeqEncoder(
            config.item_size, config.embed_dim, config.hidden_dim, config.max_seq_len
        )
        self.dagl = DAGL(config.embed_dim, config.latent_dim, config.hidden_dim)
        self.denoise_net = DenoiseNet(config.embed_dim)
        self.alpha_bars = compute_alpha_bars(
            config.T, config.beta_min, config.beta_max
        ).to(config.device)
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = self._precompute_noise_schedule()

    def _precompute_noise_schedule(self):
        # 使用线性 schedule（也可用 cosine）
        betas = torch.linspace(1e-4, 0.02, self.config.T)  # [T]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # 注册为 buffer（随模型保存）
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)

        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

    def _sample_prediction(self, seq):
        """
        反向扩散采样：从纯噪声开始，逐步去噪，得到最终 item embedding，
        再与 item_emb 矩阵点积得到 logits。
        """
        B = seq.size(0)
        device = seq.device

        # 1. 编码用户上下文（固定）
        context = self.encoder(seq)  # [B, embed_dim]

        # 2. 初始化 x_T ~ N(0, I)
        x_t = torch.randn(B, self.config.embed_dim, device=device)

        # 3. 反向扩散：t = T, T-1, ..., 1
        for t in reversed(range(0, self.config.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            with torch.no_grad():
                print(t_batch.shape)
                eps_pred = self.denoise_net(x_t, context, t_batch)  # [B, embed_dim]
            
            # 获取 alpha 系数
            sqrt_alpha_t = self.sqrt_alphas[t]          # scalar
            beta_t = self.betas[t]                      # scalar
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            sqrt_one_minus_alpha_cumprod_tm1 = self.sqrt_one_minus_alphas_cumprod[t-1] if t > 1 else 0.0

            # 计算 x_{t-1}
            # 公式来自 DDPM: x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - \bar{alpha}_t)) * eps_pred) + sigma_t * z
            # 简化版（确定性采样，z=0）常用于推荐：
            if t == 1:
                x_t = (1 / sqrt_alpha_t) * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * eps_pred)
            else:
                # 加入随机项（可选，但推荐用确定性采样以稳定结果）
                z = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t)  # 或更精确的 sigma
                x_t = (1 / sqrt_alpha_t) * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * eps_pred) + sigma_t * z

        # 4. x_0 即为预测的 target item embedding
        x_0 = x_t  # [B, embed_dim]

        # 5. 与所有 item embeddings 点积 → logits
        item_embs = self.encoder.item_emb.weight[1:]  # [num_items, embed_dim], 忽略 padding_idx=0
        logits = torch.matmul(x_0, item_embs.T)  # [B, num_items]

        return logits

    def forward(self, seq, target=None, train=True):
        """
        seq: [B, L] user historical sequence
        target: [B] next item ID (for training)
        """
        B = seq.size(0)
        c = self.encoder(seq)  # [B, D]

        if train:
            # === Training Phase ===
            # Get x0 (clean embedding of next item)
            item_emb = self.encoder.item_emb.weight  # [V, D]
            x0 = item_emb[target]  # [B, D]

            # Sample random t
            t = torch.randint(1, self.config.T + 1, (B,), device=self.config.device)  # [B]
            alpha_bar_t = self.alpha_bars[t - 1]  # [B]

            # Add noise: x_t = sqrt(α̅_t) * x0 + sqrt(1 - α̅_t) * ε
            eps = torch.randn_like(x0)
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(-1, 1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t).view(-1, 1)
            x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * eps  # [B, D]

            # Get guidance signal
            c_hat, mu, logvar = self.dagl(c, sample=True)  # [B, D]

            # Predict x0
            x0_pred = self.denoise_net(x_t, c_hat, t)  # [B, D]

            # Losses
            mse_loss = F.mse_loss(x0_pred, x0, reduction='mean')
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = mse_loss + kld_loss  # 可加 ADBO，此处简化

            return total_loss, mse_loss, kld_loss

        else:
            return self._sample_prediction(seq)
    
    @torch.no_grad()
    def generate(self, seq, num_samples=5, steps=5):
        """
        Generate diverse recommendations.
        Returns: list of top-K item IDs for each path
        """
        B = seq.size(0)
        c = self.encoder(seq)  # [B, D]
        item_emb = self.encoder.item_emb.weight  # [V, D]

        all_scores = []

        for _ in range(num_samples):
            c_hat, _, _ = self.dagl(c, sample=True)  # [B, D]

            # Start from noise
            x = torch.randn(B, self.config.embed_dim, device=self.config.device)

            # Subsample timesteps (e.g., 50 → 40 → 30 → 20 → 10)
            timesteps = torch.linspace(self.config.T, 1, steps, dtype=torch.long, device=self.config.device)

            for i, t_val in enumerate(timesteps):
                t_batch = t_val.repeat(B)
                x0_pred = self.denoise_net(x, t_batch, c_hat)

                # Deterministic DDIM-like update (simplified)
                if i == len(timesteps) - 1:
                    x = x0_pred
                else:
                    alpha_bar_t = self.alpha_bars[t_val - 1]
                    alpha_bar_next = self.alpha_bars[timesteps[i+1] - 1] if i+1 < len(timesteps) else 0
                    x = (
                        torch.sqrt(alpha_bar_next) * x0_pred +
                        torch.sqrt(1 - alpha_bar_next) * torch.randn_like(x0_pred)
                    )

            # Compute similarity with all items
            scores = torch.matmul(x, item_emb.T)  # [B, V]
            all_scores.append(scores)

        # Average or merge scores (simple average here)
        final_scores = torch.stack(all_scores, dim=0).mean(dim=0)  # [B, V]
        return final_scores