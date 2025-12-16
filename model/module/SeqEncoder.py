import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch

class SeqEncoder(nn.Module):
    def __init__(self, item_size, embed_dim, nhead=2, num_layers=2, max_seq_len=20):
        super().__init__()
        self.item_emb = nn.Embedding(item_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)  # 可学习位置编码
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_seq_len = max_seq_len

    def forward(self, seq):
        B, L = seq.shape
        x = self.item_emb(seq)  # [B, L, D]
        pos_ids = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)
        # 注意：需处理 padding mask（略）
        output = self.transformer(x)
        c = output[:, -1, :]  # 或用 [CLS] token
        return c