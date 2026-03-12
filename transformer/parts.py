# transformer/parts.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("PositionalEncoding2D: channels must be divisible by 4")
        self.channels = channels

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device

        pe = torch.zeros(1, c, h, w, device=device)  # (1, C, H, W)
        c_quarter = c // 4

        # y axis encoding
        y = torch.arange(h, device=device).float().unsqueeze(1)  # (H, 1)
        div_term_y = torch.exp(
            torch.arange(0, c_quarter, 1, device=device).float()
            * (-math.log(10000.0) / max(1, c_quarter))
        )  # (c_quarter,)

        y_sin = torch.sin(y * div_term_y.unsqueeze(0))  # (H, c_quarter)
        y_cos = torch.cos(y * div_term_y.unsqueeze(0))  # (H, c_quarter)

        # (1, c_quarter, H, W)
        pe[:, 0:c_quarter, :, :] = y_sin.permute(1, 0).unsqueeze(0).unsqueeze(-1).expand(1, c_quarter, h, w)
        pe[:, c_quarter:2*c_quarter, :, :] = y_cos.permute(1, 0).unsqueeze(0).unsqueeze(-1).expand(1, c_quarter, h, w)

        # x axis encoding
        x_pos = torch.arange(w, device=device).float().unsqueeze(1)  # (W, 1)
        div_term_x = torch.exp(
            torch.arange(0, c_quarter, 1, device=device).float()
            * (-math.log(10000.0) / max(1, c_quarter))
        )  # (c_quarter,)

        x_sin = torch.sin(x_pos * div_term_x.unsqueeze(0))  # (W, c_quarter)
        x_cos = torch.cos(x_pos * div_term_x.unsqueeze(0))  # (W, c_quarter)

        # (1, c_quarter, H, W)
        pe[:, 2*c_quarter:3*c_quarter, :, :] = x_sin.permute(1, 0).unsqueeze(0).unsqueeze(2).expand(1, c_quarter, h, w)
        pe[:, 3*c_quarter:4*c_quarter, :, :] = x_cos.permute(1, 0).unsqueeze(0).unsqueeze(2).expand(1, c_quarter, h, w)

        return x + pe

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path2 = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, N, C)
        x = x + self.drop_path1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

