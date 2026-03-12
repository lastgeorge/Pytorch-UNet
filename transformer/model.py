# transformer/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet.parts import double_conv, down, up, outconv
from .parts import PositionalEncoding2D, TransformerBlock

class Transformer_UNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # --- UNet Encoder ---
        self.inc   = double_conv(input_channels, 64)   # H, W
        self.down1 = down(64, 128)                     # H/2,  W/2
        self.down2 = down(128, 256)                    # H/4,  W/4
        self.down3 = down(256, 512)                    # H/8,  W/8
        self.down4 = down(512, embed_dim)              # H/16, W/16

        # --- Transformer bottleneck ---
        self.pos2d = PositionalEncoding2D(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop
            )
            for _ in range(depth)
        ])

        # --- UNet Decoder ---
        self.up1 = up(embed_dim + 512, 256)  # skip from down3 (512)
        self.up2 = up(256 + 256, 128)        # skip from down2 (256)
        self.up3 = up(128 + 128, 64)         # skip from down1 (128)
        self.up4 = up(64 + 64, 64)           # skip from inc (64)
        self.outc = outconv(64, output_channels)

    def forward(self, x):
        b, _, h, w = x.shape

        # Encoder
        x1 = self.inc(x)       # (B, 64,   H,    W)
        x2 = self.down1(x1)    # (B, 128,  H/2,  W/2)
        x3 = self.down2(x2)    # (B, 256,  H/4,  W/4)
        x4 = self.down3(x3)    # (B, 512,  H/8,  W/8)
        x5 = self.down4(x4)    # (B, C=embed_dim, H/16, W/16)

        # Transformer bottleneck (H/16, W/16)
        x5 = self.pos2d(x5)    # add 2D pos enc: (B, C, H/16, W/16)
        bt, c, hh, ww = x5.shape
        tokens = x5.flatten(2).transpose(1, 2)  # (B, N=hh*ww, C)

        for blk in self.blocks:
            tokens = blk(tokens)

        x5 = tokens.transpose(1, 2).reshape(bt, c, hh, ww)  # (B, C, H/16, W/16)

        # Decoder with skips
        x = self.up1(x5, x4)   # (B, 256, H/8,  W/8)
        x = self.up2(x,  x3)   # (B, 128, H/4,  W/4)
        x = self.up3(x,  x2)   # (B, 64,  H/2,  W/2)
        x = self.up4(x,  x1)   # (B, 64,  H,    W)
        x = self.outc(x)       # (B, out_ch, H, W)

        return torch.sigmoid(x)

