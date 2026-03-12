# mobilenetv3/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Compatible with both torchvision 0.13+ (weights API) and <=0.12 (pretrained=True)
try:
    from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
    from torchvision.models.mobilenetv3 import (
        MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
    )
    _HAS_WEIGHTS_API = True
except Exception:
    from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
    _HAS_WEIGHTS_API = False


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """
    2× upsample → concat(skip) → ConvBNAct×2
    Args:
        in_ch:   decoder input channels
        skip_ch: encoder feature channels to be concatenated (skip connection)
        out_ch:  block output channels
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBNAct(out_ch + skip_ch, out_ch),
            ConvBNAct(out_ch, out_ch)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Adjust spatial size if it differs due to odd sizes / stride effects
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class MobileNetV3_UNet(nn.Module):
    """
    MobileNetV3 encoder + U-Net decoder (multi-stage skips) + 1×1 head + sigmoid

    Args:
        input_channels:  number of input channels
        output_channels: number of mask channels
        variant:         'large' | 'small'
        pretrained:      whether to use torchvision pretrained weights
    """
    def __init__(self, input_channels, output_channels, variant: str = "large", pretrained: bool = True):
        super().__init__()

        # 0) Map arbitrary input channels to 3ch
        self.input_proj = nn.Conv2d(input_channels, 3, kernel_size=1)

        # 1) Build the MobileNetV3 encoder
        if variant.lower() == "small":
            if _HAS_WEIGHTS_API:
                weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                backbone = mobilenet_v3_small(weights=weights)
            else:
                backbone = mobilenet_v3_small(pretrained=pretrained)
        else:  # "large"
            if _HAS_WEIGHTS_API:
                weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
                backbone = mobilenet_v3_large(weights=weights)
            else:
                backbone = mobilenet_v3_large(pretrained=pretrained)

        self.encoder = backbone.features  # nn.Sequential

        # 2) Discover indices and channel sizes of downsampling layers via a dummy forward
        with torch.no_grad():
            old_mode = self.encoder.training
            self.encoder.eval()
            dummy = torch.zeros(1, 3, 256, 256)  # any square input works
            x = dummy
            prev_hw = x.shape[-2:]
            feat_meta = []  # list of dicts: {"idx": i, "c": channels, "hw": (h, w)} recorded only when spatial size changes
            for i, m in enumerate(self.encoder):
                x = m(x)
                hw = (x.shape[-2], x.shape[-1])
                if hw != prev_hw:
                    feat_meta.append({"idx": i, "c": x.shape[1], "hw": hw})
                    prev_hw = hw
            self.encoder.train(old_mode)

        # Use up to 5 deepest stages (typically 1/2, 1/4, 1/8, 1/16, 1/32)
        selected = feat_meta[-5:] if len(feat_meta) >= 5 else feat_meta
        self._feat_indices = [m["idx"] for m in selected]  # collect same indices during forward
        c_list = [m["c"] for m in selected]

        # Safety: replicate channels if fewer than 5 stages are present
        if len(c_list) == 0:
            # Extremely unlikely, but guard against it
            raise RuntimeError("No downsampling stages detected in MobileNetV3 encoder.")
        while len(c_list) < 5:
            c_list = [c_list[0]] + c_list  # replicate the shallowest channels at the front

        # Channel sizes from shallow → deep: c1..c5
        c1, c2, c3, c4, c5 = c_list[-5], c_list[-4], c_list[-3], c_list[-2], c_list[-1]

        # 3) Decoder (U-Net style with skip connections)
        self.up1 = UpBlock(in_ch=c5,   skip_ch=c4, out_ch=256)  # 1/32 → 1/16
        self.up2 = UpBlock(in_ch=256,  skip_ch=c3, out_ch=128)  # 1/16 → 1/8
        self.up3 = UpBlock(in_ch=128,  skip_ch=c2, out_ch=64)   # 1/8  → 1/4
        self.up4 = UpBlock(in_ch=64,   skip_ch=c1, out_ch=64)   # 1/4  → 1/2

        # 4) Final prediction head
        self.head = nn.Conv2d(64, output_channels, kernel_size=1)

    def _collect_encoder_features(self, x_rgb):
        """
        Run the encoder and collect outputs at the pre-recorded indices (self._feat_indices).
        Returns a list [f1 (shallow), f2, f3, f4, f5 (deep)] of as many features as exist (typically 4–5).
        """
        feats = []
        for i, m in enumerate(self.encoder):
            x_rgb = m(x_rgb)
            if i in self._feat_indices:
                feats.append(x_rgb)
        return feats

    def forward(self, x):
        b, _, H, W = x.shape
        x = self.input_proj(x)  # [B, 3, H, W]

        # Collect multi-scale encoder features
        feats = self._collect_encoder_features(x)
        # Expect: len(feats) == len(self._feat_indices) (typically 4–5)
        if len(feats) >= 5:
            c1, c2, c3, c4, c5 = feats[-5], feats[-4], feats[-3], feats[-2], feats[-1]
        elif len(feats) == 4:
            c1, c2, c3, c4 = feats
            c5 = c4
        elif len(feats) == 3:
            c1, c2, c3 = feats
            c4 = c3; c5 = c3
        else:
            # Minimum safety: operate even if there is only one feature map
            c1 = feats[-1]
            c2 = c1; c3 = c1; c4 = c1; c5 = c1

        # Decoder path
        x = self.up1(c5, c4)   # → 1/16
        x = self.up2(x,  c3)   # → 1/8
        x = self.up3(x,  c2)   # → 1/4
        x = self.up4(x,  c1)   # → 1/2

        # Restore to input resolution → 1×1 conv → sigmoid
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = self.head(x)
        return torch.sigmoid(x)

