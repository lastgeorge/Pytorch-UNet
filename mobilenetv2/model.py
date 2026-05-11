# mobilenetv2/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class MobileNet_UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MobileNet_UNet, self).__init__()
        # pretrained MobileNetV2 encoder
        mobilenet = mobilenet_v2(pretrained=True)

        self.input_proj = nn.Conv2d(input_channels, 3, kernel_size=1)
        self.encoder = mobilenet.features

        # decoder: ConvTranspose + Double Conv
        self.up1 = nn.ConvTranspose2d(1280, 512, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 1×1 conv
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # channel mapping
        orig_h, orig_w = x.size(2), x.size(3)
        x = self.input_proj(x)      # [B, in_ch, H, W] → [B, 3, H, W]
        # encoder
        x = self.encoder(x)         # [B, 1280, H/32, W/32]
        # decoder upsample + conv
        x = self.up1(x)             # [B, 512, H/16, W/16]
        x = self.conv1(x)
        x = self.up2(x)             # [B, 128, H/8,  W/8]
        x = self.conv2(x)
        # bilinear interpolation
        x = F.interpolate(x, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        x = self.final(x)           # [B, output_channels, H, W]
        return torch.sigmoid(x)

