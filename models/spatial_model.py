"""
OceanEmbed Spatial CNN — CNN Ocean Embedding Framework
Operates on 0.25° grid patches with 7 input channels.

Architecture:
- Input: (B, 7, H, W) multi-channel grid patch
- Encoder: U-Net style with residual blocks
- Decoder: Upsampling with skip connections
- Output: (B, 15) temperature profile at center pixel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> GELU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Two conv blocks with residual connection"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class SpatialCNNEncoder(nn.Module):
    """U-Net encoder with progressive downsampling"""
    def __init__(self, in_channels=7, base_channels=32, num_levels=3):
        super().__init__()
        self.levels = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = in_channels
        for i in range(num_levels):
            out_ch = base_channels * (2 ** i)
            self.levels.append(nn.Sequential(
                ConvBlock(channels, out_ch),
                ResBlock(out_ch),
            ))
            self.pools.append(nn.MaxPool2d(2))
            channels = out_ch

        self.out_channels = channels

    def forward(self, x):
        features = []
        for level, pool in zip(self.levels, self.pools):
            x = level(x)
            features.append(x)
            x = pool(x)
        return x, features


class SpatialCNNDecoder(nn.Module):
    """U-Net decoder with skip connections"""
    def __init__(self, encoder_channels, base_channels=32, num_levels=3):
        super().__init__()
        self.levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        channels = encoder_channels
        for i in range(num_levels - 1, -1, -1):
            skip_ch = base_channels * (2 ** i)
            self.upsamples.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.levels.append(nn.Sequential(
                ConvBlock(channels + skip_ch, skip_ch),
                ResBlock(skip_ch),
            ))
            channels = skip_ch

    def forward(self, x, skip_features):
        for upsample, level, skip in zip(self.upsamples, self.levels, reversed(skip_features)):
            x = upsample(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = level(x)
        return x


class OceanEmbedSpatialCNN(nn.Module):
    """
    Spatial CNN for 7-channel input, 15-depth output.

    Input: (B, 7, H, W) - 7 surface variables on 0.25° grid
    Output: (B, 15) - temperature at 15 depth levels
    """
    def __init__(self, in_channels=7, out_depths=15, base_channels=32):
        super().__init__()
        self.encoder = SpatialCNNEncoder(in_channels, base_channels)
        self.decoder = SpatialCNNDecoder(self.encoder.out_channels, base_channels)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(base_channels, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, out_depths),
        )
        self.residual_proj = nn.Linear(in_channels, out_depths)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        encoded, skip_features = self.encoder(x)
        decoded = self.decoder(encoded, skip_features)
        pooled = self.gap(decoded).squeeze(-1).squeeze(-1)
        output = self.head(pooled)

        center_features = x.mean(dim=(2, 3))
        residual = self.residual_proj(center_features)
        return output + 0.1 * residual

    def predict_with_uncertainty(self, x, n_forward_passes=10):
        """MC Dropout for uncertainty estimation."""
        was_training = self.training
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_forward_passes):
                predictions.append(self.forward(x))

        predictions = torch.stack(predictions)
        self.train(was_training)
        return predictions.mean(dim=0), predictions.std(dim=0)
