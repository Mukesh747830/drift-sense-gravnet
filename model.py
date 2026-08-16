"""
model.py — Fully Convolutional Siamese Network with 2D Spatial Heatmap Output

Architecture:
  1. Shared CNN backbone extracts dense features from both images.
  2. Reference features are cross-correlated against search features.
  3. A convolutional head refines the correlation map into a 2D probability heatmap.

The model outputs a spatial heatmap (not raw coordinates), eliminating the
mode-collapse problem caused by coordinate regression on repeating patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Backbone(nn.Module):
    """
    Lightweight VGG-style backbone that extracts multi-scale features.
    Input:  [B, 1, H, W]
    Output: [B, 128, H/8, W/8]  (dense feature map, stride 8)
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvBlock(1, 32, 3, 1, 1),
            ConvBlock(32, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # /2
        )
        self.layer2 = nn.Sequential(
            ConvBlock(32, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # /4
        )
        self.layer3 = nn.Sequential(
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # /8
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class DriftSenseNet(nn.Module):
    """
    Fully Convolutional Siamese Network.

    Pipeline:
      1. Backbone extracts features from search image (1000x1000 → 125x125)
      2. Backbone extracts features from reference image (1000x1000 → 125x125)
      3. Reference is adaptive-avg-pooled to a small kernel
      4. Depth-wise cross-correlation produces a raw correlation map
      5. A convolutional head refines into a 2D spatial heatmap (125x125)

    Output: [B, 1, 125, 125] heatmap (logits for spatial cross-entropy)
    """
    HEATMAP_SIZE = 125  # 1000 / 8 = 125

    def __init__(self):
        super().__init__()
        self.backbone = Backbone()

        # Squeeze the reference features into a compact kernel
        self.ref_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Project ref features into a single "template" channel for correlation
        self.ref_proj = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.search_proj = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Refinement head: takes correlation map and produces final heatmap
        self.head = nn.Sequential(
            ConvBlock(1, 32, 3, 1, 1),
            ConvBlock(32, 16, 3, 1, 1),
            nn.Conv2d(16, 1, 1, bias=True),  # Final 1x1 conv → logits
        )

    def forward(self, ref_img, search_img):
        """
        Args:
            ref_img:    [B, 1, 1000, 1000]
            search_img: [B, 1, 1000, 1000]
        Returns:
            heatmap:    [B, 1, 125, 125] — spatial logits
        """
        # Extract features
        search_feat = self.search_proj(self.backbone(search_img))  # [B, 64, 125, 125]
        ref_feat = self.ref_proj(self.backbone(ref_img))            # [B, 64, 125, 125]

        # Pool reference to compact kernel
        ref_kernel = self.ref_pool(ref_feat)  # [B, 64, 7, 7]

        # Depth-wise cross-correlation (per-batch element)
        B = search_feat.size(0)
        corr_maps = []
        for i in range(B):
            # F.conv2d with padding='same' to keep spatial dims
            s = search_feat[i:i + 1]          # [1, 64, 125, 125]
            k = ref_kernel[i:i + 1]           # [1, 64, 7, 7]
            # Group convolution: each of the 64 channels correlates independently
            corr = F.conv2d(s, k, padding=3, groups=64)  # [1, 64, 125, 125]
            corr = corr.mean(dim=1, keepdim=True)         # [1, 1, 125, 125]
            corr_maps.append(corr)

        corr_map = torch.cat(corr_maps, dim=0)  # [B, 1, 125, 125]

        # Refine correlation map into final heatmap logits
        heatmap = self.head(corr_map)  # [B, 1, 125, 125]

        return heatmap
