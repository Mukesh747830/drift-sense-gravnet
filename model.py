"""
model.py — Fully Convolutional Siamese Network with 2D Spatial Heatmap Output

Memory-optimized for RTX 5050 (8GB VRAM):
  - Images are downscaled to 256x256 internally before the backbone
  - Lightweight 2-stage backbone (stride 4) → 64x64 heatmap
  - Depth-wise cross-correlation for template matching
  - Output: [B, 1, 64, 64] spatial logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Backbone(nn.Module):
    """
    Lightweight backbone. Stride 4 total.
    Input:  [B, 1, 256, 256]
    Output: [B, 64, 64, 64]
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvBlock(1, 16, 3, 1, 1),
            ConvBlock(16, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),           # 256 → 128
        )
        self.layer2 = nn.Sequential(
            ConvBlock(16, 32, 3, 1, 1),
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),           # 128 → 64
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class DriftSenseNet(nn.Module):
    """
    Fully Convolutional Siamese Network.

    Pipeline:
      1. Both images downscaled to 256x256 (saves VRAM)
      2. Shared backbone extracts 64-ch features at stride 4 → 64x64
      3. Reference features pooled to 5x5 kernel
      4. Depth-wise cross-correlation → raw correlation map
      5. Refinement head → [B, 1, 64, 64] spatial heatmap logits

    Ground truth mapping:
      heatmap pixel (hx, hy) corresponds to image pixel (hx * 15.625, hy * 15.625)
    """
    HEATMAP_SIZE = 64    # 256 / 4 = 64
    INTERNAL_RES = 256   # images downscaled to this before backbone

    def __init__(self):
        super().__init__()
        self.backbone = Backbone()

        # Pool reference features to a compact correlation kernel
        self.ref_pool = nn.AdaptiveAvgPool2d((5, 5))

        # 1x1 projections to align feature spaces
        self.ref_proj = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.search_proj = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Refinement head: correlation map → final heatmap logits
        self.head = nn.Sequential(
            ConvBlock(1, 16, 3, 1, 1),
            ConvBlock(16, 8, 3, 1, 1),
            nn.Conv2d(8, 1, 1, bias=True),
        )

    def forward(self, ref_img, search_img):
        """
        Args:
            ref_img:    [B, 1, H, W]  (any size, will be resized)
            search_img: [B, 1, H, W]  (any size, will be resized)
        Returns:
            heatmap:    [B, 1, 64, 64] spatial logits
        """
        R = self.INTERNAL_RES

        # Downscale to 256x256 to fit in VRAM
        ref = F.interpolate(ref_img, size=(R, R), mode='bilinear', align_corners=False)
        search = F.interpolate(search_img, size=(R, R), mode='bilinear', align_corners=False)

        # Extract features through shared backbone
        search_feat = self.search_proj(self.backbone(search))   # [B, 32, 64, 64]
        ref_feat = self.ref_proj(self.backbone(ref))            # [B, 32, 64, 64]

        # Pool reference to compact kernel
        ref_kernel = self.ref_pool(ref_feat)  # [B, 32, 5, 5]

        # Depth-wise cross-correlation (per batch element)
        B = search_feat.size(0)
        corr_maps = []
        for i in range(B):
            s = search_feat[i:i + 1]           # [1, 32, 64, 64]
            k = ref_kernel[i:i + 1]            # [1, 32, 5, 5]
            corr = F.conv2d(s, k, padding=2, groups=32)  # [1, 32, 64, 64]
            corr = corr.mean(dim=1, keepdim=True)         # [1, 1, 64, 64]
            corr_maps.append(corr)

        corr_map = torch.cat(corr_maps, dim=0)  # [B, 1, 64, 64]

        # Refine into final heatmap
        heatmap = self.head(corr_map)  # [B, 1, 64, 64]

        return heatmap
