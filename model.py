import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Lightweight ResNet-like backbone
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False), # 1/2 resolution
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            SimpleConvBlock(16, 16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 1/4 resolution
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SimpleConvBlock(32, 32)
        )
        
    def forward(self, x):
        return self.net(x)

class GravNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = FeatureExtractor()
        
    def create_spatial_gravity_mask(self, h, w, center_y, center_x, sigma=200):
        y = torch.arange(0, h, dtype=torch.float32)
        x = torch.arange(0, w, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        mask = torch.exp(-dist_sq / (2 * sigma**2))
        return mask

    def forward(self, ref_img, search_img):
        # ref_img: [B, 1, 1000, 1000]
        # search_img: [B, 1, 1000, 1000]
        
        # STRICT CONSTRAINT: Downsample reference image BEFORE CNN using mode='area'
        ref_shrunk = F.interpolate(ref_img, size=(100, 100), mode='area') # [B, 1, 100, 100]
        
        # Extract features
        ref_feat = self.extractor(ref_shrunk) # [B, 32, 25, 25] (stride 4)
        search_feat = self.extractor(search_img) # [B, 32, 250, 250] (stride 4)
        
        B = ref_feat.shape[0]
        heatmaps = []
        
        for i in range(B):
            # Cross-correlation: slide ref_feat kernel over search_feat
            kernel = ref_feat[i:i+1] # [1, C, H_r, W_r]
            feat = search_feat[i:i+1] # [1, C, H_s, W_s]
            
            # Use conv2d for cross-correlation
            # To maintain the center mapping properly, we pad by kernel_size // 2
            # Here kernel size is 25x25, so padding=12
            pad = kernel.shape[2] // 2
            corr = F.conv2d(feat, kernel, padding=pad) 
            heatmaps.append(corr)
            
        heatmap = torch.cat(heatmaps, dim=0) # [B, 1, 250, 250]
        
        # Upsample heatmap to original size (1000x1000)
        heatmap = F.interpolate(heatmap, size=(1000, 1000), mode='bilinear', align_corners=False)
        
        # Apply Spatial Gravity Mask (center 500,500)
        mask = self.create_spatial_gravity_mask(1000, 1000, 500.0, 500.0).to(heatmap.device)
        masked_heatmap = heatmap * mask.unsqueeze(0).unsqueeze(0)
        
        return masked_heatmap
