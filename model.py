import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class GravNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = FeatureExtractor()

    def forward(self, ref_img, search_img):
        # 1. Feature Extraction
        # Shrink to 100x100 and crop the center 30x30 (contains 3x3 DRAM grid peaks).
        # This drastically reduces kernel size for lightning-fast training!
        ref_shrunk = F.interpolate(ref_img, size=(100, 100), mode='area')
        ref_shrunk = ref_shrunk[:, :, 35:65, 35:65]
        
        # Extract features
        ref_feat = self.extractor(ref_shrunk)      # -> 15x15 kernel!
        search_feat = self.extractor(search_img)   # -> 500x500 feature map
        
        # 2. Batched Cross-Correlation
        B = ref_feat.size(0)
        heatmaps = []
        for i in range(B):
            r = ref_feat[i].unsqueeze(0) 
            s = search_feat[i].unsqueeze(0) 
            
            # Kernel is 15x15 (Odd size). Symmetric padding perfectly preserves 500x500 size!
            heatmap = F.conv2d(s, r, padding=7)
            heatmaps.append(heatmap)
            
        heatmap = torch.cat(heatmaps, dim=0)
        
        # Interpolate correlation heatmap back to 1000x1000 pixel space
        heatmap = F.interpolate(heatmap, size=(1000, 1000), mode='bilinear', align_corners=False)
        
        # 3. Simply return the raw 2D probability heatmap
        return heatmap
