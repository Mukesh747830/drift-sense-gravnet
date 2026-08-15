import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1)
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
        # Shrink to 100x100 and crop the center 30x30.
        ref_shrunk = F.interpolate(ref_img, size=(100, 100), mode='area')
        ref_shrunk = ref_shrunk[:, :, 35:65, 35:65]
        
        # Extract features (stride=1 means NO DOWNSAMPLING, 100% Phase Preservation!)
        ref_feat = self.extractor(ref_shrunk)      # -> 30x30 kernel
        search_feat = self.extractor(search_img)   # -> 1000x1000 feature map
        
        # 2. Batched Cross-Correlation
        B = ref_feat.size(0)
        heatmaps = []
        for i in range(B):
            r = ref_feat[i].unsqueeze(0) 
            s = search_feat[i].unsqueeze(0) 
            
            # Kernel is 30x30. To get exactly 1000x1000 output and perfect coordinate alignment,
            # we pad asymmetrically: left=15, right=14, top=15, bottom=14.
            s_padded = F.pad(s, (15, 14, 15, 14))
            heatmap = F.conv2d(s_padded, r, padding=0)
            heatmaps.append(heatmap)
            
        heatmap = torch.cat(heatmaps, dim=0)
        
        # Output is NATIVELY 1000x1000! No interpolation required!
        # 3. Simply return the raw 2D probability heatmap
        return heatmap
