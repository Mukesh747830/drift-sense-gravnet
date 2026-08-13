import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class GravNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = FeatureExtractor()

    def create_spatial_gravity_mask(self, h, w, center_x, center_y, sigma=200.0):
        y = torch.arange(0, h, dtype=torch.float32)
        x = torch.arange(0, w, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        # Mathematically add the log of the mask instead of multiplying
        log_mask = -dist_sq / (2 * sigma**2)
        return log_mask

    def forward(self, ref_img, search_img, return_logits=False):
        # 1. Feature Extraction
        ref_shrunk = F.interpolate(ref_img, size=(100, 100), mode='area')
        
        ref_feat = self.extractor(ref_shrunk)      # -> 25x25
        search_feat = self.extractor(search_img)   # -> 250x250
        
        # 2. Batched Cross-Correlation
        B = ref_feat.size(0)
        heatmaps = []
        for b in range(B):
            kernel = ref_feat[b:b+1]
            feat = search_feat[b:b+1]
            
            pad = kernel.shape[2] // 2 
            corr = F.conv2d(feat, kernel, padding=pad) 
            heatmaps.append(corr)
            
        heatmap = torch.cat(heatmaps, dim=0)
        
        # Interpolate correlation heatmap back to 1000x1000 pixel space
        heatmap = F.interpolate(heatmap, size=(1000, 1000), mode='bilinear', align_corners=False)
        
        # 3. Apply Spatial Gravity Mask (Center Bias)
        _, _, H, W = heatmap.shape
        log_mask = self.create_spatial_gravity_mask(H, W, W/2, H/2).to(heatmap.device)
        
        # NORMALIZE the heatmap to [0, 10] so the Gravity Mask can effectively overpower the noise!
        # Without this, raw logits of 20,000+ completely ignore the subtle mask penalty.
        heatmap_min = heatmap.amin(dim=(2,3), keepdim=True)
        heatmap_max = heatmap.amax(dim=(2,3), keepdim=True)
        heatmap_norm = 10.0 * (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
        
        # We multiply the log_mask by 50.0 to make it a strict tie-breaker that strongly isolates the center peak
        masked_heatmap = heatmap_norm + log_mask.unsqueeze(0).unsqueeze(0) * 50.0
        
        # 4. Soft-Argmax Regression Head
        flat_logits = masked_heatmap.view(B, -1)
        weights = F.softmax(flat_logits, dim=-1)
        
        # Use actual pixel coordinates (0 to W-1) directly to fix the grid scaling bug!
        y_grid = torch.arange(0, H, dtype=torch.float32, device=masked_heatmap.device)
        x_grid = torch.arange(0, W, dtype=torch.float32, device=masked_heatmap.device)
        grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        pred_x = torch.sum(weights * grid_x.reshape(-1), dim=-1)
        pred_y = torch.sum(weights * grid_y.reshape(-1), dim=-1)
        
        coords = torch.stack([pred_x, pred_y], dim=-1)
        
        if return_logits:
            return coords, heatmap.view(B, -1)
            
        return coords
