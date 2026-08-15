import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # stride=1 perfectly preserves the true subpixel phase without aliasing.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

def fft_cross_correlate(s, r):
    B, C, H, W = s.shape
    _, _, kH, kW = r.shape
    
    pad_H = H + kH - 1
    pad_W = W + kW - 1
    
    orig_dtype = s.dtype
    s = s.to(torch.float32)
    r = r.to(torch.float32)
    
    S = torch.fft.fft2(s, s=(pad_H, pad_W))
    R = torch.fft.fft2(r, s=(pad_H, pad_W))
    
    out = torch.fft.ifft2(S * R.conj()).real
    out = out.to(orig_dtype)
    out = out.sum(dim=1, keepdim=True)
    
    return out[:, :, 0:H-kH+1, 0:W-kW+1]

class GravNet(nn.Module):
    def __init__(self):
        super(GravNet, self).__init__()
        self.extractor = FeatureExtractor()
        
    def forward(self, ref_img, search_img):
        ref_shrunk = F.interpolate(ref_img, size=(100, 100), mode='area')
        ref_shrunk = ref_shrunk[:, :, 35:65, 35:65]
        
        ref_feat = self.extractor(ref_shrunk)      # -> [B, 32, 30, 30]
        search_feat = self.extractor(search_img)   # -> [B, 32, H, W]
        
        heatmap = fft_cross_correlate(search_feat, ref_feat)
        return heatmap
