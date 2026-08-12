import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import GravNet

def subpixel_peak(heatmap, y, x):
    """
    Fits a 2D quadratic surface to the 3x3 neighborhood around the peak
    to compute subpixel coordinates.
    """
    H, W = heatmap.shape
    if y == 0 or y == H-1 or x == 0 or x == W-1:
        return float(x), float(y)
        
    patch = heatmap[y-1:y+2, x-1:x+2]
    
    # 2D quadratic fit formulas (Taylor expansion based offset)
    # dx = (f(x+1) - f(x-1)) / (2 * (2f(x) - f(x-1) - f(x+1)))
    dx_num = patch[1, 2] - patch[1, 0]
    dx_den = 2 * (2 * patch[1, 1] - patch[1, 0] - patch[1, 2])
    
    dy_num = patch[2, 1] - patch[0, 1]
    dy_den = 2 * (2 * patch[1, 1] - patch[0, 1] - patch[2, 1])
    
    dx = dx_num / (dx_den + 1e-8) if dx_den != 0 else 0
    dy = dy_num / (dy_den + 1e-8) if dy_den != 0 else 0
    
    # Clip subpixel shift to valid bounds
    dx = np.clip(dx, -0.5, 0.5)
    dy = np.clip(dy, -0.5, 0.5)
    
    return float(x + dx), float(y + dy)

def run_inference(ref_path, search_path, weights_path='gravnet_weights.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GravNet().to(device)
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except FileNotFoundError:
        print(f"Warning: {weights_path} not found. Running with untrained weights (Random guess).")
        
    model.eval()
    
    transform = transforms.ToTensor()
    ref_img = transform(Image.open(ref_path).convert('L')).unsqueeze(0).to(device)
    search_img = transform(Image.open(search_path).convert('L')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        heatmap = model(ref_img, search_img)
        heatmap = heatmap.squeeze().cpu().numpy()
        
    # Find integer peak coordinate
    y_int, x_int = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # Subpixel refinement
    x_sub, y_sub = subpixel_peak(heatmap, y_int, x_int)
    
    return x_sub, y_sub

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grav-Net Inference Script")
    parser.add_argument('ref_img', help='Path to Reference Image')
    parser.add_argument('search_img', help='Path to Search Image')
    args = parser.parse_args()
    
    x, y = run_inference(args.ref_img, args.search_img)
    print(f"Predicted Center: ({x:.2f}, {y:.2f})")
