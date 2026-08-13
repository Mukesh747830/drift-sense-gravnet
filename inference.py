import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import GravNet

def run_inference(ref_path, search_path, weights_path='gravnet_weights.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GravNet().to(device)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Weights file '{weights_path}' not found! The script was silently failing and guessing (500, 500) randomly.")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    transform = transforms.ToTensor()
    ref_img = transform(Image.open(ref_path).convert('L')).unsqueeze(0).to(device)
    search_img = transform(Image.open(search_path).convert('L')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Match training environment precision perfectly
        with torch.amp.autocast('cuda'):
            pred_coords = model(ref_img, search_img)
            pred_coords = pred_coords.squeeze().cpu().numpy()
        
    # Un-normalize coordinates back to original pixel dimension (1000)
    x_sub = pred_coords[0] * 1000.0
    y_sub = pred_coords[1] * 1000.0
    
    return x_sub, y_sub

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grav-Net Inference Script")
    parser.add_argument('ref_img', help='Path to Reference Image')
    parser.add_argument('search_img', help='Path to Search Image')
    args = parser.parse_args()
    
    x, y = run_inference(args.ref_img, args.search_img)
    print(f"Predicted Center: ({x:.2f}, {y:.2f})")
