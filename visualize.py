import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import GravNet
from eval import create_distance_penalty

def visualize_prediction(model_path, search_path, ref_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GravNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    search_img = Image.open(search_path).convert('L')
    ref_img = Image.open(ref_path).convert('L')
    
    search_tensor = torch.tensor(np.array(search_img)/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    ref_tensor = torch.tensor(np.array(ref_img)/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred_heatmap = model(ref_tensor, search_tensor).squeeze(1) # [1, 1000, 1000]
        
        heatmap_min = pred_heatmap.amin(dim=(1,2), keepdim=True)
        heatmap_max = pred_heatmap.amax(dim=(1,2), keepdim=True)
        heatmap_norm = 10.0 * (pred_heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
        
        # 1. Macro Tie-Breaker
        dist_sq = create_distance_penalty(1000, 1000, 500, 500, device=device)
        masked_heatmap = heatmap_norm - dist_sq.unsqueeze(0) * 0.1
        
        flat_idx = masked_heatmap.view(1, -1).argmax(dim=-1).item()
        my = flat_idx // 1000
        mx = flat_idx % 1000
        
        # 2. Micro Subpixel Extractor (Unmasked)
        y_start, y_end = max(0, my-5), min(1000, my+6)
        x_start, x_end = max(0, mx-5), min(1000, mx+6)
        
        window = pred_heatmap[0, y_start:y_end, x_start:x_end]
        win_idx = window.argmax().item()
        win_y = win_idx // window.shape[1]
        win_x = win_idx % window.shape[1]
        
        pred_y = y_start + win_y
        pred_x = x_start + win_x

    box_size = 100 
    search_cv2 = cv2.cvtColor(np.array(search_img), cv2.COLOR_GRAY2BGR)
    
    top_left = (int(pred_x - box_size/2), int(pred_y - box_size/2))
    bottom_right = (int(pred_x + box_size/2), int(pred_y + box_size/2))
    
    cv2.rectangle(search_cv2, top_left, bottom_right, (0, 0, 255), 2)

    plt.figure(figsize=(10, 5))
    plt.title(f"Predicted Center: ({pred_x}, {pred_y})")
    plt.imshow(cv2.cvtColor(search_cv2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_prediction(
        model_path="gravnet_weights.pt",
        search_path="dataset/search/0000.png",
        ref_path="dataset/reference/0000.png"
    )
