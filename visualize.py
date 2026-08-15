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
            heatmap = model(ref_tensor, search_tensor).squeeze() # 971x971
            
    heatmap_np = heatmap.cpu().numpy()
    
    # 971x971 center is 485
    dist_sq = create_distance_penalty(971, 971, 485, 485, device=device)
    masked_heatmap = heatmap - dist_sq * 0.1
    
    macro_idx = masked_heatmap.view(-1).argmax().item()
    macro_y = macro_idx // 971
    macro_x = macro_idx % 971
    
    y_start, y_end = max(0, macro_y - 5), min(971, macro_y + 6)
    x_start, x_end = max(0, macro_x - 5), min(971, macro_x + 6)
    
    if (y_end - y_start) == 11 and (x_end - x_start) == 11:
        window = heatmap_np[y_start:y_end, x_start:x_end]
        win_idx = window.argmax()
        win_y = win_idx // window.shape[1]
        win_x = win_idx % window.shape[1]
        
        pred_y = y_start + win_y - 35.0
        pred_x = x_start + win_x - 35.0
    else:
        pred_y = macro_y - 35.0
        pred_x = macro_x - 35.0
    box_size = 100 
    search_cv2 = cv2.cvtColor(np.array(search_img), cv2.COLOR_GRAY2BGR)
    
    top_left = (int(pred_x - box_size/2), int(pred_y - box_size/2))
    bottom_right = (int(pred_x + box_size/2), int(pred_y + box_size/2))
    
    cv2.rectangle(search_cv2, top_left, bottom_right, (0, 0, 255), 2)

    plt.figure(figsize=(10, 5))
    plt.title(f"Predicted Center: ({pred_x}, {pred_y})")
    plt.imshow(cv2.cvtColor(search_cv2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Save the picture so the user can see it!
    plt.savefig('visualized_output.png', bbox_inches='tight')
    plt.show()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GravNet Prediction")
    parser.add_argument("--search", type=str, default="dataset/search/0000.png", help="Path to search image")
    parser.add_argument("--ref", type=str, default="dataset/reference/0000.png", help="Path to reference image")
    parser.add_argument("--weights", type=str, default="gravnet_weights.pt", help="Path to model weights")
    
    args = parser.parse_args()
    
    visualize_prediction(
        model_path=args.weights,
        search_path=args.search,
        ref_path=args.ref
    )
