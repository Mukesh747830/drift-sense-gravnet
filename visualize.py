import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import GravNet
from eval import create_spatial_gravity_mask

def visualize_prediction(model_path, search_path, ref_path):
    # 1. Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GravNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load images
    search_img = Image.open(search_path).convert('L')
    ref_img = Image.open(ref_path).convert('L')
    
    # Convert to tensors
    search_tensor = torch.tensor(np.array(search_img)/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    ref_tensor = torch.tensor(np.array(ref_img)/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Get Prediction Heatmap
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred_heatmap = model(ref_tensor, search_tensor).squeeze(1) # [1, 1000, 1000]
        
        gravity_mask = create_spatial_gravity_mask(1000, 1000, 500, 500, device=device)
        masked_heatmap = torch.sigmoid(pred_heatmap) * gravity_mask.unsqueeze(0)
        
        flat_idx = masked_heatmap.view(1, -1).argmax(dim=-1).item()
        pred_y = flat_idx // 1000
        pred_x = flat_idx % 1000

    # 4. Draw the bounding box using OpenCV
    box_size = 100 
    search_cv2 = cv2.cvtColor(np.array(search_img), cv2.COLOR_GRAY2BGR)
    
    top_left = (int(pred_x - box_size/2), int(pred_y - box_size/2))
    bottom_right = (int(pred_x + box_size/2), int(pred_y + box_size/2))
    
    cv2.rectangle(search_cv2, top_left, bottom_right, (0, 0, 255), 2)

    # 5. Show the result
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
