import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import GravNet

def visualize_prediction(model_path, search_path, ref_path):
    # 1. Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GravNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load images
    search_img = Image.open(search_path).convert('L')
    ref_img = Image.open(ref_path).convert('L')
    
    # Convert to tensors (adjust normalization based on your dataset loader)
    search_tensor = torch.tensor(np.array(search_img)/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    ref_tensor = torch.tensor(np.array(ref_img)/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Get Prediction
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred_coords = model(ref_tensor, search_tensor)
        
        # Safely unpack the [1, 2] coordinate tensor
        pred_x, pred_y = pred_coords[0][0].item(), pred_coords[0][1].item()

    # 4. Draw the bounding box using OpenCV
    # The reference is 1000x1000, search is 1000x1000 (10x downsampled)
    # So the reference takes up a 100x100 pixel area in the search image.
    box_size = 100 
    search_cv2 = cv2.cvtColor(np.array(search_img), cv2.COLOR_GRAY2BGR)
    
    top_left = (int(pred_x - box_size/2), int(pred_y - box_size/2))
    bottom_right = (int(pred_x + box_size/2), int(pred_y + box_size/2))
    
    # Draw a RED rectangle (BGR format: 0, 0, 255)
    cv2.rectangle(search_cv2, top_left, bottom_right, (0, 0, 255), 2)

    # 5. Show the result
    plt.figure(figsize=(10, 5))
    plt.title(f"Predicted Center: ({pred_x:.1f}, {pred_y:.1f})")
    plt.imshow(cv2.cvtColor(search_cv2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Test it on the first image in your dataset
    visualize_prediction(
        model_path="gravnet_weights.pt",
        search_path="dataset/search/0000.png",
        ref_path="dataset/reference/0000.png"
    )
