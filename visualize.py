"""
visualize.py — Visual verification of DriftSenseNet predictions.

3-panel display:
  Left:   Reference image (1 nm/px)
  Center: Search image (10 nm/px) with predicted bounding box (RED) + GT marker (GREEN)
  Right:  Raw predicted heatmap
"""

import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from model import DriftSenseNet

HEATMAP_SIZE = DriftSenseNet.HEATMAP_SIZE  # 64
IMG_SIZE = 1000
SCALE_FACTOR = IMG_SIZE / HEATMAP_SIZE     # 15.625


def visualize_prediction(model_path, search_path, ref_path, gt_x=None, gt_y=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DriftSenseNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    search_img = Image.open(search_path).convert('L')
    ref_img = Image.open(ref_path).convert('L')

    search_arr = np.array(search_img) / 255.0
    ref_arr = np.array(ref_img) / 255.0

    search_tensor = torch.tensor(search_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    ref_tensor = torch.tensor(ref_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            heatmap = model(ref_tensor, search_tensor)  # [1, 1, 64, 64]

    # Argmax extraction
    flat = heatmap.reshape(1, -1)
    flat_idx = torch.argmax(flat, dim=1).item()
    hm_y = flat_idx // HEATMAP_SIZE
    hm_x = flat_idx % HEATMAP_SIZE
    pred_x = (hm_x + 0.5) * SCALE_FACTOR
    pred_y = (hm_y + 0.5) * SCALE_FACTOR

    # Draw on search image
    box_size = 100
    search_cv2 = cv2.cvtColor(np.array(search_img), cv2.COLOR_GRAY2BGR)

    top_left = (int(pred_x - box_size / 2), int(pred_y - box_size / 2))
    bottom_right = (int(pred_x + box_size / 2), int(pred_y + box_size / 2))
    cv2.rectangle(search_cv2, top_left, bottom_right, (0, 0, 255), 2)

    err = None
    if gt_x is not None and gt_y is not None:
        cv2.circle(search_cv2, (int(gt_x), int(gt_y)), 8, (0, 255, 0), 2)
        err = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

    # Normalize heatmap for display
    hm_np = heatmap[0, 0].cpu().float().numpy()
    hm_np = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min() + 1e-8)

    # Plot 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].set_title("Reference (1 nm/px)", fontsize=14, pad=10)
    axes[0].imshow(ref_img, cmap='gray')
    axes[0].axis('off')

    title = f"Search (10 nm/px) | Pred: ({pred_x:.1f}, {pred_y:.1f})"
    if err is not None:
        title += f" | Err: {err:.1f}px"
    axes[1].set_title(title, fontsize=14, pad=10)
    axes[1].imshow(cv2.cvtColor(search_cv2, cv2.COLOR_BGR2RGB))
    axes[1].axis('off')

    axes[2].set_title("Predicted Heatmap (64x64)", fontsize=14, pad=10)
    axes[2].imshow(hm_np, cmap='hot', interpolation='bilinear')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('visualized_output.png', dpi=150, bbox_inches='tight')
    print(f"Saved: visualized_output.png")
    print(f"Predicted center: ({pred_x:.1f}, {pred_y:.1f})")
    if err is not None:
        print(f"Euclidean error: {err:.1f} px")
    plt.show()


if __name__ == "__main__":
    import random
    labels_path = Path("dataset/labels.json")
    
    if not labels_path.exists():
        print("Dataset not found. Run dataset_generator.py first.")
    else:
        with open(labels_path, 'r') as f:
            data = json.load(f)
            
        if len(data) > 0:
            # Pick a random sample from the dataset
            sample = random.choice(data)
            
            search_file = f"dataset/{sample['search_image']}"
            ref_file = f"dataset/{sample['ref_image']}"
            
            print(f"Visualizing random sample: {search_file}")
            
            visualize_prediction(
                model_path="gravnet_weights.pt",
                search_path=search_file,
                ref_path=ref_file,
                gt_x=sample['gt_x'],
                gt_y=sample['gt_y'],
            )