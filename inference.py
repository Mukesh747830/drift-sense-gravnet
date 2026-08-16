"""
inference.py — Standalone Localization Inference Script

Loads the pre-trained DriftSenseNet and performs inference on a directory
of test image pairs (Search and Reference images). Extracts the predicted
(x, y) drift coordinates from the 2D spatial heatmap via argmax.

Usage:
    python inference.py --input_dir <path_to_test_data> --output_dir <path_to_save_results>
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms
from PIL import Image

from model import DriftSenseNet

HEATMAP_SIZE = DriftSenseNet.HEATMAP_SIZE
IMG_SIZE = 1000
SCALE_FACTOR = IMG_SIZE / HEATMAP_SIZE


def extract_prediction(heatmap):
    """Extract (pred_x, pred_y) from the heatmap."""
    flat = heatmap.reshape(1, -1)
    flat_idx = torch.argmax(flat, dim=1).item()

    hm_y = flat_idx // HEATMAP_SIZE
    hm_x = flat_idx % HEATMAP_SIZE

    pred_x = (hm_x + 0.5) * SCALE_FACTOR
    pred_y = (hm_y + 0.5) * SCALE_FACTOR

    return pred_x, pred_y


def main():
    parser = argparse.ArgumentParser(description="Applied Materials Drift-Sense Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to test images directory containing 'search' and 'reference' subdirectories.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output predictions CSV.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_dir = input_dir / "search"
    ref_dir = input_dir / "reference"

    if not search_dir.exists() or not ref_dir.exists():
        print(f"ERROR: The input directory must contain 'search' and 'reference' subfolders.")
        return

    weights_path = Path("gravnet_weights.pt")
    if not weights_path.exists():
        print("ERROR: gravnet_weights.pt not found. Ensure the weights are in the same directory.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading DriftSenseNet on {device}...")

    model = DriftSenseNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.ToTensor()
    
    # Get all image files in the search directory
    image_files = sorted([f.name for f in search_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']])
    
    print(f"Found {len(image_files)} image pairs for inference.")
    
    results = []

    with torch.no_grad():
        for filename in image_files:
            search_path = search_dir / filename
            ref_path = ref_dir / filename
            
            if not ref_path.exists():
                print(f"WARNING: Reference image for {filename} not found. Skipping.")
                continue
                
            try:
                search_img = Image.open(search_path).convert('L')
                ref_img = Image.open(ref_path).convert('L')
                
                search_tensor = transform(search_img).unsqueeze(0).to(device)
                ref_tensor = transform(ref_img).unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda'):
                    pred_heatmap = model(ref_tensor, search_tensor)
                    
                pred_x, pred_y = extract_prediction(pred_heatmap[0])
                
                results.append({
                    "filename": filename,
                    "pred_x": round(pred_x, 2),
                    "pred_y": round(pred_y, 2)
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save results to CSV
    output_csv = output_dir / "predictions.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\nInference complete! Saved {len(results)} predictions to {output_csv}")


if __name__ == "__main__":
    main()
