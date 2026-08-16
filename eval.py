"""
eval.py — Argmax Heatmap Extraction & Euclidean Error Evaluation

Extracts predictions from the model's 2D spatial heatmap output:
  1. torch.argmax on the flattened heatmap (using .reshape to prevent contiguity errors)
  2. Unravels flat index back into 2D (y, x) coordinates
  3. Scales from heatmap coords to 1000x1000 search image coords
  4. Calculates and prints Euclidean pixel error against ground truth
"""

import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import DriftSenseNet

# Constants — must match model.py and train.py
HEATMAP_SIZE = DriftSenseNet.HEATMAP_SIZE  # 125
IMG_SIZE = 1000
SCALE_FACTOR = IMG_SIZE / HEATMAP_SIZE     # 8.0


def extract_prediction(heatmap, heatmap_h, heatmap_w):
    """
    Extract (x, y) prediction from a single heatmap tensor.

    Uses .reshape() (not .view()) to avoid contiguity errors.
    Unravels the flattened argmax index back to 2D.
    Scales from heatmap coordinates to 1000x1000 image coordinates.

    Args:
        heatmap:   [1, H, W] or [H, W] tensor (logits)
        heatmap_h: height of the heatmap
        heatmap_w: width of the heatmap

    Returns:
        (pred_x, pred_y) in 1000x1000 image coordinates
    """
    # Flatten using .reshape() for contiguity safety
    flat = heatmap.reshape(1, -1)  # [1, H*W]

    # Argmax on the flattened heatmap
    flat_idx = torch.argmax(flat, dim=1).item()

    # Unravel flat index → 2D coordinates
    hm_y = flat_idx // heatmap_w
    hm_x = flat_idx % heatmap_w

    # Scale from heatmap coords to 1000x1000 image coords
    pred_x = (hm_x + 0.5) * SCALE_FACTOR
    pred_y = (hm_y + 0.5) * SCALE_FACTOR

    return pred_x, pred_y


def evaluate():
    json_path = Path('dataset/labels.json')
    if not json_path.exists():
        print("ERROR: dataset/labels.json not found. Run dataset_generator.py first.")
        return

    weights_path = Path('gravnet_weights.pt')
    if not weights_path.exists():
        print("ERROR: gravnet_weights.pt not found. Run train.py first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load model
    model = DriftSenseNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.ToTensor()

    with open(json_path, 'r') as f:
        data = json.load(f)

    errors = []
    times = []
    error_buckets = {'<5px': 0, '<10px': 0, '<20px': 0, '>=20px': 0}

    num_eval = min(len(data), 5000)
    print(f"Evaluating {num_eval} images...\n")

    batch_size = 8

    with torch.no_grad():
        for i in tqdm(range(0, num_eval, batch_size), desc="Evaluating"):
            batch_items = data[i:i + batch_size]

            ref_imgs = []
            search_imgs = []
            for item in batch_items:
                ref_path = json_path.parent / item['ref_image']
                search_path = json_path.parent / item['search_image']
                ref_imgs.append(transform(Image.open(ref_path).convert('L')))
                search_imgs.append(transform(Image.open(search_path).convert('L')))

            ref_batch = torch.stack(ref_imgs).to(device)
            search_batch = torch.stack(search_imgs).to(device)

            start_time = time.time()

            with torch.amp.autocast('cuda'):
                pred_heatmap = model(ref_batch, search_batch)  # [B, 1, 125, 125]

            elapsed = time.time() - start_time
            times.append(elapsed / len(batch_items))

            # Extract predictions for each item in the batch
            for j, item in enumerate(batch_items):
                heatmap_j = pred_heatmap[j]  # [1, 125, 125]
                pred_x, pred_y = extract_prediction(
                    heatmap_j, HEATMAP_SIZE, HEATMAP_SIZE
                )

                gt_x = item['gt_x']
                gt_y = item['gt_y']

                # Euclidean pixel error
                err = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                errors.append(err)

                # Bucket classification
                if err < 5:
                    error_buckets['<5px'] += 1
                elif err < 10:
                    error_buckets['<10px'] += 1
                elif err < 20:
                    error_buckets['<20px'] += 1
                else:
                    error_buckets['>=20px'] += 1

    # ── Results ──────────────────────────────────────────────────────
    errors = np.array(errors)
    avg_error = np.mean(errors)
    median_error = np.median(errors)
    p90_error = np.percentile(errors, 90)
    p95_error = np.percentile(errors, 95)
    max_error = np.max(errors)
    avg_time = np.mean(times)

    total = len(errors)

    print("\n" + "=" * 50)
    print("       DRIFT-SENSE EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Images evaluated:     {total}")
    print(f"  Heatmap size:         {HEATMAP_SIZE}x{HEATMAP_SIZE}")
    print(f"  Scale factor:         {SCALE_FACTOR:.1f}x")
    print("-" * 50)
    print(f"  Mean Euclidean Error:  {avg_error:.2f} px")
    print(f"  Median Error:          {median_error:.2f} px")
    print(f"  90th Percentile:       {p90_error:.2f} px")
    print(f"  95th Percentile:       {p95_error:.2f} px")
    print(f"  Max Error:             {max_error:.2f} px")
    print("-" * 50)
    print(f"  < 5 px:  {error_buckets['<5px']:>5d} / {total}  "
          f"({100 * error_buckets['<5px'] / total:.1f}%)")
    print(f"  < 10 px: {error_buckets['<10px']:>5d} / {total}  "
          f"({100 * error_buckets['<10px'] / total:.1f}%)")
    print(f"  < 20 px: {error_buckets['<20px']:>5d} / {total}  "
          f"({100 * error_buckets['<20px'] / total:.1f}%)")
    print(f"  >= 20 px:{error_buckets['>=20px']:>5d} / {total}  "
          f"({100 * error_buckets['>=20px'] / total:.1f}%)")
    print("-" * 50)
    print(f"  Avg Inference Time:    {avg_time * 1000:.1f} ms/pair")
    print("=" * 50)


if __name__ == '__main__':
    evaluate()
