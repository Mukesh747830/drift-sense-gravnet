"""
eval.py — Argmax Heatmap Extraction & Euclidean Error Evaluation

Extracts predictions from the model's 64x64 spatial heatmap:
  1. torch.argmax on heatmap.reshape(1, -1) (contiguity-safe)
  2. Unravels flat index to 2D (y, x)
  3. Scales from heatmap coords to 1000x1000 search image coords
  4. Calculates Euclidean pixel error
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import DriftSenseNet

# Constants — must match model.py and train.py
HEATMAP_SIZE = DriftSenseNet.HEATMAP_SIZE  # 64
IMG_SIZE = 1000
SCALE_FACTOR = IMG_SIZE / HEATMAP_SIZE     # 15.625


def extract_prediction(heatmap):
    """
    Extract (pred_x, pred_y) in 1000x1000 image coords from a heatmap tensor.

    Uses .reshape() (not .view()) to avoid contiguity errors.
    """
    flat = heatmap.reshape(1, -1)  # [1, 4096]
    flat_idx = torch.argmax(flat, dim=1).item()

    hm_y = flat_idx // HEATMAP_SIZE
    hm_x = flat_idx % HEATMAP_SIZE

    # Map heatmap pixel center to image coords
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
                pred_heatmap = model(ref_batch, search_batch)  # [B, 1, 64, 64]

            elapsed = time.time() - start_time
            times.append(elapsed / len(batch_items))

            for j, item in enumerate(batch_items):
                heatmap_j = pred_heatmap[j]  # [1, 64, 64]
                pred_x, pred_y = extract_prediction(heatmap_j)

                gt_x = item['gt_x']
                gt_y = item['gt_y']

                err = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                errors.append(err)

                if err < 5:
                    error_buckets['<5px'] += 1
                elif err < 10:
                    error_buckets['<10px'] += 1
                elif err < 20:
                    error_buckets['<20px'] += 1
                else:
                    error_buckets['>=20px'] += 1

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
    print(f"  Scale factor:         {SCALE_FACTOR:.3f}x")
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
