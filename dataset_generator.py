"""
dataset_generator.py — The Fingerprint Fix
Generates 5000 synthetic DRAM wafer image pairs with unique "fingerprints"
(contact drops, blob noise) so every crop is distinguishable.
"""

import json
import random
import numpy as np
from scipy.ndimage import morphological_gradient, gaussian_filter
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm


def generate_blob_noise(height, width, num_blobs=30, intensity=0.08):
    """
    Generates low-frequency blob variations to simulate faint resist thickness
    or underlying topography variations. This breaks perfect periodicity.
    """
    canvas = np.zeros((height, width), dtype=np.float32)
    for _ in range(num_blobs):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        radius = random.randint(200, 800)
        strength = random.uniform(-intensity, intensity)

        y, x = np.ogrid[max(0, cy - radius):min(height, cy + radius),
                         max(0, cx - radius):min(width, cx + radius)]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        mask = dist_sq < radius ** 2
        blob = np.zeros_like(dist_sq, dtype=np.float32)
        blob[mask] = strength * (1.0 - dist_sq[mask] / (radius ** 2))
        canvas[max(0, cy - radius):min(height, cy + radius),
               max(0, cx - radius):min(width, cx + radius)] += blob

    return canvas


def generate_layout_canvas(width=10000, height=10000, style="DRAM",
                           drop_rate=0.08, seed=None):
    """
    Creates a macro-layout canvas with streets and distinct memory banks.
    Injects unique fingerprints:
      1. Low-frequency blob noise on the base canvas
      2. Randomly dropped contacts (5-10%)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    canvas = np.zeros((height, width), dtype=np.float32)
    # Base background level
    canvas += 0.20

    # === FINGERPRINT 1: Inject blob noise into the base ===
    blob_noise = generate_blob_noise(height, width, num_blobs=40, intensity=0.06)
    canvas += blob_noise

    bank_size = 2000
    street_width = 200
    pitch = bank_size + street_width

    contact_positions = []  # Track contacts for selective dropping

    for by in range(0, height, pitch):
        for bx in range(0, width, pitch):
            if by + bank_size > height or bx + bank_size > width:
                continue

            # Draw the bank background
            canvas[by:by + bank_size, bx:bx + bank_size] = 0.30

            if style == "DRAM":
                # Word lines (horizontal) — 50px thick, every 100px
                for w in range(0, bank_size, 100):
                    canvas[by + w:by + w + 50, bx:bx + bank_size] = 0.55
                # Bit lines (vertical) — 50px thick, every 100px
                for b in range(0, bank_size, 100):
                    canvas[by:by + bank_size, bx + b:bx + b + 50] = 0.55
                # Contacts at intersections — bright 50x50 squares
                for w in range(0, bank_size, 100):
                    for b in range(0, bank_size, 100):
                        contact_positions.append((by + w, bx + b))
            else:
                # FinFET: Dense vertical fins
                for f in range(0, bank_size, 50):
                    canvas[by:by + bank_size, bx + f:bx + f + 25] = 0.70
                # Horizontal gate bars
                for g in range(0, bank_size, 500):
                    canvas[by + g:by + g + 50, bx:bx + bank_size] = 1.0
                    if by + g + 130 <= by + bank_size:
                        canvas[by + g + 80:by + g + 130, bx:bx + bank_size] = 1.0

    # === FINGERPRINT 2: Draw contacts, randomly dropping 5-10% ===
    actual_drop_rate = random.uniform(0.05, 0.10)
    for (cy, cx) in contact_positions:
        if rng.random() < actual_drop_rate:
            continue  # Drop this contact — unique fingerprint!
        canvas[cy:cy + 50, cx:cx + 50] = 1.0

    # Re-apply blob noise on top so it modulates everything
    canvas += blob_noise * 0.3
    canvas = np.clip(canvas, 0.0, 1.0)

    return canvas


def apply_sem_physics(image_array, dose="high"):
    """Simulates SEM edge effects, electron noise, and beam blur."""
    # Edge Enhancement (SEM effect)
    edges = morphological_gradient(image_array, size=(3, 3))
    image_array = np.clip(image_array + edges * 0.4, 0, 1)

    # Shot Noise (independent per image)
    noise_std = 0.04 if dose == "high" else 0.12
    noise = np.random.normal(0, noise_std, image_array.shape)
    noisy_array = np.clip(image_array + noise, 0, 1)

    # Gaussian Blur (Point Spread Function)
    img = Image.fromarray((noisy_array * 255).astype(np.uint8))
    blur_radius = 0.8 if dose == "high" else 1.2
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    return img


def generate_dataset(num_pairs=5000, out_dir="dataset"):
    out_dir = Path(out_dir)
    ref_dir = out_dir / "reference"
    search_dir = out_dir / "search"
    ref_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)

    labels = []

    # Generate 2 canvases with DIFFERENT random seeds (different fingerprints)
    print("Generating DRAM canvas with unique fingerprints...")
    canvas_dram = generate_layout_canvas(style="DRAM", seed=42)
    print("Generating FinFET canvas with unique fingerprints...")
    canvas_finfet = generate_layout_canvas(style="FinFET", seed=99)

    for i in tqdm(range(num_pairs), desc=f"Generating {num_pairs} pairs"):
        style = "DRAM" if random.random() > 0.4 else "FinFET"
        canvas = canvas_dram if style == "DRAM" else canvas_finfet
        ch, cw = canvas.shape

        # Ground truth center in the massive canvas
        cx = random.randint(1000, cw - 1000)
        cy = random.randint(1000, ch - 1000)

        # Reference (1 nm/px): 1000x1000 crop centered at (cx, cy)
        ref_x = max(0, cx - 500)
        ref_y = max(0, cy - 500)
        ref_x = min(ref_x, cw - 1000)
        ref_y = min(ref_y, ch - 1000)
        ref_crop = canvas[ref_y:ref_y + 1000, ref_x:ref_x + 1000].copy()

        # Search (10 nm/px): covers 10000x10000 canvas area, downsampled to 1000x1000
        # Random offset so reference isn't always centered
        offset_x = random.randint(-3500, 3500)
        offset_y = random.randint(-3500, 3500)

        search_x = cx + offset_x - 5000
        search_y = cy + offset_y - 5000

        # Clamp to canvas bounds
        search_x = max(0, min(cw - 10000, search_x))
        search_y = max(0, min(ch - 10000, search_y))

        search_crop = canvas[search_y:search_y + 10000,
                             search_x:search_x + 10000].copy()

        # Downsample search by 10x → 1000x1000
        search_pil = Image.fromarray((search_crop * 255).astype(np.uint8))
        search_pil = search_pil.resize((1000, 1000), Image.Resampling.BILINEAR)
        search_arr = np.array(search_pil) / 255.0

        # Apply independent SEM Physics
        ref_img = apply_sem_physics(ref_crop, dose="high")
        search_img = apply_sem_physics(search_arr, dose="low")

        # Ground truth: center of reference in the 1000x1000 search image
        # ref center in canvas coords = (ref_x + 500, ref_y + 500)
        # position in search_crop = (ref_x + 500 - search_x, ref_y + 500 - search_y)
        # scaled to 1000x1000 = divide by 10
        gt_x = (ref_x + 500 - search_x) / 10.0
        gt_y = (ref_y + 500 - search_y) / 10.0

        # Clamp GT to valid range (should already be valid, but safety check)
        gt_x = max(0.0, min(999.0, gt_x))
        gt_y = max(0.0, min(999.0, gt_y))

        ref_path = ref_dir / f"{i:04d}.png"
        search_path = search_dir / f"{i:04d}.png"

        ref_img.save(ref_path)
        search_img.save(search_path)

        labels.append({
            "id": i,
            "style": style,
            "ref_image": str(ref_path.relative_to(out_dir)).replace('\\', '/'),
            "search_image": str(search_path.relative_to(out_dir)).replace('\\', '/'),
            "gt_x": round(gt_x, 4),
            "gt_y": round(gt_y, 4)
        })

    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nDataset generated: {num_pairs} pairs at ./{out_dir}")
    print(f"Labels saved to {out_dir / 'labels.json'}")


if __name__ == "__main__":
    generate_dataset(5000, "dataset")