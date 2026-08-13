import json
import random
import numpy as np
from scipy.ndimage import morphological_gradient
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm

def generate_dram_canvas(width=10000, height=10000):
    """Creates a massive perfect grid representing DRAM structures."""
    canvas = np.zeros((height, width), dtype=np.float32)
    # Draw grid (Word lines and Bit lines)
    canvas[::100, :] = 0.5 
    canvas[:, ::100] = 0.5
    # Draw bright contacts at intersections
    canvas[::100, ::100] = 1.0
    return canvas

def apply_sem_physics(image_array, dose="high"):
    """Simulates SEM edge effects, electron noise, and beam blur."""
    # 1. Edge Enhancement (3D effect)
    edges = morphological_gradient(image_array, size=(3, 3))
    image_array = np.clip(image_array + (edges * 0.5), 0, 1)
    
    # 2. Shot Noise
    noise_std = 0.05 if dose == "high" else 0.2
    noise = np.random.normal(0, noise_std, image_array.shape)
    noisy_array = np.clip(image_array + noise, 0, 1)
    
    # 3. Gaussian Blur (Point Spread Function)
    img = Image.fromarray((noisy_array * 255).astype(np.uint8))
    blur_radius = 1.0 if dose == "high" else 1.5
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    
    return img

def generate_dataset(num_pairs=5000, out_dir="dataset", style="DRAM"):
    out_dir = Path(out_dir)
    ref_dir = out_dir / "reference"
    search_dir = out_dir / "search"
    ref_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    
    labels = []
    
    print(f"Initializing {style} canvas (this takes a moment)...")
    # Generate the massive canvas ONCE outside the loop
    canvas = generate_dram_canvas(10000, 10000)
        
    print("Downsampling search canvas...")
    # Pre-calculate low-res search base once
    search_canvas_img = Image.fromarray((canvas * 255).astype(np.uint8))
    search_img_low_res = search_canvas_img.resize((1000, 1000), Image.Resampling.BILINEAR)
    search_img_arr = np.array(search_img_low_res) / 255.0

    # Run the fast loop with a progress bar
    for i in tqdm(range(num_pairs), desc=f"Generating {num_pairs} Image Pairs"):
        # Select a random 1000x1000 crop for the Reference image
        ref_x = random.randint(0, 9000)
        ref_y = random.randint(0, 9000)
        ref_crop = canvas[ref_y:ref_y+1000, ref_x:ref_x+1000]
        
        # Apply physics filters
        ref_img = apply_sem_physics(ref_crop, dose="high")
        search_img = apply_sem_physics(search_img_arr, dose="low")
        
        # Ground truth calculation
        gt_x = ref_x / 10.0 + 50.0  
        gt_y = ref_y / 10.0 + 50.0
        
        # Save files
        ref_path = ref_dir / f"{i:04d}.png"
        search_path = search_dir / f"{i:04d}.png"
        
        ref_img.save(ref_path)
        search_img.save(search_path)
        
        labels.append({
            "id": i,
            "ref_image": str(ref_path.relative_to(out_dir)).replace('\\', '/'),
            "search_image": str(search_path.relative_to(out_dir)).replace('\\', '/'),
            "gt_x": gt_x,
            "gt_y": gt_y
        })
        
    # Save exact coordinates for PyTorch
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=4)
        
    print(f"\nDataset completely generated at ./{out_dir}")

if __name__ == "__main__":
    # Kicks off the dataset generation for 5000 pairs
    generate_dataset(5000, "dataset", "DRAM")