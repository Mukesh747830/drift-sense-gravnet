import os
import json
import random
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from scipy.ndimage import morphological_gradient

def generate_dram_canvas(width=10000, height=10000, pitch=50, line_width=10):
    canvas = np.zeros((height, width), dtype=np.float32)
    # Word lines (horizontal)
    for y in range(0, height, pitch):
        canvas[y:y+line_width, :] = 0.5
    # Bit lines (vertical)
    for x in range(0, width, pitch):
        canvas[:, x:x+line_width] = np.maximum(canvas[:, x:x+line_width], 0.5)
    # Contacts
    for y in range(0, height, pitch):
        for x in range(0, width, pitch):
            canvas[y:y+line_width, x:x+line_width] = 1.0
    return canvas

def generate_finfet_canvas(width=10000, height=10000, fin_pitch=30, gate_pitch=200, fin_width=10, gate_width=20):
    canvas = np.zeros((height, width), dtype=np.float32)
    # Fins (vertical)
    for x in range(0, width, fin_pitch):
        canvas[:, x:x+fin_width] = 0.6
    # Gates (horizontal)
    for y in range(0, height, gate_pitch):
        canvas[y:y+gate_width, :] = 1.0
    return canvas

def apply_sem_physics(image_array, dose="high"):
    # Morphological gradient for edge brightening
    gradient = morphological_gradient(image_array, size=(3, 3))
    edge_enhanced = np.clip(image_array + 0.5 * gradient, 0, 1)
    
    # Dose noise (Poisson/Gaussian shot noise)
    if dose == "high":
        noise = np.random.normal(0, 0.05, edge_enhanced.shape)
    else: # low dose, more noise
        noise = np.random.normal(0, 0.2, edge_enhanced.shape)
        
    noisy = np.clip(edge_enhanced + noise, 0, 1)
    
    # Convert to PIL for blur
    noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
    
    # Gaussian blur (PSF)
    if dose == "high":
        noisy_img = noisy_img.filter(ImageFilter.GaussianBlur(radius=1.0))
    else:
        noisy_img = noisy_img.filter(ImageFilter.GaussianBlur(radius=1.5))
        
    return noisy_img

def generate_dataset(num_pairs=30, out_dir="dataset", style="DRAM"):
    out_dir = Path(out_dir)
    ref_dir = out_dir / "reference"
    search_dir = out_dir / "search"
    ref_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    
    labels = []
    
    print(f"Generating {num_pairs} pairs of {style} data...")
    for i in range(num_pairs):
        if style == "DRAM":
            canvas = generate_dram_canvas(10000, 10000)
        else:
            canvas = generate_finfet_canvas(10000, 10000)
            
        # Select a 1000x1000 crop for the Reference image from the 1nm/px canvas
        ref_x = random.randint(0, 9000)
        ref_y = random.randint(0, 9000)
        
        ref_crop = canvas[ref_y:ref_y+1000, ref_x:ref_x+1000]
        
        # Apply high dose noise to the 1000x1000 reference FIRST (per requirement)
        ref_img = apply_sem_physics(ref_crop, dose="high")
        
        # Search image is the whole 10000x10000 canvas, downsampled to 1000x1000
        search_canvas_img = Image.fromarray((canvas * 255).astype(np.uint8))
        search_img_low_res = search_canvas_img.resize((1000, 1000), Image.Resampling.BILINEAR)
        search_img_arr = np.array(search_img_low_res) / 255.0
        
        # Apply low dose noise to search image
        search_img = apply_sem_physics(search_img_arr, dose="low")
        
        # Ground truth coordinates in the Search image
        # The reference image is 1000x1000 in the 10000x10000 canvas, spanning [ref_x, ref_x+1000]
        # In the 1000x1000 search image, it spans [ref_x/10, (ref_x+1000)/10]
        # The center is precisely ref_x/10 + 50
        gt_x = ref_x / 10.0 + 50.0  
        gt_y = ref_y / 10.0 + 50.0
        
        ref_path = ref_dir / f"{i:04d}.png"
        search_path = search_dir / f"{i:04d}.png"
        
        ref_img.save(ref_path)
        search_img.save(search_path)
        
        labels.append({
            "id": i,
            "ref_image": str(ref_path.relative_to(out_dir)),
            "search_image": str(search_path.relative_to(out_dir)),
            "gt_x": gt_x,
            "gt_y": gt_y
        })
        
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=4)
        
    print(f"Dataset generated at {out_dir}")

if __name__ == "__main__":
    generate_dataset(30, "dataset", "DRAM")
