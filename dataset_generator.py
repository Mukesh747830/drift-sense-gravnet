import json
import random
import numpy as np
from scipy.ndimage import morphological_gradient, rotate, zoom
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm

def generate_layout_canvas(width=12000, height=12000, style="DRAM"):
    """Creates a macro-layout canvas with streets and distinct memory banks."""
    canvas = np.zeros((height, width), dtype=np.float32)
    # Background level
    canvas += 0.2
    
    bank_size = 2000
    street_width = 200
    pitch = bank_size + street_width
    
    for by in range(0, height, pitch):
        for bx in range(0, width, pitch):
            if by + bank_size > height or bx + bank_size > width: continue
            
            # Draw the bank background
            canvas[by:by+bank_size, bx:bx+bank_size] = 0.3
            
            # Fill bank with patterns
            if style == "DRAM":
                # Word lines and bit lines
                canvas[by:by+bank_size:100, bx:bx+bank_size] = 0.6
                canvas[by:by+bank_size, bx:bx+bank_size:100] = 0.6
                # Contacts
                canvas[by:by+bank_size:100, bx:bx+bank_size:100] = 1.0
            else: # FinFET
                # Dense vertical fins
                canvas[by:by+bank_size, bx:bx+bank_size:50] = 0.7
                # Occasional horizontal gate bars
                canvas[by:by+bank_size:500, bx:bx+bank_size] = 1.0
                canvas[by+20:by+bank_size:500, bx:bx+bank_size] = 1.0
                
    return canvas

def apply_sem_physics(image_array, dose="high"):
    # Edge Enhancement (SEM effect)
    edges = morphological_gradient(image_array, size=(3, 3))
    image_array = np.clip(image_array + (edges * 0.5), 0, 1)
    
    # Shot Noise (Independent)
    noise_std = 0.05 if dose == "high" else 0.15
    noise = np.random.normal(0, noise_std, image_array.shape)
    noisy_array = np.clip(image_array + noise, 0, 1)
    
    # Gaussian Blur
    img = Image.fromarray((noisy_array * 255).astype(np.uint8))
    blur_radius = 1.0 if dose == "high" else 1.5
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    
    return img

def augment_search(img_arr):
    """Apply realistic degradation: rotation and scaling to search array"""
    angle = random.uniform(-2, 2)
    scale = random.uniform(0.95, 1.05)
    
    # Rotate
    img_arr = rotate(img_arr, angle, reshape=False, mode='nearest')
    
    # Scale
    if scale != 1.0:
        h, w = img_arr.shape
        img_arr = zoom(img_arr, scale, mode='nearest')
        nh, nw = img_arr.shape
        # Center crop or pad back to original size
        if scale > 1.0:
            dy, dx = (nh - h) // 2, (nw - w) // 2
            img_arr = img_arr[dy:dy+h, dx:dx+w]
        else:
            pad_y, pad_x = (h - nh) // 2, (w - nw) // 2
            padded = np.zeros((h, w), dtype=img_arr.dtype)
            padded[pad_y:pad_y+nh, pad_x:pad_x+nw] = img_arr
            img_arr = padded
            
    return img_arr

def generate_dataset(num_pairs=300, out_dir="dataset"):
    out_dir = Path(out_dir)
    ref_dir = out_dir / "reference"
    search_dir = out_dir / "search"
    ref_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    
    labels = []
    
    print("Initializing DRAM and FinFET layouts...")
    canvas_dram = generate_layout_canvas(style="DRAM")
    canvas_finfet = generate_layout_canvas(style="FinFET")
    
    for i in tqdm(range(num_pairs), desc=f"Generating {num_pairs} Image Pairs"):
        style = "DRAM" if random.random() > 0.5 else "FinFET"
        canvas = canvas_dram if style == "DRAM" else canvas_finfet
        
        # Ground truth center in the massive canvas
        cx = random.randint(1000, 11000)
        cy = random.randint(1000, 11000)
        
        # Reference (1 nm/px): 1000x1000 crop centered at cx, cy
        ref_x = cx - 500
        ref_y = cy - 500
        ref_crop = canvas[ref_y:ref_y+1000, ref_x:ref_x+1000].copy()
        
        # Search (10 nm/px): 1000x1000 macro view (covers 10000x10000 canvas area)
        # Choose a random offset for the search box so the reference isn't perfectly centered
        offset_x = random.randint(-4000, 4000)
        offset_y = random.randint(-4000, 4000)
        
        search_cx = cx + offset_x
        search_cy = cy + offset_y
        
        search_x = search_cx - 5000
        search_y = search_cy - 5000
        
        # Clamp to bounds (re-adjust cx/cy if clamped)
        search_x = max(0, min(2000, search_x))
        search_y = max(0, min(2000, search_y))
        
        search_crop = canvas[search_y:search_y+10000, search_x:search_x+10000].copy()
        
        # Downsample search by 10x
        search_img_low_res = Image.fromarray((search_crop * 255).astype(np.uint8)).resize((1000, 1000), Image.Resampling.BILINEAR)
        search_arr = np.array(search_img_low_res) / 255.0
        
        # Apply Augmentations to search image before SEM physics
        search_arr = augment_search(search_arr)
        
        # Apply independent SEM Physics
        ref_img = apply_sem_physics(ref_crop, dose="high")
        search_img = apply_sem_physics(search_arr, dose="low")
        
        # Calculate final ground truth coordinates in the 1000x1000 search image
        # The reference center (cx, cy) is located at (cx - search_x) inside search_crop
        # Scaled down by 10, the center is at (cx - search_x) / 10.0
        gt_x = (cx - search_x) / 10.0
        gt_y = (cy - search_y) / 10.0
        
        ref_path = ref_dir / f"{i:04d}.png"
        search_path = search_dir / f"{i:04d}.png"
        
        ref_img.save(ref_path)
        search_img.save(search_path)
        
        labels.append({
            "id": i,
            "style": style,
            "ref_image": str(ref_path.relative_to(out_dir)).replace('\\', '/'),
            "search_image": str(search_path.relative_to(out_dir)).replace('\\', '/'),
            "gt_x": gt_x,
            "gt_y": gt_y
        })
        
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=4)
        
    print(f"\nDataset completely generated at ./{out_dir}")

if __name__ == "__main__":
    generate_dataset(300, "dataset")