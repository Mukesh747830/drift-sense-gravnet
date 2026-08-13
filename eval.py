import json
import time
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from model import GravNet
import os

def evaluate():
    json_path = Path('dataset/labels.json')
    if not json_path.exists():
        print("Dataset not found.")
        return
        
    weights_path = 'gravnet_weights.pt'
    if not os.path.exists(weights_path):
        print(f"Weights file {weights_path} not found! Did you train the model?")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GravNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Run eval with train mode disabled for speed, but remember:
    # If the network wasn't trained correctly, the batchnorm stats might cause issues.
    model.eval()
    
    transform = transforms.ToTensor()
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    errors = []
    phase_errors = []
    times = []
    
    print(f"Evaluating {len(data)} images...")
    
    batch_size = 2 # Keeping inference batch size small to avoid OOM, adjust if desired
    for i in range(0, len(data), batch_size):
        batch_items = data[i:i+batch_size]
        
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
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # pred_coords is explicitly in [0.0, 1000.0] range now
                pred_coords = model(ref_batch, search_batch).cpu().numpy()
        end_time = time.time()
        
        for j, item in enumerate(batch_items):
            # NO NEED to multiply by 1000.0! The model.py already did it.
            pred_x = pred_coords[j][0] 
            pred_y = pred_coords[j][1] 
            
            gt_x = item['gt_x']
            gt_y = item['gt_y']
            
            # Absolute Error (Misleading on repeating DRAM grid)
            err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            errors.append(err)
            
            # Subpixel Phase Error: The TRUE metric for repeating structures!
            px_diff = abs((pred_x - gt_x) % 10)
            py_diff = abs((pred_y - gt_y) % 10)
            
            phase_err_x = min(px_diff, 10 - px_diff)
            phase_err_y = min(py_diff, 10 - py_diff)
            
            phase_err = np.sqrt(phase_err_x**2 + phase_err_y**2)
            phase_errors.append(phase_err)
            
            times.append((end_time - start_time) / len(batch_items))
            
    avg_error = np.mean(errors)
    avg_phase_err = np.mean(phase_errors)
    avg_time = np.mean(times)
    
    success_rate = sum(1 for e in phase_errors if e < 2.0) / len(phase_errors) * 100
    
    print("-" * 40)
    print(f"Absolute Origin Error (Illusion): {avg_error:.3f} px")
    print(f"Subpixel Phase Error (True Drift): {avg_phase_err:.3f} px")
    print(f"Success Rate (<2px drift err): {success_rate:.1f}%")
    print(f"Average Inference Time: {avg_time:.3f} s/pair")

if __name__ == '__main__':
    evaluate()
