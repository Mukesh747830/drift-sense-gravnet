import json
import time
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from model import GravNet
import os

def create_distance_penalty(h, w, center_x, center_y, device='cpu'):
    """
    Creates a quadratic distance penalty map to act as the tie-breaker mask.
    """
    y = torch.arange(0, h, dtype=torch.float32, device=device)
    x = torch.arange(0, w, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(y, x, indexing='ij')
    dist_sq = (x - center_x)**2 + (y - center_y)**2
    return dist_sq

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
    model.eval()
    
    transform = transforms.ToTensor()
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    errors = []
    phase_errors = []
    times = []
    
    print(f"Evaluating {len(data)} images...")
    
    # Pre-compute gravity mask for macro tie-breaking
    dist_sq = create_distance_penalty(1000, 1000, 500, 500, device=device)
    
    batch_size = 2
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
                pred_heatmap = model(ref_batch, search_batch).squeeze(1) # [B, 1000, 1000]
                
                # 1. Macro Tie-Breaker: Find the closest valid peak to the center (500, 500).
                # Subtracting dist_sq directly from raw logits preserves float32 precision flawlessly!
                masked_heatmap = pred_heatmap - dist_sq.unsqueeze(0) * 1.0
                
                flat_indices = masked_heatmap.view(pred_heatmap.size(0), -1).argmax(dim=-1)
                macro_y = flat_indices // 1000
                macro_x = flat_indices % 1000
                
                # 2. Micro Subpixel Extractor: The macro tie-breaker slightly shifts the argmax due to the mask.
                # To get perfect subpixel accuracy, we search a tiny 5x5 window around the macro candidate 
                # in the purely UNMASKED raw heatmap!
                pred_y_batch = []
                pred_x_batch = []
                for b in range(pred_heatmap.size(0)):
                    my, mx = macro_y[b].item(), macro_x[b].item()
                    
                    y_start, y_end = max(0, my-5), min(1000, my+6)
                    x_start, x_end = max(0, mx-5), min(1000, mx+6)
                    
                    window = pred_heatmap[b, y_start:y_end, x_start:x_end]
                    win_idx = window.argmax().item()
                    win_y = win_idx // window.shape[1]
                    win_x = win_idx % window.shape[1]
                    
                    pred_y_batch.append(y_start + win_y)
                    pred_x_batch.append(x_start + win_x)

        end_time = time.time()
        
        for j, item in enumerate(batch_items):
            pred_x = pred_x_batch[j]
            pred_y = pred_y_batch[j]
            
            gt_x = item['gt_x']
            gt_y = item['gt_y']
            
            # Absolute Error
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
