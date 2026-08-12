import json
import time
import torch
import numpy as np
from pathlib import Path
from inference import run_inference

def evaluate():
    json_path = Path('dataset/labels.json')
    if not json_path.exists():
        print("Dataset not found. Please run dataset_generator.py first.")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    errors = []
    times = []
    
    print(f"Evaluating {len(data)} images...")
    
    for item in data:
        ref_path = json_path.parent / item['ref_image']
        search_path = json_path.parent / item['search_image']
        
        start_time = time.time()
        pred_x, pred_y = run_inference(str(ref_path), str(search_path))
        end_time = time.time()
        
        gt_x = item['gt_x']
        gt_y = item['gt_y']
        
        err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        errors.append(err)
        times.append(end_time - start_time)
        
        print(f"Image {item['id']}: Error = {err:.3f} px (Time = {times[-1]:.3f}s)")
        
    avg_error = np.mean(errors)
    avg_time = np.mean(times)
    
    # Tolerance for success is 2 pixels
    success_rate = sum(1 for e in errors if e < 2.0) / len(errors) * 100
    
    print("-" * 30)
    print(f"Average Error: {avg_error:.3f} px")
    print(f"Success Rate (<2px err): {success_rate:.1f}%")
    print(f"Average Inference Time: {avg_time:.3f} s/pair")

if __name__ == '__main__':
    evaluate()
