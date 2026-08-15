import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random

from model import GravNet

class DriftSenseDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transforms.ToTensor()
        self.json_dir = Path(json_path).parent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        ref_path = self.json_dir / item['ref_image']
        search_path = self.json_dir / item['search_image']
        
        ref_img = Image.open(ref_path).convert('L')
        search_img = Image.open(search_path).convert('L')
        
        ref_tensor = self.transform(ref_img)
        search_tensor = self.transform(search_img)
        
        gt_x = item['gt_x']
        gt_y = item['gt_y']
        
        return ref_tensor, search_tensor, torch.tensor([gt_x, gt_y], dtype=torch.float32)

def create_normalized_2d_gaussian(h, w, center_x, center_y, sigma=2.0, device='cpu'):
    """
    Creates a sharply defined 2D Gaussian normalized to sum to 1.0.
    sigma=2.0 prevents overlap with identical adjacent peaks that are 10 pixels away.
    """
    y = torch.arange(0, h, dtype=torch.float32, device=device)
    x = torch.arange(0, w, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(y, x, indexing='ij')
    dist_sq = (x - center_x)**2 + (y - center_y)**2
    gaussian = torch.exp(-dist_sq / (2 * sigma**2))
    return gaussian / (gaussian.sum() + 1e-8)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = GravNet().to(device)
    
    dataset = DriftSenseDataset('dataset/labels.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 15
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (ref, search, gt_coords) in enumerate(pbar):
            ref, search, gt_coords = ref.to(device), search.to(device), gt_coords.to(device)
            B = ref.size(0)
            
            # Crop search_img to 200x200 around gt_coords to make stride=1 training instantaneous
            search_crops = torch.zeros(B, 1, 200, 200, device=device)
            target_heatmaps = torch.zeros(B, 171 * 171, device=device)
            
            for b in range(B):
                gx, gy = gt_coords[b,0].item(), gt_coords[b,1].item()
                # Random integer jitter so the target isn't always perfectly centered
                jx, jy = random.randint(-20, 20), random.randint(-20, 20)
                sx = int(gx) - 100 + jx
                sy = int(gy) - 100 + jy
                
                # Handle boundaries
                sx = max(0, min(1000 - 200, sx))
                sy = max(0, min(1000 - 200, sy))
                
                search_crops[b] = search[b, :, sy:sy+200, sx:sx+200]
                
                peak_x = gx - sx + 35.0
                peak_y = gy - sy + 35.0
                
                # Vectorized generation of repeating grid of targets (171x171)
                # The DRAM grid repeats every 10 pixels. 
                y, x = torch.meshgrid(
                    torch.arange(171, dtype=torch.float32, device=device),
                    torch.arange(171, dtype=torch.float32, device=device),
                    indexing='ij'
                )
                
                dx = torch.arange(-80, 81, 10, dtype=torch.float32, device=device)
                dy = torch.arange(-80, 81, 10, dtype=torch.float32, device=device)
                
                cx = peak_x + dx
                cy = peak_y + dy
                
                # Compute distance squared from all grid peaks [len(cy), len(cx), 171, 171]
                # Then sum the exp() over the peaks
                # Since the peaks are far apart, summing probabilities is valid.
                dist_sq = (x.unsqueeze(0).unsqueeze(0) - cx.unsqueeze(1).unsqueeze(2).unsqueeze(3))**2 + \
                          (y.unsqueeze(0).unsqueeze(0) - cy.unsqueeze(0).unsqueeze(2).unsqueeze(3))**2
                          
                target = torch.exp(-dist_sq / (2.0 * 2.0**2)).sum(dim=(0, 1))
                target = target / target.sum()
                target_heatmaps[b] = target.view(-1)
            
            with torch.amp.autocast('cuda'):
                pred_heatmap = model(ref, search_crops)
                pred_heatmap = pred_heatmap.reshape(B, -1)
                loss = F.cross_entropy(pred_heatmap, target_heatmaps)
            
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        torch.save(model.state_dict(), 'gravnet_weights.pt')
        print("Model weights saved.")

if __name__ == '__main__':
    train()
