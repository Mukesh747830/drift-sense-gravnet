import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np

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

def create_2d_gaussian(h, w, center_x, center_y, sigma=10.0, device='cpu'):
    y = torch.arange(0, h, dtype=torch.float32, device=device)
    x = torch.arange(0, w, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(y, x, indexing='ij')
    dist_sq = (x - center_x)**2 + (y - center_y)**2
    return torch.exp(-dist_sq / (2 * sigma**2))

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = GravNet().to(device)
    
    dataset = DriftSenseDataset('dataset/labels.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 15
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (ref, search, gt_coords) in enumerate(pbar):
            ref = ref.to(device)
            search = search.to(device)
            
            # Generate 2D Gaussian targets on the GPU for speed
            B = ref.size(0)
            target_heatmaps = torch.zeros((B, 1000, 1000), device=device)
            for b in range(B):
                target_heatmaps[b] = create_2d_gaussian(1000, 1000, gt_coords[b,0], gt_coords[b,1], sigma=10.0, device=device)
            
            with torch.amp.autocast('cuda'):
                pred_heatmap = model(ref, search).squeeze(1) # [B, 1000, 1000]
                loss = criterion(pred_heatmap, target_heatmaps)
            
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
