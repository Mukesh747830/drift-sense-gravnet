import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        
        # Convert raw [0, 1000] pixel coordinates to a flattened 1D index
        # We clamp to 999 to avoid out-of-bounds index for a 1000x1000 image
        gt_x = int(np.clip(item['gt_x'], 0, 999))
        gt_y = int(np.clip(item['gt_y'], 0, 999))
        
        # The flattened index corresponding to (gt_y, gt_x)
        target_class = gt_y * 1000 + gt_x
        
        return ref_tensor, search_tensor, target_class

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = GravNet().to(device)
    
    dataset = DriftSenseDataset('dataset/labels.json')
    
    # We use batch_size=32 and persistent_workers to speed up training massively
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Quick test over 15 epochs
    epochs = 15
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (ref, search, target_class) in enumerate(pbar):
            ref = ref.to(device)
            search = search.to(device)
            target_class = target_class.to(device)
            
            with torch.amp.autocast('cuda'):
                # We request the raw, un-softmaxed flat logits from model.py
                _, flat_logits = model(ref, search, return_logits=True)
                
                # Spatial Cross Entropy over the 1,000,000 pixels!
                # This mathematically prohibits the "All-Zeroes Heatmap" shortcut
                # because the probabilities MUST sum to 1.
                loss = F.cross_entropy(flat_logits, target_class)
            
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), 'gravnet_weights.pt')
        print("Model weights saved.")

if __name__ == '__main__':
    train()
