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

from model import GravNet

class DriftSenseDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transforms.ToTensor()
        self.json_dir = Path(json_path).parent

    def __len__(self):
        return len(self.data)

    def create_target_heatmap(self, h, w, gt_y, gt_x, sigma=10.0):
        # Create a synthesized 2D Gaussian heatmap target for MSE loss
        y = torch.arange(0, h, dtype=torch.float32)
        x = torch.arange(0, w, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        dist_sq = (x - gt_x)**2 + (y - gt_y)**2
        heatmap = torch.exp(-dist_sq / (2 * sigma**2))
        return heatmap

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Resolve paths relative to json file location
        ref_path = self.json_dir / item['ref_image']
        search_path = self.json_dir / item['search_image']
        
        ref_img = Image.open(ref_path).convert('L')
        search_img = Image.open(search_path).convert('L')
        
        ref_tensor = self.transform(ref_img)
        search_tensor = self.transform(search_img)
        
        gt_x = item['gt_x']
        gt_y = item['gt_y']
        
        target_heatmap = self.create_target_heatmap(1000, 1000, gt_y, gt_x)
        
        return ref_tensor, search_tensor, target_heatmap.unsqueeze(0)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = GravNet().to(device)
    
    dataset_path = Path('dataset/labels.json')
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found. Run dataset_generator.py first.")
        return
        
    dataset = DriftSenseDataset(dataset_path)
    
    # OS Multiprocessing safety & optimizations
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=2, 
        persistent_workers=True, # Prevents Windows worker re-creation overhead
        pin_memory=True
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.MSELoss()
    
    accumulation_steps = 8 # Effective batch size 16
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (ref, search, target_heatmap) in progress_bar:
            ref = ref.to(device)
            search = search.to(device)
            target_heatmap = target_heatmap.to(device)
            
            # AMP Training to save VRAM
            with torch.amp.autocast('cuda'):
                pred_heatmap = model(ref, search)
                loss = criterion(pred_heatmap, target_heatmap)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            epoch_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.6f}"})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), 'gravnet_weights.pt')
    print("Training complete. Weights saved to gravnet_weights.pt")

if __name__ == '__main__':
    # Required for Windows multiprocessing safety
    train()
