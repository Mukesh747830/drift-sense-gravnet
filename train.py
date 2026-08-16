"""
train.py — Heatmap Training with Spatial Cross-Entropy Loss

Converts raw (gt_x, gt_y) from labels.json into 2D Gaussian target heatmaps.
Uses spatial cross-entropy loss for robust training on repeating structures.

Optimized for RTX 5050 (8GB VRAM): batch_size=16, epochs=15, mixed precision.
"""

import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from model import DriftSenseNet


# ──────────────────────────────────────────────────────────────────────
# Constants — derived from model.py
# ──────────────────────────────────────────────────────────────────────
HEATMAP_SIZE = DriftSenseNet.HEATMAP_SIZE  # 64
IMG_SIZE = 1000
SCALE_FACTOR = IMG_SIZE / HEATMAP_SIZE     # 15.625
GAUSSIAN_SIGMA = 1.5                       # in heatmap pixels


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
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

        ref_tensor = self.transform(ref_img)       # [1, 1000, 1000]
        search_tensor = self.transform(search_img)  # [1, 1000, 1000]

        gt_x = item['gt_x']
        gt_y = item['gt_y']

        # Convert ground truth from 1000x1000 coords to heatmap coords
        hm_x = gt_x / SCALE_FACTOR
        hm_y = gt_y / SCALE_FACTOR

        # Generate 2D Gaussian target heatmap
        target_heatmap = self._make_gaussian_heatmap(hm_x, hm_y)

        return ref_tensor, search_tensor, target_heatmap

    @staticmethod
    def _make_gaussian_heatmap(cx, cy):
        """
        Creates a [HEATMAP_SIZE x HEATMAP_SIZE] Gaussian heatmap centered at (cx, cy).
        Normalized to sum to 1.0 for use as a probability distribution.
        """
        y = torch.arange(0, HEATMAP_SIZE, dtype=torch.float32)
        x = torch.arange(0, HEATMAP_SIZE, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        gaussian = torch.exp(-dist_sq / (2.0 * GAUSSIAN_SIGMA ** 2))

        # Normalize to probability distribution
        gaussian = gaussian / (gaussian.sum() + 1e-8)
        return gaussian  # [64, 64]


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = DriftSenseNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    dataset = DriftSenseDataset('dataset/labels.json')
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 15

    print(f"\nStarting training: {epochs} epochs, {len(dataloader)} batches/epoch")
    print(f"Heatmap: {HEATMAP_SIZE}x{HEATMAP_SIZE}, Scale: {SCALE_FACTOR:.3f}x\n")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for ref, search, target_heatmap in pbar:
            ref = ref.to(device, non_blocking=True)
            search = search.to(device, non_blocking=True)
            target_heatmap = target_heatmap.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                # Forward: model outputs [B, 1, 64, 64]
                pred_heatmap = model(ref, search)

                B = pred_heatmap.size(0)
                pred_flat = pred_heatmap.reshape(B, -1)      # [B, 4096]
                target_flat = target_heatmap.reshape(B, -1)   # [B, 4096]

                # Spatial cross-entropy: -sum(target * log_softmax(pred))
                log_pred = F.log_softmax(pred_flat, dim=1)
                loss = -(target_flat * log_pred).sum(dim=1).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} — Avg Loss: {avg_loss:.4f}, LR: {lr:.6f}")

        torch.save(model.state_dict(), 'gravnet_weights.pt')
        print("Weights saved.\n")

    print("Training complete.")


if __name__ == '__main__':
    train()
