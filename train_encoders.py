"""
Encoder Pre-training Script for M²BeamLLM.
Reference: Section III-F, Eq. (19).

Pre-trains the multimodal encoders on single-frame beam classification
before the full M²BeamLLM fine-tuning.

IMPORTANT: Per paper Eq. (19), encoder pre-training uses ONLY cross-entropy
loss. The alignment loss (Eq. 21) is only used during full model fine-tuning.

Usage:
    python train_encoders.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_config, get_device
from models.robust_m2beamllm import EncoderPretrainModel
from utils.dataset import create_dataloaders
from utils.metrics import top_k_accuracy


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train encoders for one epoch. Only cross-entropy loss (Eq. 19)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["images"][:, 0:1].to(device)
        radars = batch["radars"][:, 0:1].to(device)
        lidars = batch["lidars"][:, 0:1].to(device)
        gps_data = batch["gps"][:, 0:1].to(device)
        targets = batch["beam_history"][:, 0].to(device)

        optimizer.zero_grad()

        logits, raw_features = model(images, radars, lidars, gps_data)

        # Only cross-entropy loss (Eq. 19) — no alignment loss!
        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100*total_correct/total_samples:.1f}%"
        )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate encoder pre-training."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images = batch["images"][:, 0:1].to(device)
        radars = batch["radars"][:, 0:1].to(device)
        lidars = batch["lidars"][:, 0:1].to(device)
        gps_data = batch["gps"][:, 0:1].to(device)
        targets = batch["beam_history"][:, 0].to(device)

        logits, _ = model(images, radars, lidars, gps_data)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    cfg = get_config()
    device = get_device(cfg)
    print(f"Device: {device}")

    H, T = cfg.train.H_standard, cfg.train.T_standard
    train_loader, test_loader = create_dataloaders(cfg, H=H, T=T)

    model = EncoderPretrainModel(
        feature_dim=cfg.model.feature_dim,
        num_beams=cfg.data.num_beams,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.encoder_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.train.lr_decay_step, gamma=cfg.train.lr_decay_factor
    )

    best_acc = 0.0
    patience = 8
    patience_counter = 0
    print(f"\n{'='*60}")
    print(f"  Encoder Pre-training (Eq. 19: cross-entropy only)")
    print(f"  Epochs: {cfg.train.encoder_epochs}")
    print(f"  LR: {cfg.train.encoder_lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.train.encoder_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{cfg.train.encoder_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {100*train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {100*val_acc:.1f}% | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(cfg.train.checkpoint_dir, "encoder_pretrained.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.encoder.state_dict(),
                "val_acc": val_acc,
            }, save_path)
            print(f"  → Saved best encoder (Val Acc: {100*val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print(f"\nEncoder pre-training complete. Best Val Acc: {100*best_acc:.1f}%")


if __name__ == "__main__":
    main()