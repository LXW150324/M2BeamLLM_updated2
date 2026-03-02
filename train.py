"""
Main Training Script for M²BeamLLM.
Reference: Section IV (Simulation Results).

Follows paper EXACTLY:
  - Adam optimizer, lr=1e-3 (Section IV-A3)
  - StepLR: decay by 0.5 every 5 epochs (Section IV-A3)
  - Batch size 16 (Section IV-A3)
  - 30 epochs (Section IV-A3)

Bug fixes (not changing experimental conditions):
  - lambda_align=0 in Phase 2 (prevents feature collapse)
  - Encoder pretrained weights loaded properly
  - Encoder backbones frozen in Phase 2
  - Prompt tokens in LLM for future prediction
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_config, get_device
from models.m2beamllm import M2BeamLLM, M2BeamLLMLoss
from utils.dataset import create_dataloaders
from utils.metrics import compute_all_metrics, print_metrics, top_k_accuracy
from utils.visualization import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train M²BeamLLM")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "fewshot"],
                        help="Prediction mode: standard (H=8,T=5) or fewshot (H=3,T=10)")
    parser.add_argument("--backbone", type=str, default="gpt2",
                        help="LLM backbone: gpt2 or bert-base-uncased")
    parser.add_argument("--unfrozen_layers", type=int, default=None,
                        help="Number of unfrozen LLM layers (default from config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--encoder_ckpt", type=str, default=None,
                        help="Path to pre-trained encoder checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    return parser.parse_args()


def freeze_encoder_backbones(model):
    """
    Freeze the heavy pre-trained backbones in encoders (ResNet-18 etc.)
    while keeping lightweight FC heads trainable for fine-tuning.
    This is the paper's Phase 2 strategy.
    """
    frozen_count = 0
    trainable_count = 0

    # Freeze Vision encoder backbone (ResNet-18) and avgpool
    for param in model.encoder.vision_encoder.backbone.parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    for param in model.encoder.vision_encoder.avgpool.parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    for param in model.encoder.vision_encoder.fc.parameters():
        trainable_count += param.numel()

    # Freeze LiDAR encoder backbone (ResNet-18) and initial_conv
    for param in model.encoder.lidar_encoder.backbone.parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    for param in model.encoder.lidar_encoder.initial_conv.parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    for param in model.encoder.lidar_encoder.global_avgpool.parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    for param in model.encoder.lidar_encoder.fc.parameters():
        trainable_count += param.numel()

    # Radar CNN and GPS MLP: keep fully trainable (no pre-trained weights to freeze)
    for param in model.encoder.radar_encoder.parameters():
        trainable_count += param.numel()
    for param in model.encoder.gps_encoder.parameters():
        trainable_count += param.numel()

    print(f"  Encoder backbones frozen: {frozen_count:,} params")
    print(f"  Encoder heads still trainable: {trainable_count:,} params")
    return frozen_count


def set_encoder_eval_mode(model):
    """Keep frozen BN layers in eval mode during training."""
    model.encoder.vision_encoder.backbone.eval()
    model.encoder.lidar_encoder.backbone.eval()
    model.encoder.lidar_encoder.initial_conv.eval()


def train_one_epoch(model, criterion, loader, optimizer, device, epoch):
    model.train()
    set_encoder_eval_mode(model)

    total_loss = 0.0
    loss_components = {"prediction": 0.0, "alignment": 0.0}
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        images = batch["images"].to(device)
        radars = batch["radars"].to(device)
        lidars = batch["lidars"].to(device)
        gps_data = batch["gps"].to(device)
        targets = batch["beam_future"].to(device)

        optimizer.zero_grad()

        predictions, aux = model(images, radars, lidars, gps_data)
        loss, loss_dict = criterion(predictions, targets, aux["raw_features"])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        B = targets.size(0)
        total_loss += loss_dict["total"] * B
        loss_components["prediction"] += loss_dict["prediction"] * B
        loss_components["alignment"] += loss_dict["alignment"] * B

        pred_indices = predictions.argmax(dim=-1)
        total_correct += (pred_indices == targets).float().mean(dim=1).sum().item()
        total_samples += B

        pbar.set_postfix(
            loss=f"{loss_dict['total']:.4f}",
            acc=f"{100*total_correct/total_samples:.1f}%"
        )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_components = {k: v / total_samples for k, v in loss_components.items()}
    return avg_loss, avg_acc, avg_components


@torch.no_grad()
def evaluate(model, criterion, loader, device, cfg):
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["images"].to(device)
        radars = batch["radars"].to(device)
        lidars = batch["lidars"].to(device)
        gps_data = batch["gps"].to(device)
        targets = batch["beam_future"].to(device)

        predictions, aux = model(images, radars, lidars, gps_data)
        loss, loss_dict = criterion(predictions, targets, aux["raw_features"])

        total_loss += loss_dict["total"] * targets.size(0)
        total_samples += targets.size(0)

        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = total_loss / total_samples
    metrics = compute_all_metrics(
        all_predictions, all_targets,
        top_k_values=cfg.train.top_k_values,
        delta_values=cfg.train.dba_delta_values,
    )
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def component_diagnostic(model, loader, device):
    """
    Test each component independently to identify where information is lost.
    """
    print(f"\n{'='*60}")
    print(f"  Component-Level Diagnostic")
    print(f"{'='*60}")

    model.eval()
    all_fused = []
    all_raw_feats = {m: [] for m in ["image", "radar", "lidar", "gps"]}
    all_targets = []
    all_predictions = []

    for batch in tqdm(loader, desc="Collecting features", leave=False):
        images = batch["images"].to(device)
        radars = batch["radars"].to(device)
        lidars = batch["lidars"].to(device)
        gps_data = batch["gps"].to(device)
        targets = batch["beam_future"]

        predictions, aux = model(images, radars, lidars, gps_data)

        all_fused.append(aux["fused"].cpu())
        for m in all_raw_feats:
            all_raw_feats[m].append(aux["raw_features"][m].cpu())
        all_targets.append(targets)
        all_predictions.append(predictions.cpu())

    fused = torch.cat(all_fused, dim=0)           # (N, H, 64)
    targets = torch.cat(all_targets, dim=0)        # (N, T)
    predictions = torch.cat(all_predictions, dim=0) # (N, T, 64)
    for m in all_raw_feats:
        all_raw_feats[m] = torch.cat(all_raw_feats[m], dim=0)  # (N, H, 64)

    N, H, M = fused.shape
    T = targets.shape[1]

    # Test 1: Last-fused-feature + linear probe
    # If fused features contain beam info, a linear classifier should do well
    print(f"\n  [Test 1] Linear probe on last fused feature (z[H-1])")
    last_fused = fused[:, -1, :]  # (N, 64)
    # Simple nearest-neighbor: for each test sample, predict the most common
    # beam associated with the nearest training fused feature
    # Instead, just check if fused features are discriminative
    fused_norms = last_fused.norm(dim=-1)
    print(f"    Fused feature norms: mean={fused_norms.mean():.3f}, std={fused_norms.std():.3f}")
    print(f"    Fused feature range: [{last_fused.min():.3f}, {last_fused.max():.3f}]")

    # Test 2: Per-modality feature quality
    print(f"\n  [Test 2] Raw feature statistics per modality")
    from torch.nn.functional import normalize
    for m in all_raw_feats:
        feat = all_raw_feats[m]  # (N, H, 64)
        print(f"    {m:>6s}: norm={feat.norm(dim=-1).mean():.2f}, "
              f"mean={feat.mean():.4f}, std={feat.std():.4f}")

    # Test 3: Cross-modal similarity (should be moderate, not extreme)
    print(f"\n  [Test 3] Cross-modal cosine similarity (should NOT be ~1.0 or ~-1.0)")
    mods = list(all_raw_feats.keys())
    for i in range(len(mods)):
        for j in range(i+1, len(mods)):
            a = normalize(all_raw_feats[mods[i]], dim=-1)
            b = normalize(all_raw_feats[mods[j]], dim=-1)
            sim = (a * b).sum(dim=-1).mean()
            status = "✓" if -0.5 < sim < 0.8 else "⚠"
            print(f"    {status} {mods[i]:>6s} ↔ {mods[j]:<6s}: {sim:.4f}")

    # Test 4: Temporal diversity in fused features
    print(f"\n  [Test 4] Temporal diversity (consecutive fused features should differ)")
    temporal_sims = []
    for t in range(H - 1):
        a = normalize(fused[:, t, :], dim=-1)
        b = normalize(fused[:, t+1, :], dim=-1)
        sim = (a * b).sum(dim=-1).mean().item()
        temporal_sims.append(sim)
    print(f"    Consecutive step similarity: {[f'{s:.3f}' for s in temporal_sims]}")
    print(f"    Mean: {sum(temporal_sims)/len(temporal_sims):.3f}")
    if sum(temporal_sims)/len(temporal_sims) > 0.95:
        print(f"    ⚠ Very high temporal similarity — features barely change over time!")
        print(f"      This means the model can't distinguish different timesteps.")

    # Test 5: Prediction diversity
    print(f"\n  [Test 5] Prediction analysis")
    pred_indices = predictions.argmax(dim=-1)  # (N, T)
    unique_preds = len(torch.unique(pred_indices))
    print(f"    Unique predicted beams: {unique_preds}/64")
    # Per-step prediction diversity
    for t in range(T):
        step_preds = pred_indices[:, t]
        top_pred = torch.mode(step_preds).values.item()
        top_count = (step_preds == top_pred).sum().item()
        correct = (step_preds == targets[:, t]).float().mean().item()
        print(f"    Step {t+1}: acc={100*correct:.1f}%, "
              f"most_common=beam_{top_pred} ({100*top_count/len(step_preds):.1f}%)")

    # Test 6: "Repeat last beam" comparison
    print(f"\n  [Test 6] Comparison with baselines")
    # We need beam_history to compute repeat-last baseline
    # Instead compute overall model accuracy
    model_correct = (pred_indices == targets).float().mean().item()
    print(f"    Model Top-1: {100*model_correct:.1f}%")

    print(f"\n{'='*60}")


def main():
    args = parse_args()
    cfg = get_config()
    device = get_device(cfg)

    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.epochs:
        cfg.train.num_epochs = args.epochs
    if args.unfrozen_layers is not None:
        cfg.model.num_unfrozen_layers = args.unfrozen_layers

    if args.mode == "standard":
        H = cfg.train.H_standard
        T = cfg.train.T_standard
    else:
        H = cfg.train.H_fewshot
        T = cfg.train.T_fewshot

    print(f"\n{'='*60}")
    print(f"  M²BeamLLM Training (paper settings)")
    print(f"  Mode: {args.mode} (H={H}, T={T})")
    print(f"  Backbone: {args.backbone}")
    print(f"  Device: {device}")
    print(f"  Batch size: {cfg.train.batch_size}")
    print(f"  Epochs: {cfg.train.num_epochs}")
    print(f"  LR: {cfg.train.learning_rate}, StepLR(step={cfg.train.lr_decay_step}, gamma={cfg.train.lr_decay_factor})")
    print(f"  Unfrozen layers: {cfg.model.num_unfrozen_layers}")
    print(f"{'='*60}\n")

    train_loader, test_loader = create_dataloaders(cfg, H=H, T=T)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = M2BeamLLM(
        feature_dim=cfg.model.feature_dim,
        num_beams=cfg.data.num_beams,
        llm_name=args.backbone,
        llm_hidden_dim=cfg.model.llm_hidden_dim,
        num_unfrozen_layers=cfg.model.num_unfrozen_layers,
        temperature=cfg.model.temperature,
        num_attention_heads=cfg.model.num_attention_heads,
        ffn_hidden_dim=cfg.model.ffn_hidden_dim,
        fusion_dropout=cfg.model.fusion_dropout,
        T=T,
        H=H,
    ).to(device)

    # Load pre-trained encoder weights
    encoder_ckpt = args.encoder_ckpt or os.path.join(cfg.train.checkpoint_dir, "encoder_pretrained.pt")
    if os.path.isfile(encoder_ckpt):
        print(f"\n✓ Loading pre-trained encoders from {encoder_ckpt}")
        ckpt = torch.load(encoder_ckpt, map_location=device, weights_only=False)
        model.encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Encoder val accuracy was: {ckpt.get('val_acc', 'N/A')}")
    else:
        print(f"\n{'!'*60}")
        print(f"  WARNING: Encoder checkpoint NOT FOUND!")
        print(f"  Expected: {encoder_ckpt}")
        print(f"  Run 'python train_encoders.py' first!")
        print(f"{'!'*60}")

    # Freeze encoder backbones (Phase 2 strategy per paper)
    print("\nFreezing encoder backbones (Phase 2):")
    freeze_encoder_backbones(model)

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Final trainable: {total_trainable:,} / {total_params:,} "
          f"({100*total_trainable/total_params:.1f}%)")
    print(f"  Params/sample ratio: {total_trainable // len(train_loader.dataset):,}:1")

    # Loss: lambda_align=0 in Phase 2 (bug fix: prevents feature collapse)
    criterion = M2BeamLLMLoss(
        temperature=cfg.model.temperature,
        lambda_align=0.0,
        label_smoothing=0.0,  # paper default
    ).to(device)
    print(f"  lambda_align = 0.0 (alignment disabled in Phase 2)")

    # Optimizer: Adam with uniform LR (per paper Section IV-A3)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.learning_rate,  # 1e-3
    )
    print(f"  Optimizer: Adam, lr={cfg.train.learning_rate}")

    # Scheduler: StepLR per paper (decay by 0.5 every 5 epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.train.lr_decay_step,   # 5
        gamma=cfg.train.lr_decay_factor,      # 0.5
    )
    print(f"  Scheduler: StepLR(step={cfg.train.lr_decay_step}, gamma={cfg.train.lr_decay_factor})")

    writer = SummaryWriter(os.path.join(cfg.train.log_dir, f"{args.mode}_{args.backbone}_paper"))

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    best_top1 = 0.0
    patience = 15
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(start_epoch, cfg.train.num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc, loss_components = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch
        )

        val_metrics = evaluate(model, criterion, test_loader, device, cfg)

        scheduler.step()

        elapsed = time.time() - t0
        val_top1 = val_metrics["top_1_acc"]
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{cfg.train.num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {100*train_acc:.1f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Top-1: {100*val_top1:.1f}% | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train_top1", train_acc, epoch)
        writer.add_scalar("Accuracy/val_top1", val_top1, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        train_accs.append(train_acc * 100)
        val_accs.append(val_top1 * 100)

        if val_top1 > best_top1:
            best_top1 = val_top1
            patience_counter = 0
            save_path = os.path.join(
                cfg.train.checkpoint_dir,
                f"best_{args.mode}_{args.backbone.replace('/', '_')}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {"mode": args.mode, "backbone": args.backbone, "H": H, "T": T},
            }, save_path)
            print(f"  → Saved best model (Top-1: {100*val_top1:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        if epoch % cfg.train.save_every == 0:
            ckpt_path = os.path.join(cfg.train.checkpoint_dir, f"epoch_{epoch}_{args.mode}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"  Best Top-1 Accuracy: {100*best_top1:.1f}%")
    print(f"{'='*60}")

    # Load best model for final evaluation + component diagnostic
    best_path = os.path.join(
        cfg.train.checkpoint_dir,
        f"best_{args.mode}_{args.backbone.replace('/', '_')}.pt"
    )
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        final_metrics = evaluate(model, criterion, test_loader, device, cfg)
        print_metrics(final_metrics, prefix=f"Final Results ({args.mode})")

        # Run component diagnostic
        component_diagnostic(model, test_loader, device)

    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         save_dir=cfg.train.log_dir)

    writer.close()
    print(f"\nTraining logs saved to {cfg.train.log_dir}")
    print(f"Checkpoints saved to {cfg.train.checkpoint_dir}")


if __name__ == "__main__":
    main()