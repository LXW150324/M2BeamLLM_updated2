"""
Evaluation Script for M²BeamLLM.
Computes Top-K accuracy and DBA-Score on the test set.
Supports ablation studies on modality combinations.

Usage:
    python evaluate.py --checkpoint checkpoints/best_standard_gpt2.pt
    python evaluate.py --checkpoint checkpoints/best_standard_gpt2.pt --ablation
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_config, get_device
from models.m2beamllm import M2BeamLLM, M2BeamLLMLoss
from utils.dataset import create_dataloaders
from utils.metrics import compute_all_metrics, print_metrics
from utils.visualization import plot_topk_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate M²BeamLLM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "fewshot"])
    parser.add_argument("--ablation", action="store_true",
                        help="Run modality ablation study")
    return parser.parse_args()


@torch.no_grad()
def evaluate_full(model, loader, device, cfg, modality_mask=None):
    """
    Evaluate model on test set with optional modality masking.

    Args:
        modality_mask: Dict of {modality: bool}, True to include.
                       None means all modalities active.
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["images"].to(device)
        radars = batch["radars"].to(device)
        lidars = batch["lidars"].to(device)
        gps_data = batch["gps"].to(device)
        targets = batch["beam_future"].to(device)

        # Apply modality masking for ablation
        if modality_mask is not None:
            if not modality_mask.get("image", True):
                images = torch.zeros_like(images)
            if not modality_mask.get("radar", True):
                radars = torch.zeros_like(radars)
            if not modality_mask.get("lidar", True):
                lidars = torch.zeros_like(lidars)
            if not modality_mask.get("gps", True):
                gps_data = torch.zeros_like(gps_data)

        predictions, _ = model(images, radars, lidars, gps_data)
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_all_metrics(
        all_predictions, all_targets,
        top_k_values=cfg.train.top_k_values,
        delta_values=cfg.train.dba_delta_values,
    )

    return metrics


def run_ablation(model, loader, device, cfg):
    """
    Run modality ablation study (Section IV-E).
    Tests various modality combinations.
    """
    configurations = {
        "Image only": {"image": True, "radar": False, "lidar": False, "gps": False},
        "Radar only": {"image": False, "radar": True, "lidar": False, "gps": False},
        "GPS only": {"image": False, "radar": False, "lidar": False, "gps": True},
        "Image+Radar": {"image": True, "radar": True, "lidar": False, "gps": False},
        "Image+Radar+LiDAR": {"image": True, "radar": True, "lidar": True, "gps": False},
        "All modalities": {"image": True, "radar": True, "lidar": True, "gps": True},
    }

    results = {}
    for name, mask in configurations.items():
        print(f"\nEvaluating: {name}")
        metrics = evaluate_full(model, loader, device, cfg, modality_mask=mask)
        results[name] = metrics
        print(f"  Top-1: {metrics['top_1_acc'] * 100:.1f}%  "
              f"Top-3: {metrics['top_3_acc'] * 100:.1f}%")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  Modality Ablation Results")
    print(f"{'=' * 80}")
    print(f"{'Configuration':<25} {'Top-1':>8} {'Top-2':>8} {'Top-3':>8} {'DBA(K=1,Δ=1)':>14}")
    print(f"{'-' * 80}")
    for name, m in results.items():
        print(f"{name:<25} {m['top_1_acc'] * 100:>7.1f}% {m['top_2_acc'] * 100:>7.1f}% "
              f"{m['top_3_acc'] * 100:>7.1f}% {m.get('dba_k1_d1', 0):>13.4f}")

    # Plot comparison
    plot_topk_comparison(results, k_values=[1, 2, 3], save_dir=cfg.train.log_dir)

    return results


def main():
    args = parse_args()
    cfg = get_config()
    device = get_device(cfg)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get model config from checkpoint
    ckpt_cfg = ckpt.get("config", {})
    H = ckpt_cfg.get("H", cfg.train.H_standard if args.mode == "standard" else cfg.train.H_fewshot)
    T = ckpt_cfg.get("T", cfg.train.T_standard if args.mode == "standard" else cfg.train.T_fewshot)
    backbone = ckpt_cfg.get("backbone", "gpt2")

    print(f"Mode: {args.mode} (H={H}, T={T})")
    print(f"Backbone: {backbone}")
    print(f"Device: {device}")

    # Create test dataloader
    _, test_loader = create_dataloaders(cfg, H=H, T=T)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Build model
    model = M2BeamLLM(
        feature_dim=cfg.model.feature_dim,
        num_beams=cfg.data.num_beams,
        llm_name=backbone,
        llm_hidden_dim=cfg.model.llm_hidden_dim,
        num_unfrozen_layers=cfg.model.num_unfrozen_layers,
        temperature=cfg.model.temperature,
        num_attention_heads=cfg.model.num_attention_heads,
        ffn_hidden_dim=cfg.model.ffn_hidden_dim,
        fusion_dropout=cfg.model.fusion_dropout,
        T=T,
        H=H,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    if args.ablation:
        # Modality ablation study
        run_ablation(model, test_loader, device, cfg)
    else:
        # Standard evaluation
        metrics = evaluate_full(model, test_loader, device, cfg)
        print_metrics(metrics, prefix=f"Test Results ({args.mode})")


if __name__ == "__main__":
    main()