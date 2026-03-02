"""
Comprehensive Evaluation Script for Robust M²BeamLLM.
Runs all experiments (E1-E4) and stress tests (S1-S3) from the paper.

Usage:
    python evaluate_robust.py --checkpoint best_robust_standard_none.pt
    python evaluate_robust.py --checkpoint best_robust_standard_none.pt --experiment E1
    python evaluate_robust.py --checkpoint best_robust_standard_none.pt --experiment all
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_config, get_device
from models.robust_m2beamllm import RobustM2BeamLLM, RobustM2BeamLLMLoss
from utils.dataset import create_dataloaders
from utils.metrics import compute_all_metrics, print_metrics
from utils.stress_test import (
    inject_asynchrony, apply_degradation,
    compute_complexity_analysis, print_complexity_table,
)
from utils.training_utils import (
    build_model,
    build_supervised_optimizer,
    freeze_encoder_backbones,
    set_encoder_eval_mode,
    evaluate,
    run_stress_test_s1,
    run_stress_test_s2,
    run_stress_test_s3,
    split_domain_shift_loaders,
    run_gradient_contamination_experiment,
    run_reliability_calibration,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Robust M²BeamLLM")
    p.add_argument("--checkpoint", required=True,
                   help="Model checkpoint filename or path")
    p.add_argument("--mode", choices=["standard", "fewshot"], default="standard")
    p.add_argument("--backbone", default="gpt2")
    p.add_argument("--experiment",
                   choices=["none", "E1", "E2", "E3", "E4", "S1", "S2", "S3", "all"],
                   default="all")
    p.add_argument("--e3_epochs", type=int, default=3,
                   help="Quick fine-tuning epochs when E3 baseline checkpoint is missing")
    return p.parse_args()


# ===========================================================================
# E2: Delay-Regime Specialization
# ===========================================================================

def run_e2_delay_regime(model: RobustM2BeamLLM,
                        criterion: RobustM2BeamLLMLoss,
                        test_loader, device, cfg) -> dict:
    """E2: Delay-Regime Specialization Effect."""
    print(f"\n{'='*60}")
    print(f"  E2: Delay-Regime Specialization Effect")
    print(f"{'='*60}")

    results = {}
    for delay in [0, 50, 100, 200]:
        delay_config = {m: float(delay) for m in ["image", "radar", "lidar", "gps"]}
        metrics = evaluate(model, criterion, test_loader, device, cfg,
                           staleness_ms=delay_config)
        results[f"delay_{delay}ms"] = metrics
        print(f"  Delay {delay:4d} ms | "
              f"Top-1: {metrics['top_1_acc']*100:.1f}% | "
              f"Worst-10%: {metrics.get('worst_10_acc', 0)*100:.1f}%")
    return results


# ===========================================================================
# E3: LLM Necessity
# ===========================================================================

def run_e3_llm_necessity(model: RobustM2BeamLLM,
                         criterion: RobustM2BeamLLMLoss,
                         train_loader, test_loader, device, cfg,
                         checkpoint_path: str, backbone: str,
                         quick_epochs: int = 3) -> dict:
    """E3: LLM Necessity Verification."""
    print(f"\n{'='*60}")
    print(f"  E3: LLM Necessity Verification")
    print(f"{'='*60}")

    def build_class_weights(loader, num_beams, H, T):
        ds = loader.dataset
        base_ds = ds.dataset if hasattr(ds, "dataset") else ds
        beams = getattr(base_ds, "beams", None)
        if beams is None:
            return None
        future = np.array(beams[:, H:H + T], dtype=np.int64).reshape(-1)
        counts = np.bincount(future, minlength=num_beams).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        freq = counts / counts.sum()
        w = 1.0 / np.sqrt(freq)
        w = np.clip(w, 0.0, np.percentile(w, 95))
        w = w / max(w.mean(), 1e-8)
        return torch.tensor(w, dtype=torch.float32, device=device)

    def quick_supervised_finetune(m, crit, loader, epochs: int):
        opt = build_supervised_optimizer(m, cfg)
        for ep in range(1, epochs + 1):
            m.train()
            set_encoder_eval_mode(m)
            align_scale = min(1.0, ep / max(1, cfg.train.align_warmup_epochs))
            total = 0.0
            n = 0
            for batch in loader:
                images = batch["images"].to(device)
                radars = batch["radars"].to(device)
                lidars = batch["lidars"].to(device)
                gps = batch["gps"].to(device)
                targets = batch["beam_future"].to(device)

                opt.zero_grad()
                preds, aux = m(images, radars, lidars, gps)
                loss, _ = crit(
                    preds, targets, aux,
                    align_loss_module=m.alignment_loss_fn,
                    align_scale=align_scale,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                total += loss.item() * targets.size(0)
                n += targets.size(0)
            print(f"    Quick finetune epoch {ep}/{epochs} | loss={total/max(n,1):.4f}")

    base_ds = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset
    H = getattr(base_ds, "H", cfg.train.H_standard)
    T = getattr(base_ds, "T", cfg.train.T_standard)
    in_loader, out_loader = split_domain_shift_loaders(
        test_loader, holdout_ratio=cfg.stress_test.domain_shift_holdout
    )

    # LLM metrics (already loaded model)
    llm_full = evaluate(model, criterion, test_loader, device, cfg)
    llm_in = evaluate(model, criterion, in_loader, device, cfg)
    llm_out = evaluate(model, criterion, out_loader, device, cfg)
    print(f"  LLM + PEFT | Full Top-1: {llm_full['top_1_acc']*100:.1f}% | "
          f"Domain drop: {(llm_in['top_1_acc']-llm_out['top_1_acc'])*100:.1f}%")

    # Vanilla Transformer baseline
    vanilla = build_model(cfg, "vanilla_transformer", device, H, T)
    freeze_encoder_backbones(vanilla)
    vanilla_ckpt = checkpoint_path.replace("_none.pt", "_A5.pt")
    if not os.path.isfile(vanilla_ckpt):
        vanilla_ckpt = os.path.join(cfg.train.checkpoint_dir, f"best_robust_standard_A5.pt")

    vanilla_criterion = RobustM2BeamLLMLoss(
        feature_dim=cfg.model.feature_dim,
        lambda_align=cfg.train.lambda_align,
        class_weights=build_class_weights(train_loader, cfg.data.num_beams, H, T),
        focal_gamma=cfg.train.focal_gamma,
        lambda_moe_balance=cfg.train.lambda_moe_balance,
    ).to(device)

    if os.path.isfile(vanilla_ckpt):
        ckpt = torch.load(vanilla_ckpt, map_location=device, weights_only=False)
        vanilla.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded vanilla baseline checkpoint: {vanilla_ckpt}")
    else:
        print(f"  Vanilla baseline checkpoint not found, running quick finetune ({quick_epochs} epochs)")
        quick_supervised_finetune(vanilla, vanilla_criterion, train_loader, quick_epochs)

    vanilla_full = evaluate(vanilla, vanilla_criterion, test_loader, device, cfg)
    vanilla_in = evaluate(vanilla, vanilla_criterion, in_loader, device, cfg)
    vanilla_out = evaluate(vanilla, vanilla_criterion, out_loader, device, cfg)
    print(f"  Vanilla Transformer | Full Top-1: {vanilla_full['top_1_acc']*100:.1f}% | "
          f"Domain drop: {(vanilla_in['top_1_acc']-vanilla_out['top_1_acc'])*100:.1f}%")

    return {
        "llm_peft_full": llm_full,
        "llm_peft_in_domain": llm_in,
        "llm_peft_out_domain": llm_out,
        "llm_peft_domain_drop": llm_in["top_1_acc"] - llm_out["top_1_acc"],
        "vanilla_full": vanilla_full,
        "vanilla_in_domain": vanilla_in,
        "vanilla_out_domain": vanilla_out,
        "vanilla_domain_drop": vanilla_in["top_1_acc"] - vanilla_out["top_1_acc"],
    }


# ===========================================================================
# E4: Tail Robustness
# ===========================================================================

def run_e4_tail_robustness(model: RobustM2BeamLLM,
                           criterion: RobustM2BeamLLMLoss,
                           test_loader, device, cfg) -> dict:
    """E4: Tail Robustness Metrics under various stress conditions."""
    print(f"\n{'='*60}")
    print(f"  E4: Tail Robustness Metrics")
    print(f"{'='*60}")

    results = {}

    # Clean
    metrics = evaluate(model, criterion, test_loader, device, cfg)
    results["clean"] = metrics
    print(f"\n  Clean:")
    print(f"    Top-1: {metrics['top_1_acc']*100:.1f}%")
    print(f"    CVaR-10: {metrics.get('cvar_10', 0):.4f}")
    print(f"    Worst-10%: {metrics.get('worst_10_acc', 0)*100:.1f}%")

    # Under degradation
    for alpha in [0.3, 0.6, 0.9]:
        metrics = evaluate(
            model, criterion, test_loader, device, cfg,
            degradation={
                "severity": alpha,
                "stress_config": cfg.stress_test,
            },
        )
        results[f"degraded_{alpha}"] = metrics
        print(f"\n  Degradation α={alpha}:")
        print(f"    Top-1: {metrics['top_1_acc']*100:.1f}%")
        print(f"    CVaR-10: {metrics.get('cvar_10', 0):.4f}")
        print(f"    Worst-10%: {metrics.get('worst_10_acc', 0)*100:.1f}%")

    # Percentile curve
    print(f"\n  Percentile Degradation Curve (clean):")
    for k, v in sorted(results["clean"].items()):
        if k.startswith("percentile"):
            print(f"    {k}: {v*100:.1f}%")

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    cfg = get_config()
    device = get_device(cfg)

    H = cfg.train.H_standard if args.mode == "standard" else cfg.train.H_fewshot
    T = cfg.train.T_standard if args.mode == "standard" else cfg.train.T_fewshot

    print(f"  Mode: {args.mode} (H={H}, T={T})")
    print(f"  Checkpoint: {args.checkpoint}")

    # Data
    train_loader, test_loader = create_dataloaders(cfg, H=H, T=T)

    # Model
    model = build_model(cfg, args.backbone, device, H, T)
    freeze_encoder_backbones(model)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(cfg.train.checkpoint_dir, ckpt_path)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Loaded checkpoint")
    else:
        print(f"⚠ Checkpoint not found: {ckpt_path}")

    criterion = RobustM2BeamLLMLoss(
        feature_dim=cfg.model.feature_dim,
        lambda_align=cfg.train.lambda_align,
        focal_gamma=cfg.train.focal_gamma,
        lambda_moe_balance=cfg.train.lambda_moe_balance,
    ).to(device)

    # C4: Complexity
    complexity = compute_complexity_analysis(model)
    print_complexity_table(complexity)

    # Standard metrics
    metrics = evaluate(model, criterion, test_loader, device, cfg)
    print_metrics(metrics, "Standard Evaluation")

    # Reliability calibration
    run_reliability_calibration(model, test_loader, device)

    # Run requested experiments
    all_results = {"standard": metrics}
    exp = args.experiment

    if exp in ("E1", "all"):
        all_results["E1"] = run_gradient_contamination_experiment(
            model, train_loader, criterion, device
        )
    if exp in ("E2", "all"):
        all_results["E2"] = run_e2_delay_regime(
            model, criterion, test_loader, device, cfg
        )
    if exp in ("E3", "all"):
        all_results["E3"] = run_e3_llm_necessity(
            model, criterion, train_loader, test_loader, device, cfg,
            checkpoint_path=ckpt_path, backbone=args.backbone,
            quick_epochs=args.e3_epochs,
        )
    if exp in ("E4", "all"):
        all_results["E4"] = run_e4_tail_robustness(
            model, criterion, test_loader, device, cfg
        )
    if exp in ("S1", "all"):
        all_results["S1"] = run_stress_test_s1(
            model, criterion, test_loader, device, cfg
        )
    if exp in ("S2", "all"):
        all_results["S2"] = run_stress_test_s2(
            model, criterion, test_loader, device, cfg
        )
    if exp in ("S3", "all"):
        all_results["S3"] = run_stress_test_s3(
            model, criterion, test_loader, device, cfg
        )

    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    results_path = os.path.join(cfg.train.log_dir, f"eval_results_{args.mode}.json")
    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n✓ Results saved: {results_path}")


if __name__ == "__main__":
    main()
