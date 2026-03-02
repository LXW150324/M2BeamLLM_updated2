"""
Main Training Script for Robust M²BeamLLM.
Implements two-stage training (C3) and full evaluation suite (C5).

Stage 1: Self-supervised pretraining (no beam labels)
Stage 2: Supervised beam prediction with PEFT + reliability-weighted alignment

Usage:
    python train_robust.py --stage both
    python train_robust.py --stage ssl
    python train_robust.py --stage supervised
    python train_robust.py --ablation A3
    python train_robust.py --stress_test S1
    python train_robust.py --run_all_experiments
    python train_robust.py --complexity_only
"""

import os
import sys
import time
import argparse
import json
import random
import re
import glob
from typing import Dict, Optional
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_config, get_device, Config
from models.robust_m2beamllm import RobustM2BeamLLM, RobustM2BeamLLMLoss
from utils.dataset import create_dataloaders
from utils.metrics import print_metrics
from utils.stress_test import (
    compute_complexity_analysis, print_complexity_table,
    benchmark_inference_latency, estimate_flops_per_sample, print_complexity_benchmark,
)
from utils.visualization import (
    plot_ablation_results,
    plot_complexity_breakdown,
    plot_c4_latency_benchmark,
    plot_ssl_pretraining_curves,
    plot_training_curves,
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
    run_e2_delay_regime_specialization,
    run_gradient_contamination_experiment,
    run_reliability_calibration,
    run_reliability_calibration_paper,
    run_reliability_monotonicity_s2,
    subsample_loader_by_fraction,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Robust M²BeamLLM")
    p.add_argument("--stage", choices=["ssl", "supervised", "both"], default="both")
    p.add_argument("--mode", choices=["standard", "fewshot"], default="standard")
    p.add_argument("--dataset", choices=["deepsense", "deepmimo", "viwi"], default=None,
                   help="Dataset protocol/preset for preprocessed window .npy files")
    p.add_argument("--data_root", type=str, default=None,
                   help="Override preprocessed dataset root (contains train_*.npy / test_*.npy)")
    p.add_argument("--backbone", default="gpt2")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--label_ratio", type=float, default=None,
                   help="Supervised label ratio for E3 (percent 1-100 or fraction 0-1)")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--encoder_ckpt", type=str, default=None)
    p.add_argument("--ssl_ckpt", type=str, default=None)
    p.add_argument("--skip_encoder_ckpt", action="store_true",
                   help="Skip loading encoder pretrained checkpoint")
    p.add_argument("--skip_ssl_ckpt", action="store_true",
                   help="Skip loading SSL pretrained checkpoint in supervised stage")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ablation", choices=["none", "A1", "A2", "A3", "A4", "A5"],
                   default="none")
    p.add_argument("--stress_test", choices=["none", "S1", "S2", "S3", "all"],
                   default="none")
    p.add_argument("--run_all_experiments", action="store_true")
    p.add_argument("--run_e2", action="store_true",
                   help="Run E2 delay-regime specialization comparison")
    p.add_argument("--e2_a1_ckpt", type=str, default=None,
                   help="Path to trained A1 checkpoint for E2 comparison")
    p.add_argument("--run_e4_paper", action="store_true",
                   help="Run paper-definition E4 reliability calibration")
    p.add_argument("--run_e4_s2_monotonicity", action="store_true",
                   help="Run paper-definition E4 monotonicity check under S2 severities")
    p.add_argument("--complexity_only", action="store_true")
    p.add_argument("--complexity_benchmark", action="store_true",
                   help="Also benchmark FLOPs/latency for C4")
    p.add_argument("--latency_warmup_iters", type=int, default=None)
    p.add_argument("--latency_iters", type=int, default=None)
    p.add_argument("--s1_modality_specific", action="store_true",
                   help="Run modality-specific delays in S1")
    return p.parse_args()


def set_global_seed(seed: int):
    """Best-effort reproducibility."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_state_dict_compatible(module: torch.nn.Module,
                               state_dict: Dict[str, torch.Tensor],
                               module_name: str = "module"):
    """Load checkpoint state_dict with shape-mismatch tolerance."""
    target_state = module.state_dict()
    filtered = {}
    unexpected = []
    mismatched = []

    for key, value in state_dict.items():
        if key not in target_state:
            unexpected.append(key)
            continue
        tgt = target_state[key]
        if tuple(tgt.shape) != tuple(value.shape):
            mismatched.append((key, tuple(value.shape), tuple(tgt.shape)))
            continue
        filtered[key] = value

    result = module.load_state_dict(filtered, strict=False)

    print(f"  {module_name}: loaded {len(filtered)}/{len(state_dict)} compatible tensors")
    if result.missing_keys:
        preview = ", ".join(result.missing_keys[:5])
        extra = " ..." if len(result.missing_keys) > 5 else ""
        print(f"  {module_name}: missing {len(result.missing_keys)} keys [{preview}{extra}]")
    if unexpected:
        preview = ", ".join(unexpected[:5])
        extra = " ..." if len(unexpected) > 5 else ""
        print(f"  {module_name}: unexpected {len(unexpected)} keys [{preview}{extra}]")
    if mismatched:
        print(f"  {module_name}: skipped {len(mismatched)} shape-mismatched keys")
        for key, src_shape, dst_shape in mismatched[:8]:
            print(f"    - {key}: ckpt{src_shape} != model{dst_shape}")

    return result, unexpected, mismatched


def set_stage2_frozen_llm_eval_mode(model: RobustM2BeamLLM, head_only: bool):
    """Keep frozen LLM submodules in eval mode while trainable ones stay in train mode."""
    if not hasattr(model, "llm_backbone"):
        return

    llm = getattr(model.llm_backbone, "llm", None)
    if llm is not None:
        llm.eval()
        blocks = getattr(llm, "h", None)
        if blocks is not None:
            for block in blocks:
                if any(p.requires_grad for p in block.parameters()):
                    block.train()
                else:
                    block.eval()
        ln_f = getattr(llm, "ln_f", None)
        if ln_f is not None:
            if any(p.requires_grad for p in ln_f.parameters()):
                ln_f.train()
            else:
                ln_f.eval()

    moe_layers = getattr(model.llm_backbone, "moe_lora_layers", None)
    if moe_layers is not None:
        moe_layers.eval()
        for layer in moe_layers:
            if any(p.requires_grad for p in layer.parameters()):
                layer.train()
            else:
                layer.eval()


# ===========================================================================
# Stage 1: SSL Pretraining
# ===========================================================================

def train_ssl_epoch(model: RobustM2BeamLLM, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cfg: Config) -> float:
    """Train one SSL epoch."""
    model.train()
    set_encoder_eval_mode(model)
    total_loss = 0.0
    n_samples = 0

    for batch in tqdm(loader, desc=f"SSL Epoch {epoch}", leave=False):
        images = batch["images"].to(device)
        radars = batch["radars"].to(device)
        lidars = batch["lidars"].to(device)
        gps = batch["gps"].to(device)

        raw_features = model.encoder(images, radars, lidars, gps)

        if cfg.ssl.use_explicit_alignment:
            aligned, _ = model.async_alignment(raw_features)
        else:
            aligned = raw_features

        if cfg.ssl.use_reliability_gating:
            _, _, gated = model.reliability_estimator(aligned)
        else:
            gated = aligned

        fused_tokens = model.fusion(gated)
        B, _, M = fused_tokens.shape
        H_hist = images.size(1)
        fused = fused_tokens.reshape(B, H_hist, model.fusion.num_tokens, M).mean(dim=2)

        ssl_loss, _ = model.ssl_objectives(
            aligned, fused,
            lambda_mr=cfg.ssl.lambda_mr,
            lambda_cm=cfg.ssl.lambda_cm,
            lambda_tf=cfg.ssl.lambda_tf,
        )

        optimizer.zero_grad()
        ssl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += ssl_loss.item() * images.size(0)
        n_samples += images.size(0)

    return total_loss / max(n_samples, 1)


def run_ssl_pretraining(model: RobustM2BeamLLM, train_loader: DataLoader,
                        cfg: Config, device: torch.device) -> str:
    """Stage 1: SSL pretraining loop."""
    print(f"\n{'='*60}")
    print(f"  Stage 1: Self-Supervised Pretraining")
    print(f"{'='*60}")

    ssl_train_keys = ["fusion", "ssl_objectives", "encoder.radar", "encoder.gps"]
    if cfg.ssl.use_explicit_alignment:
        ssl_train_keys.append("async_alignment")
    if cfg.ssl.use_reliability_gating:
        ssl_train_keys.append("reliability")

    ssl_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and any(k in name for k in ssl_train_keys)
    ]

    optimizer = torch.optim.Adam(ssl_params, lr=cfg.ssl.ssl_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.ssl.ssl_epochs
    )

    ssl_losses = []
    ssl_lrs = []
    for epoch in range(1, cfg.ssl.ssl_epochs + 1):
        loss = train_ssl_epoch(model, train_loader, optimizer, device, epoch, cfg)
        scheduler.step()
        ssl_losses.append(float(loss))
        ssl_lrs.append(float(optimizer.param_groups[0]["lr"]))
        print(f"  SSL Epoch {epoch}/{cfg.ssl.ssl_epochs} | "
              f"Loss: {loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    plot_ssl_pretraining_curves(ssl_losses, ssl_lrs, save_dir=cfg.train.log_dir)

    ssl_path = os.path.join(cfg.train.checkpoint_dir, "ssl_pretrained.pt")
    torch.save({"model_state_dict": model.state_dict()}, ssl_path)
    print(f"  SSL checkpoint saved: {ssl_path}")
    return ssl_path


# ===========================================================================
# Stage 2: Supervised Training
# ===========================================================================

def train_one_epoch(model: RobustM2BeamLLM,
                    criterion: RobustM2BeamLLMLoss,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    cfg: Config,
                    epoch: int,
                    ablation: str = "none",
                    head_only_warmup: bool = False,
                    align_warmup_epochs: int = 5,
                    moe_warmup_epochs: int = 0,
                    warmup_delay_epochs: int = 0):
    """Train one supervised epoch."""
    model.train()
    set_encoder_eval_mode(model)
    set_stage2_frozen_llm_eval_mode(model, head_only=head_only_warmup)

    total_loss = 0.0
    loss_components = {
        "beam": 0.0,
        "alignment": 0.0,
        "moe_balance": 0.0,
    }
    total_correct = 0
    total_samples = 0

    use_reliability_align = (ablation != "A3")

    def _ramp(epoch_idx: int, warmup_epochs: int, delay_epochs: int = 0) -> float:
        if warmup_epochs <= 0:
            return 1.0
        if epoch_idx <= delay_epochs:
            return 0.0
        return min(1.0, max(0.0, (epoch_idx - delay_epochs - 1) / float(warmup_epochs)))

    align_scale = _ramp(epoch, align_warmup_epochs, delay_epochs=warmup_delay_epochs)
    moe_scale = _ramp(epoch, moe_warmup_epochs, delay_epochs=warmup_delay_epochs)

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False):
        images = batch["images"].to(device)
        radars = batch["radars"].to(device)
        lidars = batch["lidars"].to(device)
        gps = batch["gps"].to(device)
        targets = batch["beam_future"].to(device)
        B = targets.size(0)

        optimizer.zero_grad()
        predictions, aux = model(images, radars, lidars, gps)

        loss, loss_dict = criterion(
            predictions, targets, aux,
            use_reliability_align=use_reliability_align,
            align_loss_module=model.alignment_loss_fn,
            align_scale=align_scale,
            moe_scale=moe_scale,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_dict["total"] * B
        for k in loss_components:
            loss_components[k] += loss_dict.get(k, 0.0) * B

        pred_indices = predictions.argmax(dim=-1)
        total_correct += (pred_indices == targets).float().mean(dim=1).sum().item()
        total_samples += B

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    avg_comp = {k: v / max(total_samples, 1) for k, v in loss_components.items()}
    avg_comp["align_coeff"] = align_scale * criterion.lambda_align
    avg_comp["moe_balance_coeff"] = moe_scale * criterion.lambda_moe_balance
    avg_comp["beam_coeff"] = criterion.lambda_beam
    avg_comp["label_smoothing"] = getattr(criterion, "label_smoothing", 0.0)
    return avg_loss, avg_acc, avg_comp


def apply_cli_ablation(model: RobustM2BeamLLM, ablation: str):
    """Apply single ablation to the provided model."""
    if ablation == "A1":
        for mod_refine in model.async_alignment.refinement.values():
            for adapter in mod_refine.regime_adapters:
                for p in adapter.parameters():
                    p.requires_grad = False
                    p.zero_()
            for bias in mod_refine.regime_bias:
                for p in bias.parameters():
                    p.requires_grad = False
                    p.zero_()
        print("  Applied A1: removed delay-regime conditioning.")
    elif ablation == "A2":
        model.async_alignment.set_repair_mode("zero_fill")
        model.async_alignment.repair.eta_raw.requires_grad = False
        print("  Applied A2: removed short-gap repair (zero-fill).")
    elif ablation == "A4":
        if hasattr(model.llm_backbone, "moe_lora_layers"):
            for moe_layer in model.llm_backbone.moe_lora_layers:
                for key in ["q", "v"]:
                    moe_layer[key].set_shared_expert(True, expert_idx=0)
                    for p in moe_layer[key].router.parameters():
                        p.requires_grad = False
            print("  Applied A4: removed MoE routing (single shared LoRA expert).")


def apply_stage2_curriculum(model: RobustM2BeamLLM, cfg: Config, epoch: int) -> dict:
    """
    Progressive Stage-2 curriculum:
    head-only -> MoE-only -> top-LLM+MoE -> full-model.
    """
    cache_name = "_stage2_curriculum_reqgrad_cache"
    meta_name = "_stage2_curriculum_meta"
    phase_name = "_stage2_curriculum_phase"
    if not hasattr(model, cache_name):
        cache = {}
        llm_layer_ids = set()
        for name, p in model.named_parameters():
            if name.startswith("llm_backbone.llm.") or "llm_backbone.moe_lora_layers" in name:
                cache[name] = bool(p.requires_grad)
                m = re.search(r"llm_backbone\.llm\.h\.(\d+)\.", name)
                if m and p.requires_grad:
                    llm_layer_ids.add(int(m.group(1)))
        setattr(model, cache_name, cache)
        top_layer = max(llm_layer_ids) if llm_layer_ids else None
        setattr(model, meta_name, {"top_llm_layer": top_layer})
        setattr(model, phase_name, None)

    head_only_epochs = max(int(getattr(cfg.train, "llm_warmup_epochs", 0)), 0)
    moe_only_epochs = max(int(getattr(cfg.train, "moe_only_warmup_epochs", 0)), 0)
    llm_top1_epochs = max(int(getattr(cfg.train, "llm_top1_warmup_epochs", 0)), 0)
    e_head_end = head_only_epochs
    e_moe_end = e_head_end + moe_only_epochs
    e_top1_end = e_moe_end + llm_top1_epochs

    if epoch <= e_head_end:
        phase_type = "head_only"
    elif epoch <= e_moe_end:
        phase_type = "moe_only"
    elif epoch <= e_top1_end:
        phase_type = "top_llm_only"
    else:
        phase_type = "full"

    head_only = (phase_type == "head_only")
    cache = getattr(model, cache_name)
    meta = getattr(model, meta_name)
    top_llm_layer = meta.get("top_llm_layer")
    for name, p in model.named_parameters():
        if name not in cache:
            continue
        orig_trainable = cache[name]
        if not orig_trainable:
            p.requires_grad = False
            continue

        is_moe = "llm_backbone.moe_lora_layers" in name
        is_ln_f = name.startswith("llm_backbone.llm.ln_f.")
        m_layer = re.search(r"llm_backbone\.llm\.h\.(\d+)\.", name)
        llm_layer_idx = int(m_layer.group(1)) if m_layer else None

        if phase_type == "head_only":
            p.requires_grad = False
        elif phase_type == "moe_only":
            p.requires_grad = is_moe
        elif phase_type == "top_llm_only":
            p.requires_grad = is_moe or is_ln_f or (
                llm_layer_idx is not None and top_llm_layer is not None and llm_layer_idx == top_llm_layer
            )
        else:
            p.requires_grad = True

    if phase_type == "head_only":
        phase = f"head-only (epoch {epoch}/{head_only_epochs})"
    elif phase_type == "moe_only":
        phase = f"moe-only (epoch {epoch - e_head_end}/{moe_only_epochs})"
    elif phase_type == "top_llm_only":
        phase = f"top-layer+MoE (epoch {epoch - e_moe_end}/{llm_top1_epochs})"
    else:
        phase = "full-model"

    changed = phase != getattr(model, phase_name)
    setattr(model, phase_name, phase)
    full_model_start_epoch = e_top1_end + 1
    aux_ramp_delay_epochs = max(full_model_start_epoch - 1, 0)
    return {
        "changed": changed,
        "phase": phase,
        "phase_type": phase_type,
        "head_only": head_only,
        "aux_ramp_delay_epochs": aux_ramp_delay_epochs,
    }


# ===========================================================================
# Ablation Study (A1-A5)
# ===========================================================================

def run_ablation_study(cfg: Config, device: torch.device, backbone: str,
                       train_loader: DataLoader, test_loader: DataLoader,
                       H: int, T: int):
    """Run systematic ablation study A1-A5."""
    print(f"\n{'='*60}")
    print(f"  Systematic Ablation Study (A1-A5)")
    print(f"{'='*60}")

    ablation_results = {}

    def make_model(backbone_name: str = backbone):
        m = build_model(cfg, backbone_name, device, H, T)
        freeze_encoder_backbones(m)
        return m

    criterion = RobustM2BeamLLMLoss(
        feature_dim=cfg.model.feature_dim,
        lambda_align=cfg.train.lambda_align,
        lambda_beam=cfg.train.lambda_beam,
        label_smoothing=cfg.train.label_smoothing,
        lambda_moe_balance=cfg.train.lambda_moe_balance,
    ).to(device)

    max_epochs = min(cfg.train.num_epochs, 10)

    # --- Baseline ---
    print(f"\n  --- Baseline (Full Model) ---")
    model = make_model()
    optimizer = build_supervised_optimizer(model, cfg)
    for epoch in range(1, max_epochs + 1):
        train_one_epoch(
            model, criterion, train_loader, optimizer, device, cfg, epoch,
            align_warmup_epochs=cfg.train.align_warmup_epochs,
        )
    metrics = evaluate(model, criterion, test_loader, device, cfg)
    ablation_results["Full"] = metrics
    print_metrics(metrics, "Full Model")

    # --- A1: Remove delay-regime conditioning ---
    print(f"\n  --- A1: Remove Delay-Regime Conditioning ---")
    model_a1 = make_model()
    for mod_refine in model_a1.async_alignment.refinement.values():
        for adapter in mod_refine.regime_adapters:
            for p in adapter.parameters():
                p.requires_grad = False
                p.zero_()
        for bias in mod_refine.regime_bias:
            for p in bias.parameters():
                p.requires_grad = False
                p.zero_()
    optimizer = build_supervised_optimizer(model_a1, cfg)
    for epoch in range(1, max_epochs + 1):
        train_one_epoch(
            model_a1, criterion, train_loader, optimizer, device, cfg, epoch,
            align_warmup_epochs=cfg.train.align_warmup_epochs,
        )
    metrics = evaluate(model_a1, criterion, test_loader, device, cfg)
    ablation_results["A1_no_regime"] = metrics
    print_metrics(metrics, "A1: No Delay-Regime")

    # --- A2: Remove short-gap repair ---
    print(f"\n  --- A2: Remove Short-Gap Repair ---")
    model_a2 = make_model()
    model_a2.async_alignment.set_repair_mode("zero_fill")
    model_a2.async_alignment.repair.eta_raw.requires_grad = False
    optimizer = build_supervised_optimizer(model_a2, cfg)
    for epoch in range(1, max_epochs + 1):
        train_one_epoch(
            model_a2, criterion, train_loader, optimizer, device, cfg, epoch,
            align_warmup_epochs=cfg.train.align_warmup_epochs,
        )
    metrics = evaluate(model_a2, criterion, test_loader, device, cfg)
    ablation_results["A2_no_repair"] = metrics
    print_metrics(metrics, "A2: No Short-Gap Repair")

    # --- A3: Remove reliability-weighted alignment ---
    print(f"\n  --- A3: Remove Reliability-Weighted Alignment ---")
    model_a3 = make_model()
    optimizer = build_supervised_optimizer(model_a3, cfg)
    for epoch in range(1, max_epochs + 1):
        train_one_epoch(
            model_a3, criterion, train_loader, optimizer, device, cfg, epoch,
            ablation="A3", align_warmup_epochs=cfg.train.align_warmup_epochs,
        )
    metrics = evaluate(model_a3, criterion, test_loader, device, cfg, ablation="A3")
    ablation_results["A3_no_rw_align"] = metrics
    print_metrics(metrics, "A3: No Reliability-Weighted Alignment")

    # --- A4: Remove MoE routing ---
    print(f"\n  --- A4: Remove MoE Routing ---")
    model_a4 = make_model()
    if hasattr(model_a4.llm_backbone, "moe_lora_layers"):
        for moe_layer in model_a4.llm_backbone.moe_lora_layers:
            for key in ["q", "v"]:
                moe_layer[key].set_shared_expert(True, expert_idx=0)
                for p in moe_layer[key].router.parameters():
                    p.requires_grad = False
    optimizer = build_supervised_optimizer(model_a4, cfg)
    for epoch in range(1, max_epochs + 1):
        train_one_epoch(
            model_a4, criterion, train_loader, optimizer, device, cfg, epoch,
            align_warmup_epochs=cfg.train.align_warmup_epochs,
        )
    metrics = evaluate(model_a4, criterion, test_loader, device, cfg)
    ablation_results["A4_no_moe"] = metrics
    print_metrics(metrics, "A4: No MoE Routing")

    # --- A5: Replace LLM with vanilla Transformer ---
    print(f"\n  --- A5: Replace LLM with Vanilla Transformer ---")
    model_a5 = make_model("vanilla_transformer")
    optimizer = build_supervised_optimizer(model_a5, cfg)
    for epoch in range(1, max_epochs + 1):
        train_one_epoch(
            model_a5, criterion, train_loader, optimizer, device, cfg, epoch,
            align_warmup_epochs=cfg.train.align_warmup_epochs,
        )
    metrics = evaluate(model_a5, criterion, test_loader, device, cfg)
    ablation_results["A5_vanilla_transformer"] = metrics
    print_metrics(metrics, "A5: Vanilla Transformer")

    return ablation_results


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    cfg = get_config()

    if args.dataset or args.data_root:
        cfg.apply_dataset_preset(args.dataset or cfg.data.dataset_name, data_root=args.data_root)

    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
    if args.seed is not None:
        cfg.train.seed = int(args.seed)
    if args.latency_warmup_iters is not None:
        cfg.train.complexity_latency_warmup_iters = int(max(0, args.latency_warmup_iters))
    if args.latency_iters is not None:
        cfg.train.complexity_latency_iters = int(max(1, args.latency_iters))
    if args.s1_modality_specific:
        cfg.stress_test.s1_include_modality_specific_delays = True

    set_global_seed(cfg.train.seed)
    device = get_device(cfg)

    H = cfg.train.H_standard if args.mode == "standard" else cfg.train.H_fewshot
    T = cfg.train.T_standard if args.mode == "standard" else cfg.train.T_fewshot
    effective_backbone = "vanilla_transformer" if args.ablation == "A5" else args.backbone

    print(f"\n{'='*60}")
    print(f"  Robust M²BeamLLM Training")
    print(f"  Stage: {args.stage} | Mode: {args.mode} (H={H}, T={T})")
    print(f"  Dataset: {cfg.data.dataset_name} | Data Root: {cfg.data.data_root}")
    print(f"  Backbone: {effective_backbone} | Device: {device}")
    print(f"  Ablation: {args.ablation}")
    print(f"  Seed: {cfg.train.seed}")
    if effective_backbone == "vanilla_transformer":
        print(
            f"  A5 Vanilla: L={cfg.model.vanilla_num_layers}, "
            f"H={cfg.model.vanilla_num_heads}, d={cfg.model.vanilla_hidden_dim}"
        )
    print(f"{'='*60}\n")

    # Data
    train_loader_full, test_loader = create_dataloaders(cfg, H=H, T=T)
    train_loader = train_loader_full
    label_ratio_frac = 1.0
    if args.label_ratio is not None:
        ratio_val = float(args.label_ratio)
        label_ratio_frac = ratio_val / 100.0 if ratio_val > 1.0 else ratio_val
        label_ratio_frac = max(0.0, min(1.0, label_ratio_frac))
        if label_ratio_frac < 1.0:
            train_loader = subsample_loader_by_fraction(
                train_loader_full, label_ratio_frac, seed=cfg.train.seed, shuffle=True,
            )
            print(f"  Supervised label ratio: {label_ratio_frac*100:.1f}%")
    label_ratio_pct = int(round(label_ratio_frac * 100))

    def _jsonify(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        return str(obj)

    run_tag = f"{args.mode}_{args.ablation}_seed{cfg.train.seed}_lrat{label_ratio_pct}"

    def _save_log_json(stem: str, payload: dict):
        path = os.path.join(cfg.train.log_dir, f"{stem}_{run_tag}.json")
        with open(path, "w") as f:
            json.dump(_jsonify(payload), f, indent=2)
        print(f"  Saved {stem}: {path}")
        return path

    # Model
    model = build_model(cfg, effective_backbone, device, H, T)

    # Encoder checkpoint
    encoder_ckpt = args.encoder_ckpt or os.path.join(
        cfg.train.checkpoint_dir, "encoder_pretrained.pt"
    )
    if args.skip_encoder_ckpt:
        print("  Skipping encoder checkpoint loading.")
    elif os.path.isfile(encoder_ckpt):
        print(f"  Loading encoders from {encoder_ckpt}")
        ckpt = torch.load(encoder_ckpt, map_location=device, weights_only=False)
        raw_state = ckpt["model_state_dict"]
        encoder_state = {}
        has_prefixed_keys = any(k.startswith("encoder.") for k in raw_state.keys())
        if has_prefixed_keys:
            for k, v in raw_state.items():
                if k.startswith("encoder."):
                    encoder_state[k[len("encoder."):]] = v
        else:
            encoder_state = dict(raw_state)
        load_state_dict_compatible(model.encoder, encoder_state, module_name="encoder")

    freeze_encoder_backbones(model)

    # C4: Complexity
    if args.complexity_only:
        complexity = compute_complexity_analysis(model)
        print_complexity_table(complexity)
        plot_complexity_breakdown(complexity, save_dir=cfg.train.log_dir)
        if args.complexity_benchmark:
            bench = benchmark_inference_latency(
                model, device=device, H=H, T=T,
                warmup_iters=cfg.train.complexity_latency_warmup_iters,
                iters=cfg.train.complexity_latency_iters,
            )
            bench["flops_per_sample"] = estimate_flops_per_sample(model, device=device, H=H, T=T)
            print_complexity_benchmark(bench)
            plot_c4_latency_benchmark(bench, save_dir=cfg.train.log_dir)
        return

    complexity = compute_complexity_analysis(model)
    print_complexity_table(complexity)
    plot_complexity_breakdown(complexity, save_dir=cfg.train.log_dir)

    # Loss — clean CE + alignment + MoE balance only
    criterion = RobustM2BeamLLMLoss(
        feature_dim=cfg.model.feature_dim,
        lambda_align=cfg.train.lambda_align,
        lambda_beam=cfg.train.lambda_beam,
        label_smoothing=cfg.train.label_smoothing,
        lambda_moe_balance=cfg.train.lambda_moe_balance,
    ).to(device)

    # ========== Stage 1: SSL ==========
    if args.stage in ("ssl", "both"):
        run_ssl_pretraining(model, train_loader_full, cfg, device)

    ssl_ckpt = args.ssl_ckpt or os.path.join(cfg.train.checkpoint_dir, "ssl_pretrained.pt")
    if args.stage == "supervised":
        if args.skip_ssl_ckpt:
            print("  Skipping SSL checkpoint loading.")
        elif os.path.isfile(ssl_ckpt):
            print(f"  Loading SSL checkpoint: {ssl_ckpt}")
            ckpt = torch.load(ssl_ckpt, map_location=device, weights_only=False)
            load_state_dict_compatible(model, ckpt["model_state_dict"], module_name="ssl_init")
        else:
            print("  SSL checkpoint not found, starting without SSL initialization.")

    # ========== Stage 2: Supervised ==========
    if args.stage in ("supervised", "both"):
        print(f"\n{'='*60}")
        print(f"  Stage 2: Supervised Beam Prediction")
        print(f"{'='*60}")

        if args.ablation in ("A1", "A2", "A4"):
            apply_cli_ablation(model, args.ablation)

        optimizer = build_supervised_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.lr_decay_step,
            gamma=cfg.train.lr_decay_factor,
        )

        best_top1 = 0.0
        patience_counter = 0
        patience = 15
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(1, cfg.train.num_epochs + 1):
            t0 = time.time()
            curriculum = apply_stage2_curriculum(model, cfg, epoch)
            if curriculum["changed"]:
                print(f"  Stage2 curriculum: {curriculum['phase']}")

            train_loss, train_acc, loss_comp = train_one_epoch(
                model, criterion, train_loader, optimizer, device, cfg, epoch,
                ablation=args.ablation,
                head_only_warmup=curriculum["head_only"],
                align_warmup_epochs=cfg.train.align_warmup_epochs,
                moe_warmup_epochs=cfg.train.moe_balance_warmup_epochs,
                warmup_delay_epochs=curriculum.get(
                    "aux_ramp_delay_epochs", cfg.train.llm_warmup_epochs
                ),
            )
            val_metrics = evaluate(
                model, criterion, test_loader, device, cfg,
                ablation=args.ablation,
            )
            scheduler.step()

            val_top1 = val_metrics["top_1_acc"]
            elapsed = time.time() - t0

            print(f"Epoch {epoch:3d}/{cfg.train.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {100*train_acc:.1f}% | "
                  f"Val Top-1: {100*val_top1:.1f}% | "
                  f"L_beam: {loss_comp['beam']:.4f} "
                  f"L_align: {loss_comp['alignment']:.4f} "
                  f"L_moe: {loss_comp['moe_balance']:.4f} "
                  f"(la={loss_comp['align_coeff']:.3f}, lm={loss_comp['moe_balance_coeff']:.3f}) | "
                  f"{elapsed:.1f}s")

            train_losses.append(train_loss)
            val_losses.append(val_metrics["loss"])
            train_accs.append(train_acc * 100)
            val_accs.append(val_top1 * 100)

            if val_top1 > best_top1:
                best_top1 = val_top1
                patience_counter = 0
                save_path = os.path.join(
                    cfg.train.checkpoint_dir,
                    f"best_robust_{args.mode}_{args.ablation}.pt",
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                }, save_path)
                print(f"  -> Best model saved (Top-1: {100*val_top1:.1f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Load best
        best_path = os.path.join(
            cfg.train.checkpoint_dir,
            f"best_robust_{args.mode}_{args.ablation}.pt",
        )
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            load_state_dict_compatible(model, ckpt["model_state_dict"], module_name="best_ckpt")

        final_metrics = evaluate(
            model, criterion, test_loader, device, cfg, ablation=args.ablation,
        )
        print_metrics(final_metrics,
                      f"Final Results ({args.mode}, ablation={args.ablation})")

        calib_summary = run_reliability_calibration(model, test_loader, device, save_dir=cfg.train.log_dir)
        if args.run_e4_paper:
            run_reliability_calibration_paper(
                model, test_loader, device, cfg, save_dir=cfg.train.log_dir
            )
        if args.run_e4_s2_monotonicity:
            run_reliability_monotonicity_s2(
                model, test_loader, device, cfg, save_dir=cfg.train.log_dir
            )

        plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                             save_dir=cfg.train.log_dir)
        _save_log_json("final_metrics", {
            "meta": {
                "stage": args.stage,
                "mode": args.mode,
                "ablation": args.ablation,
                "backbone": effective_backbone,
                "seed": int(cfg.train.seed),
                "label_ratio_pct": label_ratio_pct,
                "H": int(H), "T": int(T),
            },
            "final_metrics": final_metrics,
            "best_val_top1": float(best_top1),
        })

    # ========== Stress Tests ==========
    if args.stress_test in ("S1", "all"):
        s1_results = run_stress_test_s1(model, criterion, test_loader, device, cfg)
        _save_log_json("stress_S1", {"meta": {"seed": int(cfg.train.seed)}, "results": s1_results})
    if args.stress_test in ("S2", "all"):
        s2_results = run_stress_test_s2(model, criterion, test_loader, device, cfg)
        _save_log_json("stress_S2", {"meta": {"seed": int(cfg.train.seed)}, "results": s2_results})
    if args.stress_test in ("S3", "all"):
        s3_results = run_stress_test_s3(model, criterion, test_loader, device, cfg)
        _save_log_json("stress_S3", {"meta": {"seed": int(cfg.train.seed)}, "results": s3_results})

    # ========== E2: Delay-Regime Specialization ==========
    if args.run_e2:
        def _default_e2_ckpt():
            pattern = os.path.join(cfg.train.checkpoint_dir, f"best_robust_{args.mode}_A1*.pt")
            cands = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
            return cands[0] if cands else None

        a1_ckpt_path = args.e2_a1_ckpt or _default_e2_ckpt()
        if not a1_ckpt_path or not os.path.isfile(a1_ckpt_path):
            print("  E2 skipped: A1 checkpoint not found.")
        else:
            model_a1 = build_model(cfg, effective_backbone, device, H, T)
            freeze_encoder_backbones(model_a1)
            apply_cli_ablation(model_a1, "A1")
            ckpt_a1 = torch.load(a1_ckpt_path, map_location=device, weights_only=False)
            load_state_dict_compatible(model_a1, ckpt_a1["model_state_dict"], module_name="e2_a1_ckpt")
            e2_results = run_e2_delay_regime_specialization(
                model, model_a1, criterion, test_loader, device, cfg, save_dir=cfg.train.log_dir
            )
            _save_log_json("E2_delay_regime_specialization", {
                "meta": {"seed": int(cfg.train.seed), "mode": args.mode},
                "results": e2_results,
            })

    # ========== E1 + Ablations ==========
    if args.run_all_experiments:
        run_gradient_contamination_experiment(
            model, train_loader, criterion, device, save_dir=cfg.train.log_dir
        )
        ablation_results = run_ablation_study(
            cfg, device, args.backbone, train_loader, test_loader, H, T,
        )
        results_path = os.path.join(cfg.train.log_dir, f"ablation_results_{run_tag}.json")
        serializable = {
            k: {mk: float(mv) for mk, mv in v.items()}
            for k, v in ablation_results.items()
        }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        plot_ablation_results(ablation_results, save_dir=cfg.train.log_dir)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
