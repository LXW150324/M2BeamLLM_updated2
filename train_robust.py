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
    apply_degradation,
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
    build_beam_transition_log_prior,
    build_supervised_optimizer,
    compute_pairwise_reranker_loss,
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
                   help="Supervised label ratio for E3 (percent 1-100 or fraction 0-1). Stage1 SSL uses full train set.")
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
                   help="Run E2 delay-regime specialization comparison (baseline vs A1 checkpoint)")
    p.add_argument("--e2_a1_ckpt", type=str, default=None,
                   help="Path to trained A1 checkpoint for E2 comparison (optional)")
    p.add_argument("--run_e4_paper", action="store_true",
                   help="Run paper-definition E4 reliability calibration: Corr(sigma^2, e_mod)")
    p.add_argument("--run_e4_s2_monotonicity", action="store_true",
                   help="Run paper-definition E4 monotonicity check under S2 severities")
    p.add_argument("--complexity_only", action="store_true")
    p.add_argument("--complexity_benchmark", action="store_true",
                   help="Also benchmark FLOPs/latency for C4 (best-effort)")
    p.add_argument("--latency_warmup_iters", type=int, default=None)
    p.add_argument("--latency_iters", type=int, default=None)
    p.add_argument("--s1_modality_specific", action="store_true",
                   help="Run modality-specific delays in S1 (paper-faithful, slower)")
    p.add_argument("--use_beam_history", action="store_true",
                   help="Enable optional beam-history conditioned extension branch")
    p.add_argument("--use_ar_decoder", action="store_true",
                   help="Enable optional autoregressive beam decoder extension")
    p.add_argument("--use_temporal_smoothing", action="store_true",
                   help="Enable inference-time temporal smoothing on beam logits")
    p.add_argument("--use_transition_rerank", action="store_true",
                   help="Enable inference-time transition-prior logit reranking")
    p.add_argument("--use_constrained_reranker", action="store_true",
                   help="Enable inference-time constrained Top-K reranker")
    p.add_argument("--use_pairwise_reranker", action="store_true",
                   help="Enable trainable Top-K pairwise reranker (R1) and use it at eval time")
    p.add_argument("--use_viterbi_smoother", action="store_true",
                   help="Enable inference-time Viterbi-style temporal smoother over Top-K lattice")
    p.add_argument("--clean_top1_preset", action="store_true",
                   help="Prioritize clean Top-1: simpler Stage2 loss/no eval postprocess + enable beam history")
    p.add_argument("--ts_alpha", type=float, default=None,
                   help="Override temporal smoothing alpha (0..1)")
    p.add_argument("--rerank_lambda", type=float, default=None,
                   help="Override transition rerank strength")
    p.add_argument("--cr_lambda", type=float, default=None,
                   help="Override constrained reranker strength")
    p.add_argument("--pairwise_lambda", type=float, default=None,
                   help="Override train/eval pairwise reranker strength")
    return p.parse_args()


def set_global_seed(seed: int):
    """Best-effort reproducibility for high-variance Stage2 training."""
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
    """
    Load a checkpoint state_dict while skipping keys that are missing,
    unexpected, or shape-mismatched with the current module.

    This keeps older checkpoints usable after small architecture changes
    (e.g. extra router context features) without silently pretending that
    all weights were restored.
    """
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

    print(
        f"  {module_name}: loaded {len(filtered)}/{len(state_dict)} compatible tensors"
    )
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
        if len(mismatched) > 8:
            print(f"    ... {len(mismatched) - 8} more")

    return result, unexpected, mismatched


def set_stage2_frozen_llm_eval_mode(model: RobustM2BeamLLM, head_only: bool):
    """
    Keep frozen LLM/MoE submodules in eval mode (dropout off) while preserving
    train mode for currently trainable LLM blocks/adapters.
    """
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


def build_class_weights(train_loader: DataLoader, num_beams: int, H: int, T: int,
                        device: torch.device) -> torch.Tensor | None:
    """
    Build class-balanced CE weights from train beam labels.
    Uses inverse-sqrt frequency with clipping for stability.
    """
    counts, freq = _get_train_future_class_stats(train_loader, num_beams, H, T)
    if counts is None or freq is None:
        return None

    weights = 1.0 / np.sqrt(freq)
    # Clip extreme tails so rare classes do not destabilize training.
    weights = np.clip(weights, 0.0, np.percentile(weights, 95))
    weights = weights / max(weights.mean(), 1e-8)

    major = int(np.argmax(counts))
    top_idx = np.argsort(counts)[-5:][::-1]
    top_msg = ", ".join(f"{int(i)}:{freq[i]*100:.1f}%" for i in top_idx)
    eff_classes = int((counts > 0).sum())
    print(
        f"  Class-balance: major beam={major} freq={freq[major]*100:.2f}% | "
        f"active={eff_classes}/{num_beams} | top5={top_msg}"
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_class_prior(train_loader: DataLoader, num_beams: int, H: int, T: int,
                      device: torch.device) -> torch.Tensor | None:
    """Empirical train prior p(y) for anti-collapse regularization."""
    _, freq = _get_train_future_class_stats(train_loader, num_beams, H, T)
    if freq is None:
        return None
    return torch.tensor(freq, dtype=torch.float32, device=device)


def build_valid_beam_mask(train_loader: DataLoader, num_beams: int, H: int, T: int,
                          device: torch.device) -> torch.Tensor | None:
    """
    Return a boolean mask over beam IDs that are observed at least once in
    train future labels. IDs unseen in training are masked out from logits.
    """
    ds = getattr(train_loader, "dataset", None)
    beams = getattr(ds, "beams", None) if ds is not None else None
    if beams is None:
        print("  Valid-beam mask disabled: train beams unavailable.")
        return None
    future = np.array(beams[:, H:H + T], dtype=np.int64).reshape(-1)
    if future.size == 0:
        print("  Valid-beam mask disabled: empty future labels.")
        return None
    counts = np.bincount(future, minlength=num_beams).astype(np.int64)
    mask = counts > 0
    invalid = np.where(~mask)[0].tolist()
    print(f"  Valid beams: {int(mask.sum())}/{num_beams} (masked unseen: {invalid})")
    return torch.tensor(mask, dtype=torch.bool, device=device)


def _get_train_future_class_stats(train_loader: DataLoader, num_beams: int, H: int, T: int
                                  ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (counts, frequency) over future beam labels in training windows."""
    ds = getattr(train_loader, "dataset", None)
    beams = getattr(ds, "beams", None) if ds is not None else None
    if beams is None:
        print("  Class-balanced loss disabled: train beams unavailable.")
        return None, None

    future = np.array(beams[:, H:H + T], dtype=np.int64).reshape(-1)
    if future.size == 0:
        print("  Class-balanced loss disabled: empty future labels.")
        return None, None

    counts = np.bincount(future, minlength=num_beams).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid divide-by-zero for unseen classes
    freq = counts / counts.sum()
    return counts, freq


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

        # Proposal-faithful Stage 1 default: pretrain encoders/fusion/SSL heads.
        # Explicit alignment and reliability weighting are introduced in Stage 2.
        if cfg.ssl.use_explicit_alignment:
            aligned, _ = model.async_alignment(raw_features)
        else:
            aligned = raw_features

        if cfg.ssl.use_reliability_gating:
            _, _, gated = model.reliability_estimator(aligned)
        else:
            gated = aligned

        fused_tokens = model.fusion(gated)  # (B, H*num_tokens, M)
        # SSL temporal forecasting is defined over timesteps (B, H, M), not
        # flattened fusion tokens. Pool tokens within each timestep.
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
    print("  Stage1 modules:",
          f"alignment={'on' if cfg.ssl.use_explicit_alignment else 'off'},",
          f"reliability={'on' if cfg.ssl.use_reliability_gating else 'off'}")

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
                    focal_warmup_epochs: int = 0,
                    moe_warmup_epochs: int = 0,
                    prior_warmup_epochs: int = 0,
                    warmup_delay_epochs: int = 0,
                    ar_warmup_epochs: int = 0,
                    ar_warmup_delay_epochs: int = 0,
                    ar_tf_decay_epochs: int = 0,
                    ar_tf_min_ratio: float = 0.0,
                    reliability_monopoly_warmup_epochs: int = 0,
                    pairwise_reranker_warmup_epochs: int = 0,
                    pairwise_reranker_delay_epochs: int = 0,
                    modality_dropout_enabled: bool = False,
                    modality_dropout_start_epoch: int = 1,
                    modality_dropout_probs: Optional[Dict[str, float]] = None,
                    corruption_enabled: bool = False,
                    corruption_start_epoch: int = 1,
                    corruption_apply_prob: float = 0.0,
                    corruption_combined_prob: float = 0.0):
    """Train one supervised epoch."""
    model.train()
    set_encoder_eval_mode(model)
    set_stage2_frozen_llm_eval_mode(model, head_only=head_only_warmup)

    total_loss = 0.0
    loss_components = {
        "beam": 0.0,
        "alignment": 0.0,
        "moe_balance": 0.0,
        "prior_match": 0.0,
        "cvar": 0.0,
        "reliability_monopoly": 0.0,
        "pairwise_rerank": 0.0,
        "ar_eff_scale": 0.0,
        "ar_tf_ratio": 0.0,
        "pairwise_rerank_coverage": 0.0,
        "pairwise_rerank_hardcase": 0.0,
        "moddrop_any": 0.0,
        "moddrop_gps": 0.0,
        "corrupt_any": 0.0,
        "corrupt_combined": 0.0,
        "corrupt_severity": 0.0,
    }
    total_correct = 0
    total_samples = 0
    pred_counts = torch.zeros(model.num_beams, dtype=torch.long)

    use_reliability_align = (ablation != "A3")
    detach_reliability_align = bool(
        use_reliability_align and
        epoch <= int(max(0, getattr(cfg.train, "reliability_align_detach_epochs", 0)))
    )
    def _ramp(epoch_idx: int, warmup_epochs: int, delay_epochs: int = 0) -> float:
        if warmup_epochs <= 0:
            return 1.0
        if epoch_idx <= delay_epochs:
            return 0.0
        return min(1.0, max(0.0, (epoch_idx - delay_epochs - 1) / float(warmup_epochs)))

    # Delay auxiliary/focal ramps until after head-only warmup to avoid pushing
    # the classifier into a prior-collapse regime before the backbone is active.
    align_scale = _ramp(epoch, align_warmup_epochs, delay_epochs=warmup_delay_epochs)
    moe_scale = _ramp(epoch, moe_warmup_epochs, delay_epochs=warmup_delay_epochs)
    focal_gamma = criterion.focal_gamma * _ramp(
        epoch, focal_warmup_epochs, delay_epochs=warmup_delay_epochs
    )
    prior_scale = _ramp(epoch, prior_warmup_epochs, delay_epochs=0)
    cvar_scale = _ramp(epoch, getattr(cfg.train, "cvar_warmup_epochs", 0), delay_epochs=0)
    rel_mono_scale = _ramp(
        epoch, reliability_monopoly_warmup_epochs, delay_epochs=warmup_delay_epochs
    )
    pairwise_scale = _ramp(
        epoch, pairwise_reranker_warmup_epochs, delay_epochs=pairwise_reranker_delay_epochs
    )
    ar_scale = _ramp(epoch, ar_warmup_epochs, delay_epochs=ar_warmup_delay_epochs)
    tf_ramp = _ramp(epoch, ar_tf_decay_epochs, delay_epochs=ar_warmup_delay_epochs)
    ar_tf_ratio = 1.0 - (1.0 - float(ar_tf_min_ratio)) * tf_ramp

    def _maybe_apply_modality_dropout(images, radars, lidars, gps, B: int):
        if (not modality_dropout_enabled) or (epoch < int(modality_dropout_start_epoch)):
            return images, radars, lidars, gps, {"any": 0.0, "gps": 0.0}
        probs = modality_dropout_probs or {}
        p_img = max(0.0, min(1.0, float(probs.get("image", 0.0))))
        p_rad = max(0.0, min(1.0, float(probs.get("radar", 0.0))))
        p_lid = max(0.0, min(1.0, float(probs.get("lidar", 0.0))))
        p_gps = max(0.0, min(1.0, float(probs.get("gps", 0.0))))
        dev = images.device
        drop = torch.stack([
            (torch.rand(B, device=dev) < p_img),
            (torch.rand(B, device=dev) < p_rad),
            (torch.rand(B, device=dev) < p_lid),
            (torch.rand(B, device=dev) < p_gps),
        ], dim=1)  # (B,4)
        # Keep at least one modality per sample.
        all_drop = drop.all(dim=1)
        if all_drop.any():
            idx = torch.where(all_drop)[0]
            drop[idx, 0] = False  # keep image as fallback
        img_mask = (~drop[:, 0]).float().view(B, 1, 1, 1, 1)
        rad_mask = (~drop[:, 1]).float().view(B, 1, 1, 1, 1)
        lid_mask = (~drop[:, 2]).float().view(B, 1, 1, 1, 1)
        gps_mask = (~drop[:, 3]).float().view(B, 1, 1)
        return (
            images * img_mask,
            radars * rad_mask,
            lidars * lid_mask,
            gps * gps_mask,
            {
                "any": float(drop.any(dim=1).float().mean().item()),
                "gps": float(drop[:, 3].float().mean().item()),
            }
        )

    def _sample_corruption_severity() -> float:
        # Mid-severity corruption is the current failure zone, so oversample it.
        u = random.random()
        if u < 0.20:
            lo, hi = 0.05, 0.20
        elif u < 0.70:
            lo, hi = 0.20, 0.60
        else:
            lo, hi = 0.60, 1.00
        return float(lo + (hi - lo) * random.random())

    def _maybe_apply_stage2_corruption(images, radars, lidars, gps, beam_history, B: int):
        if (not corruption_enabled) or (epoch < int(corruption_start_epoch)):
            return images, radars, lidars, gps, None, {
                "any": 0.0, "combined": 0.0, "severity": 0.0,
            }
        if random.random() >= float(max(0.0, min(1.0, corruption_apply_prob))):
            return images, radars, lidars, gps, None, {
                "any": 0.0, "combined": 0.0, "severity": 0.0,
            }

        combined = random.random() < float(max(0.0, min(1.0, corruption_combined_prob)))
        target_modalities = ["image", "radar", "lidar", "gps"] if combined else [
            random.choice(["image", "radar", "lidar", "gps"])
        ]
        severity = _sample_corruption_severity()
        degraded = apply_degradation(
            {
                "images": images,
                "radars": radars,
                "lidars": lidars,
                "gps": gps,
                "beam_history": beam_history if beam_history is not None else torch.zeros(
                    (B, model.H), dtype=torch.long, device=images.device
                ),
            },
            severity=severity,
            target_modalities=target_modalities,
            stress_config=cfg.stress_test,
            data_root=cfg.data.data_root,
        )
        return (
            degraded["images"],
            degraded["radars"],
            degraded["lidars"],
            degraded["gps"],
            degraded.get("_degradation_cues"),
            {
                "any": 1.0,
                "combined": 1.0 if combined else 0.0,
                "severity": severity,
            },
        )

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False):
        images = batch["images"].to(device)
        radars = batch["radars"].to(device)
        lidars = batch["lidars"].to(device)
        gps = batch["gps"].to(device)
        targets = batch["beam_future"].to(device)
        beam_history = batch.get("beam_history")
        if beam_history is not None:
            beam_history = beam_history.to(device)
        B = targets.size(0)
        images, radars, lidars, gps, moddrop_diag = _maybe_apply_modality_dropout(
            images, radars, lidars, gps, B
        )
        images, radars, lidars, gps, degradation_cues, corrupt_diag = _maybe_apply_stage2_corruption(
            images, radars, lidars, gps, beam_history, B
        )

        optimizer.zero_grad()
        use_ar = bool(getattr(model, "use_autoregressive_decoder", False))
        predictions, aux = model(
            images, radars, lidars, gps,
            degradation_cues=degradation_cues,
            beam_history=beam_history,
            beam_future_targets=targets if use_ar else None,
            teacher_forcing=use_ar,
            teacher_forcing_ratio=ar_tf_ratio,
            ar_logit_scale_override=(ar_scale if use_ar else None),
        )
        predictions_for_loss = predictions
        valid_mask = getattr(criterion, "valid_beam_mask", None)
        if valid_mask is not None:
            vm = valid_mask.to(device=predictions.device)
            if vm.numel() == predictions.size(-1):
                predictions_for_loss = predictions.masked_fill((~vm).view(1, 1, -1), -1e4)
        loss, loss_dict = criterion(
            predictions_for_loss, targets, aux,
            use_reliability_align=use_reliability_align,
            align_loss_module=model.alignment_loss_fn,
            align_scale=align_scale,
            focal_gamma_override=focal_gamma,
            detach_reliability_align_weights=detach_reliability_align,
            moe_scale=moe_scale,
            prior_match_scale=prior_scale,
            cvar_scale=cvar_scale,
            reliability_monopoly_scale=rel_mono_scale,
        )
        pairwise_loss, pairwise_diag = compute_pairwise_reranker_loss(
            model=model,
            predictions=predictions_for_loss,
            targets=targets,
            beam_history=beam_history,
            aux=aux,
            staleness_ms=None,
            gps_hist=gps,
            cfg=cfg,
            epoch=epoch,
            scale_override=pairwise_scale,
        )
        total_loss_tensor = loss + pairwise_loss
        loss_dict["total"] = loss_dict.get("total", float(loss.item())) + float(pairwise_loss.detach().item())
        loss_dict["pairwise_rerank"] = float(pairwise_diag.get("loss", 0.0))
        loss_dict["pairwise_rerank_coeff"] = float(pairwise_diag.get("coeff", 0.0))
        loss_dict["pairwise_rerank_coverage"] = float(pairwise_diag.get("coverage", 0.0))
        loss_dict["pairwise_rerank_hardcase"] = float(pairwise_diag.get("hardcase", 0.0))

        total_loss_tensor.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_dict["total"] * B
        for k in loss_components:
            if k in (
                "ar_eff_scale", "ar_tf_ratio",
                "pairwise_rerank_coverage", "pairwise_rerank_hardcase",
                "moddrop_any", "moddrop_gps",
                "corrupt_any", "corrupt_combined", "corrupt_severity",
            ):
                continue
            loss_components[k] += loss_dict.get(k, 0.0) * B
        ar_diag = aux.get("ar_decoder", {})
        if ar_diag.get("enabled", False):
            loss_components["ar_eff_scale"] += float(ar_diag.get("effective_logit_scale", 0.0)) * B
            loss_components["ar_tf_ratio"] += float(ar_diag.get("teacher_forcing_ratio", 0.0)) * B
        loss_components["pairwise_rerank_coverage"] += float(loss_dict.get("pairwise_rerank_coverage", 0.0)) * B
        loss_components["pairwise_rerank_hardcase"] += float(loss_dict.get("pairwise_rerank_hardcase", 0.0)) * B
        loss_components["moddrop_any"] += float(moddrop_diag.get("any", 0.0)) * B
        loss_components["moddrop_gps"] += float(moddrop_diag.get("gps", 0.0)) * B
        loss_components["corrupt_any"] += float(corrupt_diag.get("any", 0.0)) * B
        loss_components["corrupt_combined"] += float(corrupt_diag.get("combined", 0.0)) * B
        loss_components["corrupt_severity"] += float(corrupt_diag.get("severity", 0.0)) * B

        pred_indices = predictions_for_loss.argmax(dim=-1)
        pred_counts += torch.bincount(
            pred_indices.reshape(-1).detach().cpu(),
            minlength=model.num_beams,
        )
        total_correct += (pred_indices == targets).float().mean(dim=1).sum().item()
        total_samples += B

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    avg_comp = {k: v / max(total_samples, 1) for k, v in loss_components.items()}
    avg_comp["align_coeff"] = align_scale * criterion.lambda_align
    avg_comp["moe_balance_coeff"] = moe_scale * criterion.lambda_moe_balance
    avg_comp["prior_match_coeff"] = prior_scale * criterion.lambda_prior_match
    avg_comp["cvar_coeff"] = cvar_scale * getattr(criterion, "lambda_cvar", 0.0)
    avg_comp["reliability_monopoly_coeff"] = rel_mono_scale * criterion.lambda_reliability_monopoly
    avg_comp["pairwise_rerank_coeff"] = float(getattr(cfg.train, "pairwise_reranker_lambda", 0.0)) * pairwise_scale
    avg_comp["beam_coeff"] = criterion.lambda_beam
    avg_comp["focal_gamma"] = focal_gamma
    avg_comp["label_smoothing"] = getattr(criterion, "label_smoothing", 0.0)
    avg_comp["beam_soft_target_lambda"] = getattr(criterion, "beam_soft_target_lambda", 0.0)
    avg_comp["detach_reliability_align"] = 1.0 if detach_reliability_align else 0.0
    avg_comp["ar_scale"] = ar_scale
    avg_comp["ar_tf_ratio_sched"] = ar_tf_ratio
    avg_comp["pairwise_rerank_scale"] = pairwise_scale
    total_pred = int(pred_counts.sum().item())
    avg_comp["train_pred_major_share"] = (
        pred_counts.max().item() / max(total_pred, 1)
    )
    avg_comp["train_pred_unique"] = int((pred_counts > 0).sum().item())
    return avg_loss, avg_acc, avg_comp


def apply_cli_ablation(model: RobustM2BeamLLM, ablation: str):
    """
    Apply single ablation to the provided model for main training flow.
    (run_ablation_study has its own per-model ablation logic.)
    """
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
        print("  Applied A1: removed delay-regime conditioning (shared alignment MLP).")
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
        else:
            print("  Warning: A4 requested but selected backbone has no MoE adapters.")


def apply_stage2_curriculum(model: RobustM2BeamLLM, cfg: Config, epoch: int) -> dict:
    """
    Progressive Stage-2 curriculum to reduce collapse and transition shock:
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
        phase = f"head-only (LLM/MoE frozen, dropout off, epoch {epoch}/{head_only_epochs})"
    elif phase_type == "moe_only":
        phase_idx = epoch - e_head_end
        phase = f"moe-only (MoE trainable, LLM frozen, epoch {phase_idx}/{moe_only_epochs})"
    elif phase_type == "top_llm_only":
        phase_idx = epoch - e_moe_end
        phase = (
            f"top-layer+MoE (last LLM block + ln_f, epoch {phase_idx}/{llm_top1_epochs})"
        )
    else:
        phase = "full-model (LLM/MoE restored)"
    changed = phase != getattr(model, phase_name)
    setattr(model, phase_name, phase)
    full_model_start_epoch = e_top1_end + 1
    aux_ramp_delay_epochs = max(full_model_start_epoch - 1, 0)
    # AR decoder starts at top-layer+MoE phase to avoid head-only memorization
    # while still giving enough training time before the probe ends.
    ar_ramp_delay_epochs = max(e_moe_end, 0)
    return {
        "changed": changed,
        "phase": phase,
        "phase_type": phase_type,
        "head_only": head_only,
        "aux_ramp_delay_epochs": aux_ramp_delay_epochs,
        "ar_ramp_delay_epochs": ar_ramp_delay_epochs,
    }


# ===========================================================================
# Ablation Study (A1-A5)
# ===========================================================================

def run_ablation_study(cfg: Config, device: torch.device, backbone: str,
                       train_loader: DataLoader, test_loader: DataLoader,
                       H: int, T: int, class_weights: torch.Tensor | None = None):
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
        class_weights=class_weights,
        focal_gamma=cfg.train.focal_gamma,
        label_smoothing=getattr(cfg.train, "label_smoothing", 0.0),
        beam_soft_target_lambda=getattr(cfg.train, "beam_soft_target_lambda", 0.0),
        beam_soft_target_tau=getattr(cfg.train, "beam_soft_target_tau", 1.5),
        lambda_moe_balance=cfg.train.lambda_moe_balance,
        lambda_cvar=getattr(cfg.train, "lambda_cvar", 0.0),
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
                # True A4 behavior: use one shared LoRA expert (no condition routing).
                moe_layer[key].set_shared_expert(True, expert_idx=0)
                for p in moe_layer[key].router.parameters():
                    p.requires_grad = False
    else:
        print("  Warning: selected backbone has no MoE adapters; A4 acts as baseline.")
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

    if args.clean_top1_preset:
        # Explicit enhancement mode for clean Top-1. This is intentionally
        # separate from the paper-faithful default path.
        cfg.model.use_beam_history = True
        cfg.model.use_autoregressive_decoder = False
        cfg.model.use_pairwise_reranker = False
        cfg.model.history_anchor_scale = 0.35
        cfg.model.history_anchor_decay = 0.78
        cfg.model.history_anchor_big_jump_penalty = 0.15

        cfg.train.use_class_balanced_loss = False
        cfg.train.mask_unseen_beams = True
        cfg.train.focal_gamma = 0.0
        cfg.train.label_smoothing = 0.0
        cfg.train.lambda_moe_balance = 0.0
        cfg.train.lambda_prior_match = 0.0
        cfg.train.lambda_reliability_monopoly = 0.0
        cfg.train.lambda_align = min(float(cfg.train.lambda_align), 0.2)
        cfg.train.beam_step_weights = [1.35, 1.15, 1.00, 0.90, 0.80]

        cfg.train.eval_temporal_smoothing_enable = False
        cfg.train.eval_transition_rerank_enable = False
        cfg.train.eval_constrained_reranker_enable = False
        cfg.train.eval_pairwise_reranker_enable = False
        cfg.train.eval_viterbi_smoother_enable = False
        cfg.reliability.softmax_temperature = 1.0
        cfg.reliability.uniform_mix = 0.0
        cfg.reliability.reliability_min = 1e-4

        print(
            "  Applied clean_top1_preset: history on, extra Stage2 losses off, "
            "history-anchor on, step-weighted beam loss on, eval postprocess off."
        )

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
    if args.use_beam_history:
        cfg.model.use_beam_history = True
    if args.use_ar_decoder:
        cfg.model.use_autoregressive_decoder = True
    if args.use_pairwise_reranker:
        cfg.model.use_pairwise_reranker = True
        cfg.train.eval_pairwise_reranker_enable = True
    if args.use_temporal_smoothing:
        cfg.train.eval_temporal_smoothing_enable = True
    if args.use_transition_rerank:
        cfg.train.eval_transition_rerank_enable = True
    if args.use_constrained_reranker:
        cfg.train.eval_constrained_reranker_enable = True
    if args.use_viterbi_smoother:
        cfg.train.eval_viterbi_smoother_enable = True
    if args.ts_alpha is not None:
        cfg.train.eval_temporal_smoothing_alpha = float(args.ts_alpha)
    if args.rerank_lambda is not None:
        cfg.train.eval_transition_rerank_lambda = float(args.rerank_lambda)
    if args.cr_lambda is not None:
        cfg.train.eval_constrained_reranker_lambda = float(args.cr_lambda)
    if args.pairwise_lambda is not None:
        cfg.train.pairwise_reranker_lambda = float(args.pairwise_lambda)
        cfg.train.eval_pairwise_reranker_lambda = float(args.pairwise_lambda)

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
    print(f"  Clean Top1 Preset: {'on' if args.clean_top1_preset else 'off'}")
    print(
        "  Proposal-Faithful Default: "
        f"{'yes' if not args.clean_top1_preset else 'no (using enhancement preset)'}"
    )
    print(f"  History Cond: {'on' if getattr(cfg.model, 'use_beam_history', False) else 'off'}")
    print(
        f"  Frozen LLM Layers: {cfg.model.num_unfrozen_layers} | "
        f"Adapter Layers: {getattr(cfg.model, 'num_adapter_layers', 0)}"
    )
    print(f"  AR Decoder: {'on' if getattr(cfg.model, 'use_autoregressive_decoder', False) else 'off'}")
    print(f"  Pairwise Reranker (R1): {'on' if getattr(cfg.model, 'use_pairwise_reranker', False) else 'off'}")
    print(f"  History Anchor: {getattr(cfg.model, 'history_anchor_scale', 0.0):.2f}")
    print(
        f"  Eval Postprocess: TS={'on' if cfg.train.eval_temporal_smoothing_enable else 'off'}"
        f"(α={cfg.train.eval_temporal_smoothing_alpha:.2f}), "
        f"TransRerank={'on' if cfg.train.eval_transition_rerank_enable else 'off'}"
        f"(λ={cfg.train.eval_transition_rerank_lambda:.2f}), "
        f"CR={'on' if cfg.train.eval_constrained_reranker_enable else 'off'}"
        f"(λ={cfg.train.eval_constrained_reranker_lambda:.2f}), "
        f"PR={'on' if cfg.train.eval_pairwise_reranker_enable else 'off'}"
        f"(λ={cfg.train.eval_pairwise_reranker_lambda:.2f}), "
        f"Viterbi={'on' if cfg.train.eval_viterbi_smoother_enable else 'off'}"
    )
    print(f"  Seed: {cfg.train.seed}")
    if effective_backbone == "vanilla_transformer":
        print(
            "  A5 Vanilla (paper-aligned): "
            f"L={cfg.model.vanilla_num_layers}, H={cfg.model.vanilla_num_heads}, "
            f"d={cfg.model.vanilla_hidden_dim}, ffn={cfg.model.vanilla_ffn_hidden_dim}"
        )
    if args.complexity_benchmark:
        print(f"  C4 Benchmark: on (warmup={cfg.train.complexity_latency_warmup_iters}, iters={cfg.train.complexity_latency_iters})")
    print(f"{'='*60}\n")

    # Data
    train_loader_full, test_loader = create_dataloaders(cfg, H=H, T=T)
    train_loader = train_loader_full  # supervised loader (may be subsampled)
    label_ratio_frac = 1.0
    if args.label_ratio is not None:
        ratio_val = float(args.label_ratio)
        label_ratio_frac = ratio_val / 100.0 if ratio_val > 1.0 else ratio_val
        label_ratio_frac = max(0.0, min(1.0, label_ratio_frac))
        if label_ratio_frac <= 0.0:
            raise ValueError("--label_ratio must be > 0")
        if label_ratio_frac < 1.0:
            train_loader = subsample_loader_by_fraction(
                train_loader_full,
                label_ratio_frac,
                seed=cfg.train.seed,
                shuffle=True,
                weighted_sampling=bool(getattr(cfg.train, "use_weighted_sampler", False)),
                H=H,
                T=T,
                sampler_power=float(getattr(cfg.train, "weighted_sampler_power", 0.5)),
            )
            print(
                f"  Supervised label ratio: {label_ratio_frac*100:.1f}% "
                f"({len(train_loader.dataset)}/{len(train_loader_full.dataset)} windows)"
            )
        else:
            print("  Supervised label ratio: 100.0% (full train set)")
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
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                return obj.item()
            except Exception:
                pass
        return str(obj)

    run_tag_parts = [
        args.mode,
        args.ablation,
        f"seed{cfg.train.seed}",
        f"lrat{label_ratio_pct}",
    ]
    if args.clean_top1_preset:
        run_tag_parts.append("clean")
    if getattr(cfg.model, "use_beam_history", False):
        run_tag_parts.append("hist")
    if getattr(cfg.model, "use_autoregressive_decoder", False):
        run_tag_parts.append("ar")
    if getattr(cfg.model, "use_pairwise_reranker", False):
        run_tag_parts.append("pr")
    if getattr(cfg.train, "eval_temporal_smoothing_enable", False):
        run_tag_parts.append("ts")
    if getattr(cfg.train, "eval_transition_rerank_enable", False):
        run_tag_parts.append("rr")
    if getattr(cfg.train, "eval_constrained_reranker_enable", False):
        run_tag_parts.append("cr")
    if getattr(cfg.train, "eval_pairwise_reranker_enable", False):
        run_tag_parts.append("prr")
    if getattr(cfg.train, "eval_viterbi_smoother_enable", False):
        run_tag_parts.append("vt")
    run_tag = "_".join(run_tag_parts)

    def _save_log_json(stem: str, payload: dict):
        path = os.path.join(cfg.train.log_dir, f"{stem}_{run_tag}.json")
        with open(path, "w") as f:
            json.dump(_jsonify(payload), f, indent=2)
        print(f"  Saved {stem}: {path}")
        return path

    # Model
    model = build_model(cfg, effective_backbone, device, H, T)

    # Optional inference-time transition prior (for logit reranking)
    if getattr(cfg.train, "eval_transition_rerank_enable", False):
        print("  Building transition prior for inference reranking...")
        transition_log_prior = build_beam_transition_log_prior(
            train_loader,
            num_beams=cfg.data.num_beams,
            pseudocount=getattr(cfg.train, "eval_transition_prior_pseudocount", 1.0),
        )
        # Store on both cfg (for newly created ablation models) and current model.
        setattr(cfg.train, "_eval_transition_log_prior", transition_log_prior)
        setattr(model, "eval_transition_log_prior", transition_log_prior)
        print(f"  Transition prior built: shape={tuple(transition_log_prior.shape)}")

    # Encoder checkpoint
    encoder_ckpt = args.encoder_ckpt or os.path.join(
        cfg.train.checkpoint_dir, "encoder_pretrained.pt"
    )
    if args.skip_encoder_ckpt:
        print("  Skipping encoder checkpoint loading (--skip_encoder_ckpt).")
    elif os.path.isfile(encoder_ckpt):
        print(f"✓ Loading encoders from {encoder_ckpt}")
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
                warmup_iters=getattr(cfg.train, "complexity_latency_warmup_iters", 10),
                iters=getattr(cfg.train, "complexity_latency_iters", 30),
            )
            bench["flops_per_sample"] = estimate_flops_per_sample(model, device=device, H=H, T=T)
            print_complexity_benchmark(bench)
            plot_c4_latency_benchmark(bench, save_dir=cfg.train.log_dir)
            _save_log_json("complexity_benchmark", {"meta": {"seed": int(cfg.train.seed), "H": int(H), "T": int(T)}, "results": bench})
        return

    complexity = compute_complexity_analysis(model)
    print_complexity_table(complexity)
    plot_complexity_breakdown(complexity, save_dir=cfg.train.log_dir)
    if args.complexity_benchmark:
        bench = benchmark_inference_latency(
            model, device=device, H=H, T=T,
            warmup_iters=getattr(cfg.train, "complexity_latency_warmup_iters", 10),
            iters=getattr(cfg.train, "complexity_latency_iters", 30),
        )
        bench["flops_per_sample"] = estimate_flops_per_sample(model, device=device, H=H, T=T)
        print_complexity_benchmark(bench)
        plot_c4_latency_benchmark(bench, save_dir=cfg.train.log_dir)
        _save_log_json("complexity_benchmark", {"meta": {"seed": int(cfg.train.seed), "H": int(H), "T": int(T)}, "results": bench})

    # Loss
    class_weights = None
    class_prior = None
    valid_beam_mask = None
    step_weights = None
    if cfg.train.use_class_balanced_loss and not bool(getattr(cfg.train, "use_weighted_sampler", False)):
        class_weights = build_class_weights(
            train_loader, cfg.data.num_beams, H, T, device
        )
    elif cfg.train.use_class_balanced_loss:
        print("  Class-balanced CE weights: off (weighted sampler already handles long-tail sampling).")
    if getattr(cfg.train, "mask_unseen_beams", False):
        valid_beam_mask = build_valid_beam_mask(
            train_loader, cfg.data.num_beams, H, T, device
        )
    if getattr(cfg.train, "beam_step_weights", None):
        step_weights = torch.tensor(
            list(cfg.train.beam_step_weights), dtype=torch.float32, device=device
        )
        print(f"  Beam step weights: {step_weights.detach().cpu().tolist()}")
    class_prior = build_class_prior(train_loader, cfg.data.num_beams, H, T, device)
    criterion = RobustM2BeamLLMLoss(
        feature_dim=cfg.model.feature_dim,
        lambda_align=cfg.train.lambda_align,
        lambda_beam=cfg.train.lambda_beam,
        class_weights=class_weights,
        class_prior=class_prior,
        valid_beam_mask=valid_beam_mask,
        step_weights=step_weights,
        focal_gamma=cfg.train.focal_gamma,
        label_smoothing=getattr(cfg.train, "label_smoothing", 0.0),
        beam_soft_target_lambda=getattr(cfg.train, "beam_soft_target_lambda", 0.0),
        beam_soft_target_tau=getattr(cfg.train, "beam_soft_target_tau", 1.5),
        lambda_moe_balance=cfg.train.lambda_moe_balance,
        lambda_prior_match=cfg.train.lambda_prior_match,
        lambda_cvar=getattr(cfg.train, "lambda_cvar", 0.0),
        lambda_reliability_monopoly=cfg.train.lambda_reliability_monopoly,
        reliability_monopoly_cap=cfg.train.reliability_monopoly_cap,
    ).to(device)

    # ========== Stage 1: SSL ==========
    if args.stage in ("ssl", "both"):
        run_ssl_pretraining(model, train_loader_full, cfg, device)

    ssl_ckpt = args.ssl_ckpt or os.path.join(cfg.train.checkpoint_dir, "ssl_pretrained.pt")
    if args.stage == "supervised":
        if args.skip_ssl_ckpt:
            print("  Skipping SSL checkpoint loading (--skip_ssl_ckpt).")
        elif os.path.isfile(ssl_ckpt):
            print(f"✓ Loading SSL checkpoint: {ssl_ckpt}")
            ckpt = torch.load(ssl_ckpt, map_location=device, weights_only=False)
            load_state_dict_compatible(model, ckpt["model_state_dict"], module_name="ssl_init")
        else:
            print("  SSL checkpoint not found, supervised stage starts without SSL initialization.")

    # ========== Stage 2: Supervised ==========
    if args.stage in ("supervised", "both"):
        print(f"\n{'='*60}")
        print(f"  Stage 2: Supervised Beam Prediction")
        print(f"{'='*60}")
        if getattr(cfg.train, "stage2_modality_dropout_enable", False):
            print(
                "  Stage2 modality dropout (train-only): "
                f"img={cfg.train.moddrop_image_prob:.2f}, "
                f"rad={cfg.train.moddrop_radar_prob:.2f}, "
                f"lid={cfg.train.moddrop_lidar_prob:.2f}, "
                f"gps={cfg.train.moddrop_gps_prob:.2f} "
                f"(start epoch {cfg.train.stage2_modality_dropout_start_epoch})"
            )
        if getattr(cfg.train, "stage2_corruption_enable", False):
            print(
                "  Stage2 corruption aug (train-only): "
                f"p={cfg.train.stage2_corruption_apply_prob:.2f}, "
                f"combined_p={cfg.train.stage2_corruption_combined_prob:.2f} "
                f"(start epoch {cfg.train.stage2_corruption_start_epoch})"
            )

        if args.ablation in ("A1", "A2", "A4"):
            apply_cli_ablation(model, args.ablation)

        optimizer = build_supervised_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.lr_decay_step,
            gamma=cfg.train.lr_decay_factor,
        )

        best_top1 = 0.0
        suffix_parts = []
        if getattr(cfg.model, "use_beam_history", False):
            suffix_parts.append("hist")
        if getattr(cfg.model, "use_autoregressive_decoder", False):
            suffix_parts.append("ar")
        if getattr(cfg.model, "use_pairwise_reranker", False):
            suffix_parts.append("pr")
        ckpt_suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
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
                focal_warmup_epochs=cfg.train.focal_warmup_epochs,
                moe_warmup_epochs=cfg.train.moe_balance_warmup_epochs,
                prior_warmup_epochs=cfg.train.prior_match_warmup_epochs,
                warmup_delay_epochs=curriculum.get(
                    "aux_ramp_delay_epochs", cfg.train.llm_warmup_epochs
                ),
                ar_warmup_epochs=getattr(cfg.train, "ar_decoder_warmup_epochs", 0),
                ar_warmup_delay_epochs=curriculum.get("ar_ramp_delay_epochs", 0),
                ar_tf_decay_epochs=getattr(cfg.train, "ar_teacher_forcing_decay_epochs", 0),
                ar_tf_min_ratio=getattr(cfg.train, "ar_teacher_forcing_min_ratio", 0.0),
                reliability_monopoly_warmup_epochs=getattr(
                    cfg.train, "reliability_monopoly_warmup_epochs", 0
                ),
                pairwise_reranker_warmup_epochs=getattr(cfg.train, "pairwise_reranker_warmup_epochs", 0),
                pairwise_reranker_delay_epochs=curriculum.get("ar_ramp_delay_epochs", 0),
                modality_dropout_enabled=bool(
                    getattr(cfg.train, "stage2_modality_dropout_enable", False)
                    and (
                        getattr(cfg.model, "use_beam_history", False)
                        or getattr(cfg.model, "use_autoregressive_decoder", False)
                    )
                ),
                modality_dropout_start_epoch=getattr(
                    cfg.train, "stage2_modality_dropout_start_epoch", 1
                ),
                modality_dropout_probs={
                    "image": getattr(cfg.train, "moddrop_image_prob", 0.0),
                    "radar": getattr(cfg.train, "moddrop_radar_prob", 0.0),
                    "lidar": getattr(cfg.train, "moddrop_lidar_prob", 0.0),
                    "gps": getattr(cfg.train, "moddrop_gps_prob", 0.0),
                },
                corruption_enabled=bool(getattr(cfg.train, "stage2_corruption_enable", False)),
                corruption_start_epoch=getattr(cfg.train, "stage2_corruption_start_epoch", 1),
                corruption_apply_prob=getattr(cfg.train, "stage2_corruption_apply_prob", 0.0),
                corruption_combined_prob=getattr(cfg.train, "stage2_corruption_combined_prob", 0.0),
            )
            val_metrics = evaluate(
                model, criterion, test_loader, device, cfg,
                ablation=args.ablation,
            )
            scheduler.step()

            val_top1 = val_metrics["top_1_acc"]
            val_step1 = val_metrics.get("step_1_acc", None)
            rl_top1 = val_metrics.get("repeat_last_top_1_acc", None)
            rl_step1 = val_metrics.get("repeat_last_step_1_acc", None)
            lrs = [pg["lr"] for pg in optimizer.param_groups]
            lr_str = "/".join(f"{x:.6f}" for x in lrs)
            elapsed = time.time() - t0
            baseline_msg = ""
            if rl_top1 is not None:
                baseline_msg += (
                    f" | RL Top-1: {100*rl_top1:.1f}% "
                    f"(Δ={100*(val_top1-rl_top1):+.1f}%)"
                )
            if (val_step1 is not None) and (rl_step1 is not None):
                baseline_msg += f" | S1Δ={100*(val_step1-rl_step1):+.1f}%"

            print(f"Epoch {epoch:3d}/{cfg.train.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {100*train_acc:.1f}% | "
                  f"Val Top-1: {100*val_top1:.1f}% | "
                  f"L_beam: {loss_comp['beam']:.4f} "
                  f"L_align: {loss_comp['alignment']:.4f} "
                  f"L_moe: {loss_comp['moe_balance']:.4f} "
                  f"L_prior: {loss_comp['prior_match']:.4f} "
                  f"L_cvar: {loss_comp.get('cvar', 0.0):.4f} "
                  f"L_rmono: {loss_comp.get('reliability_monopoly', 0.0):.4f} "
                  f"L_rank: {loss_comp.get('pairwise_rerank', 0.0):.4f} "
                  f"(λa={loss_comp['align_coeff']:.3f}, λm={loss_comp['moe_balance_coeff']:.3f}, "
                  f"λp={loss_comp['prior_match_coeff']:.3f}, λcv={loss_comp.get('cvar_coeff', 0.0):.3f}, γ={loss_comp['focal_gamma']:.2f}, "
                  f"LS={loss_comp.get('label_smoothing', 0.0):.2f}, "
                  f"ST={loss_comp.get('beam_soft_target_lambda', 0.0):.2f}, "
                  f"λr={loss_comp.get('reliability_monopoly_coeff', 0.0):.3f}, "
                  f"λrk={loss_comp.get('pairwise_rerank_coeff', 0.0):.3f}, "
                  f"DR={'on' if loss_comp.get('detach_reliability_align', 0.0) > 0.5 else 'off'}, "
                  f"ARs={loss_comp.get('ar_eff_scale', 0.0):.3f}, TF={loss_comp.get('ar_tf_ratio', 0.0):.2f}, "
                  f"MdG={100*loss_comp.get('moddrop_gps', 0.0):.0f}%, "
                  f"Corr={100*loss_comp.get('corrupt_any', 0.0):.0f}%/"
                  f"{100*loss_comp.get('corrupt_combined', 0.0):.0f}%@{loss_comp.get('corrupt_severity', 0.0):.2f}, "
                  f"CovK={100*loss_comp.get('pairwise_rerank_coverage', 0.0):.0f}%, "
                  f"HC={100*loss_comp.get('pairwise_rerank_hardcase', 0.0):.0f}%) | "
                  f"PredMajor: {100*loss_comp['train_pred_major_share']:.1f}% "
                  f"U={loss_comp['train_pred_unique']} | "
                  f"LRs: {lr_str} | {elapsed:.1f}s"
                  f"{baseline_msg}")

            if (val_step1 is not None) and (rl_step1 is not None) and (val_step1 < rl_step1):
                if epoch == 1 or epoch == cfg.train.num_epochs or epoch % 5 == 0:
                    print(
                        "  Warning: step_1_acc is below repeat-last baseline. "
                        "This usually means the model is underusing beam history "
                        "(try --clean_top1_preset or --use_beam_history)."
                    )

            train_losses.append(train_loss)
            val_losses.append(val_metrics["loss"])
            train_accs.append(train_acc * 100)
            val_accs.append(val_top1 * 100)

            if val_top1 > best_top1:
                best_top1 = val_top1
                patience_counter = 0
                save_path = os.path.join(
                    cfg.train.checkpoint_dir,
                    f"best_robust_{args.mode}_{args.ablation}{ckpt_suffix}.pt",
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                }, save_path)
                print(f"  → Best model saved (Top-1: {100*val_top1:.1f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Load best
        best_path = os.path.join(
            cfg.train.checkpoint_dir,
            f"best_robust_{args.mode}_{args.ablation}{ckpt_suffix}.pt",
        )
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            load_state_dict_compatible(model, ckpt["model_state_dict"], module_name="best_ckpt")

        final_metrics = evaluate(
            model, criterion, test_loader, device, cfg, ablation=args.ablation,
        )
        print_metrics(final_metrics,
                      f"Final Results ({args.mode}, ablation={args.ablation}, "
                      f"history_cond={'on' if getattr(cfg.model, 'use_beam_history', False) else 'off'}, "
                      f"ar_decoder={'on' if getattr(cfg.model, 'use_autoregressive_decoder', False) else 'off'}, "
                      f"pairwise_reranker={'on' if getattr(cfg.model, 'use_pairwise_reranker', False) else 'off'})")
        if "repeat_last_top_1_acc" in final_metrics:
            rl_top1 = final_metrics["repeat_last_top_1_acc"]
            gap_top1 = final_metrics.get("top1_minus_repeat_last", final_metrics["top_1_acc"] - rl_top1)
            print(
                f"  Baseline check: repeat-last Top-1={100*rl_top1:.1f}% | "
                f"model gap={100*gap_top1:+.1f}%"
            )
        if "repeat_last_step_1_acc" in final_metrics and "step_1_acc" in final_metrics:
            gap_s1 = final_metrics.get(
                "step1_minus_repeat_last",
                final_metrics["step_1_acc"] - final_metrics["repeat_last_step_1_acc"],
            )
            print(
                f"  Baseline check: repeat-last step1={100*final_metrics['repeat_last_step_1_acc']:.1f}% | "
                f"model step1 gap={100*gap_s1:+.1f}%"
            )

        calib_summary = run_reliability_calibration(model, test_loader, device, save_dir=cfg.train.log_dir)
        e4_paper_summary = None
        e4_mono_summary = None
        if args.run_e4_paper:
            e4_paper_summary = run_reliability_calibration_paper(
                model, test_loader, device, cfg, save_dir=cfg.train.log_dir
            )
            _save_log_json("e4_reliability_calibration_paper", {
                "meta": {"seed": int(cfg.train.seed), "mode": args.mode, "ablation": args.ablation},
                "results": e4_paper_summary,
            })
        if args.run_e4_s2_monotonicity:
            e4_mono_summary = run_reliability_monotonicity_s2(
                model, test_loader, device, cfg, save_dir=cfg.train.log_dir
            )
            _save_log_json("e4_reliability_monotonicity", {
                "meta": {"seed": int(cfg.train.seed), "mode": args.mode, "ablation": args.ablation},
                "results": e4_mono_summary,
            })
        plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                             save_dir=cfg.train.log_dir)
        _save_log_json("final_metrics", {
            "meta": {
                "stage": args.stage,
                "mode": args.mode,
                "ablation": args.ablation,
                "backbone": effective_backbone,
                "history_cond": bool(getattr(cfg.model, "use_beam_history", False)),
                "ar_decoder": bool(getattr(cfg.model, "use_autoregressive_decoder", False)),
                "pairwise_reranker": bool(getattr(cfg.model, "use_pairwise_reranker", False)),
                "seed": int(cfg.train.seed),
                "label_ratio_pct": label_ratio_pct,
                "H": int(H), "T": int(T),
                "temporal_smoothing": bool(getattr(cfg.train, "eval_temporal_smoothing_enable", False)),
                "temporal_smoothing_alpha": float(getattr(cfg.train, "eval_temporal_smoothing_alpha", 0.0)),
                "transition_rerank": bool(getattr(cfg.train, "eval_transition_rerank_enable", False)),
                "transition_rerank_lambda": float(getattr(cfg.train, "eval_transition_rerank_lambda", 0.0)),
                "constrained_reranker": bool(getattr(cfg.train, "eval_constrained_reranker_enable", False)),
                "constrained_reranker_lambda": float(getattr(cfg.train, "eval_constrained_reranker_lambda", 0.0)),
                "pairwise_reranker_eval": bool(getattr(cfg.train, "eval_pairwise_reranker_enable", False)),
                "pairwise_reranker_eval_lambda": float(getattr(cfg.train, "eval_pairwise_reranker_lambda", 0.0)),
                "pairwise_reranker_train_lambda": float(getattr(cfg.train, "pairwise_reranker_lambda", 0.0)),
                "viterbi_smoother": bool(getattr(cfg.train, "eval_viterbi_smoother_enable", False)),
            },
            "final_metrics": final_metrics,
            "best_val_top1": float(best_top1),
        })
        # Save a compact calibration summary (no raw arrays to keep JSON size small).
        compact_calib = {"mods": {}}
        for mod, sm in (calib_summary or {}).get("mods", {}).items():
            compact_calib["mods"][mod] = {
                "corr": sm.get("corr"),
                "pval": sm.get("pval"),
                "bins": sm.get("bins"),
                "bin_acc": sm.get("bin_acc"),
                "bin_counts": sm.get("bin_counts"),
                "reliability_mean": float(np.mean(sm.get("reliability", []))) if len(sm.get("reliability", [])) else None,
                "reliability_std": float(np.std(sm.get("reliability", []))) if len(sm.get("reliability", [])) else None,
                "logvar_mean": float(np.mean(sm.get("log_variance", []))) if len(sm.get("log_variance", [])) else None,
                "logvar_std": float(np.std(sm.get("log_variance", []))) if len(sm.get("log_variance", [])) else None,
            }
        _save_log_json("reliability_calibration", compact_calib)

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

    # ========== E2: Delay-Regime Specialization (baseline vs A1) ==========
    if args.run_e2:
        print(f"\n{'='*60}")
        print("  Preparing E2 comparison model (A1)")
        print(f"{'='*60}")

        def _default_e2_ckpt():
            pattern = os.path.join(cfg.train.checkpoint_dir, f"best_robust_{args.mode}_A1*.pt")
            cands = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
            return cands[0] if cands else None

        a1_ckpt_path = args.e2_a1_ckpt or _default_e2_ckpt()
        if not a1_ckpt_path or (not os.path.isfile(a1_ckpt_path)):
            print("  E2 skipped: A1 checkpoint not found. Train A1 first or pass --e2_a1_ckpt.")
        else:
            print(f"  Loading A1 checkpoint for E2: {a1_ckpt_path}")
            model_a1 = build_model(cfg, effective_backbone, device, H, T)
            freeze_encoder_backbones(model_a1)
            apply_cli_ablation(model_a1, "A1")
            ckpt_a1 = torch.load(a1_ckpt_path, map_location=device, weights_only=False)
            load_state_dict_compatible(model_a1, ckpt_a1["model_state_dict"], module_name="e2_a1_ckpt")

            # Share eval-time transition prior if enabled.
            trans_log_prior = getattr(model, "eval_transition_log_prior", None)
            if trans_log_prior is not None:
                setattr(model_a1, "eval_transition_log_prior", trans_log_prior)

            e2_results = run_e2_delay_regime_specialization(
                model, model_a1, criterion, test_loader, device, cfg, save_dir=cfg.train.log_dir
            )
            _save_log_json("E2_delay_regime_specialization", {
                "meta": {
                    "seed": int(cfg.train.seed),
                    "mode": args.mode,
                    "baseline_ablation": args.ablation,
                    "a1_ckpt": a1_ckpt_path,
                },
                "results": e2_results,
            })

    # ========== E1 + Ablations ==========
    if args.run_all_experiments:
        run_gradient_contamination_experiment(
            model, train_loader, criterion, device, save_dir=cfg.train.log_dir
        )

        ablation_results = run_ablation_study(
            cfg, device, args.backbone, train_loader, test_loader, H, T,
            class_weights=class_weights,
        )
        results_path = os.path.join(cfg.train.log_dir, f"ablation_results_{run_tag}.json")
        serializable = {
            k: {mk: float(mv) for mk, mv in v.items()}
            for k, v in ablation_results.items()
        }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nAblation results saved: {results_path}")
        plot_ablation_results(ablation_results, save_dir=cfg.train.log_dir)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
