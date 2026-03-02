"""
Shared training and evaluation utilities for Robust M²BeamLLM.
Used by both train_robust.py and evaluate_robust.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset, WeightedRandomSampler
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List

from configs.config import Config, get_device
from models.robust_m2beamllm import RobustM2BeamLLM, RobustM2BeamLLMLoss
from utils.metrics import compute_all_metrics
from utils.stress_test import inject_asynchrony, apply_degradation
from utils.visualization import (
    plot_e2_delay_regime_specialization,
    plot_e4_reliability_monotonicity,
    plot_e4_reliability_paper_calibration,
    plot_e1_gradient_contamination,
    plot_reliability_calibration_summary,
    plot_s1_modality_delay_results,
    plot_s2_reliability_diagnostics,
    plot_s1_stress_results,
    plot_s2_stress_results,
    plot_s3_domain_shift_results,
)


# ===========================================================================
# Model construction
# ===========================================================================

def build_model(cfg: Config, backbone: str, device: torch.device,
                H: int, T: int) -> RobustM2BeamLLM:
    """Build Robust M²BeamLLM model from config."""
    model = RobustM2BeamLLM(
        feature_dim=cfg.model.feature_dim,
        num_beams=cfg.data.num_beams,
        llm_name=backbone,
        llm_hidden_dim=cfg.model.llm_hidden_dim,
        num_unfrozen_layers=cfg.model.num_unfrozen_layers,
        num_adapter_layers=getattr(cfg.model, "num_adapter_layers", 2),
        T=T, H=H,
        # C1 params
        buffer_size=cfg.async_align.buffer_size,
        staleness_embed_dim=cfg.async_align.staleness_embed_dim,
        max_staleness_ms=cfg.async_align.max_staleness_ms,
        repair_stale_threshold_ms=cfg.async_align.repair_stale_threshold_ms,
        repair_max_consecutive_missing=getattr(cfg.async_align, "repair_max_consecutive_missing", 3),
        num_delay_regimes=cfg.async_align.num_delay_regimes,
        delay_boundaries=cfg.async_align.delay_regime_boundaries,
        residual_hidden=cfg.async_align.residual_mlp_hidden,
        regime_adapter_dim=cfg.async_align.regime_adapter_dim,
        # C2 params
        variance_head_hidden=cfg.reliability.variance_head_hidden,
        reliability_min=cfg.reliability.reliability_min,
        reliability_logvar_clip=getattr(cfg.reliability, "logvar_clip", None),
        reliability_softmax_temperature=getattr(cfg.reliability, "softmax_temperature", 1.0),
        reliability_uniform_mix=getattr(cfg.reliability, "uniform_mix", 0.0),
        reliability_ema_beta=getattr(cfg.reliability, "ema_beta", 0.0),
        reliability_pair_conf_threshold=getattr(cfg.reliability, "pair_conf_threshold", 0.0),
        alignment_proj_dim=cfg.reliability.alignment_projection_dim,
        num_fusion_tokens=cfg.model.num_fusion_tokens,
        # C3 params
        num_experts=cfg.moe_lora.num_experts,
        lora_rank=cfg.moe_lora.lora_rank,
        lora_alpha=cfg.moe_lora.lora_alpha,
        lora_dropout=cfg.moe_lora.lora_dropout,
        router_extra_dim=(8 if getattr(cfg.moe_lora, "use_router_extra_context", False) else 0),
        # Fusion params
        num_attention_heads=cfg.model.num_attention_heads,
        fusion_dropout=cfg.model.fusion_dropout,
        ffn_hidden_dim=cfg.model.ffn_hidden_dim,
        # A5 baseline params
        vanilla_hidden_dim=getattr(cfg.model, "vanilla_hidden_dim", cfg.model.llm_hidden_dim),
        vanilla_num_layers=cfg.model.vanilla_num_layers,
        vanilla_num_heads=cfg.model.vanilla_num_heads,
        vanilla_ffn_hidden_dim=cfg.model.vanilla_ffn_hidden_dim,
        vanilla_dropout=cfg.model.vanilla_dropout,
    ).to(device)
    return model


def build_supervised_optimizer(model: RobustM2BeamLLM, cfg: Config) -> torch.optim.Optimizer:
    """
    Build Stage-2 optimizer with differential learning rates.
    The unfrozen pretrained LLM layers use a much smaller LR than heads/fusion.
    """
    main_params = []
    lora_params = []
    llm_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("llm_backbone.llm."):
            llm_params.append(p)
        elif "llm_backbone.moe_lora_layers" in name:
            lora_params.append(p)
        else:
            main_params.append(p)

    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": cfg.train.learning_rate})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": cfg.train.lora_learning_rate})
    if llm_params:
        param_groups.append({"params": llm_params, "lr": cfg.train.llm_learning_rate})

    def _count(params):
        return sum(p.numel() for p in params)

    print("  Stage2 optimizer groups:")
    print(
        f"    main: {len(main_params)} tensors, {_count(main_params):,} params, "
        f"lr={cfg.train.learning_rate:g}, wd={cfg.train.weight_decay:g}"
    )
    if lora_params:
        print(
            f"    lora: {len(lora_params)} tensors, {_count(lora_params):,} params, "
            f"lr={cfg.train.lora_learning_rate:g}, wd={cfg.train.weight_decay:g}"
        )
    if llm_params:
        print(
            f"    llm : {len(llm_params)} tensors, {_count(llm_params):,} params, "
            f"lr={cfg.train.llm_learning_rate:g}, wd={cfg.train.weight_decay:g}"
        )

    return torch.optim.AdamW(
        param_groups,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )


def freeze_encoder_backbones(model: RobustM2BeamLLM) -> int:
    """
    Freeze pretrained encoder backbones (Phase 2 strategy).
    Returns total number of frozen parameters.
    """
    frozen = 0
    for param in model.encoder.vision_encoder.backbone.parameters():
        param.requires_grad = False
        frozen += param.numel()
    for param in model.encoder.vision_encoder.avgpool.parameters():
        param.requires_grad = False
        frozen += param.numel()
    for param in model.encoder.lidar_encoder.backbone.parameters():
        param.requires_grad = False
        frozen += param.numel()
    for param in model.encoder.lidar_encoder.initial_conv.parameters():
        param.requires_grad = False
        frozen += param.numel()
    for param in model.encoder.lidar_encoder.global_avgpool.parameters():
        param.requires_grad = False
        frozen += param.numel()
    print(f"  Encoder backbones frozen: {frozen:,} params")
    return frozen


def set_encoder_eval_mode(model: RobustM2BeamLLM):
    """Keep frozen BN layers in eval mode during training."""
    model.encoder.vision_encoder.backbone.eval()
    model.encoder.lidar_encoder.backbone.eval()
    model.encoder.lidar_encoder.initial_conv.eval()


# ===========================================================================
# Evaluation
# ===========================================================================

@torch.no_grad()
def build_beam_transition_log_prior(loader: DataLoader,
                                    num_beams: int,
                                    pseudocount: float = 1.0) -> torch.Tensor:
    """
    Estimate first-order beam transition log-prior from train data.
    Uses beam_history + beam_future when available.
    """
    pc = float(max(pseudocount, 1e-6))
    counts = torch.full((num_beams, num_beams), pc, dtype=torch.float64, device="cpu")

    for batch in tqdm(loader, desc="Build transition prior", leave=False):
        fut = batch["beam_future"].detach().cpu().long()
        hist = batch.get("beam_history")
        if hist is not None:
            hist = hist.detach().cpu().long()
            seq = torch.cat([hist, fut], dim=1)
        else:
            seq = fut
        if seq.size(1) < 2:
            continue

        prev = seq[:, :-1].reshape(-1)
        nxt = seq[:, 1:].reshape(-1)
        valid = (prev >= 0) & (prev < num_beams) & (nxt >= 0) & (nxt < num_beams)
        prev = prev[valid]
        nxt = nxt[valid]
        if prev.numel() == 0:
            continue
        flat = prev * num_beams + nxt
        binc = torch.bincount(flat, minlength=num_beams * num_beams)
        counts += binc.reshape(num_beams, num_beams).to(dtype=counts.dtype)

    probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs.log().to(dtype=torch.float32, device="cpu")


def _sample_postprocess_context(aux: Optional[Dict],
                                staleness_ms,
                                gps_hist: Optional[torch.Tensor],
                                cfg: Config,
                                device: torch.device,
                                batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-sample context used by constrained reranking / smoothing.
    Returns (low_confidence, stale_norm, speed_norm), each shape (B,).
    """
    low_conf = torch.zeros(batch_size, device=device)
    stale_norm = torch.zeros(batch_size, device=device)
    speed_norm = torch.zeros(batch_size, device=device)

    if aux is not None and isinstance(aux, dict) and "log_variances" in aux:
        try:
            mods = [m for m in ["image", "radar", "lidar", "gps"] if m in aux["log_variances"]]
            if mods:
                lv_stack = torch.stack([aux["log_variances"][m] for m in mods], dim=-1).to(device)  # (B,H,M)
                lv_mean = lv_stack.mean(dim=(1, 2))
                clip = getattr(cfg.reliability, "logvar_clip", None)
                denom = float(clip) if (clip is not None and clip > 1e-6) else 2.5
                low_conf = torch.sigmoid(lv_mean / denom).clamp(0.0, 1.0)
        except Exception:
            pass

    if isinstance(staleness_ms, dict) and len(staleness_ms) > 0:
        try:
            vals = []
            for _, v in staleness_ms.items():
                if torch.is_tensor(v):
                    t = v.to(device=device, dtype=torch.float32)
                    if t.dim() == 0:
                        t = t.view(1).expand(batch_size)
                    elif t.dim() >= 2:
                        t = t.float().mean(dim=1)
                    vals.append(t)
                else:
                    vals.append(torch.full((batch_size,), float(v), device=device))
            if vals:
                stale_ms = torch.stack(vals, dim=0).mean(dim=0)
                max_stale = float(getattr(cfg.async_align, "max_staleness_ms", 300.0))
                stale_norm = (stale_ms / max(max_stale, 1e-6)).clamp(0.0, 1.0)
        except Exception:
            pass

    if gps_hist is not None and torch.is_tensor(gps_hist) and gps_hist.dim() == 3 and gps_hist.size(1) >= 2:
        try:
            gps_h = gps_hist.to(device=device, dtype=torch.float32)
            dxy = gps_h[:, 1:, :] - gps_h[:, :-1, :]
            speed = dxy.pow(2).sum(dim=-1).sqrt().mean(dim=1)  # normalized GPS units / frame
            med = speed.detach().median().clamp_min(1e-6)
            speed_norm = (speed / (2.0 * med)).clamp(0.0, 1.0)
        except Exception:
            pass

    return low_conf, stale_norm, speed_norm


def _beam_distance(a: torch.Tensor, b: torch.Tensor, num_beams: int) -> torch.Tensor:
    """Absolute beam-index distance normalized to [0,1]."""
    return (a.float() - b.float()).abs() / float(max(num_beams - 1, 1))


def _apply_valid_beam_mask(logits: torch.Tensor, criterion: RobustM2BeamLLMLoss) -> torch.Tensor:
    """
    Mask beam IDs that are marked invalid by criterion.valid_beam_mask.
    Returns original logits when mask is absent/mismatched.
    """
    valid_mask = getattr(criterion, "valid_beam_mask", None)
    if valid_mask is None:
        return logits
    if valid_mask.numel() != logits.size(-1):
        return logits
    vm = valid_mask.to(device=logits.device)
    return logits.masked_fill((~vm).view(1, 1, -1), -1e4)


def _extract_modal_reliability_means(aux: Optional[Dict],
                                     device: torch.device,
                                     batch_size: int) -> torch.Tensor:
    """
    Returns per-sample mean reliability weights for 4 modalities (B,4).
    Falls back to uniform if unavailable.
    """
    mods = ["image", "radar", "lidar", "gps"]
    default = torch.full((batch_size, 4), 0.25, device=device)
    if not isinstance(aux, dict):
        return default
    rw = aux.get("reliability_weights")
    if not isinstance(rw, dict):
        return default
    vals = []
    try:
        for m in mods:
            v = rw.get(m)
            if not torch.is_tensor(v):
                return default
            t = v.to(device=device, dtype=torch.float32)
            if t.dim() == 1:
                vals.append(t)
            else:
                vals.append(t.mean(dim=1))
        rel = torch.stack(vals, dim=-1)
        rel_sum = rel.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (rel / rel_sum).clamp(0.0, 1.0)
    except Exception:
        return default


def _extract_modal_staleness_means(staleness_ms,
                                   cfg: Config,
                                   device: torch.device,
                                   batch_size: int) -> torch.Tensor:
    """
    Returns normalized per-modality staleness means (B,4) in [0,1].
    """
    mods = ["image", "radar", "lidar", "gps"]
    out = torch.zeros(batch_size, 4, device=device)
    if not isinstance(staleness_ms, dict):
        return out
    denom = max(float(getattr(cfg.async_align, "max_staleness_ms", 300.0)), 1e-6)
    for j, m in enumerate(mods):
        v = staleness_ms.get(m)
        if v is None:
            continue
        try:
            if torch.is_tensor(v):
                t = v.to(device=device, dtype=torch.float32)
                if t.dim() == 0:
                    t = t.view(1).expand(batch_size)
                elif t.dim() >= 2:
                    t = t.mean(dim=1)
                out[:, j] = (t / denom).clamp(0.0, 1.0)
            else:
                out[:, j] = float(v) / denom
        except Exception:
            continue
    return out.clamp(0.0, 1.0)


def _match_feature_dim(feat: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Pad or truncate last dim to target_dim so reranker feature layout can evolve
    without breaking old checkpoints/configs.
    """
    cur = feat.size(-1)
    if cur == target_dim:
        return feat
    if cur > target_dim:
        return feat[..., :target_dim]
    pad = torch.zeros(*feat.shape[:-1], target_dim - cur, device=feat.device, dtype=feat.dtype)
    return torch.cat([feat, pad], dim=-1)


def _build_pairwise_topk_features(logits_t: torch.Tensor,
                                  topi: torch.Tensor,
                                  topv: torch.Tensor,
                                  prev_ref: torch.Tensor,
                                  hist: Optional[torch.Tensor],
                                  rel_means: torch.Tensor,
                                  modal_stale: torch.Tensor,
                                  low_conf: torch.Tensor,
                                  stale_norm: torch.Tensor,
                                  speed_norm: torch.Tensor,
                                  num_beams: int,
                                  target_dim: int) -> torch.Tensor:
    """
    Build engineered candidate features for Top-K pairwise reranker.
    Returns (B,K,F). Features include base scores, temporal continuity,
    beam-distance geometry, and reliability/staleness context.
    """
    B, K = topi.shape
    probs_t = torch.softmax(logits_t, dim=-1)
    top_probs = probs_t.gather(dim=-1, index=topi)

    rank_norm = (
        torch.arange(K, device=logits_t.device, dtype=logits_t.dtype)
        .view(1, K).expand(B, K) / float(max(K - 1, 1))
    )

    top1v = topv[:, :1]
    top2v = topv[:, 1:2] if K >= 2 else top1v
    margin_top1 = (top1v - topv).clamp_min(0.0)
    margin_top2 = (top2v - topv).clamp_min(0.0)
    margin_scale = (top1v - topv[:, -1:]).abs().clamp_min(1e-6)
    margin_top1_n = (margin_top1 / margin_scale).clamp(0.0, 2.0)
    margin_top2_n = (margin_top2 / margin_scale).clamp(0.0, 2.0)

    dist_prev = _beam_distance(topi, prev_ref.unsqueeze(1), num_beams)
    is_prev = (topi == prev_ref.unsqueeze(1)).float()
    is_neighbor = ((topi - prev_ref.unsqueeze(1)).abs() <= 1).float()
    big_jump = ((topi - prev_ref.unsqueeze(1)).abs() >= 4).float()

    dist_hist_mean = torch.zeros_like(dist_prev)
    dist_hist_last2 = torch.zeros_like(dist_prev)
    repeat_freq = torch.zeros_like(dist_prev)
    neigh_freq = torch.zeros_like(dist_prev)
    dir_consistency = torch.zeros_like(dist_prev)
    if hist is not None and hist.numel() > 0:
        histf = hist.float()
        hist_mean = histf.mean(dim=1, keepdim=True).expand(B, K)
        dist_hist_mean = _beam_distance(topi, hist_mean, num_beams)

        hist_last2 = histf[:, -min(2, hist.size(1)):].mean(dim=1, keepdim=True).expand(B, K)
        dist_hist_last2 = _beam_distance(topi, hist_last2, num_beams)

        repeat_freq = (hist.unsqueeze(-1) == topi.unsqueeze(1)).float().mean(dim=1)
        neigh_freq = ((hist.unsqueeze(-1) - topi.unsqueeze(1)).abs() <= 1).float().mean(dim=1)

        if hist.size(1) >= 2:
            prev_step_dir = (hist[:, -1] - hist[:, -2]).sign().float().unsqueeze(1)
            cand_dir = (topi - hist[:, -1:].expand(B, K)).sign().float()
            dir_consistency = (cand_dir == prev_step_dir).float()

    rel_entropy = -(rel_means.clamp_min(1e-6) * rel_means.clamp_min(1e-6).log()).sum(dim=-1)
    rel_entropy = rel_entropy / torch.log(torch.tensor(4.0, device=logits_t.device))
    rel_max = rel_means.max(dim=-1).values

    # Broadcast per-sample context to candidate dimension.
    lc = low_conf.view(B, 1).expand(B, K)
    st = stale_norm.view(B, 1).expand(B, K)
    sp = speed_norm.view(B, 1).expand(B, K)
    re = rel_entropy.view(B, 1).expand(B, K)
    rmax = rel_max.view(B, 1).expand(B, K)

    rel_b = rel_means.unsqueeze(1).expand(B, K, 4)
    stale_b = modal_stale.unsqueeze(1).expand(B, K, 4)

    feats = [
        topv,                    # 1
        top_probs,               # 1
        rank_norm,               # 1
        margin_top1_n,           # 1
        margin_top2_n,           # 1
        dist_prev,               # 1
        dist_hist_mean,          # 1
        dist_hist_last2,         # 1
        is_prev,                 # 1
        is_neighbor,             # 1
        repeat_freq,             # 1
        neigh_freq,              # 1
        dir_consistency,         # 1
        big_jump,                # 1
        lc, st, sp, re, rmax,    # 5
    ]
    feat = torch.stack(feats, dim=-1)  # (B,K,19)
    feat = torch.cat([feat, rel_b, stale_b, (lc * st).unsqueeze(-1)], dim=-1)  # +4 +4 +1 = 28
    return _match_feature_dim(feat, target_dim)


def _apply_trainable_pairwise_reranker(logits: torch.Tensor,
                                       beam_history: Optional[torch.Tensor],
                                       aux: Optional[Dict],
                                       staleness_ms,
                                       gps_hist: Optional[torch.Tensor],
                                       model: RobustM2BeamLLM,
                                       cfg: Config) -> torch.Tensor:
    """
    Inference-time trainable pairwise reranker (R1). Applies Top-K rerank scores
    as a residual logit correction. Designed to compose with Viterbi smoothing.
    """
    if not bool(getattr(cfg.train, "eval_pairwise_reranker_enable", False)):
        return logits
    reranker = getattr(model, "topk_pairwise_reranker", None)
    if reranker is None:
        return logits

    B, T, C = logits.shape
    if B == 0 or T == 0:
        return logits
    K = int(max(1, min(C, getattr(cfg.train, "eval_pairwise_reranker_k", 5))))
    base_lambda = float(max(0.0, getattr(cfg.train, "eval_pairwise_reranker_lambda", 0.0)))
    if base_lambda <= 0.0:
        return logits

    low_conf, stale_norm, speed_norm = _sample_postprocess_context(
        aux, staleness_ms, gps_hist, cfg, logits.device, B
    )
    rel_means = _extract_modal_reliability_means(aux, logits.device, B)
    modal_stale = _extract_modal_staleness_means(staleness_ms, cfg, logits.device, B)

    stale_scale = float(getattr(cfg.train, "eval_pairwise_reranker_stale_scale", 0.0))
    lowconf_scale = float(getattr(cfg.train, "eval_pairwise_reranker_lowconf_scale", 0.0))
    lam_eff = base_lambda * (1.0 + stale_scale * stale_norm + lowconf_scale * low_conf)
    lam_eff = lam_eff / (1.0 + 0.5 * speed_norm)  # allow faster motion to override smoothing/rerank bias

    hist = None
    if beam_history is not None and torch.is_tensor(beam_history) and beam_history.numel() > 0:
        hist = beam_history.to(device=logits.device).long().clamp(min=0, max=C - 1)
        prev_ref = hist[:, -1].clone()
    else:
        prev_ref = logits[:, 0].argmax(dim=-1)

    target_dim = int(getattr(cfg.model, "pairwise_reranker_feature_dim", 24))
    out = logits.clone()
    for t in range(T):
        topv, topi = out[:, t].topk(K, dim=-1)
        feat = _build_pairwise_topk_features(
            out[:, t], topi, topv, prev_ref,
            hist, rel_means, modal_stale,
            low_conf, stale_norm, speed_norm,
            C, target_dim,
        )
        rerank_scores = reranker(feat).to(dtype=out.dtype)  # (B,K)
        bonus = lam_eff.unsqueeze(1).to(dtype=out.dtype) * rerank_scores
        out[:, t].scatter_add_(dim=-1, index=topi, src=bonus)
        prev_ref = out[:, t].argmax(dim=-1)
    return out


def compute_pairwise_reranker_loss(model: RobustM2BeamLLM,
                                   predictions: torch.Tensor,
                                   targets: torch.Tensor,
                                   beam_history: Optional[torch.Tensor],
                                   aux: Optional[Dict],
                                   staleness_ms,
                                   gps_hist: Optional[torch.Tensor],
                                   cfg: Config,
                                   epoch: int = 1,
                                   scale_override: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Train-time pairwise hinge loss for Top-K reranker (R1).
    Uses detached base logits so the candidate generator remains the primary model.
    """
    if not bool(getattr(cfg.model, "use_pairwise_reranker", False)):
        z = predictions.new_zeros(())
        return z, {"loss": 0.0, "coeff": 0.0, "coverage": 0.0, "hardcase": 0.0}
    reranker = getattr(model, "topk_pairwise_reranker", None)
    if reranker is None:
        z = predictions.new_zeros(())
        return z, {"loss": 0.0, "coeff": 0.0, "coverage": 0.0, "hardcase": 0.0}

    B, T, C = predictions.shape
    K = int(max(1, min(C, getattr(cfg.train, "pairwise_reranker_k", 5))))
    base_coeff = float(max(0.0, getattr(cfg.train, "pairwise_reranker_lambda", 0.0)))
    if scale_override is None:
        delay = int(max(0, getattr(cfg.train, "pairwise_reranker_delay_epochs", 0)))
        warm = int(max(0, getattr(cfg.train, "pairwise_reranker_warmup_epochs", 0)))
        if epoch <= delay:
            ramp = 0.0
        elif warm <= 0:
            ramp = 1.0
        else:
            ramp = min(1.0, max(0.0, (epoch - delay - 1) / float(warm)))
        coeff = base_coeff * ramp
    else:
        coeff = base_coeff * max(float(scale_override), 0.0)
    if coeff <= 0.0:
        z = predictions.new_zeros(())
        return z, {"loss": 0.0, "coeff": coeff, "coverage": 0.0, "hardcase": 0.0}

    margin = float(getattr(cfg.train, "pairwise_reranker_margin", 0.2))
    hard_w = float(max(1.0, getattr(cfg.train, "pairwise_reranker_hardcase_weight", 2.0)))
    target_dim = int(getattr(cfg.model, "pairwise_reranker_feature_dim", 24))

    logits = predictions.detach()  # keep candidate generator untouched by pairwise loss
    low_conf, stale_norm, speed_norm = _sample_postprocess_context(
        aux, staleness_ms, gps_hist, cfg, logits.device, B
    )
    rel_means = _extract_modal_reliability_means(aux, logits.device, B)
    modal_stale = _extract_modal_staleness_means(staleness_ms, cfg, logits.device, B)

    hist = None
    if beam_history is not None and torch.is_tensor(beam_history) and beam_history.numel() > 0:
        hist = beam_history.to(device=logits.device).long().clamp(min=0, max=C - 1)
        prev_ref = hist[:, -1].clone()
    else:
        prev_ref = logits[:, 0].argmax(dim=-1)

    losses = []
    covered = 0.0
    hardcase = 0.0
    total_steps = float(B * T)

    for t in range(T):
        topv, topi = logits[:, t].topk(K, dim=-1)  # (B,K)
        feat = _build_pairwise_topk_features(
            logits[:, t], topi, topv, prev_ref,
            hist, rel_means, modal_stale,
            low_conf, stale_norm, speed_norm,
            C, target_dim,
        )
        scores = reranker(feat)  # (B,K)
        tgt = targets[:, t].long()
        hit = (topi == tgt.unsqueeze(1))
        hit_any = hit.any(dim=1)
        covered += float(hit_any.float().sum().item())

        top1_wrong = (topi[:, 0] != tgt)
        hard_mask = hit_any & top1_wrong
        hardcase += float(hard_mask.float().sum().item())

        if hit_any.any():
            pos_idx = hit.float().argmax(dim=1)
            pos_scores = scores.gather(1, pos_idx.unsqueeze(1)).squeeze(1)
            neg_scores = scores.masked_fill(hit, float("-inf")).max(dim=1).values
            step_loss = F.relu(margin - pos_scores + neg_scores)
            weights = torch.where(hard_mask, torch.full_like(step_loss, hard_w), torch.ones_like(step_loss))
            step_loss = step_loss * weights
            valid = hit_any.float()
            denom = valid.sum().clamp_min(1.0)
            losses.append((step_loss * valid).sum() / denom)

        # Keep temporal context aligned with base generator trajectory during training.
        prev_ref = logits[:, t].argmax(dim=-1)

    if len(losses) == 0:
        z = predictions.new_zeros(())
        return z, {"loss": 0.0, "coeff": coeff, "coverage": covered / max(total_steps, 1.0), "hardcase": hardcase / max(total_steps, 1.0)}

    raw_loss = torch.stack(losses).mean()
    return coeff * raw_loss, {
        "loss": float(raw_loss.detach().item()),
        "coeff": coeff,
        "coverage": covered / max(total_steps, 1.0),
        "hardcase": hardcase / max(total_steps, 1.0),
    }


def _apply_constrained_topk_reranker(logits: torch.Tensor,
                                     beam_history: Optional[torch.Tensor],
                                     aux: Optional[Dict],
                                     staleness_ms,
                                     gps_hist: Optional[torch.Tensor],
                                     cfg: Config) -> torch.Tensor:
    """Heuristic constrained Top-K reranker (Layer 2), feature-based and inference-time only."""
    if not bool(getattr(cfg.train, "eval_constrained_reranker_enable", False)):
        return logits

    B, T, C = logits.shape
    if B == 0 or T == 0:
        return logits

    K = int(max(1, min(C, getattr(cfg.train, "eval_constrained_reranker_k", 5))))
    base_lambda = float(max(0.0, getattr(cfg.train, "eval_constrained_reranker_lambda", 0.0)))
    if base_lambda <= 0.0:
        return logits

    rank_penalty = float(getattr(cfg.train, "eval_rerank_rank_penalty", 0.0))
    dprev_penalty = float(getattr(cfg.train, "eval_rerank_dist_prev_penalty", 0.0))
    dhist_penalty = float(getattr(cfg.train, "eval_rerank_dist_hist_penalty", 0.0))
    neigh_bonus = float(getattr(cfg.train, "eval_rerank_neighbor_bonus", 0.0))
    repeat_bonus = float(getattr(cfg.train, "eval_rerank_hist_repeat_bonus", 0.0))
    margin_penalty = float(getattr(cfg.train, "eval_rerank_margin_penalty", 0.0))
    stale_scale = float(getattr(cfg.train, "eval_rerank_staleness_scale", 0.0))
    lowconf_scale = float(getattr(cfg.train, "eval_rerank_lowconf_scale", 0.0))

    low_conf, stale_norm, _ = _sample_postprocess_context(
        aux, staleness_ms, gps_hist, cfg, logits.device, B
    )
    lam_eff = base_lambda * (1.0 + stale_scale * stale_norm + lowconf_scale * low_conf)  # (B,)

    hist = None
    hist_mean = None
    if beam_history is not None and torch.is_tensor(beam_history) and beam_history.numel() > 0:
        hist = beam_history.to(device=logits.device).long().clamp(min=0, max=C - 1)
        hist_mean = hist.float().mean(dim=1, keepdim=True)

    if hist is not None:
        prev_ref = hist[:, -1].clone()
    else:
        prev_ref = logits[:, 0].argmax(dim=-1)

    rank_norm_template = (
        torch.arange(K, device=logits.device, dtype=logits.dtype)
        / float(max(K - 1, 1))
    )

    out = logits.clone()
    for t in range(T):
        topv, topi = out[:, t].topk(K, dim=-1)  # (B,K)
        top1 = topv[:, :1]
        margin = (top1 - topv).clamp_min(0.0)
        denom = margin[:, -1:].clamp_min(1e-6)
        margin_norm = (margin / denom).clamp(0.0, 1.0)
        rank_norm = rank_norm_template.view(1, K).expand(B, K)

        dist_prev = _beam_distance(topi, prev_ref.unsqueeze(1), C)
        dist_hist = _beam_distance(topi, hist_mean.expand(B, K), C) if hist_mean is not None else torch.zeros_like(dist_prev)
        is_neighbor = (topi == prev_ref.unsqueeze(1)) | ((topi - prev_ref.unsqueeze(1)).abs() <= 1)
        neighbor_feat = is_neighbor.float()

        repeat_feat = torch.zeros_like(dist_prev)
        if hist is not None:
            exact_rep = (hist.unsqueeze(-1) == topi.unsqueeze(1)).float().mean(dim=1)
            repeat_feat = exact_rep

        score_feat = (
            - rank_penalty * rank_norm
            - dprev_penalty * dist_prev
            - dhist_penalty * dist_hist
            + neigh_bonus * neighbor_feat
            + repeat_bonus * repeat_feat
            - margin_penalty * margin_norm
        )
        bonus = lam_eff.unsqueeze(1).to(dtype=out.dtype) * score_feat
        out[:, t].scatter_add_(dim=-1, index=topi, src=bonus)
        prev_ref = out[:, t].argmax(dim=-1)

    return out


def _apply_viterbi_topk_smoother(logits: torch.Tensor,
                                 beam_history: Optional[torch.Tensor],
                                 aux: Optional[Dict],
                                 staleness_ms,
                                 gps_hist: Optional[torch.Tensor],
                                 cfg: Config) -> torch.Tensor:
    """Viterbi-style temporal smoothing over top-K lattice (Layer 3)."""
    if not bool(getattr(cfg.train, "eval_viterbi_smoother_enable", False)):
        return logits

    B, T, C = logits.shape
    if B == 0 or T <= 1:
        return logits

    K = int(max(1, min(C, getattr(cfg.train, "eval_viterbi_k", 5))))
    path_bonus = float(getattr(cfg.train, "eval_viterbi_path_bonus", 0.0))
    if path_bonus <= 0.0:
        return logits

    alpha0 = float(getattr(cfg.train, "eval_viterbi_base_alpha", 0.0))
    beta0 = float(getattr(cfg.train, "eval_viterbi_big_jump_penalty", 0.0))
    jump_thr = int(max(1, getattr(cfg.train, "eval_viterbi_big_jump_threshold", 4)))
    stale_gain = float(getattr(cfg.train, "eval_viterbi_staleness_gain", 0.0))
    lowconf_gain = float(getattr(cfg.train, "eval_viterbi_lowconf_gain", 0.0))
    speed_relax = float(getattr(cfg.train, "eval_viterbi_speed_relax_gain", 0.0))

    low_conf, stale_norm, speed_norm = _sample_postprocess_context(
        aux, staleness_ms, gps_hist, cfg, logits.device, B
    )
    alpha_eff = alpha0 * (1.0 + stale_gain * stale_norm + lowconf_gain * low_conf) / (1.0 + speed_relax * speed_norm)
    beta_eff = beta0 * (1.0 + 0.5 * stale_norm + 0.5 * low_conf)

    out = logits.clone()
    topv, topi = out.topk(K, dim=-1)  # (B,T,K)

    hist_last = None
    if beam_history is not None and torch.is_tensor(beam_history) and beam_history.numel() > 0:
        hist_last = beam_history.to(device=out.device).long().clamp(min=0, max=C - 1)[:, -1]

    for b in range(B):
        cand_ids = topi[b]   # (T,K)
        emit = topv[b]       # (T,K)
        dp = torch.full((T, K), float("-inf"), device=out.device, dtype=out.dtype)
        back = torch.zeros((T, K), device=out.device, dtype=torch.long)

        # Initial node scores: optionally penalize large jump from last historical beam.
        init = emit[0].clone()
        if hist_last is not None:
            d0 = _beam_distance(cand_ids[0], hist_last[b].expand_as(cand_ids[0]), C)
            init = init - alpha_eff[b].to(dtype=out.dtype) * d0
        dp[0] = init

        for t in range(1, T):
            prev_ids = cand_ids[t - 1].view(K, 1)
            cur_ids = cand_ids[t].view(1, K)
            dist = _beam_distance(prev_ids, cur_ids, C)  # (K,K)
            big_jump = ((prev_ids - cur_ids).abs() > jump_thr).float()
            trans = (
                - alpha_eff[b].to(dtype=out.dtype) * dist
                - beta_eff[b].to(dtype=out.dtype) * big_jump
            )  # (K,K)

            scores = dp[t - 1].view(K, 1) + trans  # (K,K)
            best_prev_scores, best_prev_idx = scores.max(dim=0)
            dp[t] = emit[t] + best_prev_scores
            back[t] = best_prev_idx

        path_pos = torch.zeros(T, dtype=torch.long, device=out.device)
        path_pos[-1] = dp[-1].argmax()
        for t in range(T - 2, -1, -1):
            path_pos[t] = back[t + 1, path_pos[t + 1]]
        path_ids = cand_ids[torch.arange(T, device=out.device), path_pos]  # (T,)

        out[b, torch.arange(T, device=out.device), path_ids] += path_bonus

    return out


def _maybe_postprocess_eval_predictions(predictions: torch.Tensor,
                                        beam_history: Optional[torch.Tensor],
                                        aux: Optional[Dict],
                                        staleness_ms,
                                        gps_hist: Optional[torch.Tensor],
                                        model: RobustM2BeamLLM,
                                        cfg: Config) -> torch.Tensor:
    """
    Optional inference-time postprocessing:
    - temporal smoothing (causal EMA on logits across future steps)
    - transition-prior rerank (logit correction using Markov transition prior)
    """
    use_ts = bool(getattr(cfg.train, "eval_temporal_smoothing_enable", False))
    use_rr = bool(getattr(cfg.train, "eval_transition_rerank_enable", False))
    use_cr = bool(getattr(cfg.train, "eval_constrained_reranker_enable", False))
    use_pr = bool(getattr(cfg.train, "eval_pairwise_reranker_enable", False))
    use_vs = bool(getattr(cfg.train, "eval_viterbi_smoother_enable", False))
    if not (use_ts or use_rr or use_cr or use_pr or use_vs):
        return predictions

    logits = predictions.clone()
    B, T, C = logits.shape

    if use_ts and T > 1:
        alpha = float(getattr(cfg.train, "eval_temporal_smoothing_alpha", 0.0))
        alpha = max(0.0, min(1.0, alpha))
        if alpha > 0.0:
            smoothed = logits.clone()
            for t in range(1, T):
                smoothed[:, t] = (1.0 - alpha) * smoothed[:, t] + alpha * smoothed[:, t - 1]
            logits = smoothed

    if use_rr:
        trans_log_prior = getattr(model, "eval_transition_log_prior", None)
        if trans_log_prior is None:
            trans_log_prior = getattr(cfg.train, "_eval_transition_log_prior", None)
        if trans_log_prior is not None:
            trans_log_prior = trans_log_prior.to(device=logits.device, dtype=logits.dtype)
            lam = float(getattr(cfg.train, "eval_transition_rerank_lambda", 0.0))
            lam = max(0.0, lam)
            prev_temp = float(getattr(cfg.train, "eval_transition_rerank_prev_temp", 1.0))
            prev_temp = max(prev_temp, 1e-4)
            if lam > 0.0:
                if beam_history is not None and beam_history.numel() > 0:
                    last_beam = beam_history[:, -1].long().clamp(min=0, max=C - 1)
                    logits[:, 0] = logits[:, 0] + lam * trans_log_prior[last_beam]
                for t in range(1, T):
                    prev_probs = torch.softmax(logits[:, t - 1] / prev_temp, dim=-1)
                    logits[:, t] = logits[:, t] + lam * (prev_probs @ trans_log_prior)

    if use_cr:
        logits = _apply_constrained_topk_reranker(
            logits, beam_history, aux, staleness_ms, gps_hist, cfg
        )
    if use_pr:
        logits = _apply_trainable_pairwise_reranker(
            logits, beam_history, aux, staleness_ms, gps_hist, model, cfg
        )
    if use_vs:
        logits = _apply_viterbi_topk_smoother(
            logits, beam_history, aux, staleness_ms, gps_hist, cfg
        )

    return logits

@torch.no_grad()
def evaluate(model: RobustM2BeamLLM,
             criterion: RobustM2BeamLLMLoss,
             loader: DataLoader,
             device: torch.device,
             cfg: Config,
             ablation: str = "none",
             staleness_ms: Optional[Dict[str, float]] = None,
             missing_masks: Optional[Dict] = None,
             burst_length: int = 0,
             burst_modalities: Optional[list] = None,
             degradation: Optional[Dict] = None) -> Dict[str, float]:
    """
    Evaluate model on test set with optional stress conditions.

    Args:
        model: trained model
        criterion: loss function
        loader: test DataLoader
        device: torch device
        cfg: full config
        ablation: ablation condition name ("none", "A1", ..., "A5")
        staleness_ms: Dict[modality->delay_ms] for S1 stress test
        missing_masks: not used directly (computed from staleness)
        degradation: Dict with 'severity' and 'target_modalities' for S2 stress
    Returns:
        metrics: Dict of all evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_powers = []
    total_loss = 0.0
    total_samples = 0
    repeat_last_total_correct = 0.0
    repeat_last_total_count = 0.0
    repeat_last_step_correct = None
    repeat_last_step_count = None
    mods = ["image", "radar", "lidar", "gps"]
    diag_stats = {
        "rel_sum": {m: 0.0 for m in mods},
        "rel_sq_sum": {m: 0.0 for m in mods},
        "logvar_sum": {m: 0.0 for m in mods},
        "logvar_sq_sum": {m: 0.0 for m in mods},
        "cue_sum": {m: 0.0 for m in mods},
        "cue_sq_sum": {m: 0.0 for m in mods},
        "count": 0.0,
        "max_w_sum": 0.0,
        "entropy_sum": 0.0,
        "obs_keep_sum": 0.0,
    }

    use_reliability_align = (ablation != "A3")

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["images"].to(device)
        radars = batch["radars"].to(device)
        lidars = batch["lidars"].to(device)
        gps = batch["gps"].to(device)
        targets = batch["beam_future"].to(device)
        beam_history = batch.get("beam_history")
        if beam_history is not None:
            beam_history = beam_history.to(device)
        power_future = batch.get("power_future")
        if power_future is not None:
            power_future = power_future.to(device)

        # Build modifiable batch dict
        batch_dict = {
            "images": images,
            "radars": radars,
            "lidars": lidars,
            "gps": gps,
        }
        s_ms = None
        m_masks = None

        # S1: Asynchrony stress
        if staleness_ms is not None or burst_length > 0:
            batch_dict, s_ms, m_masks = inject_asynchrony(
                batch_dict,
                delay_ms=staleness_ms,
                burst_length=burst_length,
                burst_modalities=burst_modalities,
            )

        # S2: Degradation stress
        if degradation is not None:
            deg_kwargs = dict(degradation)
            deg_kwargs.setdefault("data_root", cfg.data.data_root)
            batch_dict = apply_degradation(batch_dict, **deg_kwargs)

        # Forward
        predictions, aux = model(
            batch_dict["images"], batch_dict["radars"],
            batch_dict["lidars"], batch_dict["gps"],
            staleness_ms=s_ms, missing_masks=m_masks,
            degradation_cues=batch_dict.get("_degradation_cues"),
            beam_history=beam_history,
        )
        loss, _ = criterion(
            predictions, targets, aux,
            use_reliability_align=use_reliability_align,
        )
        predictions_eval = predictions

        if "reliability_weights" in aux and "log_variances" in aux:
            rel_stack = []
            for mod in mods:
                w = aux["reliability_weights"][mod].detach()
                lv = aux["log_variances"][mod].detach()
                qc = aux.get("quality_cues", {}).get(mod)
                if qc is None:
                    qc = torch.zeros_like(w).unsqueeze(-1)
                qc = qc.detach().squeeze(-1)
                diag_stats["rel_sum"][mod] += float(w.sum().item())
                diag_stats["rel_sq_sum"][mod] += float((w * w).sum().item())
                diag_stats["logvar_sum"][mod] += float(lv.sum().item())
                diag_stats["logvar_sq_sum"][mod] += float((lv * lv).sum().item())
                diag_stats["cue_sum"][mod] += float(qc.sum().item())
                diag_stats["cue_sq_sum"][mod] += float((qc * qc).sum().item())
                rel_stack.append(w)
            rel_stack = torch.stack(rel_stack, dim=-1)  # (B,H,4)
            diag_stats["count"] += float(rel_stack.numel() / len(mods))
            diag_stats["max_w_sum"] += float(rel_stack.max(dim=-1).values.sum().item())
            entropy = -(rel_stack.clamp_min(1e-8) * rel_stack.clamp_min(1e-8).log()).sum(dim=-1)
            diag_stats["entropy_sum"] += float(entropy.sum().item())
            obs_keep = aux.get("observation_keep")
            if obs_keep is not None:
                diag_stats["obs_keep_sum"] += float(obs_keep.detach().sum().item())

        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)
        all_preds.append(predictions_eval.cpu())
        all_targets.append(targets.cpu())
        if power_future is not None:
            all_powers.append(power_future.cpu())

        # Strong temporal baseline: repeat the last historical beam for all future steps.
        if beam_history is not None and beam_history.numel() > 0:
            last_beam = beam_history[:, -1].long()
            repeat_pred = last_beam.unsqueeze(1).expand_as(targets)
            corr = (repeat_pred == targets).float()
            repeat_last_total_correct += float(corr.sum().item())
            repeat_last_total_count += float(corr.numel())
            step_sum = corr.sum(dim=0).detach().cpu()
            if repeat_last_step_correct is None:
                repeat_last_step_correct = step_sum.clone()
                repeat_last_step_count = torch.full_like(step_sum, corr.size(0), dtype=torch.float32)
            else:
                repeat_last_step_correct += step_sum
                repeat_last_step_count += torch.full_like(step_sum, corr.size(0), dtype=torch.float32)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_powers_tensor = torch.cat(all_powers, dim=0) if all_powers else None

    metrics = compute_all_metrics(
        all_preds, all_targets,
        top_k_values=cfg.train.top_k_values,
        delta_values=cfg.train.dba_delta_values,
        compute_tail=True,
        beam_powers=all_powers_tensor,
    )
    metrics["loss"] = total_loss / max(total_samples, 1)
    if repeat_last_total_count > 0:
        metrics["repeat_last_top_1_acc"] = float(
            repeat_last_total_correct / max(repeat_last_total_count, 1.0)
        )
        if repeat_last_step_correct is not None and repeat_last_step_count is not None:
            T = int(repeat_last_step_correct.numel())
            for t in range(T):
                metrics[f"repeat_last_step_{t+1}_acc"] = float(
                    repeat_last_step_correct[t].item() / max(repeat_last_step_count[t].item(), 1.0)
                )
            if "top_1_acc" in metrics:
                metrics["top1_minus_repeat_last"] = float(
                    metrics["top_1_acc"] - metrics["repeat_last_top_1_acc"]
                )
            if "step_1_acc" in metrics:
                metrics["step1_minus_repeat_last"] = float(
                    metrics["step_1_acc"] - metrics["repeat_last_step_1_acc"]
                )
    if diag_stats["count"] > 0:
        denom = max(diag_stats["count"], 1.0)
        for mod in mods:
            rel_mean = diag_stats["rel_sum"][mod] / denom
            rel_var = max(diag_stats["rel_sq_sum"][mod] / denom - rel_mean * rel_mean, 0.0)
            lv_mean = diag_stats["logvar_sum"][mod] / denom
            lv_var = max(diag_stats["logvar_sq_sum"][mod] / denom - lv_mean * lv_mean, 0.0)
            qc_mean = diag_stats["cue_sum"][mod] / denom
            qc_var = max(diag_stats["cue_sq_sum"][mod] / denom - qc_mean * qc_mean, 0.0)
            metrics[f"reliability_mean_{mod}"] = float(rel_mean)
            metrics[f"reliability_std_{mod}"] = float(rel_var ** 0.5)
            metrics[f"logvar_mean_{mod}"] = float(lv_mean)
            metrics[f"logvar_std_{mod}"] = float(lv_var ** 0.5)
            metrics[f"quality_cue_mean_{mod}"] = float(qc_mean)
            metrics[f"quality_cue_std_{mod}"] = float(qc_var ** 0.5)
        metrics["reliability_max_mean"] = float(diag_stats["max_w_sum"] / denom)
        metrics["reliability_entropy_mean"] = float(diag_stats["entropy_sum"] / denom)
        metrics["observation_keep_mean"] = float(diag_stats["obs_keep_sum"] / max(total_samples, 1))
    return metrics


# ===========================================================================
# Stress Test Runners (C5)
# ===========================================================================

def run_stress_test_s1(model: RobustM2BeamLLM,
                       criterion: RobustM2BeamLLMLoss,
                       test_loader: DataLoader,
                       device: torch.device,
                       cfg: Config) -> Dict:
    """S1: Asynchrony stress test — inject delays and missing bursts."""
    print(f"\n{'='*60}")
    print(f"  S1: Asynchrony Stress Test")
    print(f"{'='*60}")

    results = {}

    # Uniform delay across all modalities
    for delay in cfg.stress_test.delay_values_ms:
        delay_config = {m: delay for m in ["image", "radar", "lidar", "gps"]}
        metrics = evaluate(model, criterion, test_loader, device, cfg,
                           staleness_ms=delay_config)
        results[f"delay_{int(delay)}ms"] = metrics
        print(f"  Delay {int(delay):4d} ms | "
              f"Top-1: {metrics['top_1_acc']*100:.1f}% | "
              f"CVaR-10: {metrics.get('cvar_10', 0):.4f} | "
              f"Worst-10%: {metrics.get('worst_10_acc', 0)*100:.1f}%")

    # Modality-specific delay (paper-faithful S1 variant; disabled by default due runtime)
    if bool(getattr(cfg.stress_test, "s1_include_modality_specific_delays", False)):
        print(f"\n  Modality-specific delay test:")
        mods = ["image", "radar", "lidar", "gps"]
        for mod in mods:
            for delay in cfg.stress_test.delay_values_ms:
                if float(delay) <= 0:
                    continue
                delay_config = {m: 0.0 for m in mods}
                delay_config[mod] = float(delay)
                metrics = evaluate(model, criterion, test_loader, device, cfg,
                                   staleness_ms=delay_config)
                key = f"delay_{mod}_{int(delay)}ms"
                results[key] = metrics
                print(f"    {mod:<5s} delay={int(delay):4d} ms | "
                      f"Top-1: {metrics['top_1_acc']*100:.1f}% | "
                      f"Worst-10%: {metrics.get('worst_10_acc', 0)*100:.1f}%")

    # Burst missing test
    print(f"\n  Burst Missing Test:")
    for burst_len in cfg.stress_test.burst_lengths:
        metrics = evaluate(
            model, criterion, test_loader, device, cfg,
            staleness_ms={m: 0.0 for m in ["image", "radar", "lidar", "gps"]},
            burst_length=burst_len,
            burst_modalities=["image", "radar", "lidar", "gps"],
        )
        results[f"burst_{burst_len}"] = metrics
        print(f"  Burst len={burst_len:2d} | Top-1: {metrics['top_1_acc']*100:.1f}%")

    plot_s1_stress_results(results, save_dir=cfg.train.log_dir)
    plot_s1_modality_delay_results(results, save_dir=cfg.train.log_dir)
    return results


def run_stress_test_s2(model: RobustM2BeamLLM,
                       criterion: RobustM2BeamLLMLoss,
                       test_loader: DataLoader,
                       device: torch.device,
                       cfg: Config) -> Dict:
    """S2: Degradation stress test — apply modality corruption."""
    print(f"\n{'='*60}")
    print(f"  S2: Degradation Stress Test")
    print(f"{'='*60}")

    results = {}

    # Single modality corruption
    for mod in ["image", "radar", "lidar", "gps"]:
        print(f"\n  Single modality: {mod}")
        for alpha in cfg.stress_test.corruption_severities:
            metrics = evaluate(
                model, criterion, test_loader, device, cfg,
                degradation={
                    "severity": alpha,
                    "target_modalities": [mod],
                    "stress_config": cfg.stress_test,
                },
            )
            results[f"{mod}_alpha_{alpha:.1f}"] = metrics
            rel_mean = metrics.get(f"reliability_mean_{mod}", 0.0)
            print(f"    α={alpha:.1f} | "
                  f"Top-1: {metrics['top_1_acc']*100:.1f}% | "
                  f"Worst-10%: {metrics.get('worst_10_acc', 0)*100:.1f}% | "
                  f"w_{mod}: {rel_mean:.3f}")

    # Combined corruption (all modalities)
    print(f"\n  Combined corruption (all modalities):")
    for alpha in cfg.stress_test.corruption_severities:
        metrics = evaluate(
            model, criterion, test_loader, device, cfg,
            degradation={
                "severity": alpha,
                "target_modalities": None,
                "stress_config": cfg.stress_test,
            },
        )
        results[f"combined_alpha_{alpha:.1f}"] = metrics
        rel_summary = "/".join(
            f"{m[:3]}={metrics.get(f'reliability_mean_{m}', 0.0):.2f}"
            for m in ["image", "radar", "lidar", "gps"]
        )
        print(f"    α={alpha:.1f} | "
              f"Top-1: {metrics['top_1_acc']*100:.1f}% | "
              f"CVaR-10: {metrics.get('cvar_10', 0):.4f} | "
              f"{rel_summary}")

    plot_s2_stress_results(results, save_dir=cfg.train.log_dir)
    plot_s2_reliability_diagnostics(results, save_dir=cfg.train.log_dir)
    return results


def _build_subset_weighted_sampler(base_loader: DataLoader,
                                   indices: List[int],
                                   H: int,
                                   T: int,
                                   power: float = 0.5) -> Optional[WeightedRandomSampler]:
    ds = getattr(base_loader, "dataset", None)
    beams = getattr(ds, "beams", None) if ds is not None else None
    if beams is None:
        return None
    future = np.array(beams[:, H:H + T], dtype=np.int64)
    if future.size == 0:
        return None
    counts = np.bincount(future.reshape(-1), minlength=int(future.max()) + 1 if future.size else 1).astype(np.float64)
    freq = counts / max(counts.sum(), 1.0)
    freq = np.clip(freq, 1e-8, None)
    inv = 1.0 / np.power(freq, max(float(power), 0.0))
    sample_w = inv[future].mean(axis=1)
    sample_w = sample_w[indices]
    sample_w = sample_w / max(sample_w.mean(), 1e-8)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )


def _make_subset_loader(base_loader: DataLoader, indices: List[int],
                        shuffle: bool = False,
                        sampler: Optional[WeightedRandomSampler] = None) -> DataLoader:
    """Create a DataLoader over a subset while preserving key loader settings."""
    subset = Subset(base_loader.dataset, indices)
    return DataLoader(
        subset,
        batch_size=base_loader.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=base_loader.num_workers,
        pin_memory=base_loader.pin_memory,
        drop_last=False,
    )


def subsample_loader_by_fraction(base_loader: DataLoader,
                                 fraction: float,
                                 seed: int = 0,
                                 shuffle: bool = True,
                                 weighted_sampling: bool = False,
                                 H: Optional[int] = None,
                                 T: Optional[int] = None,
                                 sampler_power: float = 0.5) -> DataLoader:
    """
    Deterministically subsample a DataLoader's dataset for label-efficiency experiments.
    fraction in (0, 1]; values >=1 return the original loader.
    """
    frac = float(fraction)
    if frac >= 1.0:
        return base_loader
    if frac <= 0.0:
        raise ValueError(f"fraction must be > 0, got {fraction}")

    n = len(base_loader.dataset)
    keep = max(1, int(round(n * frac)))
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    indices = perm[:keep]
    sampler = None
    if weighted_sampling and H is not None and T is not None:
        sampler = _build_subset_weighted_sampler(
            base_loader, indices, H=H, T=T, power=sampler_power
        )
    return _make_subset_loader(base_loader, indices, shuffle=shuffle, sampler=sampler)


def split_domain_shift_loaders(test_loader: DataLoader,
                               holdout_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Build in-domain and out-of-domain loaders.
    Preference order:
      1) domain_ids metadata (sequence/scenario IDs) if available
      2) temporal holdout fallback
    """
    n = len(test_loader.dataset)
    if n < 2:
        return test_loader, test_loader

    # Preferred: hold out unseen domain IDs if dataset metadata is available.
    domain_ids = getattr(test_loader.dataset, "domain_ids", None)
    if domain_ids is not None:
        dom = np.array(domain_ids)
        if dom.ndim >= 1 and len(dom) == n:
            uniq = sorted(np.unique(dom).tolist())
            if len(uniq) >= 2:
                holdout_domains = max(1, int(round(len(uniq) * holdout_ratio)))
                holdout_domains = min(holdout_domains, len(uniq) - 1)
                out_domains = set(uniq[-holdout_domains:])
                in_indices = [i for i, d in enumerate(dom.tolist()) if d not in out_domains]
                out_indices = [i for i, d in enumerate(dom.tolist()) if d in out_domains]
                if in_indices and out_indices:
                    return (
                        _make_subset_loader(test_loader, in_indices, shuffle=False),
                        _make_subset_loader(test_loader, out_indices, shuffle=False),
                    )

    holdout = max(1, int(n * holdout_ratio))
    holdout = min(holdout, n - 1)

    in_indices = list(range(0, n - holdout))
    out_indices = list(range(n - holdout, n))
    in_loader = _make_subset_loader(test_loader, in_indices, shuffle=False)
    out_loader = _make_subset_loader(test_loader, out_indices, shuffle=False)
    return in_loader, out_loader


def run_stress_test_s3(model: RobustM2BeamLLM,
                       criterion: RobustM2BeamLLMLoss,
                       test_loader: DataLoader,
                       device: torch.device,
                       cfg: Config) -> Dict:
    """S3: Domain-shift stress via temporal holdout split."""
    print(f"\n{'='*60}")
    print(f"  S3: Domain Shift Stress Test")
    print(f"{'='*60}")

    in_loader, out_loader = split_domain_shift_loaders(
        test_loader, holdout_ratio=cfg.stress_test.domain_shift_holdout
    )

    in_metrics = evaluate(model, criterion, in_loader, device, cfg)
    out_metrics = evaluate(model, criterion, out_loader, device, cfg)

    acc_in = in_metrics["top_1_acc"]
    acc_out = out_metrics["top_1_acc"]
    drop = acc_in - acc_out

    print(f"  In-domain   Top-1: {acc_in*100:.1f}%")
    print(f"  Out-domain  Top-1: {acc_out*100:.1f}%")
    print(f"  Accuracy drop: {drop*100:.1f}%")

    results = {
        "in_domain": in_metrics,
        "out_domain": out_metrics,
        "domain_shift_drop_top1": drop,
    }
    plot_s3_domain_shift_results(results, save_dir=cfg.train.log_dir)
    return results


# ===========================================================================
# E1: Gradient Contamination
# ===========================================================================

def run_gradient_contamination_experiment(
    model: RobustM2BeamLLM,
    train_loader: DataLoader,
    criterion: RobustM2BeamLLMLoss,
    device: torch.device,
    save_dir: str = "logs",
) -> Dict:
    """E1: Compare alignment gradient norms under degradation."""
    from utils.stress_test import collect_gradient_norms

    print(f"\n{'='*60}")
    print(f"  E1: Gradient Contamination Visualization")
    print(f"{'='*60}")

    results = {"weighted": [], "unweighted": []}

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:
            break
        for severity in [0.0, 0.2, 0.5, 0.8, 1.0]:
            gnorms_weighted = collect_gradient_norms(
                model, batch, criterion, device,
                degraded_modality="image", severity=severity,
                use_reliability=True,
            )
            gnorms_unweighted = collect_gradient_norms(
                model, batch, criterion, device,
                degraded_modality="image", severity=severity,
                use_reliability=False,
            )
            results["weighted"].append({"severity": severity, **gnorms_weighted})
            results["unweighted"].append({"severity": severity, **gnorms_unweighted})

    print(f"\n  {'Severity':>10} | {'image rw':>12} | {'image unweighted':>16} | {'unw/rw':>8}")
    print(f"  {'-'*10} | {'-'*12} | {'-'*16} | {'-'*8}")
    for w_entry, u_entry in zip(results["weighted"], results["unweighted"]):
        g_w = w_entry.get("grad_norm_image", 0.0)
        g_u = u_entry.get("grad_norm_image", 0.0)
        ratio = g_u / max(g_w, 1e-8)
        print(f"  {w_entry['severity']:>10.1f} | {g_w:>12.6f} | {g_u:>16.6f} | {ratio:>8.2f}")

    plot_e1_gradient_contamination(results, save_dir=save_dir)
    return results


# ===========================================================================
# Reliability Calibration
# ===========================================================================

@torch.no_grad()
def run_reliability_calibration(
    model: RobustM2BeamLLM,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: str = "logs",
):
    """Verify reliability estimator calibration (C2)."""
    print(f"\n{'='*60}")
    print(f"  Reliability Calibration Diagnostic")
    print(f"{'='*60}")

    import numpy as np

    model.eval()
    mods = ["image", "radar", "lidar", "gps"]
    all_reliabilities = {m: [] for m in mods}
    all_log_vars = {m: [] for m in mods}
    all_correct = []

    for batch in tqdm(test_loader, desc="Calibration", leave=False):
        images = batch["images"].to(device)
        radars = batch["radars"].to(device)
        lidars = batch["lidars"].to(device)
        gps = batch["gps"].to(device)
        targets = batch["beam_future"].to(device)
        beam_history = batch.get("beam_history")
        if beam_history is not None:
            beam_history = beam_history.to(device)

        predictions, aux = model(images, radars, lidars, gps, beam_history=beam_history)

        pred_idx = predictions.argmax(dim=-1)
        correct = (pred_idx == targets).float().mean(dim=1)
        all_correct.append(correct.cpu())

        for mod in mods:
            all_reliabilities[mod].append(
                aux["reliability_weights"][mod].mean(dim=1).cpu()
            )
            all_log_vars[mod].append(
                aux["log_variances"][mod].mean(dim=1).cpu()
            )

    all_correct_np = torch.cat(all_correct).numpy()

    summary = {"mods": {}}
    for mod in mods:
        rel = torch.cat(all_reliabilities[mod]).numpy()
        lv = torch.cat(all_log_vars[mod]).numpy()
        print(f"\n  {mod}:")
        print(f"    Reliability: mean={rel.mean():.3f}, std={rel.std():.3f}")
        print(f"    Log-variance: mean={lv.mean():.3f}, std={lv.std():.3f}")

        corr = None
        pval = None
        try:
            from scipy.stats import pearsonr
            corr, pval = pearsonr(rel, all_correct_np)
            print(f"    Corr(w_ω, accuracy): ρ={corr:.3f}, p={pval:.4f}")
        except ImportError:
            pass

        bins = np.quantile(rel, [0, 0.25, 0.5, 0.75, 1.0])
        print(f"    Reliability-conditioned accuracy:")
        bin_accs = []
        bin_counts = []
        for i in range(len(bins) - 1):
            mask = (rel >= bins[i]) & (rel < bins[i + 1] + 1e-8)
            if mask.sum() > 0:
                bin_acc = all_correct_np[mask].mean()
                bin_accs.append(float(bin_acc))
                bin_counts.append(int(mask.sum()))
                print(f"      w ∈ [{bins[i]:.3f}, {bins[i+1]:.3f}): "
                      f"acc={bin_acc*100:.1f}% (n={mask.sum()})")
            else:
                bin_accs.append(0.0)
                bin_counts.append(0)

        summary["mods"][mod] = {
            "reliability": rel,
            "log_variance": lv,
            "correctness": all_correct_np,
            "corr": None if corr is None else float(corr),
            "pval": None if pval is None else float(pval),
            "bins": bins,
            "bin_acc": np.array(bin_accs, dtype=float),
            "bin_counts": np.array(bin_counts, dtype=int),
        }

    plot_reliability_calibration_summary(summary, save_dir=save_dir)
    return summary


@torch.no_grad()
def run_reliability_calibration_paper(
    model: RobustM2BeamLLM,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    save_dir: str = "logs",
) -> Dict:
    """
    Paper-style E4 reliability calibration:
      - Corr(σ̂²_ω, e_ω) where e_ω is per-modality beam error estimated by
        masking out all *other* modalities.
      - Reliability/sigma²-conditioned error bins.
    """
    print(f"\n{'='*60}")
    print(f"  E4 (Paper): Reliability Calibration (σ̂² vs e_ω)")
    print(f"{'='*60}")

    import numpy as np

    model.eval()
    mods = ["image", "radar", "lidar", "gps"]
    sigma2_store = {m: [] for m in mods}
    rel_store = {m: [] for m in mods}
    mod_err_store = {m: [] for m in mods}

    for batch in tqdm(test_loader, desc="E4-Calib", leave=False):
        images = batch["images"].to(device)
        radars = batch["radars"].to(device)
        lidars = batch["lidars"].to(device)
        gps = batch["gps"].to(device)
        targets = batch["beam_future"].to(device)
        beam_history = batch.get("beam_history")
        if beam_history is not None:
            beam_history = beam_history.to(device)

        B, H = images.shape[:2]
        zero_stale = {m: torch.zeros(B, H, device=device) for m in mods}

        # Full forward: collect estimated reliability/logvar from the deployed model.
        _, aux_full = model(
            images, radars, lidars, gps,
            staleness_ms=zero_stale,
            missing_masks={m: torch.zeros(B, H, device=device) for m in mods},
            beam_history=beam_history,
        )
        for m in mods:
            lv = aux_full["log_variances"][m].mean(dim=1)  # (B,)
            sigma2 = torch.exp(lv)
            sigma2_store[m].append(sigma2.cpu())
            rel_store[m].append(aux_full["reliability_weights"][m].mean(dim=1).cpu())

        # Per-modality error e_ω by removing all other modalities (paper intent).
        inputs = {"image": images, "radar": radars, "lidar": lidars, "gps": gps}
        for keep_mod in mods:
            masked_inputs = {}
            missing_masks = {}
            for m in mods:
                if m == keep_mod:
                    masked_inputs[m] = inputs[m]
                    missing_masks[m] = torch.zeros(B, H, device=device)
                else:
                    masked_inputs[m] = torch.zeros_like(inputs[m])
                    missing_masks[m] = torch.ones(B, H, device=device)
            preds_single, _ = model(
                masked_inputs["image"], masked_inputs["radar"],
                masked_inputs["lidar"], masked_inputs["gps"],
                staleness_ms=zero_stale,
                missing_masks=missing_masks,
                beam_history=beam_history,
            )
            pred_idx = preds_single.argmax(dim=-1)
            # e_ω = 1 - average exact accuracy over horizon
            e_w = 1.0 - (pred_idx == targets).float().mean(dim=1)
            mod_err_store[keep_mod].append(e_w.cpu())

    summary = {"mods": {}}
    for mod in mods:
        sigma2 = torch.cat(sigma2_store[mod]).numpy()
        rel = torch.cat(rel_store[mod]).numpy()
        err = torch.cat(mod_err_store[mod]).numpy()

        corr = pval = None
        try:
            from scipy.stats import pearsonr
            corr, pval = pearsonr(sigma2, err)
        except Exception:
            pass

        sigma2_bins = np.quantile(sigma2, [0, 0.25, 0.5, 0.75, 1.0]) if sigma2.size else np.array([])
        sigma2_bin_err = []
        sigma2_bin_counts = []
        print(f"\n  {mod}:")
        print(f"    σ̂² mean={sigma2.mean():.4f}, std={sigma2.std():.4f}")
        print(f"    e_ω mean={err.mean():.4f}, std={err.std():.4f}")
        if corr is not None:
            print(f"    Corr(σ̂²_ω, e_ω): ρ={corr:.3f}, p={pval:.4f}")
        if sigma2_bins.size >= 2:
            print(f"    Sigma²-conditioned error:")
            for i in range(len(sigma2_bins) - 1):
                mask = (sigma2 >= sigma2_bins[i]) & (sigma2 < sigma2_bins[i + 1] + 1e-8)
                if mask.sum() > 0:
                    be = float(err[mask].mean())
                    sigma2_bin_err.append(be)
                    sigma2_bin_counts.append(int(mask.sum()))
                    print(f"      σ² ∈ [{sigma2_bins[i]:.4f}, {sigma2_bins[i+1]:.4f}): "
                          f"e={be:.4f} (n={int(mask.sum())})")
                else:
                    sigma2_bin_err.append(0.0)
                    sigma2_bin_counts.append(0)

        summary["mods"][mod] = {
            "sigma2_hat": sigma2,
            "reliability": rel,
            "modality_error": err,
            "corr_sigma2_error": None if corr is None else float(corr),
            "pval_sigma2_error": None if pval is None else float(pval),
            "sigma2_bins": sigma2_bins,
            "sigma2_bin_err": np.array(sigma2_bin_err, dtype=float),
            "sigma2_bin_counts": np.array(sigma2_bin_counts, dtype=int),
        }

    plot_e4_reliability_paper_calibration(summary, save_dir=save_dir)
    return summary


@torch.no_grad()
def run_reliability_monotonicity_s2(
    model: RobustM2BeamLLM,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    save_dir: str = "logs",
) -> Dict:
    """
    Paper-style E4 monotonicity check:
    under controlled S2 corruption severity α, estimated variance σ̂² should increase.
    """
    print(f"\n{'='*60}")
    print(f"  E4 (Paper): Reliability Monotonicity under S2")
    print(f"{'='*60}")

    import numpy as np
    mods = ["image", "radar", "lidar", "gps"]
    severities = [float(a) for a in cfg.stress_test.corruption_severities]
    results = {"mods": {m: {"severity_rows": []} for m in mods}}

    for mod in mods:
        print(f"\n  Target modality: {mod}")
        for alpha in severities:
            sigma2_vals = []
            rel_vals = []
            for batch in tqdm(test_loader, desc=f"E4-S2 {mod} α={alpha:.1f}", leave=False):
                images = batch["images"].to(device)
                radars = batch["radars"].to(device)
                lidars = batch["lidars"].to(device)
                gps = batch["gps"].to(device)
                beam_history = batch.get("beam_history")
                if beam_history is not None:
                    beam_history = beam_history.to(device)

                batch_dict = apply_degradation(
                    {"images": images, "radars": radars, "lidars": lidars, "gps": gps},
                    severity=alpha,
                    target_modalities=[mod],
                    stress_config=cfg.stress_test,
                )
                preds, aux = model(
                    batch_dict["images"], batch_dict["radars"],
                    batch_dict["lidars"], batch_dict["gps"],
                    beam_history=beam_history,
                )
                del preds
                lv = aux["log_variances"][mod].mean(dim=1)
                sigma2_vals.append(torch.exp(lv).cpu())
                rel_vals.append(aux["reliability_weights"][mod].mean(dim=1).cpu())

            sigma2 = torch.cat(sigma2_vals).numpy()
            rel = torch.cat(rel_vals).numpy()
            row = {
                "alpha": float(alpha),
                "sigma2_mean": float(np.mean(sigma2)),
                "sigma2_std": float(np.std(sigma2)),
                "reliability_mean": float(np.mean(rel)),
                "reliability_std": float(np.std(rel)),
            }
            results["mods"][mod]["severity_rows"].append(row)
            print(f"    α={alpha:.1f} | σ̂²={row['sigma2_mean']:.4f} | w={row['reliability_mean']:.4f}")

        # Monotonic trend statistic (Spearman rho if available)
        x = np.array([r["alpha"] for r in results["mods"][mod]["severity_rows"]], dtype=float)
        y = np.array([r["sigma2_mean"] for r in results["mods"][mod]["severity_rows"]], dtype=float)
        rho = pval = None
        try:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(x, y)
        except Exception:
            # Fallback monotonic score in [0,1]: fraction of non-decreasing adjacent differences.
            diffs = np.diff(y)
            rho = float(np.mean(diffs >= -1e-8)) if len(diffs) else 1.0
        results["mods"][mod]["spearman_rho_sigma2_vs_alpha"] = None if rho is None else float(rho)
        if pval is not None:
            results["mods"][mod]["spearman_pval_sigma2_vs_alpha"] = float(pval)

    plot_e4_reliability_monotonicity(results, save_dir=save_dir)
    return results


def run_e2_delay_regime_specialization(
    baseline_model: RobustM2BeamLLM,
    a1_model: RobustM2BeamLLM,
    criterion: RobustM2BeamLLMLoss,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    save_dir: str = "logs",
) -> Dict:
    """
    E2: Compare regime-aware baseline vs A1 (no regime conditioning) under S1 delay stress.
    """
    print(f"\n{'='*60}")
    print(f"  E2: Delay-Regime Specialization Effect")
    print(f"{'='*60}")

    results = {"baseline": {}, "A1": {}}
    mods = ["image", "radar", "lidar", "gps"]
    for delay in cfg.stress_test.delay_values_ms:
        delay_cfg = {m: float(delay) for m in mods}
        m_base = evaluate(baseline_model, criterion, test_loader, device, cfg, staleness_ms=delay_cfg)
        m_a1 = evaluate(a1_model, criterion, test_loader, device, cfg, staleness_ms=delay_cfg)
        key = f"delay_{int(delay)}ms"
        results["baseline"][key] = m_base
        results["A1"][key] = m_a1
        print(
            f"  Delay {int(delay):4d} ms | "
            f"Top-1 base={m_base['top_1_acc']*100:.1f}% vs A1={m_a1['top_1_acc']*100:.1f}% | "
            f"Worst-10 base={m_base.get('worst_10_acc',0)*100:.1f}% vs A1={m_a1.get('worst_10_acc',0)*100:.1f}%"
        )

    plot_e2_delay_regime_specialization(results, save_dir=save_dir)
    return results
