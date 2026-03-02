"""
C5: Reproducible Robustness Evaluation Suite.
Reference: Paper Section II-E.

Three stress-test axes:
  S1: Asynchrony stress (inject delays and missing bursts)
  S2: Degradation stress (modality corruption at controlled severity)
  S3: Domain shift stress (holdout scenarios)

Four targeted experiments:
  E1: Gradient contamination visualization
  E2: Delay-regime specialization effect
  E3: LLM necessity verification
  E4: Tail robustness metrics
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from scipy import ndimage
from functools import lru_cache


# ===========================================================================
# S1: Asynchrony Stress - Inject delays and missing bursts
# ===========================================================================

def inject_asynchrony(batch: Dict[str, torch.Tensor],
                      delay_ms: Dict[str, float] = None,
                      burst_length: int = 0,
                      burst_modalities: List[str] = None,
                      frame_interval_ms: float = 100.0
                      ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Inject modality-specific delays and bursty missing intervals.

    Args:
        batch: standard data batch with images, radars, lidars, gps
        delay_ms: Dict[modality -> delay in ms]. Simulates temporal shift
                  by rolling features backward (using older frames).
        burst_length: number of consecutive frames to mask (set to zero)
        burst_modalities: which modalities get burst missing
        frame_interval_ms: time between consecutive frames
    Returns:
        modified_batch: batch with delays/missing applied
        staleness_ms: Dict[mod -> (B, H)] staleness per frame
        missing_masks: Dict[mod -> (B, H)] binary missing indicators
    """
    mod_keys = {"image": "images", "radar": "radars", "lidar": "lidars", "gps": "gps"}
    B, H = batch["images"].shape[:2]
    device = batch["images"].device

    if delay_ms is None:
        delay_ms = {m: 0.0 for m in mod_keys}
    if burst_modalities is None:
        burst_modalities = []

    staleness = {}
    missing_masks = {}
    modified = {k: v.clone() for k, v in batch.items()}

    for mod, batch_key in mod_keys.items():
        delay = delay_ms.get(mod, 0.0)
        shift = int(delay / frame_interval_ms)  # number of frames to shift

        # Create staleness tensor
        base_staleness = torch.zeros(B, H, device=device)
        if shift > 0 and batch_key in modified:
            # Roll: use older frames (simulate delay)
            data = modified[batch_key]
            shifted = torch.roll(data, shifts=shift, dims=1)
            # Zero out the first `shift` frames (no data available that old)
            shifted[:, :shift] = 0
            modified[batch_key] = shifted
            base_staleness[:, :] = delay

        staleness[mod] = base_staleness

        # Missing mask
        mask = torch.zeros(B, H, device=device)
        if mod in burst_modalities and burst_length > 0:
            # Random burst start position for each sample
            for b in range(B):
                start = torch.randint(0, max(1, H - burst_length), (1,)).item()
                end = min(start + burst_length, H)
                mask[b, start:end] = 1.0
                # Zero out missing frames
                if batch_key in modified:
                    modified[batch_key][b, start:end] = 0

        missing_masks[mod] = mask

    return modified, staleness, missing_masks


# ===========================================================================
# S2: Degradation Stress - Apply modality corruption
# ===========================================================================

def apply_camera_blur(images: torch.Tensor, severity: float,
                      max_kernel_sigma: float = 20.0) -> torch.Tensor:
    """
    Gaussian blur on camera images.
    σ_blur = α × max_kernel_sigma
    """
    if severity <= 0:
        return images
    sigma = severity * max_kernel_sigma
    # Use a Gaussian kernel
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=images.dtype, device=images.device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / max(sigma, 0.01)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable 2D convolution
    original_shape = images.shape
    if images.dim() == 5:  # (B, H, C, H_img, W_img)
        B, H, C, Hi, Wi = images.shape
        images = images.reshape(B * H, C, Hi, Wi)
    else:
        B_flat = images.shape[0]
        C = images.shape[1]

    # Pad and convolve
    pad_h = kernel_size // 2
    kernel_h = kernel_1d.view(1, 1, kernel_size, 1).expand(C, -1, -1, -1)
    kernel_w = kernel_1d.view(1, 1, 1, kernel_size).expand(C, -1, -1, -1)

    blurred = F.pad(images, (0, 0, pad_h, pad_h), mode='reflect')
    blurred = F.conv2d(blurred, kernel_h, groups=C)
    blurred = F.pad(blurred, (pad_h, pad_h, 0, 0), mode='reflect')
    blurred = F.conv2d(blurred, kernel_w, groups=C)

    if len(original_shape) == 5:
        blurred = blurred.reshape(original_shape)
    return blurred


def apply_lidar_dropout(lidars: torch.Tensor, severity: float) -> torch.Tensor:
    """
    Histogram-space LiDAR degradation proxy.
    Direct Bernoulli dropout on sparse occupancy grids can act like denoising, so
    we combine occupancy attenuation/dropout with sparse clutter injection.
    """
    if severity <= 0:
        return lidars
    occupied = (lidars > 0).float()
    drop_prob = min(float(severity) * 0.8, 0.95)
    keep_mask = (torch.rand_like(lidars) > drop_prob).float()
    attenuated = lidars * (occupied * keep_mask + (1.0 - occupied))
    attenuated = attenuated * (1.0 - 0.25 * float(severity))

    # Sparse false positives / clutter; stronger at higher severity.
    clutter_prob = 0.01 + 0.04 * float(severity)
    clutter_mask = (torch.rand_like(lidars) < clutter_prob).float()
    clutter = clutter_mask * torch.rand_like(lidars) * (0.15 + 0.35 * float(severity))
    return (attenuated + clutter).clamp(0.0, 1.0)


def apply_radar_noise(radars: torch.Tensor, severity: float,
                      max_noise_std: float = 5.0) -> torch.Tensor:
    """
    Radar degradation on normalized range-angle maps.
    The tensor is already normalized to [0,1], so use a data-relative noise scale
    instead of directly interpreting max_noise_std as raw physical meters.
    """
    if severity <= 0:
        return radars
    spatial_dims = tuple(range(radars.dim() - 2, radars.dim()))
    base_std = radars.detach().std(dim=spatial_dims, keepdim=True).clamp_min(1e-3)
    noise_std = float(severity) * float(max_noise_std) * base_std
    signal_scale = max(0.0, 1.0 - 0.25 * float(severity))
    noise = torch.randn_like(radars) * noise_std
    return (radars * signal_scale + noise).clamp(0.0, 1.0)


def apply_gps_noise(gps: torch.Tensor, severity: float,
                    max_noise_std: float = 50.0,
                    data_root: Optional[str] = None,
                    use_physical_space: bool = True) -> torch.Tensor:
    """Additive GPS noise; prefer physical-space perturbation then re-normalize."""
    if severity <= 0:
        return gps

    @lru_cache(maxsize=8)
    def _load_norm(path: str):
        norm_path = os.path.join(path, "gps_normalizer.npz")
        if not os.path.isfile(norm_path):
            return None
        data = np.load(norm_path)
        gps_min = torch.from_numpy(np.array(data["gps_min"], dtype=np.float32))
        gps_max = torch.from_numpy(np.array(data["gps_max"], dtype=np.float32))
        return gps_min, gps_max

    if use_physical_space and data_root:
        stats = _load_norm(data_root)
        if stats is not None:
            gps_min, gps_max = stats
            gps_min = gps_min.to(device=gps.device, dtype=gps.dtype)
            gps_max = gps_max.to(device=gps.device, dtype=gps.dtype)
            scale = (gps_max - gps_min).clamp_min(1e-6)
            gps_phys = gps * scale + gps_min
            noise_std = severity * max_noise_std
            noise = torch.randn_like(gps_phys) * noise_std
            gps_noisy = gps_phys + noise
            return ((gps_noisy - gps_min) / scale).clamp(0.0, 1.0)

    noise_std = severity * max_noise_std / 2000.0
    noise = torch.randn_like(gps) * noise_std
    return (gps + noise).clamp(0.0, 1.0)


def apply_degradation(batch: Dict[str, torch.Tensor],
                      severity: float,
                      target_modalities: Optional[List[str]] = None,
                      stress_config=None,
                      data_root: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Apply controlled degradation to specified modalities at severity α ∈ [0,1].

    Args:
        batch: data batch
        severity: α in [0, 1]
        target_modalities: which modalities to corrupt. None = all.
        stress_config: StressTestConfig for max noise parameters
    """
    if target_modalities is None:
        target_modalities = ["image", "radar", "lidar", "gps"]

    modified = {k: v.clone() for k, v in batch.items()}
    B = int(batch["beam_history"].shape[0]) if "beam_history" in batch else int(batch["images"].shape[0])
    H = int(batch["images"].shape[1])
    device = batch["images"].device

    cam_blur_max = stress_config.camera_blur_max if stress_config else 20.0
    lidar_dropout_max = stress_config.lidar_dropout_max if stress_config else 1.0
    radar_noise_max = stress_config.radar_noise_max if stress_config else 5.0
    gps_noise_max = stress_config.gps_noise_max if stress_config else 50.0

    if "image" in target_modalities:
        modified["images"] = apply_camera_blur(modified["images"], severity, cam_blur_max)
    if "lidar" in target_modalities:
        modified["lidars"] = apply_lidar_dropout(modified["lidars"], min(float(severity) * float(lidar_dropout_max), 1.0))
    if "radar" in target_modalities:
        modified["radars"] = apply_radar_noise(modified["radars"], severity, radar_noise_max)
    if "gps" in target_modalities:
        modified["gps"] = apply_gps_noise(
            modified["gps"], severity, gps_noise_max,
            data_root=data_root,
            use_physical_space=bool(getattr(stress_config, "gps_noise_in_physical_space", True)),
        )

    mods = ["image", "radar", "lidar", "gps"]
    degradation_cues = {}
    for mod in mods:
        cue = float(severity) if mod in target_modalities else 0.0
        degradation_cues[mod] = torch.full((B, H, 1), cue, device=device, dtype=torch.float32)
    modified["_degradation_cues"] = degradation_cues

    return modified


# ===========================================================================
# E4: Tail Robustness Metrics
# ===========================================================================

def compute_tail_metrics(per_sample_losses: np.ndarray,
                         per_sample_correct: np.ndarray,
                         tail_percentile: float = 0.1
                         ) -> Dict[str, float]:
    """
    Compute tail-risk metrics for safety-critical evaluation.

    Args:
        per_sample_losses: (N,) per-sample loss values
        per_sample_correct: (N,) binary correctness (1=correct)
        tail_percentile: fraction for worst-case analysis (default 10%)
    Returns:
        Dict with CVaR, worst-bin accuracy, worst-10% accuracy, percentile curve
    """
    N = len(per_sample_losses)
    tail_size = max(1, int(N * tail_percentile))

    # Sort by loss (descending = worst first)
    sorted_idx = np.argsort(per_sample_losses)[::-1]
    worst_losses = per_sample_losses[sorted_idx[:tail_size]]
    worst_correct = per_sample_correct[sorted_idx[:tail_size]]

    # CVaR: expected loss in worst tail_percentile
    q_threshold = np.percentile(per_sample_losses, 100 * (1 - tail_percentile))
    tail_mask = per_sample_losses >= q_threshold
    cvar = per_sample_losses[tail_mask].mean() if tail_mask.sum() > 0 else 0.0

    # Worst 10% accuracy
    worst_acc = worst_correct.mean()

    # Percentile degradation curve
    percentiles = [10, 25, 50, 75, 90]
    percentile_accs = {}
    for p in percentiles:
        threshold = np.percentile(per_sample_losses, p)
        mask = per_sample_losses <= threshold
        if mask.sum() > 0:
            percentile_accs[f"p{p}_acc"] = per_sample_correct[mask].mean()
        else:
            percentile_accs[f"p{p}_acc"] = 0.0

    return {
        f"cvar_{int(tail_percentile*100)}": float(cvar),
        f"worst_{int(tail_percentile*100)}_acc": float(worst_acc),
        "overall_acc": float(per_sample_correct.mean()),
        **percentile_accs,
    }


def compute_per_sample_metrics(predictions: torch.Tensor,
                               targets: torch.Tensor
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample loss and correctness for tail analysis.

    Args:
        predictions: (N, T, num_beams)
        targets: (N, T)
    Returns:
        per_sample_losses: (N,) CE loss per sample
        per_sample_correct: (N,) Top-1 correct fraction per sample
    """
    N, T, C = predictions.shape

    # Per-sample CE loss
    losses = F.cross_entropy(
        predictions.reshape(-1, C),
        targets.reshape(-1),
        reduction='none'
    ).reshape(N, T).mean(dim=1)  # (N,)

    # Per-sample correctness
    pred_idx = predictions.argmax(dim=-1)  # (N, T)
    correct = (pred_idx == targets).float().mean(dim=1)  # (N,)

    return losses.detach().cpu().numpy(), correct.detach().cpu().numpy()


# ===========================================================================
# E1: Gradient Contamination Visualization Data Collection
# ===========================================================================

def collect_gradient_norms(model, batch, criterion, device,
                           degraded_modality: str = "image",
                           severity: float = 0.8,
                           use_reliability: bool = True) -> Dict[str, float]:
    """
    Collect alignment gradient norms for gradient contamination analysis.

    Returns:
        Dict with gradient norms for degraded and clean modalities
    """
    model.train()
    images = batch["images"].to(device)
    radars = batch["radars"].to(device)
    lidars = batch["lidars"].to(device)
    gps = batch["gps"].to(device)
    targets = batch["beam_future"].to(device)
    beam_history = batch.get("beam_history")
    if beam_history is not None:
        beam_history = beam_history.to(device)

    # Clean forward (no grad) to measure degradation-induced representation drift
    with torch.no_grad():
        _, aux_clean = model(images, radars, lidars, gps, beam_history=beam_history)

    # Apply degradation to one modality
    degraded_batch = apply_degradation(
        {"images": images, "radars": radars, "lidars": lidars, "gps": gps},
        severity=severity,
        target_modalities=[degraded_modality],
    )

    # Forward
    predictions, aux = model(
        degraded_batch["images"], degraded_batch["radars"],
        degraded_batch["lidars"], degraded_batch["gps"],
        beam_history=beam_history,
    )

    # Compute alignment loss
    align_loss = model.alignment_loss_fn(
        aux["aligned_features"],
        aux["reliability_weights"],
        use_reliability=use_reliability,
    )

    # Backward for alignment loss only
    model.zero_grad()
    align_loss.backward(retain_graph=True)

    # Collect gradient norms per modality encoder
    grad_norms = {}
    mod_encoders = {
        "image": model.encoder.vision_encoder,
        "radar": model.encoder.radar_encoder,
        "lidar": model.encoder.lidar_encoder,
        "gps": model.encoder.gps_encoder,
    }
    for mod, enc in mod_encoders.items():
        total_norm = 0.0
        count = 0
        for p in enc.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
                count += 1
        grad_norms[f"grad_norm_{mod}"] = (total_norm ** 0.5) if count > 0 else 0.0

    # E1 representation drift proxy: ||h_bar(clean) - h_bar(degraded)|| averaged
    for mod in ["image", "radar", "lidar", "gps"]:
        h_clean = aux_clean["aligned_features"][mod]
        h_deg = aux["aligned_features"][mod]
        drift = (h_deg - h_clean).pow(2).sum(dim=-1).sqrt().mean()
        grad_norms[f"repr_drift_{mod}"] = float(drift.item())
        rel_mean = aux["reliability_weights"][mod].mean()
        lv_mean = aux["log_variances"][mod].mean()
        grad_norms[f"reliability_{mod}"] = float(rel_mean.item())
        grad_norms[f"logvar_{mod}"] = float(lv_mean.item())
        grad_norms[f"sigma2_{mod}"] = float(torch.exp(lv_mean).item())

    grad_norms["degraded_modality"] = degraded_modality
    grad_norms["severity"] = severity
    grad_norms["use_reliability_align"] = bool(use_reliability)

    model.zero_grad()
    return grad_norms


# ===========================================================================
# C4: Complexity Analysis
# ===========================================================================

def compute_complexity_analysis(model, feature_dim: int = 64,
                                num_modalities: int = 4,
                                buffer_size: int = 3,
                                num_regimes: int = 3,
                                num_experts: int = 4,
                                lora_rank: int = 16,
                                num_beams: int = 64) -> Dict:
    """
    C4: Detailed complexity breakdown of all proposed modules.
    Reference: Paper Table I.
    """
    results = {}

    def count_params(obj):
        if isinstance(obj, torch.nn.Parameter):
            return obj.numel()
        return sum(p.numel() for p in obj.parameters())

    def count_trainable(obj):
        if isinstance(obj, torch.nn.Parameter):
            return obj.numel() if obj.requires_grad else 0
        return sum(p.numel() for p in obj.parameters() if p.requires_grad)

    def row_stats(row_obj=None, synthetic: Optional[Dict[str, int]] = None):
        if synthetic is not None:
            params = int(synthetic["params"])
            trainable = int(synthetic["trainable"])
        else:
            params = count_params(row_obj)
            trainable = count_trainable(row_obj)
        memory_mib = params * 4 / (1024 ** 2)  # FP32 parameter memory in MiB
        return params, trainable, memory_mib

    llm_module = getattr(model.llm_backbone, "llm", None)
    moe_layers = getattr(model.llm_backbone, "moe_lora_layers", None)
    backbone_name = "MoE-LoRA (C3)" if llm_module is not None else "Vanilla Transformer (A5 baseline)"

    # Use backbone reference size for reporting.
    if llm_module is not None:
        llm_params = count_params(llm_module)
        llm_trainable = count_trainable(llm_module)
    else:
        llm_params = count_params(model.llm_backbone)
        llm_trainable = count_trainable(model.llm_backbone)
    llm_frozen = max(llm_params - llm_trainable, 0)

    # Top-level rows are additive and define TOTAL. Sub-rows are informational.
    top_rows = [
        {
            "module": "Async Alignment (C1)",
            "obj": model.async_alignment,
            "is_subrow": False,
            "is_ext": False,
            "is_training_only": False,
        },
        {
            "module": "Reliability Estimator (C2)",
            "obj": model.reliability_estimator,
            "is_subrow": False,
            "is_ext": False,
            "is_training_only": False,
        },
        {
            "module": "Alignment Loss Proj (C2-c)",
            "obj": model.alignment_loss_fn,
            "is_subrow": False,
            "is_ext": False,
            "is_training_only": True,
        },
        {
            "module": "Cross-Attn Fusion (C2-d)",
            "obj": model.fusion,
            "is_subrow": False,
            "is_ext": False,
            "is_training_only": False,
        },
        {
            "module": backbone_name,
            "obj": model.llm_backbone,
            "is_subrow": False,
            "is_ext": False,
            "is_training_only": False,
        },
        {
            "module": "Beam Head",
            "obj": model.beam_classifier,
            "is_subrow": False,
            "is_ext": False,
            "is_training_only": False,
        },
        {
            "module": "SSL Objectives (C3)",
            "obj": model.ssl_objectives,
            "is_subrow": False,
            "is_ext": False,
            "is_training_only": True,
        },
        {
            "module": "Encoder Backbones",
            "obj": model.encoder,
            "is_subrow": False,
            "is_ext": False,
            "is_training_only": False,
        },
    ]
    if getattr(model, "beam_history_conditioner", None) is not None:
        top_rows.append({
            "module": "Beam History Cond. (Ext)",
            "obj": model.beam_history_conditioner,
            "is_subrow": False,
            "is_ext": True,
            "is_training_only": False,
        })
    if getattr(model, "ar_beam_decoder", None) is not None:
        top_rows.append({
            "module": "AR Beam Decoder (Ext)",
            "obj": model.ar_beam_decoder,
            "is_subrow": False,
            "is_ext": True,
            "is_training_only": False,
        })
    if getattr(model, "topk_pairwise_reranker", None) is not None:
        top_rows.append({
            "module": "Top-K Pairwise Reranker (Ext)",
            "obj": model.topk_pairwise_reranker,
            "is_subrow": False,
            "is_ext": True,
            "is_training_only": False,
        })

    subrows_by_parent: Dict[str, List[Dict]] = {
        "Async Alignment (C1)": [
            {"module": "  - Staleness Embedding", "obj": model.async_alignment.staleness_embed},
            {"module": "  - Weighting (all mods)", "obj": model.async_alignment.weighting},
            {"module": "  - Short-gap Repair", "obj": model.async_alignment.repair},
            {"module": "  - Residual Refinement", "obj": model.async_alignment.refinement},
        ],
        "Encoder Backbones": [
            {"module": "  - Vision encoder", "obj": model.encoder.vision_encoder},
            {"module": "  - Radar encoder", "obj": model.encoder.radar_encoder},
            {"module": "  - LiDAR encoder", "obj": model.encoder.lidar_encoder},
            {"module": "  - GPS encoder", "obj": model.encoder.gps_encoder},
        ],
    }

    c3_subrows = []
    if llm_module is not None:
        c3_subrows.append({
            "module": "  - LLM core (frozen subset)",
            "synthetic": {"params": llm_frozen, "trainable": 0},
        })
        c3_subrows.append({
            "module": "  - LLM core (unfrozen subset, inside backbone)",
            "synthetic": {"params": llm_trainable, "trainable": llm_trainable},
        })
        if hasattr(model.llm_backbone, "input_proj"):
            c3_subrows.append({"module": "  - LLM input projection", "obj": model.llm_backbone.input_proj})
        if hasattr(model.llm_backbone, "prompt_tokens"):
            c3_subrows.append({"module": "  - LLM prompt tokens", "obj": model.llm_backbone.prompt_tokens})
        if hasattr(model.llm_backbone, "output_proj"):
            c3_subrows.append({"module": "  - LLM output projection", "obj": model.llm_backbone.output_proj})
    if moe_layers is not None:
        c3_subrows.append({"module": "  - MoE-LoRA adapters", "obj": moe_layers})
    if c3_subrows:
        subrows_by_parent[backbone_name] = c3_subrows

    breakdown = []
    top_level_total_params = 0
    top_level_total_trainable = 0
    top_level_total_params_standard = 0
    top_level_total_trainable_standard = 0
    inference_total_params = 0
    inference_total_trainable = 0
    inference_total_params_standard = 0
    inference_total_trainable_standard = 0
    training_only_top_rows = []
    has_ext_rows = False

    for row in top_rows:
        params, trainable, memory_mib = row_stats(row_obj=row.get("obj"), synthetic=row.get("synthetic"))
        row_entry = {
            "module": row["module"],
            "params": params,
            "trainable": trainable,
            "memory_mib": memory_mib,
            "is_subrow": False,
            "is_ext": row["is_ext"],
            "is_training_only": row["is_training_only"],
        }
        breakdown.append(row_entry)
        top_level_total_params += params
        top_level_total_trainable += trainable
        if not row["is_ext"]:
            top_level_total_params_standard += params
            top_level_total_trainable_standard += trainable
        if not row["is_training_only"]:
            inference_total_params += params
            inference_total_trainable += trainable
            if not row["is_ext"]:
                inference_total_params_standard += params
                inference_total_trainable_standard += trainable
        if row["is_training_only"]:
            training_only_top_rows.append(row["module"])
        if row["is_ext"]:
            has_ext_rows = True

        for sub in subrows_by_parent.get(row["module"], []):
            s_params, s_trainable, s_memory_mib = row_stats(
                row_obj=sub.get("obj"), synthetic=sub.get("synthetic")
            )
            breakdown.append({
                "module": sub["module"],
                "params": s_params,
                "trainable": s_trainable,
                "memory_mib": s_memory_mib,
                "is_subrow": True,
                "parent": row["module"],
            })

    results["breakdown"] = breakdown
    # TOTAL is the sum of additive top-level rows only (sub-rows excluded).
    results["total_params"] = top_level_total_params
    results["total_trainable"] = top_level_total_trainable
    results["total_params_standard"] = top_level_total_params_standard
    results["total_trainable_standard"] = top_level_total_trainable_standard
    results["inference_total_params"] = inference_total_params
    results["inference_total_trainable"] = inference_total_trainable
    results["inference_total_params_standard"] = inference_total_params_standard
    results["inference_total_trainable_standard"] = inference_total_trainable_standard
    results["has_ext_rows"] = has_ext_rows
    results["training_only_top_rows"] = training_only_top_rows
    results["llm_backbone_total_params"] = llm_params
    results["llm_backbone_trainable_params"] = llm_trainable
    results["llm_frozen_params"] = llm_frozen
    results["non_llm_trainable_params"] = max(results["total_trainable"] - llm_trainable, 0)
    results["non_c3_backbone_trainable_params"] = max(
        results["total_trainable"] - count_trainable(model.llm_backbone), 0
    )
    results["proposed_module_params"] = results["total_trainable"]
    results["fraction_of_backbone"] = results["total_trainable"] / max(llm_params, 1) * 100

    return results


def print_complexity_table(results: Dict):
    """Pretty-print complexity analysis as Table I."""
    print(f"\n{'='*75}")
    print(f"  COMPLEXITY BREAKDOWN (Table I)")
    print(f"{'='*75}")
    print(f"  {'Module':<35} {'Params':>12} {'Trainable':>12} {'Param Mem':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

    for item in results["breakdown"]:
        name = item["module"]
        params = item["params"]
        trainable = item["trainable"]
        mem = item["memory_mib"]
        print(f"  {name:<35} {params:>12,} {trainable:>12,} {mem:>7.2f} MiB")

    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'TOTAL (top-level, current cfg)':<35} {results['total_params']:>12,} {results['total_trainable']:>12,}")
    if results.get("has_ext_rows", False):
        print(f"  {'TOTAL (top-level, Ext off)':<35} {results['total_params_standard']:>12,} {results['total_trainable_standard']:>12,}")
    print(f"  {'TOTAL (inference-active, current)':<35} {results['inference_total_params']:>12,} {results['inference_total_trainable']:>12,}")
    if results.get("has_ext_rows", False):
        print(f"  {'TOTAL (inference-active, Ext off)':<35} {results['inference_total_params_standard']:>12,} {results['inference_total_trainable_standard']:>12,}")
    print(f"  Note: rows prefixed with '-' are sub-components and are excluded from TOTAL.")
    if results.get("training_only_top_rows"):
        print(f"  Training-time-only top rows: {', '.join(results['training_only_top_rows'])}")
        print(f"  (excluded from inference-active totals)")
    print(f"  Param memory uses MiB (FP32 parameter storage only; excludes activations/optimizer state).")

    print(f"\n  LLM backbone total params: {results.get('llm_backbone_total_params', 0):,}")
    print(f"  LLM backbone trainable (PEFT): {results.get('llm_backbone_trainable_params', 0):,}")
    print(f"  LLM backbone frozen subset: {results['llm_frozen_params']:,}")
    print(f"  Total trainable (all modules): {results['total_trainable']:,}")
    print(f"  Non-core-LLM trainable params: {results.get('non_llm_trainable_params', 0):,}")
    print(f"  Outside C3 backbone trainable params: {results.get('non_c3_backbone_trainable_params', 0):,}")
    print(f"  Trainable / LLM backbone total: {results['fraction_of_backbone']:.3f}%")
    print(f"{'='*75}")


@torch.no_grad()
def benchmark_inference_latency(model,
                                device: torch.device,
                                H: int,
                                T: int,
                                num_beams: int = 64,
                                warmup_iters: int = 10,
                                iters: int = 30) -> Dict[str, float]:
    """
    C4 helper: lightweight inference latency benchmark with dummy inputs.
    Reports wall-clock latency for the current device/config.
    """
    model.eval()
    B = 1
    # DeepSense scenario32 canonical tensor shapes used by current pipeline.
    images = torch.zeros(B, H, 3, 224, 224, device=device)
    radars = torch.zeros(B, H, 1, 64, 256, device=device)
    lidars = torch.zeros(B, H, 1, 256, 256, device=device)
    gps = torch.zeros(B, H, 2, device=device)
    beam_history = torch.zeros(B, H, dtype=torch.long, device=device)

    def _sync():
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        if hasattr(torch, "mps") and str(device) == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    # Warmup
    for _ in range(max(0, int(warmup_iters))):
        _ = model(images, radars, lidars, gps, beam_history=beam_history)
    _sync()

    timings = []
    for _ in range(max(1, int(iters))):
        t0 = time.perf_counter()
        _ = model(images, radars, lidars, gps, beam_history=beam_history)
        _sync()
        timings.append((time.perf_counter() - t0) * 1000.0)

    timings_np = np.asarray(timings, dtype=np.float64)
    return {
        "batch_size": 1,
        "H": int(H),
        "T": int(T),
        "latency_ms_mean": float(timings_np.mean()),
        "latency_ms_std": float(timings_np.std()),
        "latency_ms_p50": float(np.percentile(timings_np, 50)),
        "latency_ms_p90": float(np.percentile(timings_np, 90)),
        "latency_ms_p95": float(np.percentile(timings_np, 95)),
        "iters": int(iters),
        "warmup_iters": int(warmup_iters),
    }


@torch.no_grad()
def estimate_flops_per_sample(model,
                              device: torch.device,
                              H: int,
                              T: int) -> Optional[float]:
    """
    C4 helper: best-effort FLOPs/sample estimate using torch.profiler (if available).
    Returns total FLOPs for one forward pass on dummy input, or None if unsupported.
    """
    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception:
        return None

    model.eval()
    B = 1
    images = torch.zeros(B, H, 3, 224, 224, device=device)
    radars = torch.zeros(B, H, 1, 64, 256, device=device)
    lidars = torch.zeros(B, H, 1, 256, 256, device=device)
    gps = torch.zeros(B, H, 2, device=device)
    beam_history = torch.zeros(B, H, dtype=torch.long, device=device)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    try:
        with profile(activities=activities, record_shapes=False, with_flops=True) as prof:
            _ = model(images, radars, lidars, gps, beam_history=beam_history)
        total_flops = 0
        for evt in prof.key_averages():
            fl = getattr(evt, "flops", None)
            if fl is not None:
                total_flops += int(fl)
        return float(total_flops) if total_flops > 0 else None
    except Exception:
        return None


def print_complexity_benchmark(results: Dict):
    """Pretty-print latency/FLOPs benchmark summary."""
    if not results:
        return
    print(f"\n{'='*75}")
    print("  C4 Benchmark (Inference Runtime)")
    print(f"{'='*75}")
    if results.get("flops_per_sample") is not None:
        print(f"  FLOPs/sample (forward, B=1): {results['flops_per_sample'] / 1e9:.3f} GFLOPs")
    else:
        print("  FLOPs/sample (forward, B=1): N/A (torch.profiler flops unsupported)")
    if "latency_ms_mean" in results:
        print(
            f"  Latency (ms): mean={results['latency_ms_mean']:.3f}, "
            f"std={results['latency_ms_std']:.3f}, "
            f"p50={results['latency_ms_p50']:.3f}, "
            f"p90={results['latency_ms_p90']:.3f}, "
            f"p95={results['latency_ms_p95']:.3f}"
        )
        print(f"  Benchmark iters: warmup={results.get('warmup_iters', 0)}, run={results.get('iters', 0)}")
    print(f"{'='*75}")
