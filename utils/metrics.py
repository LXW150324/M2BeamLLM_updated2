"""
Extended evaluation metrics for Robust M²BeamLLM.
Includes standard beam prediction metrics + tail-risk metrics (C5-E4).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F


def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(1)
        targets = targets.unsqueeze(1)
    _, top_k_indices = predictions.topk(k, dim=-1)
    targets_expanded = targets.unsqueeze(-1)
    correct = (top_k_indices == targets_expanded).any(dim=-1).float()
    return correct.mean().item()


def dba_score(predictions: torch.Tensor, targets: torch.Tensor,
              k: int = 1, delta: int = 1) -> float:
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(1)
        targets = targets.unsqueeze(1)
    _, top_k_indices = predictions.topk(k, dim=-1)
    targets_expanded = targets.unsqueeze(-1).float()
    top_k_float = top_k_indices.float()
    total_dba = 0.0
    for kk in range(1, k + 1):
        top_kk = top_k_float[:, :, :kk]
        distances = torch.abs(top_kk - targets_expanded)
        min_dist, _ = distances.min(dim=-1)
        clipped = torch.clamp(min_dist / delta, max=1.0)
        y_k = 1.0 - clipped.mean().item()
        total_dba += y_k
    return total_dba / k


def compute_per_step_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Per future-step accuracy breakdown."""
    N, T, C = predictions.shape
    pred_idx = predictions.argmax(dim=-1)
    results = {}
    for t in range(T):
        acc = (pred_idx[:, t] == targets[:, t]).float().mean().item()
        results[f"step_{t+1}_acc"] = acc
    return results


def compute_communication_proxy_metrics(predictions: torch.Tensor,
                                        targets: torch.Tensor,
                                        top_k_values: List[int]) -> Dict[str, float]:
    """
    Communication-related proxy metrics (when true receive-power labels are unavailable).

    Proxies are explicitly marked as *_proxy to avoid confusion with physical measurements.
    - nrp_proxy_kK: normalized receive-power proxy using beam-index distance
      (1 - min(|pred-target|)/(C-1)) over top-K candidates.
    - sweep_overhead_proxy_kK: beam sweeping overhead ratio proxy = K / C.
    - topk_overhead_efficiency_kK: top-K accuracy normalized by overhead proxy.
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(1)
        targets = targets.unsqueeze(1)
    _, _, C = predictions.shape
    max_dist = max(C - 1, 1)
    top_k_values = sorted(set(int(k) for k in top_k_values if int(k) >= 1))

    metrics = {}
    topmax = max(top_k_values) if top_k_values else 1
    topmax = min(topmax, C)
    top_idx = predictions.topk(topmax, dim=-1).indices  # (N,T,Kmax)
    targets_exp = targets.unsqueeze(-1)
    dists = torch.abs(top_idx.float() - targets_exp.float())  # (N,T,Kmax)

    for k in top_k_values:
        kk = min(k, C)
        min_dist = dists[:, :, :kk].min(dim=-1).values
        nrp_proxy = 1.0 - torch.clamp(min_dist / float(max_dist), min=0.0, max=1.0)
        metrics[f"nrp_proxy_k{kk}"] = float(nrp_proxy.mean().item())

        overhead = float(kk) / float(C)
        metrics[f"sweep_overhead_proxy_k{kk}"] = overhead

        acc_k = top_k_accuracy(predictions, targets, k=kk)
        metrics[f"topk_overhead_efficiency_k{kk}"] = float(acc_k / max(overhead, 1e-8))

    return metrics


def compute_communication_power_metrics(predictions: torch.Tensor,
                                        beam_powers: torch.Tensor,
                                        top_k_values: List[int]) -> Dict[str, float]:
    """
    Communication metrics using true per-beam power vectors (if available).

    Args:
        predictions: (N,T,C) logits
        beam_powers: (N,T,C) true receive-power values over all beams
    Returns:
        nrp_kK: normalized receive power (best power in top-K / best power overall)
        sweep_overhead_kK: exact top-K sweep overhead ratio K/C
        power_overhead_efficiency_kK: nrp_kK normalized by overhead
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(1)
    if beam_powers.dim() == 2:
        beam_powers = beam_powers.unsqueeze(1)
    if predictions.shape != beam_powers.shape:
        raise ValueError(
            f"predictions and beam_powers shape mismatch: {tuple(predictions.shape)} vs {tuple(beam_powers.shape)}"
        )

    N, T, C = predictions.shape
    top_k_values = sorted(set(int(k) for k in top_k_values if int(k) >= 1))
    kmax = min(max(top_k_values) if top_k_values else 1, C)

    top_idx = predictions.topk(kmax, dim=-1).indices  # (N,T,Kmax)
    gathered_power = beam_powers.gather(dim=-1, index=top_idx)  # (N,T,Kmax)
    best_power = beam_powers.max(dim=-1).values.clamp_min(1e-8)  # (N,T)

    metrics = {}
    for k in top_k_values:
        kk = min(k, C)
        best_topk_power = gathered_power[..., :kk].max(dim=-1).values
        nrp = (best_topk_power / best_power).clamp(min=0.0)
        overhead = float(kk) / float(C)
        metrics[f"nrp_k{kk}"] = float(nrp.mean().item())
        metrics[f"sweep_overhead_k{kk}"] = overhead
        metrics[f"power_overhead_efficiency_k{kk}"] = float(nrp.mean().item() / max(overhead, 1e-8))
    return metrics


def compute_tail_risk_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                              tail_fraction: float = 0.1) -> Dict[str, float]:
    """
    C5-E4: Tail robustness metrics.
    - CVaR (Conditional Value-at-Risk at tail_fraction level)
    - Worst tail_fraction% accuracy
    - Percentile degradation curve
    """
    N, T, C = predictions.shape

    # Per-sample CE loss
    per_sample_loss = F.cross_entropy(
        predictions.reshape(-1, C), targets.reshape(-1), reduction='none'
    ).reshape(N, T).mean(dim=1).detach().cpu().numpy()

    # Per-sample correctness
    pred_idx = predictions.argmax(dim=-1)
    per_sample_correct = (pred_idx == targets).float().mean(dim=1).detach().cpu().numpy()

    tail_size = max(1, int(N * tail_fraction))
    sorted_idx = np.argsort(per_sample_loss)[::-1]  # worst first

    # CVaR
    q_threshold = np.percentile(per_sample_loss, 100 * (1 - tail_fraction))
    tail_mask = per_sample_loss >= q_threshold
    cvar = float(per_sample_loss[tail_mask].mean()) if tail_mask.sum() > 0 else 0.0

    # Worst X% accuracy
    worst_correct = per_sample_correct[sorted_idx[:tail_size]]
    worst_acc = float(worst_correct.mean())

    # Percentile degradation curve
    percentile_accs = {}
    for p in [10, 25, 50, 75, 90]:
        threshold = np.percentile(per_sample_loss, p)
        mask = per_sample_loss <= threshold
        if mask.sum() > 0:
            percentile_accs[f"percentile_{p}_acc"] = float(per_sample_correct[mask].mean())

    pct = int(tail_fraction * 100)
    return {
        f"cvar_{pct}": cvar,
        f"worst_{pct}_acc": worst_acc,
        **percentile_accs,
    }


def compute_all_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                        top_k_values: List[int] = None,
                        delta_values: List[int] = None,
                        compute_tail: bool = True,
                        compute_comm_proxies: bool = True,
                        beam_powers: torch.Tensor | None = None) -> Dict[str, float]:
    """Compute all metrics including tail-risk."""
    if top_k_values is None:
        top_k_values = [1, 2, 3, 5]
    if delta_values is None:
        delta_values = [1, 2]

    metrics = {}

    for k in top_k_values:
        metrics[f"top_{k}_acc"] = top_k_accuracy(predictions, targets, k=k)

    for k in top_k_values:
        for delta in delta_values:
            metrics[f"dba_k{k}_d{delta}"] = dba_score(predictions, targets, k=k, delta=delta)

    # Per-step accuracy
    step_accs = compute_per_step_accuracy(predictions, targets)
    metrics.update(step_accs)

    # Tail-risk metrics
    if compute_tail:
        tail = compute_tail_risk_metrics(predictions, targets, tail_fraction=0.1)
        metrics.update(tail)

    # Communication-related proxy metrics (explicitly marked *_proxy)
    if compute_comm_proxies:
        comm = compute_communication_proxy_metrics(predictions, targets, top_k_values)
        metrics.update(comm)
    if beam_powers is not None:
        try:
            true_comm = compute_communication_power_metrics(predictions, beam_powers, top_k_values)
            metrics.update(true_comm)
        except Exception:
            # Keep evaluation robust if optional power arrays are malformed/missing.
            pass

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    if prefix:
        print(f"\n{'='*65}")
        print(f"  {prefix}")
        print(f"{'='*65}")

    # Top-K
    print("  Top-K Accuracy:")
    for key, val in sorted(metrics.items()):
        if key.startswith("top_"):
            k = key.split("_")[1]
            print(f"    Top-{k}: {val*100:.2f}%")

    # DBA
    print("  DBA-Score:")
    for key, val in sorted(metrics.items()):
        if key.startswith("dba_"):
            parts = key.split("_")
            print(f"    {key}: {val:.4f}")

    # Per-step
    print("  Per-Step Accuracy:")
    for key, val in sorted(metrics.items()):
        if key.startswith("step_"):
            print(f"    {key}: {val*100:.1f}%")

    # Temporal baselines (if provided by evaluator)
    baseline_keys = [k for k in metrics if k.startswith("repeat_last_") or k.endswith("_minus_repeat_last")]
    if baseline_keys:
        print("  Repeat-Last Baseline:")
        if "repeat_last_top_1_acc" in metrics:
            print(f"    repeat_last_top_1_acc: {metrics['repeat_last_top_1_acc']*100:.1f}%")
        for key, val in sorted(metrics.items()):
            if key.startswith("repeat_last_step_"):
                print(f"    {key}: {val*100:.1f}%")
        if "top1_minus_repeat_last" in metrics:
            print(f"    top1_minus_repeat_last: {metrics['top1_minus_repeat_last']*100:+.1f}%")
        if "step1_minus_repeat_last" in metrics:
            print(f"    step1_minus_repeat_last: {metrics['step1_minus_repeat_last']*100:+.1f}%")

    # Tail-risk
    tail_keys = [k for k in metrics if k.startswith(("cvar", "worst", "percentile"))]
    if tail_keys:
        print("  Tail-Risk Metrics:")
        for key in sorted(tail_keys):
            if "acc" in key:
                print(f"    {key}: {metrics[key]*100:.1f}%")
            else:
                print(f"    {key}: {metrics[key]:.4f}")

    # Communication proxies
    comm_keys = [k for k in metrics if k.startswith((
        "nrp_proxy_", "sweep_overhead_proxy_", "topk_overhead_efficiency_",
        "nrp_k", "sweep_overhead_k", "power_overhead_efficiency_k"
    ))]
    if comm_keys:
        print("  Communication Metrics:")
        for key in sorted(comm_keys):
            val = metrics[key]
            if key.startswith(("nrp_proxy_", "sweep_overhead_proxy_", "nrp_k", "sweep_overhead_k")):
                print(f"    {key}: {val*100:.2f}%")
            else:
                print(f"    {key}: {val:.3f}")
