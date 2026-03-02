"""
Visualization utilities for training curves and experiment plots.
"""

import os
import re
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)


def _savefig(fig: plt.Figure, save_dir: str, filename: str):
    _ensure_dir(save_dir)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                         train_accs: List[float], val_accs: List[float],
                         save_dir: str = "logs"):
    """Plot and save training/validation loss and accuracy curves."""
    _ensure_dir(save_dir)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_accs, "b-", label="Train Top-1 Acc")
    ax2.plot(epochs, val_accs, "r-", label="Val Top-1 Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-1 Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    _savefig(fig, save_dir, "training_curves.png")


def plot_ssl_pretraining_curves(ssl_losses: List[float],
                                ssl_lrs: Optional[List[float]] = None,
                                save_dir: str = "logs"):
    """Plot Stage-1 SSL pretraining loss/LR curves."""
    if not ssl_losses:
        return
    epochs = np.arange(1, len(ssl_losses) + 1)
    if ssl_lrs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))
        ax2 = None

    ax1.plot(epochs, ssl_losses, color="tab:blue", marker="o", linewidth=2)
    ax1.set_title("Stage 1 SSL Pretraining Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SSL Loss")
    ax1.grid(True, alpha=0.3)

    if ax2 is not None:
        ax2.plot(epochs, ssl_lrs, color="tab:orange", marker="o", linewidth=2)
        ax2.set_title("Stage 1 SSL Learning Rate")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("LR")
        ax2.grid(True, alpha=0.3)

    _savefig(fig, save_dir, "stage1_ssl_curves.png")


def plot_beam_distribution(beam_indices: np.ndarray, num_beams: int = 64,
                           save_dir: str = "logs"):
    """Plot optimal beam index frequency distribution (Fig. 4)."""
    _ensure_dir(save_dir)

    plt.figure(figsize=(10, 5))
    plt.hist(beam_indices, bins=num_beams, range=(0, num_beams),
             edgecolor="black", alpha=0.7)
    plt.xlabel("Beam Index")
    plt.ylabel("Sample Count")
    plt.title("Optimal Beam Index Frequency Distribution")
    plt.grid(True, alpha=0.3)
    fig = plt.gcf()
    _savefig(fig, save_dir, "beam_distribution.png")


def plot_topk_comparison(results: Dict[str, Dict[str, float]],
                         k_values: List[int] = None,
                         save_dir: str = "logs"):
    """Plot Top-K accuracy comparison across models (Fig. 10/12)."""
    if k_values is None:
        k_values = [1, 2, 3]

    _ensure_dir(save_dir)

    x = np.arange(len(k_values))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model_name, metrics) in enumerate(results.items()):
        accs = [metrics.get(f"top_{k}_acc", 0) * 100 for k in k_values]
        ax.bar(x + i * width, accs, width, label=model_name)

    ax.set_xlabel("Top-K")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-K Accuracy Comparison")
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels([f"Top-{k}" for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    _savefig(fig, save_dir, "topk_comparison.png")


def _metric_pct(metrics: Dict, key: str) -> float:
    return float(metrics.get(key, 0.0)) * 100.0


def plot_s1_stress_results(results: Dict[str, Dict[str, float]],
                           save_dir: str = "logs"):
    """Plot S1 asynchrony stress (delay + burst-missing) curves."""
    if not results:
        return
    delay_rows = []
    burst_rows = []
    for k, v in results.items():
        m_delay = re.match(r"delay_(\d+)ms$", k)
        m_burst = re.match(r"burst_(\d+)$", k)
        if m_delay:
            delay_rows.append((int(m_delay.group(1)), v))
        elif m_burst:
            burst_rows.append((int(m_burst.group(1)), v))
    delay_rows.sort(key=lambda x: x[0])
    burst_rows.sort(key=lambda x: x[0])
    if not delay_rows and not burst_rows:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_dt, ax_db, ax_bt, ax_bb = axes.flat

    if delay_rows:
        x = [d for d, _ in delay_rows]
        y_top1 = [_metric_pct(m, "top_1_acc") for _, m in delay_rows]
        y_top5 = [_metric_pct(m, "top_5_acc") for _, m in delay_rows]
        y_w10 = [_metric_pct(m, "worst_10_acc") for _, m in delay_rows]
        y_cvar = [float(m.get("cvar_10", 0.0)) for _, m in delay_rows]
        ax_dt.plot(x, y_top1, marker="o", label="Top-1")
        if any(v > 0 for v in y_top5):
            ax_dt.plot(x, y_top5, marker="s", label="Top-5")
        ax_dt.set_title("S1 Delay Stress: Accuracy vs Delay")
        ax_dt.set_xlabel("Delay (ms)")
        ax_dt.set_ylabel("Accuracy (%)")
        ax_dt.grid(True, alpha=0.3)
        ax_dt.legend()

        ax_db.plot(x, y_w10, marker="o", label="Worst-10% Acc")
        ax_db2 = ax_db.twinx()
        ax_db2.plot(x, y_cvar, marker="s", color="tab:red", label="CVaR-10")
        ax_db.set_title("S1 Delay Stress: Tail Metrics")
        ax_db.set_xlabel("Delay (ms)")
        ax_db.set_ylabel("Worst-10% Acc (%)")
        ax_db2.set_ylabel("CVaR-10")
        ax_db.grid(True, alpha=0.3)
        lines = ax_db.get_lines() + ax_db2.get_lines()
        ax_db.legend(lines, [l.get_label() for l in lines], loc="best")
    else:
        ax_dt.axis("off")
        ax_db.axis("off")

    if burst_rows:
        x = [b for b, _ in burst_rows]
        y_top1 = [_metric_pct(m, "top_1_acc") for _, m in burst_rows]
        y_top5 = [_metric_pct(m, "top_5_acc") for _, m in burst_rows]
        y_w10 = [_metric_pct(m, "worst_10_acc") for _, m in burst_rows]
        ax_bt.plot(x, y_top1, marker="o", label="Top-1")
        if any(v > 0 for v in y_top5):
            ax_bt.plot(x, y_top5, marker="s", label="Top-5")
        ax_bt.set_title("S1 Burst Missing: Accuracy vs Burst Length")
        ax_bt.set_xlabel("Burst Length (frames)")
        ax_bt.set_ylabel("Accuracy (%)")
        ax_bt.grid(True, alpha=0.3)
        ax_bt.legend()

        ax_bb.plot(x, y_w10, marker="o", color="tab:orange")
        ax_bb.set_title("S1 Burst Missing: Worst-10% Acc")
        ax_bb.set_xlabel("Burst Length (frames)")
        ax_bb.set_ylabel("Worst-10% Acc (%)")
        ax_bb.grid(True, alpha=0.3)
    else:
        ax_bt.axis("off")
        ax_bb.axis("off")

    _savefig(fig, save_dir, "S1_asynchrony_stress.png")


def plot_s2_stress_results(results: Dict[str, Dict[str, float]],
                           save_dir: str = "logs"):
    """Plot S2 degradation stress curves (single-modality and combined)."""
    if not results:
        return
    mods = ["image", "radar", "lidar", "gps"]
    per_mod = {m: [] for m in mods}
    combined = []
    for k, v in results.items():
        m = re.match(r"^(image|radar|lidar|gps)_alpha_([0-9.]+)$", k)
        c = re.match(r"^combined_alpha_([0-9.]+)$", k)
        if m:
            per_mod[m.group(1)].append((float(m.group(2)), v))
        elif c:
            combined.append((float(c.group(1)), v))
    for m in mods:
        per_mod[m].sort(key=lambda x: x[0])
    combined.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_single_top1, ax_single_tail, ax_comb_top1, ax_comb_tail = axes.flat

    colors = {"image": "tab:blue", "radar": "tab:orange", "lidar": "tab:green", "gps": "tab:red"}
    for mod in mods:
        rows = per_mod[mod]
        if not rows:
            continue
        x = [a for a, _ in rows]
        ax_single_top1.plot(x, [_metric_pct(m, "top_1_acc") for _, m in rows],
                            marker="o", label=mod, color=colors[mod])
        ax_single_tail.plot(x, [_metric_pct(m, "worst_10_acc") for _, m in rows],
                            marker="o", label=mod, color=colors[mod])
    ax_single_top1.set_title("S2 Single-Modality Degradation: Top-1")
    ax_single_top1.set_xlabel("Severity α")
    ax_single_top1.set_ylabel("Top-1 (%)")
    ax_single_top1.grid(True, alpha=0.3)
    ax_single_top1.legend()
    ax_single_tail.set_title("S2 Single-Modality Degradation: Worst-10% Acc")
    ax_single_tail.set_xlabel("Severity α")
    ax_single_tail.set_ylabel("Worst-10% Acc (%)")
    ax_single_tail.grid(True, alpha=0.3)
    ax_single_tail.legend()

    if combined:
        x = [a for a, _ in combined]
        y1 = [_metric_pct(m, "top_1_acc") for _, m in combined]
        y5 = [_metric_pct(m, "top_5_acc") for _, m in combined]
        yw = [_metric_pct(m, "worst_10_acc") for _, m in combined]
        yc = [float(m.get("cvar_10", 0.0)) for _, m in combined]
        ax_comb_top1.plot(x, y1, marker="o", label="Top-1")
        if any(v > 0 for v in y5):
            ax_comb_top1.plot(x, y5, marker="s", label="Top-5")
        ax_comb_top1.set_title("S2 Combined Degradation: Accuracy")
        ax_comb_top1.set_xlabel("Severity α")
        ax_comb_top1.set_ylabel("Accuracy (%)")
        ax_comb_top1.grid(True, alpha=0.3)
        ax_comb_top1.legend()

        ax_comb_tail.plot(x, yw, marker="o", label="Worst-10% Acc")
        ax2 = ax_comb_tail.twinx()
        ax2.plot(x, yc, marker="s", color="tab:red", label="CVaR-10")
        ax_comb_tail.set_title("S2 Combined Degradation: Tail Metrics")
        ax_comb_tail.set_xlabel("Severity α")
        ax_comb_tail.set_ylabel("Worst-10% Acc (%)")
        ax2.set_ylabel("CVaR-10")
        ax_comb_tail.grid(True, alpha=0.3)
        lines = ax_comb_tail.get_lines() + ax2.get_lines()
        ax_comb_tail.legend(lines, [l.get_label() for l in lines], loc="best")
    else:
        ax_comb_top1.axis("off")
        ax_comb_tail.axis("off")

    _savefig(fig, save_dir, "S2_degradation_stress.png")


def plot_s2_reliability_diagnostics(results: Dict[str, Dict[str, float]],
                                    save_dir: str = "logs"):
    """Plot S2 reliability/logvar/quality-cue diagnostics vs degradation severity."""
    if not results:
        return

    mods = ["image", "radar", "lidar", "gps"]
    per_mod = {m: [] for m in mods}
    combined = []
    for k, v in results.items():
        m = re.match(r"^(image|radar|lidar|gps)_alpha_([0-9.]+)$", k)
        c = re.match(r"^combined_alpha_([0-9.]+)$", k)
        if m:
            per_mod[m.group(1)].append((float(m.group(2)), v))
        elif c:
            combined.append((float(c.group(1)), v))
    for m in mods:
        per_mod[m].sort(key=lambda x: x[0])
    combined.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_single_rel, ax_comb_rel, ax_comb_q, ax_comb_misc = axes.flat
    colors = {"image": "tab:blue", "radar": "tab:orange", "lidar": "tab:green", "gps": "tab:red"}

    for mod in mods:
        rows = per_mod[mod]
        if not rows:
            continue
        x = [a for a, _ in rows]
        y = [100.0 * float(m.get(f"reliability_mean_{mod}", 0.0)) for _, m in rows]
        ax_single_rel.plot(x, y, marker="o", color=colors[mod], label=f"{mod} degraded")
    ax_single_rel.set_title("S2 Single-Modality: Reliability of Corrupted Modality")
    ax_single_rel.set_xlabel("Severity α")
    ax_single_rel.set_ylabel("Mean Reliability (%)")
    ax_single_rel.grid(True, alpha=0.3)
    ax_single_rel.legend()

    if combined:
        x = [a for a, _ in combined]
        for mod in mods:
            y = [100.0 * float(m.get(f"reliability_mean_{mod}", 0.0)) for _, m in combined]
            ax_comb_rel.plot(x, y, marker="o", color=colors[mod], label=mod)
        ax_comb_rel.set_title("S2 Combined: Reliability Weights")
        ax_comb_rel.set_xlabel("Severity α")
        ax_comb_rel.set_ylabel("Mean Reliability (%)")
        ax_comb_rel.grid(True, alpha=0.3)
        ax_comb_rel.legend()

        for mod in mods:
            y = [float(m.get(f"quality_cue_mean_{mod}", 0.0)) for _, m in combined]
            ax_comb_q.plot(x, y, marker="o", color=colors[mod], label=mod)
        ax_comb_q.set_title("S2 Combined: Quality Cues")
        ax_comb_q.set_xlabel("Severity α")
        ax_comb_q.set_ylabel("Mean Quality Cue")
        ax_comb_q.grid(True, alpha=0.3)
        ax_comb_q.legend()

        ymax = [100.0 * float(m.get("reliability_max_mean", 0.0)) for _, m in combined]
        ent = [float(m.get("reliability_entropy_mean", 0.0)) for _, m in combined]
        ax_comb_misc.plot(x, ymax, marker="o", label="Max weight (%)")
        ax2 = ax_comb_misc.twinx()
        ax2.plot(x, ent, marker="s", color="tab:red", label="Entropy")
        ax_comb_misc.set_title("S2 Combined: Weight Concentration")
        ax_comb_misc.set_xlabel("Severity α")
        ax_comb_misc.set_ylabel("Max Reliability (%)")
        ax2.set_ylabel("Entropy")
        ax_comb_misc.grid(True, alpha=0.3)
        lines = ax_comb_misc.get_lines() + ax2.get_lines()
        ax_comb_misc.legend(lines, [l.get_label() for l in lines], loc="best")
    else:
        ax_comb_rel.axis("off")
        ax_comb_q.axis("off")
        ax_comb_misc.axis("off")

    _savefig(fig, save_dir, "S2_reliability_diagnostics.png")


def plot_s3_domain_shift_results(results: Dict[str, Dict],
                                 save_dir: str = "logs"):
    """Plot S3 domain shift in-domain vs out-domain metrics."""
    if not results:
        return
    in_m = results.get("in_domain", {})
    out_m = results.get("out_domain", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["In-domain", "Out-domain"]
    x = np.arange(len(labels))
    top1 = [_metric_pct(in_m, "top_1_acc"), _metric_pct(out_m, "top_1_acc")]
    top5 = [_metric_pct(in_m, "top_5_acc"), _metric_pct(out_m, "top_5_acc")]
    w10 = [_metric_pct(in_m, "worst_10_acc"), _metric_pct(out_m, "worst_10_acc")]
    cvar = [float(in_m.get("cvar_10", 0.0)), float(out_m.get("cvar_10", 0.0))]

    w = 0.35
    ax1.bar(x - w / 2, top1, w, label="Top-1")
    ax1.bar(x + w / 2, top5, w, label="Top-5")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("S3 Domain Shift: Accuracy")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend()

    ax2.bar(x - w / 2, w10, w, label="Worst-10% Acc")
    ax2b = ax2.twinx()
    ax2b.plot(x, cvar, color="tab:red", marker="o", linewidth=2, label="CVaR-10")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Worst-10% Acc (%)")
    ax2b.set_ylabel("CVaR-10")
    ax2.set_title("S3 Domain Shift: Tail Metrics")
    ax2.grid(True, alpha=0.3, axis="y")
    lines = ax2.get_lines() + ax2b.get_lines()
    # bar containers don't appear in get_lines; add manual legend
    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="best")

    _savefig(fig, save_dir, "S3_domain_shift_stress.png")


def plot_e1_gradient_contamination(results: Dict[str, List[Dict]],
                                   save_dir: str = "logs"):
    """Plot E1 gradient contamination comparison (weighted vs unweighted)."""
    if not results or not results.get("weighted") or not results.get("unweighted"):
        return

    def _aggregate(entries: List[Dict], key: str):
        xs = sorted({float(e.get("severity", 0.0)) for e in entries})
        ys = []
        for s in xs:
            vals = [float(e.get(key, 0.0)) for e in entries if float(e.get("severity", 0.0)) == s]
            ys.append(float(np.mean(vals)) if vals else 0.0)
        return xs, ys

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    keys = [
        ("grad_norm_image", "Image Alignment Grad Norm"),
        ("grad_norm_radar", "Radar Alignment Grad Norm"),
        ("repr_drift_image", "Image Representation Drift"),
    ]
    for ax, (k, title) in zip(axes, keys):
        xw, yw = _aggregate(results["weighted"], k)
        xu, yu = _aggregate(results["unweighted"], k)
        if xw:
            ax.plot(xw, yw, marker="o", label="Weighted")
        if xu:
            ax.plot(xu, yu, marker="s", label="Unweighted")
        ax.set_title(title)
        ax.set_xlabel("Degradation Severity")
        ax.grid(True, alpha=0.3)
        ax.legend()

    _savefig(fig, save_dir, "E1_gradient_contamination.png")

    # Additional diagnostic (closer to paper intent): reliability-conditioned bins
    # and a theoretical suppression trend proxy exp(-2 sigma^2).
    weighted = results.get("weighted", [])
    unweighted = results.get("unweighted", [])
    if weighted and unweighted:
        fig2, (ax_rbin, ax_theory) = plt.subplots(1, 2, figsize=(12, 4.5))

        def _vals(entries, key):
            return np.asarray([float(e.get(key, 0.0)) for e in entries], dtype=float)

        rw_rel = _vals(weighted, "reliability_image")
        rw_grad = _vals(weighted, "grad_norm_image")
        uw_rel = _vals(unweighted, "reliability_image")
        uw_grad = _vals(unweighted, "grad_norm_image")
        if rw_rel.size and uw_rel.size:
            # Reliability quartile bins from weighted path.
            q = np.quantile(rw_rel, [0, 0.25, 0.5, 0.75, 1.0])
            data_w, data_u, labels = [], [], []
            for i in range(4):
                lo, hi = q[i], q[i + 1]
                mw = (rw_rel >= lo) & (rw_rel < hi + 1e-12)
                mu = (uw_rel >= lo) & (uw_rel < hi + 1e-12)
                if mw.any() or mu.any():
                    data_w.append(rw_grad[mw] if mw.any() else np.array([0.0]))
                    data_u.append(uw_grad[mu] if mu.any() else np.array([0.0]))
                    labels.append(f"Q{i+1}")
            if data_w:
                pos = np.arange(len(data_w)) * 2.0
                ax_rbin.violinplot(data_w, positions=pos - 0.3, widths=0.5, showmeans=True)
                ax_rbin.violinplot(data_u, positions=pos + 0.3, widths=0.5, showmeans=True)
                ax_rbin.set_xticks(pos)
                ax_rbin.set_xticklabels(labels)
                ax_rbin.set_title("E1: Grad Norm by Reliability Quartile")
                ax_rbin.set_xlabel("Image reliability quartile (weighted bins)")
                ax_rbin.set_ylabel("Image alignment grad norm")
                # Manual legend handles
                ax_rbin.plot([], [], color="tab:blue", label="Weighted")
                ax_rbin.plot([], [], color="tab:orange", label="Unweighted")
                ax_rbin.legend(loc="best")
                ax_rbin.grid(True, alpha=0.3)

        sig2 = _vals(weighted, "sigma2_image")
        grad = rw_grad if "rw_grad" in locals() else _vals(weighted, "grad_norm_image")
        if sig2.size and grad.size:
            order = np.argsort(sig2)
            sig2_s = sig2[order]
            grad_s = grad[order]
            ax_theory.scatter(sig2, grad, s=14, alpha=0.35, label="Observed (weighted)")
            # Normalize theoretical suppression proxy into observed range for visual comparison.
            th = np.exp(-2.0 * sig2_s)
            if np.max(th) - np.min(th) > 1e-12 and np.max(grad_s) - np.min(grad_s) > 1e-12:
                th_scaled = (th - np.min(th)) / (np.max(th) - np.min(th))
                th_scaled = th_scaled * (np.max(grad_s) - np.min(grad_s)) + np.min(grad_s)
                ax_theory.plot(sig2_s, th_scaled, color="tab:red", linewidth=2,
                               label="Theory trend ∝ exp(-2σ²)")
            ax_theory.set_title("E1: Grad Suppression vs σ̂² (Image)")
            ax_theory.set_xlabel("Estimated variance σ̂²")
            ax_theory.set_ylabel("Image alignment grad norm")
            ax_theory.grid(True, alpha=0.3)
            ax_theory.legend(loc="best")

        _savefig(fig2, save_dir, "E1_gradient_contamination_diagnostic.png")


def plot_reliability_calibration_summary(summary: Dict,
                                         save_dir: str = "logs"):
    """Plot reliability calibration diagnostics (E0) from summary dict."""
    mods = list(summary.get("mods", {}).keys())
    if not mods:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, mod in zip(axes.flat, mods):
        sm = summary["mods"][mod]
        rel = np.asarray(sm.get("reliability", []), dtype=float)
        correct = np.asarray(sm.get("correctness", []), dtype=float)
        bins = np.asarray(sm.get("bins", []), dtype=float)
        bin_acc = np.asarray(sm.get("bin_acc", []), dtype=float) * 100.0
        if rel.size > 0 and correct.size > 0:
            ax.scatter(rel, correct * 100.0, s=8, alpha=0.12, color="tab:blue")
        if bins.size >= 2 and bin_acc.size == bins.size - 1:
            centers = 0.5 * (bins[:-1] + bins[1:])
            ax.plot(centers, bin_acc, color="tab:red", marker="o", linewidth=2,
                    label="Binned acc")
        ax.set_title(f"{mod} reliability")
        ax.set_xlabel("Reliability weight")
        ax.set_ylabel("Per-sample acc (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    _savefig(fig, save_dir, "E0_reliability_calibration.png")


def plot_ablation_results(ablation_results: Dict[str, Dict[str, float]],
                          save_dir: str = "logs"):
    """Plot A1-A5 ablation summary bars."""
    if not ablation_results:
        return
    preferred_order = [
        "Full",
        "A1_no_regime",
        "A2_no_repair",
        "A3_no_rw_align",
        "A4_no_moe",
        "A5_vanilla_transformer",
    ]
    keys = [k for k in preferred_order if k in ablation_results] + [
        k for k in ablation_results.keys() if k not in preferred_order
    ]
    labels = [k.replace("A1_no_regime", "A1")
                .replace("A2_no_repair", "A2")
                .replace("A3_no_rw_align", "A3")
                .replace("A4_no_moe", "A4")
                .replace("A5_vanilla_transformer", "A5")
              for k in keys]
    top1 = [_metric_pct(ablation_results[k], "top_1_acc") for k in keys]
    top5 = [_metric_pct(ablation_results[k], "top_5_acc") for k in keys]
    worst10 = [_metric_pct(ablation_results[k], "worst_10_acc") for k in keys]
    cvar = [float(ablation_results[k].get("cvar_10", 0.0)) for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(keys))
    w = 0.38

    axes[0].bar(x - w/2, top1, w, label="Top-1")
    axes[0].bar(x + w/2, top5, w, label="Top-5")
    axes[0].set_title("Ablation Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend()

    axes[1].bar(x, worst10, color="tab:orange")
    axes[1].set_title("Ablation Worst-10% Accuracy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25)
    axes[1].set_ylabel("Worst-10% Acc (%)")
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(x, cvar, color="tab:red")
    axes[2].set_title("Ablation CVaR-10")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25)
    axes[2].set_ylabel("CVaR-10")
    axes[2].grid(True, alpha=0.3, axis="y")

    _savefig(fig, save_dir, "Ablation_A1_A5_summary.png")


def plot_complexity_breakdown(results: Dict,
                              save_dir: str = "logs"):
    """Plot C4 complexity summary (top-level rows only)."""
    if not results or "breakdown" not in results:
        return
    rows = [r for r in results["breakdown"] if not r.get("is_subrow", False)]
    if not rows:
        return

    names = [r["module"] for r in rows]
    params_m = [r["params"] / 1e6 for r in rows]
    trainable_m = [r["trainable"] / 1e6 for r in rows]
    mem_mib = [r["memory_mib"] for r in rows]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, 0.45 * len(rows) + 2)))
    axes[0].barh(y, params_m, color="tab:blue")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Params (M)")
    axes[0].set_title("C4 Top-Level Params")
    axes[0].grid(True, alpha=0.3, axis="x")

    axes[1].barh(y, trainable_m, color="tab:green")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Trainable Params (M)")
    axes[1].set_title("C4 Top-Level Trainable")
    axes[1].grid(True, alpha=0.3, axis="x")

    axes[2].barh(y, mem_mib, color="tab:orange")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Param Mem (MiB, FP32)")
    axes[2].set_title("C4 Param Memory")
    axes[2].grid(True, alpha=0.3, axis="x")

    _savefig(fig, save_dir, "C4_complexity_breakdown.png")


def plot_s1_modality_delay_results(results: Dict[str, Dict[str, float]],
                                   save_dir: str = "logs"):
    """Plot S1 modality-specific delay stress curves if present."""
    rows = {}
    for k, v in (results or {}).items():
        m = re.match(r"^delay_(image|radar|lidar|gps)_(\d+)ms$", k)
        if not m:
            continue
        mod = m.group(1)
        delay = int(m.group(2))
        rows.setdefault(mod, []).append((delay, v))
    if not rows:
        return
    for mod in rows:
        rows[mod].sort(key=lambda x: x[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    colors = {"image": "tab:blue", "radar": "tab:orange", "lidar": "tab:green", "gps": "tab:red"}
    for mod, entries in rows.items():
        x = [d for d, _ in entries]
        y1 = [_metric_pct(m, "top_1_acc") for _, m in entries]
        yw = [_metric_pct(m, "worst_10_acc") for _, m in entries]
        ax1.plot(x, y1, marker="o", label=mod, color=colors.get(mod))
        ax2.plot(x, yw, marker="o", label=mod, color=colors.get(mod))
    ax1.set_title("S1 Modality-Specific Delay: Top-1")
    ax1.set_xlabel("Delay (ms)")
    ax1.set_ylabel("Top-1 (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_title("S1 Modality-Specific Delay: Worst-10% Acc")
    ax2.set_xlabel("Delay (ms)")
    ax2.set_ylabel("Worst-10% Acc (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    _savefig(fig, save_dir, "S1_asynchrony_stress_modality_delay.png")


def plot_e2_delay_regime_specialization(results: Dict[str, Dict[str, Dict[str, float]]],
                                        save_dir: str = "logs"):
    """Plot E2 baseline vs A1 under delay stress for Top-1 and Worst-10% accuracy."""
    if not results:
        return
    baseline = results.get("baseline", {})
    a1 = results.get("A1", {})

    def _delay_rows(d: Dict[str, Dict[str, float]]):
        rows = []
        for k, v in d.items():
            m = re.match(r"^delay_(\d+)ms$", k)
            if m:
                rows.append((int(m.group(1)), v))
        return sorted(rows, key=lambda x: x[0])

    b_rows = _delay_rows(baseline)
    a_rows = _delay_rows(a1)
    if not b_rows and not a_rows:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    if b_rows:
        xb = [d for d, _ in b_rows]
        ax1.plot(xb, [_metric_pct(m, "top_1_acc") for _, m in b_rows], marker="o", label="Regime-aware")
        ax2.plot(xb, [_metric_pct(m, "worst_10_acc") for _, m in b_rows], marker="o", label="Regime-aware")
    if a_rows:
        xa = [d for d, _ in a_rows]
        ax1.plot(xa, [_metric_pct(m, "top_1_acc") for _, m in a_rows], marker="s", label="A1 no-regime")
        ax2.plot(xa, [_metric_pct(m, "worst_10_acc") for _, m in a_rows], marker="s", label="A1 no-regime")
    ax1.set_title("E2 Delay-Regime Specialization: Top-1 vs Delay")
    ax1.set_xlabel("Delay (ms)")
    ax1.set_ylabel("Top-1 (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_title("E2 Delay-Regime Specialization: Worst-10% vs Delay")
    ax2.set_xlabel("Delay (ms)")
    ax2.set_ylabel("Worst-10% Acc (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    _savefig(fig, save_dir, "E2_delay_regime_specialization.png")


def plot_e4_reliability_paper_calibration(summary: Dict,
                                          save_dir: str = "logs"):
    """Plot paper-style E4 calibration: sigma^2 vs per-modality error."""
    mods = list((summary or {}).get("mods", {}).keys())
    if not mods:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, mod in zip(axes.flat, mods):
        sm = summary["mods"][mod]
        sigma2 = np.asarray(sm.get("sigma2_hat", []), dtype=float)
        err = np.asarray(sm.get("modality_error", []), dtype=float)
        bins = np.asarray(sm.get("sigma2_bins", []), dtype=float)
        bin_err = np.asarray(sm.get("sigma2_bin_err", []), dtype=float)
        if sigma2.size and err.size:
            ax.scatter(sigma2, err, s=7, alpha=0.12, color="tab:blue")
        if bins.size >= 2 and bin_err.size == bins.size - 1:
            centers = 0.5 * (bins[:-1] + bins[1:])
            ax.plot(centers, bin_err, color="tab:red", marker="o", linewidth=2, label="Binned error")
        ax.set_title(f"{mod}: σ̂² vs eω")
        ax.set_xlabel("Estimated variance σ̂²")
        ax.set_ylabel("Per-modality beam error")
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend(loc="best")
    _savefig(fig, save_dir, "E4_reliability_calibration_paper.png")


def plot_e4_reliability_monotonicity(results: Dict,
                                     save_dir: str = "logs"):
    """Plot S2 severity vs mean estimated variance/reliability for monotonicity check."""
    mods = list((results or {}).get("mods", {}).keys())
    if not mods:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax_var, ax_rel = axes
    for mod in mods:
        rows = sorted(results["mods"][mod].get("severity_rows", []), key=lambda x: x["alpha"])
        if not rows:
            continue
        x = [float(r["alpha"]) for r in rows]
        y_var = [float(r.get("sigma2_mean", 0.0)) for r in rows]
        y_rel = [float(r.get("reliability_mean", 0.0)) for r in rows]
        ax_var.plot(x, y_var, marker="o", label=mod)
        ax_rel.plot(x, y_rel, marker="o", label=mod)
    ax_var.set_title("E4 Monotonicity: σ̂² vs Corruption Severity")
    ax_var.set_xlabel("Severity α")
    ax_var.set_ylabel("Mean σ̂²")
    ax_var.grid(True, alpha=0.3)
    ax_var.legend()
    ax_rel.set_title("E4 Monotonicity: Reliability vs Corruption Severity")
    ax_rel.set_xlabel("Severity α")
    ax_rel.set_ylabel("Mean reliability wω")
    ax_rel.grid(True, alpha=0.3)
    ax_rel.legend()
    _savefig(fig, save_dir, "E4_reliability_monotonicity.png")


def plot_c4_latency_benchmark(results: Dict,
                              save_dir: str = "logs"):
    """Plot optional C4 latency/FLOPs benchmark summary."""
    if not results:
        return
    labels = []
    vals = []
    if "latency_ms_mean" in results:
        labels.append("Latency (ms)")
        vals.append(float(results["latency_ms_mean"]))
    if "flops_per_sample" in results and results["flops_per_sample"] is not None:
        labels.append("FLOPs/sample (G)")
        vals.append(float(results["flops_per_sample"]) / 1e9)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, vals, color=["tab:blue", "tab:orange"][:len(vals)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("C4 Latency / FLOPs Benchmark")
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, save_dir, "C4_latency_flops_benchmark.png")
