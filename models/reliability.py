"""
C2: Reliability-Guided Fusion with Reliability-Weighted Alignment Regularization.
Reference: Paper Section II-B, Section III-B.

Components:
  (a) Reliability estimator: uncertainty-aware precision-based weighting
  (b) Coarse modality gating (before fusion)
  (c) Reliability-weighted alignment regularization (training-time protection)
  (d) Cross-attention fusion for LLM token interface

Theoretical support: Gradient Contamination Bound (Section III-B).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ReliabilityEstimator(nn.Module):
    """
    (a) Uncertainty-aware reliability estimation.

    For each modality ω, produces:
        log σ̂²_ω(t) = f_var_ω([h̄_ω(t); q_ω(t); m_ω(t)])
        w_ω(t) ∝ exp(-log σ̂²_ω) = 1 / σ̂²_ω

    This is grounded in a precision-based interpretation:
      high reliability = low estimated feature uncertainty.
    """

    MODALITIES = ["image", "radar", "lidar", "gps"]

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 64,
                 reliability_min: float = 0.01,
                 logvar_clip: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 uniform_mix: float = 0.0,
                 ema_beta: float = 0.0):
        super().__init__()
        self.reliability_min = reliability_min
        self.logvar_clip = float(logvar_clip) if logvar_clip is not None else None
        self.softmax_temperature = float(max(softmax_temperature, 1e-3))
        self.uniform_mix = float(max(0.0, min(0.5, uniform_mix)))
        self.ema_beta = float(max(0.0, min(0.99, ema_beta)))

        # Per-modality variance head: f_var_ω
        # Input: [h̄_ω(t); q_ω(t); m_ω(t)]
        # q_ω = quality cues (1-dim: missing indicator, staleness, etc.)
        # m_ω = binary missing mask
        input_dim = feature_dim + 2  # feature + quality_cue + missing_mask
        self.var_heads = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),  # log σ̂²
            ) for m in self.MODALITIES
        })

        # Zero-init last layer of each var_head so all log_var=0 at init
        # → exp(-0)=1 for all modalities → uniform reliability weights (0.25 each)
        for m in self.MODALITIES:
            last_layer = self.var_heads[m][-1]  # the Linear(hidden//2, 1)
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(self, aligned_features: Dict[str, torch.Tensor],
                quality_cues: Optional[Dict[str, torch.Tensor]] = None,
                missing_masks: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            aligned_features: Dict[mod -> (B, H, feature_dim)]
            quality_cues:     Dict[mod -> (B, H, 1)] optional quality indicators
            missing_masks:    Dict[mod -> (B, H)] binary missing flags
        Returns:
            reliability_weights: Dict[mod -> (B, H)] normalized weights w_ω
            log_variances:       Dict[mod -> (B, H)] log σ̂²_ω
            gated_features:      Dict[mod -> (B, H, feature_dim)] reliability-gated
        """
        B, H, M = aligned_features[self.MODALITIES[0]].shape
        device = aligned_features[self.MODALITIES[0]].device

        if quality_cues is None:
            quality_cues = {m: torch.zeros(B, H, 1, device=device) for m in self.MODALITIES}
        if missing_masks is None:
            missing_masks = {m: torch.zeros(B, H, device=device) for m in self.MODALITIES}

        log_vars = {}
        for mod in self.MODALITIES:
            feat = aligned_features[mod]          # (B, H, M)
            q = quality_cues[mod]                  # (B, H, 1)
            m = missing_masks[mod].unsqueeze(-1)   # (B, H, 1)
            var_input = torch.cat([feat, q, m], dim=-1)  # (B, H, M+2)
            log_vars[mod] = self.var_heads[mod](var_input).squeeze(-1)  # (B, H)

        # Precision-based weights using predicted log-variance:
        # w_ω ∝ exp(-log σ̂²_ω) = 1/σ̂²_ω. Using -log_var directly avoids the
        # overly sharp exp(-exp(log_var)) saturation that can collapse to
        # clamp-min weights for most modalities.
        if self.logvar_clip is not None:
            log_vars_used = {
                m: log_vars[m].clamp(-self.logvar_clip, self.logvar_clip)
                for m in self.MODALITIES
            }
        else:
            log_vars_used = log_vars

        stacked = torch.stack(
            [-(log_vars_used[m] / self.softmax_temperature) for m in self.MODALITIES], dim=-1
        )  # (B, H, |Ω|)
        weights_stacked = F.softmax(stacked, dim=-1)  # (B, H, |Ω|)
        if self.uniform_mix > 0:
            n_mod = float(len(self.MODALITIES))
            weights_stacked = (1.0 - self.uniform_mix) * weights_stacked + \
                self.uniform_mix * (1.0 / n_mod)

        if self.ema_beta > 0.0 and H > 1:
            smoothed = [weights_stacked[:, 0]]
            beta = self.ema_beta
            for t in range(1, H):
                smoothed.append(beta * smoothed[-1] + (1.0 - beta) * weights_stacked[:, t])
            weights_stacked = torch.stack(smoothed, dim=1)

        # Clamp minimum reliability
        weights_stacked = weights_stacked.clamp(min=self.reliability_min)
        weights_stacked = weights_stacked / weights_stacked.sum(dim=-1, keepdim=True)

        reliability_weights = {}
        gated_features = {}
        for i, mod in enumerate(self.MODALITIES):
            w = weights_stacked[:, :, i]  # (B, H)
            reliability_weights[mod] = w
            # (b) Coarse modality gating
            gated_features[mod] = aligned_features[mod] * w.unsqueeze(-1)

        return reliability_weights, log_vars, gated_features


class ReliabilityWeightedAlignmentLoss(nn.Module):
    """
    (c) Reliability-weighted alignment regularization.

    L_align = Σ_t Σ_{ω≠ω'} w_ω(t)·w_{ω'}(t) · ||h̄_ω(t) - P_{ω←ω'}(h̄_{ω'}(t))||²

    When either modality is unreliable, alignment is suppressed,
    preventing noise from corrupting the shared representation space.

    Gradient Contamination Bound (Proposition):
        Var[||G_rw||] / Var[||G_std||] ≤ exp(-2(σ²_ω + σ²_{ω'})) / Z²
    """

    MODALITIES = ["image", "radar", "lidar", "gps"]

    def __init__(self, feature_dim: int = 64, proj_dim: int = 64,
                 pair_conf_threshold: float = 0.0):
        super().__init__()
        self.pair_conf_threshold = float(max(0.0, min(1.0, pair_conf_threshold)))
        # Lightweight pairwise projections P_{ω←ω'}
        self.projections = nn.ModuleDict()
        for i, m1 in enumerate(self.MODALITIES):
            for j, m2 in enumerate(self.MODALITIES):
                if i != j:
                    key = f"{m1}_from_{m2}"
                    self.projections[key] = nn.Linear(feature_dim, proj_dim, bias=False)

    def forward(self, aligned_features: Dict[str, torch.Tensor],
                reliability_weights: Dict[str, torch.Tensor],
                use_reliability: bool = True,
                detach_reliability: bool = False) -> torch.Tensor:
        """
        Args:
            aligned_features:    Dict[mod -> (B, H, feature_dim)]
            reliability_weights: Dict[mod -> (B, H)]
            use_reliability: if False, set w=1 for ablation (A3)
        Returns:
            L_align: scalar alignment loss
        """
        total_loss = 0.0
        count = 0

        for i, m1 in enumerate(self.MODALITIES):
            for j, m2 in enumerate(self.MODALITIES):
                if i >= j:
                    continue

                h1 = aligned_features[m1]  # (B, H, M)
                h2 = aligned_features[m2]  # (B, H, M)

                # Project m2 -> m1 space
                key = f"{m1}_from_{m2}"
                h2_proj = self.projections[key](h2)  # (B, H, proj_dim)
                h1_proj = h1[..., :h2_proj.shape[-1]]  # match dim if needed

                # Pairwise alignment error
                diff = (h1_proj - h2_proj).pow(2).sum(dim=-1)  # (B, H)

                if use_reliability:
                    # Reliability weighting: w_ω · w_{ω'}
                    rw1 = reliability_weights[m1].detach() if detach_reliability else reliability_weights[m1]
                    rw2 = reliability_weights[m2].detach() if detach_reliability else reliability_weights[m2]
                    w = rw1 * rw2  # (B, H)
                    if self.pair_conf_threshold > 0:
                        pair_mask = (
                            (rw1 >= self.pair_conf_threshold) &
                            (rw2 >= self.pair_conf_threshold)
                        ).float()
                        if float(pair_mask.sum().item()) > 0:
                            w = w * pair_mask
                    weighted_diff = (w * diff).mean()
                else:
                    weighted_diff = diff.mean()

                total_loss += weighted_diff
                count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0)


class CrossAttentionFusion(nn.Module):
    """
    (d) Cross-attention fusion for LLM token interface.

    Fuses gated modality features into M compact tokens using cross-attention:
        Z(t) = F({h^gated_ω(t)}_ω) ∈ R^{M×d}

    Uses learnable query tokens that attend to modality features.
    """

    def __init__(self, feature_dim: int = 64, num_fusion_tokens: int = 4,
                 num_heads: int = 4, dropout: float = 0.1, ffn_hidden: int = 256):
        super().__init__()
        self.num_tokens = num_fusion_tokens
        self.feature_dim = feature_dim

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_fusion_tokens, feature_dim) * 0.02)

        # Cross-attention: queries=fusion_tokens, keys/values=modality_features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)

        # Self-attention among fusion tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(feature_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(feature_dim)

    def forward(self, gated_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gated_features: Dict[mod -> (B, H, feature_dim)] reliability-gated features
        Returns:
            Z: (B, H, num_tokens, feature_dim) fused tokens per timestep
               Reshaped to (B, H*num_tokens, feature_dim) for LLM input
        """
        mods = list(gated_features.keys())
        B, H, M = gated_features[mods[0]].shape

        all_tokens = []
        for t in range(H):
            # Stack modality features at time t: (B, |Ω|, M)
            kv = torch.stack([gated_features[m][:, t, :] for m in mods], dim=1)

            # Query tokens
            queries = self.query_tokens.expand(B, -1, -1)  # (B, num_tokens, M)

            # Cross-attention: queries attend to modality features
            ca_out, _ = self.cross_attn(queries, kv, kv)
            ca_out = self.norm1(queries + ca_out)

            # Self-attention among fusion tokens
            sa_out, _ = self.self_attn(ca_out, ca_out, ca_out)
            sa_out = self.norm2(ca_out + sa_out)

            # FFN
            ffn_out = self.ffn(sa_out)
            tokens = self.norm3(sa_out + ffn_out)  # (B, num_tokens, M)
            all_tokens.append(tokens)

        # Stack: (B, H, num_tokens, M)
        Z = torch.stack(all_tokens, dim=1)
        # Reshape for LLM: (B, H*num_tokens, M)
        return Z.reshape(B, H * self.num_tokens, M)
