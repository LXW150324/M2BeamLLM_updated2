"""
C1: Delay-Regime-Aware Asynchronous Alignment for Beam Prediction.
Reference: Paper Section II-A, Section III-A.

Aligns each modality into a decision-time representation using:
  (a) Windowed candidate retrieval
  (b) Staleness embedding and temporal weighting
  (c) Short-gap repair (first-order Taylor extrapolation)
  (d) Residual alignment refinement with delay-regime conditioning

Theoretical support: Staleness-Induced Beam Misalignment Bound (Section III-A).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SinusoidalStalenessEmbedding(nn.Module):
    """
    Learnable staleness embedding ϕ(Δt) using sinusoidal positional encoding
    followed by a small MLP. Maps continuous staleness values to embeddings.
    """

    def __init__(self, embed_dim: int = 32, max_staleness_ms: float = 300.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_staleness_ms = max_staleness_ms

        # Sinusoidal frequencies (fixed)
        half_dim = embed_dim // 2
        freqs = torch.exp(torch.arange(half_dim).float() * -(math.log(10000.0) / half_dim))
        self.register_buffer("freqs", freqs)

        # MLP to project sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, staleness_ms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            staleness_ms: (...) staleness in milliseconds
        Returns:
            (..., embed_dim) staleness embedding
        """
        # Normalize to [0, 1]
        t = staleness_ms.clamp(0, self.max_staleness_ms) / self.max_staleness_ms
        # Sinusoidal encoding
        t_expanded = t.unsqueeze(-1) * self.freqs * math.pi  # (..., half_dim)
        sin_enc = torch.sin(t_expanded)
        cos_enc = torch.cos(t_expanded)
        emb = torch.cat([sin_enc, cos_enc], dim=-1)  # (..., embed_dim)
        return self.mlp(emb)


class StalenessAwareWeighting(nn.Module):
    """
    (a)+(b): Windowed candidate retrieval with staleness-aware attention weighting.
    For each modality, computes soft mixture over K+1 buffered observations:
        α_{ω,τ}(t) = softmax_τ(s_{ω,τ}(t))
        h̃_ω(t) = Σ_τ α_{ω,τ}(t) h_ω(t-τ)
    """

    def __init__(self, feature_dim: int = 64, staleness_embed_dim: int = 32):
        super().__init__()
        # Attention score: v^T tanh(W_h h + W_d ϕ(Δt) + b)
        self.W_h = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_d = nn.Linear(staleness_embed_dim, feature_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(feature_dim))
        self.v = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, h_buffer: torch.Tensor,
                staleness_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_buffer:        (B, K+1, feature_dim) buffered encoded features
            staleness_embed: (B, K+1, staleness_embed_dim)
        Returns:
            h_tilde: (B, feature_dim) aligned feature
            alpha:   (B, K+1) attention weights
        """
        # Score: (B, K+1, feature_dim)
        score_input = torch.tanh(self.W_h(h_buffer) + self.W_d(staleness_embed) + self.b)
        # (B, K+1, 1) -> (B, K+1)
        scores = self.v(score_input).squeeze(-1)
        alpha = F.softmax(scores, dim=-1)  # (B, K+1)
        # Soft mixture
        h_tilde = (alpha.unsqueeze(-1) * h_buffer).sum(dim=1)  # (B, feature_dim)
        return h_tilde, alpha


class ShortGapRepair(nn.Module):
    """
    (c): Short-gap repair via first-order Taylor extrapolation in feature space.
        ĥ_ω(t) = h̃_ω(t-1) + η_ω (h̃_ω(t-1) - h̃_ω(t-2))
    where η_ω = σ(η_raw) ∈ [0,1] is a per-modality learnable scalar.

    Physical interpretation:
        η=0 → zero-order hold (repeat last valid feature)
        η=1 → full linear extrapolation
    Stability: ||ĥ_ω(t)|| ≤ 3·C_norm (bounded by layer-norm bound).
    """

    def __init__(self, num_modalities: int = 4):
        super().__init__()
        # Per-modality learnable extrapolation strength, init so η ≈ 0.5
        self.eta_raw = nn.Parameter(torch.zeros(num_modalities))

    def forward(self, h_prev1: torch.Tensor, h_prev2: torch.Tensor,
                modality_idx: int) -> torch.Tensor:
        """
        Args:
            h_prev1: (B, feature_dim) h̃_ω(t-1)
            h_prev2: (B, feature_dim) h̃_ω(t-2)
            modality_idx: index of the modality
        Returns:
            h_repaired: (B, feature_dim)
        """
        eta = torch.sigmoid(self.eta_raw[modality_idx])
        return h_prev1 + eta * (h_prev1 - h_prev2)


class ResidualAlignmentRefinement(nn.Module):
    """
    (d): Residual alignment refinement with delay-regime conditioning.
        h̄_ω(t) = h̃_ω(t) + MLP_ω([h̃_ω(t); ϕ(Δt); m_ω(t)])

    Delay-regime conditioning: discretize staleness into R regimes,
    activate regime-specific FiLM parameters inside MLP.
    """

    def __init__(self, feature_dim: int = 64, staleness_embed_dim: int = 32,
                 hidden_dim: int = 128, num_regimes: int = 3,
                 regime_adapter_dim: int = 32):
        super().__init__()
        # Input: [h̃; ϕ(Δt); m_ω] where m_ω is 1-dim binary
        input_dim = feature_dim + staleness_embed_dim + 1

        # Main refinement MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        # Regime-specific FiLM (Feature-wise Linear Modulation) adapters
        # γ and β for each regime
        self.regime_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(regime_adapter_dim, hidden_dim),
                nn.Tanh(),
            ) for _ in range(num_regimes)
        ])
        self.regime_bias = nn.ModuleList([
            nn.Linear(regime_adapter_dim, hidden_dim)
            for _ in range(num_regimes)
        ])
        # Regime embedding
        self.regime_embed = nn.Embedding(num_regimes, regime_adapter_dim)

        # Zero-init fc2 so residual starts at 0 (module is identity at init)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h_tilde: torch.Tensor, staleness_embed: torch.Tensor,
                missing_mask: torch.Tensor, regime_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_tilde:         (B, feature_dim) preliminary aligned feature
            staleness_embed: (B, staleness_embed_dim)
            missing_mask:    (B, 1) binary mask (1 if repaired/missing)
            regime_idx:      (B,) integer delay regime index
        Returns:
            h_bar: (B, feature_dim) refined aligned feature
        """
        # Concat input
        x = torch.cat([h_tilde, staleness_embed, missing_mask], dim=-1)  # (B, input_dim)

        # MLP forward
        h = self.act(self.fc1(x))  # (B, hidden_dim)

        # FiLM conditioning: γ * h + β for each regime
        r_embed = self.regime_embed(regime_idx)  # (B, regime_adapter_dim)

        # Gather regime-specific modulation (batch-wise)
        gamma = torch.zeros_like(h)
        beta = torch.zeros_like(h)
        for r in range(len(self.regime_adapters)):
            mask_r = (regime_idx == r).unsqueeze(-1).float()  # (B, 1)
            gamma += mask_r * (1.0 + self.regime_adapters[r](r_embed))
            beta += mask_r * self.regime_bias[r](r_embed)

        h = gamma * h + beta  # FiLM modulation
        h = self.dropout(h)

        residual = self.fc2(h)  # (B, feature_dim)
        return h_tilde + residual  # Residual connection


class AsyncAlignment(nn.Module):
    """
    Complete C1 module: Delay-Regime-Aware Asynchronous Alignment.
    Integrates (a)-(d) for all modalities.
    """

    MODALITIES = ["image", "radar", "lidar", "gps"]

    def __init__(self, feature_dim: int = 64, buffer_size: int = 3,
                 staleness_embed_dim: int = 32, max_staleness_ms: float = 300.0,
                 repair_stale_threshold_ms: Optional[float] = None,
                 repair_max_consecutive_missing: Optional[int] = 3,
                 num_regimes: int = 3, residual_hidden: int = 128,
                 regime_adapter_dim: int = 32,
                 delay_boundaries: list = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.buffer_size = buffer_size
        self.num_regimes = num_regimes
        self.max_staleness_ms = max_staleness_ms
        self.repair_stale_threshold_ms = (
            max_staleness_ms if repair_stale_threshold_ms is None
            else float(repair_stale_threshold_ms)
        )
        self.repair_max_consecutive_missing = (
            None if repair_max_consecutive_missing is None
            else int(max(0, repair_max_consecutive_missing))
        )
        self.delay_boundaries = delay_boundaries or [50.0, 150.0]
        self.repair_mode = "learned"

        # Shared staleness embedding
        self.staleness_embed = SinusoidalStalenessEmbedding(
            staleness_embed_dim, max_staleness_ms
        )

        # Per-modality staleness-aware weighting
        self.weighting = nn.ModuleDict({
            m: StalenessAwareWeighting(feature_dim, staleness_embed_dim)
            for m in self.MODALITIES
        })

        # Short-gap repair (shared, per-modality η)
        self.repair = ShortGapRepair(num_modalities=len(self.MODALITIES))

        # Per-modality residual alignment refinement
        self.refinement = nn.ModuleDict({
            m: ResidualAlignmentRefinement(
                feature_dim, staleness_embed_dim, residual_hidden,
                num_regimes, regime_adapter_dim
            ) for m in self.MODALITIES
        })

    def set_repair_mode(self, mode: str = "learned"):
        """Switch short-gap repair behavior (used for ablations)."""
        valid = {"learned", "zero_fill"}
        if mode not in valid:
            raise ValueError(f"repair mode must be one of {valid}, got {mode}")
        self.repair_mode = mode

    def compute_delay_regime(self, staleness_ms: torch.Tensor) -> torch.Tensor:
        """Discretize staleness into regime indices."""
        regime = torch.zeros_like(staleness_ms, dtype=torch.long)
        for i, boundary in enumerate(self.delay_boundaries):
            regime += (staleness_ms >= boundary).long()
        return regime.clamp(0, self.num_regimes - 1)

    def forward(self, features: Dict[str, torch.Tensor],
                staleness_ms: Optional[Dict[str, torch.Tensor]] = None,
                missing_masks: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Args:
            features: Dict[modality -> (B, H, feature_dim)] encoded features
            staleness_ms: Dict[modality -> (B, H)] staleness in ms per frame.
                          If None, assume synchronized (Δt=0).
            missing_masks: Dict[modality -> (B, H)] binary (1=missing).
                           If None, assume all present.
        Returns:
            aligned_features: Dict[modality -> (B, H, feature_dim)]
            info: Dict with attention weights, regime indices, etc.
        """
        B, H, M = features[self.MODALITIES[0]].shape
        device = features[self.MODALITIES[0]].device

        # Defaults for synchronized evaluation
        if staleness_ms is None:
            staleness_ms = {m: torch.zeros(B, H, device=device) for m in self.MODALITIES}
        if missing_masks is None:
            missing_masks = {m: torch.zeros(B, H, device=device) for m in self.MODALITIES}

        aligned = {}
        info = {"attention_weights": {}, "regimes": {}, "repair_flags": {}}

        for mi, mod in enumerate(self.MODALITIES):
            feat = features[mod]          # (B, H, M)
            stale = staleness_ms[mod]     # (B, H)
            miss = missing_masks[mod]     # (B, H)

            aligned_frames = []
            attn_weights_list = []
            repair_flags_list = []
            short_gap_repair_flags_list = []
            long_gap_missing_flags_list = []
            consecutive_missing = torch.zeros(B, dtype=torch.long, device=device)

            for t in range(H):
                # (a) Windowed candidate retrieval: gather K+1 frames
                K = self.buffer_size
                start_idx = max(0, t - K)
                buf_indices = list(range(start_idx, t + 1))
                # Pad if needed
                while len(buf_indices) < K + 1:
                    buf_indices = [buf_indices[0]] + buf_indices

                h_buf = torch.stack([feat[:, i, :] for i in buf_indices], dim=1)  # (B, K+1, M)
                s_buf = torch.stack([stale[:, i] for i in buf_indices], dim=1)    # (B, K+1)

                # (b) Staleness embedding
                s_embed = self.staleness_embed(s_buf)  # (B, K+1, embed_dim)

                # (b) Staleness-aware weighting
                h_tilde, alpha = self.weighting[mod](h_buf, s_embed)  # (B, M), (B, K+1)
                attn_weights_list.append(alpha)

                # (c) Short-gap repair for missing or severely stale frames.
                # PDF proposal triggers repair for missing or Δt > Δt_max.
                is_missing = (miss[:, t] > 0.5)
                consecutive_missing = torch.where(
                    is_missing,
                    consecutive_missing + 1,
                    torch.zeros_like(consecutive_missing),
                )
                if self.repair_max_consecutive_missing is None:
                    short_missing = is_missing
                else:
                    short_missing = is_missing & (consecutive_missing <= self.repair_max_consecutive_missing)
                long_missing = is_missing & (~short_missing)
                is_stale = stale[:, t] > self.repair_stale_threshold_ms
                needs_repair = short_missing | is_stale  # (B,)
                repair_flags_list.append(needs_repair.float())
                short_gap_repair_flags_list.append(short_missing.float())
                long_gap_missing_flags_list.append(long_missing.float())

                if needs_repair.any():
                    if self.repair_mode == "zero_fill":
                        h_repaired = torch.zeros_like(h_tilde)
                    elif t >= 2:
                        # First-order feature extrapolation (proposal default)
                        h_prev1 = aligned_frames[-1]
                        h_prev2 = aligned_frames[-2] if len(aligned_frames) >= 2 else h_prev1
                        h_repaired = self.repair(h_prev1, h_prev2, mi)
                    elif t >= 1 and len(aligned_frames) >= 1:
                        # Early-timestep fallback: zero-order hold
                        h_repaired = aligned_frames[-1]
                    else:
                        h_repaired = torch.zeros_like(h_tilde)

                    repair_mask = needs_repair.unsqueeze(-1).float()
                    h_tilde = (1.0 - repair_mask) * h_tilde + repair_mask * h_repaired

                # (d) Residual alignment refinement with regime conditioning
                stale_t = stale[:, t]  # (B,)
                regime = self.compute_delay_regime(stale_t)  # (B,)
                s_embed_t = self.staleness_embed(stale_t)    # (B, embed_dim)
                m_mask = needs_repair.unsqueeze(-1).float()   # (B, 1)

                h_bar = self.refinement[mod](h_tilde, s_embed_t, m_mask, regime)
                aligned_frames.append(h_bar)

            aligned[mod] = torch.stack(aligned_frames, dim=1)  # (B, H, M)
            info["attention_weights"][mod] = torch.stack(attn_weights_list, dim=1)
            info["regimes"][mod] = self.compute_delay_regime(stale[:, -1])
            info["repair_flags"][mod] = torch.stack(repair_flags_list, dim=1)
            info.setdefault("short_gap_repair_flags", {})[mod] = torch.stack(short_gap_repair_flags_list, dim=1)
            info.setdefault("long_gap_missing_flags", {})[mod] = torch.stack(long_gap_missing_flags_list, dim=1)

        return aligned, info
