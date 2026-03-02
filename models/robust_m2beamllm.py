"""
Robust M²BeamLLM: Complete framework integrating all innovations (C1-C3).
Reference: Paper Fig. 2, Sections II-III.

Inference pipeline at decision time t:
  1. Encode per-modality observations (encoders.py)
  2. Delay-aware async alignment (C1: async_alignment.py)
  3. Reliability estimation and gating (C2: reliability.py)
  4. Cross-attention fusion -> LLM tokens (C2-d: reliability.py)
  5. LLM inference with MoE-LoRA PEFT (C3: moe_lora.py)
  6. Beam classification head -> beam ranking

Training:
  Stage 1: SSL pretraining (no beam labels)
  Stage 2: Supervised beam prediction + reliability-weighted alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.encoders import MultimodalEncoder
from models.async_alignment import AsyncAlignment
from models.reliability import (
    ReliabilityEstimator,
    ReliabilityWeightedAlignmentLoss,
    CrossAttentionFusion,
)
from models.moe_lora import PEFTLLMBackbone, SSLObjectives, VanillaTransformerBackbone


class BeamHistoryConditioner(nn.Module):
    """
    Optional history-conditioned residual branch (extension).
    Encodes past beam indices and produces per-step feature/logit residuals.
    """

    def __init__(
        self,
        num_beams: int,
        feature_dim: int,
        H: int,
        T: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H = int(H)
        self.T = int(T)
        self.feature_dim = int(feature_dim)
        self.num_beams = int(num_beams)

        self.beam_embed = nn.Embedding(num_beams, feature_dim)
        self.pos_embed = nn.Embedding(max(H, 1), feature_dim)
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

        self.feature_gate = nn.Linear(feature_dim, feature_dim)
        self.feature_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, T * feature_dim),
        )
        self.logit_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, T * num_beams),
        )

        # Start from a moderate contribution so history is useful but does not
        # instantly dominate the multimodal path on the first few updates.
        self.feature_res_scale = nn.Parameter(torch.tensor(0.5))
        self.logit_res_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, beam_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            beam_history: (B, H_hist) int64 beam indices
        Returns:
            Dict with feature/logit residuals and diagnostics.
        """
        if beam_history is None:
            raise ValueError("beam_history is required when history conditioning is enabled.")
        if beam_history.dim() != 2:
            raise ValueError(f"Expected beam_history shape (B, H), got {tuple(beam_history.shape)}")

        x_idx = beam_history.long()
        if x_idx.size(1) > self.H:
            x_idx = x_idx[:, -self.H:]
        B, L = x_idx.shape
        device = x_idx.device

        pos = torch.arange(L, device=device)
        x = self.beam_embed(x_idx) + self.pos_embed(pos).unsqueeze(0)  # (B, L, D)
        x = self.dropout(x)
        _, h_n = self.gru(x)
        ctx = self.norm(h_n[-1])  # (B, D)

        feat_gate = torch.sigmoid(self.feature_gate(ctx)).unsqueeze(1)  # (B,1,D)
        feat_res = self.feature_head(ctx).reshape(B, self.T, self.feature_dim)
        logit_res = self.logit_head(ctx).reshape(B, self.T, self.num_beams)

        return {
            "ctx": ctx,
            "feature_gate": feat_gate,
            "feature_residual": feat_res,
            "logit_residual": logit_res,
            "feature_scale": self.feature_res_scale,
            "logit_scale": self.logit_res_scale,
        }


class AutoregressiveBeamDecoder(nn.Module):
    """
    Optional autoregressive beam decoder (extension).
    Uses previous beam token + per-step multimodal context to refine logits.
    """

    def __init__(
        self,
        num_beams: int,
        feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_beams = int(num_beams)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.beam_embed = nn.Embedding(num_beams, feature_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.init_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.Tanh(),
        )
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_beams),
        )

        # Residual scale so the base classifier remains dominant early in training.
        self.logit_res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        context_features: torch.Tensor,
        beam_history: Optional[torch.Tensor] = None,
        teacher_forcing_targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            context_features: (B, T, D)
            beam_history: (B, H) or None
            teacher_forcing_targets: (B, T) or None
            teacher_forcing_ratio: scheduled-sampling ratio in [0,1]
        Returns:
            Dict containing ar_logits residual and diagnostics.
        """
        B, T, D = context_features.shape
        if D != self.feature_dim:
            raise ValueError(
                f"AR decoder feature_dim mismatch: got {D}, expected {self.feature_dim}"
            )

        if beam_history is not None and beam_history.numel() > 0:
            bh = beam_history.long()
            prev_beam = bh[:, -1]
            hist_emb = self.beam_embed(bh).mean(dim=1)
        else:
            prev_beam = torch.zeros(B, dtype=torch.long, device=context_features.device)
            hist_emb = torch.zeros(B, self.feature_dim, device=context_features.device)

        ctx_global = context_features.mean(dim=1)
        h = self.init_proj(torch.cat([ctx_global, hist_emb], dim=-1))

        tf_ratio = float(max(0.0, min(1.0, teacher_forcing_ratio)))
        logits_steps = []
        for t in range(T):
            prev_emb = self.beam_embed(prev_beam)
            x = self.in_proj(torch.cat([context_features[:, t, :], prev_emb], dim=-1))
            h = self.gru_cell(x, h)
            h_norm = self.state_norm(h)
            logits_t = self.out_proj(torch.cat([h_norm, context_features[:, t, :]], dim=-1))
            logits_steps.append(logits_t)

            pred_next = logits_t.argmax(dim=-1)
            if teacher_forcing_targets is not None:
                gt_next = teacher_forcing_targets[:, t].long()
                if tf_ratio >= 1.0:
                    prev_beam = gt_next
                elif tf_ratio <= 0.0:
                    prev_beam = pred_next
                else:
                    use_tf = (torch.rand(B, device=context_features.device) < tf_ratio)
                    prev_beam = torch.where(use_tf, gt_next, pred_next)
            else:
                prev_beam = pred_next

        ar_logits = torch.stack(logits_steps, dim=1)  # (B, T, C)

        return {
            "ar_logits": ar_logits,
            "logit_scale": self.logit_res_scale,
            "state_norm": h.norm(dim=-1).mean().detach(),
            "teacher_forcing_ratio": tf_ratio,
        }


class TopKPairwiseReranker(nn.Module):
    """
    Optional trainable Top-K reranker (extension).
    Consumes engineered candidate features and outputs scalar rerank scores.

    This is designed to be used on top of the base model's Top-K candidates:
      s_final = s_base + lambda * s_rerank
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        # Residual scaling keeps the reranker conservative early in training.
        self.score_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (..., F)
        Returns:
            scores: (...,)
        """
        if features.size(-1) != self.feature_dim:
            raise ValueError(
                f"TopKPairwiseReranker feature mismatch: got {features.size(-1)}, "
                f"expected {self.feature_dim}"
            )
        scores = self.net(features).squeeze(-1)
        return torch.tanh(self.score_scale) * scores


class RobustM2BeamLLM(nn.Module):
    """
    Full Robust M²BeamLLM framework with C1-C3 innovations.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_beams: int = 64,
        llm_name: str = "gpt2",
        llm_hidden_dim: int = 768,
        num_unfrozen_layers: int = 0,
        num_adapter_layers: int = 2,
        T: int = 5,
        H: int = 8,
        pretrained_encoders: bool = True,
        # C1 params
        buffer_size: int = 3,
        staleness_embed_dim: int = 32,
        max_staleness_ms: float = 300.0,
        repair_stale_threshold_ms: Optional[float] = None,
        repair_max_consecutive_missing: Optional[int] = 3,
        num_delay_regimes: int = 3,
        delay_boundaries: list = None,
        residual_hidden: int = 128,
        regime_adapter_dim: int = 32,
        # C2 params
        variance_head_hidden: int = 64,
        reliability_min: float = 1e-4,
        reliability_logvar_clip: Optional[float] = None,
        reliability_softmax_temperature: float = 1.0,
        reliability_uniform_mix: float = 0.0,
        reliability_ema_beta: float = 0.0,
        reliability_pair_conf_threshold: float = 0.0,
        reliability_degradation_cue_mix: float = 0.0,
        reliability_degradation_prior_scale: float = 0.0,
        degradation_observation_keep_scale: float = 0.0,
        degradation_observation_keep_min: float = 1.0,
        degradation_temporal_prior_scale: float = 0.0,
        degradation_temporal_big_jump_penalty: float = 0.0,
        degradation_temporal_neighbor_radius: int = 2,
        degradation_temporal_prior_max_boost: float = 0.0,
        history_anchor_scale: float = 0.0,
        history_anchor_decay: float = 0.8,
        history_anchor_big_jump_penalty: float = 0.0,
        history_anchor_neighbor_radius: int = 2,
        alignment_proj_dim: int = 64,
        num_fusion_tokens: int = 4,
        # C3 params
        num_experts: int = 4,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        router_extra_dim: int = 0,
        # Fusion params
        num_attention_heads: int = 4,
        fusion_dropout: float = 0.1,
        ffn_hidden_dim: int = 256,
        # A5 / E3 baseline params
        vanilla_hidden_dim: int = 1024,
        vanilla_num_layers: int = 24,
        vanilla_num_heads: int = 16,
        vanilla_ffn_hidden_dim: int = 5120,
        vanilla_dropout: float = 0.1,
        # Extension: optional beam-history conditioning for stronger clean Top-1
        use_beam_history: bool = False,
        beam_history_dropout: float = 0.1,
        # Extension: optional autoregressive beam decoder
        use_autoregressive_decoder: bool = False,
        ar_decoder_hidden_dim: int = 128,
        ar_decoder_dropout: float = 0.1,
        gps_ema_beta: float = 0.0,
        # Extension: optional trainable Top-K pairwise reranker
        use_pairwise_reranker: bool = False,
        pairwise_reranker_feature_dim: int = 24,
        pairwise_reranker_hidden_dim: int = 64,
        pairwise_reranker_dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_beams = num_beams
        self.T = T
        self.H = H
        self.num_delay_regimes = num_delay_regimes
        self.use_beam_history = bool(use_beam_history)
        self.use_autoregressive_decoder = bool(use_autoregressive_decoder)
        self.use_pairwise_reranker = bool(use_pairwise_reranker)
        self.reliability_degradation_cue_mix = float(
            max(0.0, min(1.0, reliability_degradation_cue_mix))
        )
        self.reliability_degradation_prior_scale = float(
            max(0.0, reliability_degradation_prior_scale)
        )
        self.degradation_observation_keep_scale = float(
            max(0.0, degradation_observation_keep_scale)
        )
        self.degradation_observation_keep_min = float(
            max(0.0, min(1.0, degradation_observation_keep_min))
        )
        self.degradation_temporal_prior_scale = float(
            max(0.0, degradation_temporal_prior_scale)
        )
        self.degradation_temporal_big_jump_penalty = float(
            max(0.0, degradation_temporal_big_jump_penalty)
        )
        self.degradation_temporal_neighbor_radius = int(
            max(0, degradation_temporal_neighbor_radius)
        )
        self.degradation_temporal_prior_max_boost = float(
            max(0.0, degradation_temporal_prior_max_boost)
        )
        self.history_anchor_scale = float(max(0.0, history_anchor_scale))
        self.history_anchor_decay = float(max(0.0, min(1.0, history_anchor_decay)))
        self.history_anchor_big_jump_penalty = float(max(0.0, history_anchor_big_jump_penalty))
        self.history_anchor_neighbor_radius = int(max(0, history_anchor_neighbor_radius))
        self.router_extra_dim = int(max(0, router_extra_dim))

        # 1. Multimodal Encoders
        self.encoder = MultimodalEncoder(
            feature_dim, pretrained_encoders, gps_ema_beta=gps_ema_beta
        )

        # 2. C1: Async Alignment
        self.async_alignment = AsyncAlignment(
            feature_dim=feature_dim,
            buffer_size=buffer_size,
            staleness_embed_dim=staleness_embed_dim,
            max_staleness_ms=max_staleness_ms,
            repair_stale_threshold_ms=repair_stale_threshold_ms,
            repair_max_consecutive_missing=repair_max_consecutive_missing,
            num_regimes=num_delay_regimes,
            residual_hidden=residual_hidden,
            regime_adapter_dim=regime_adapter_dim,
            delay_boundaries=delay_boundaries or [50.0, 150.0],
        )

        # 3. C2: Reliability Estimation
        self.reliability_estimator = ReliabilityEstimator(
            feature_dim=feature_dim,
            hidden_dim=variance_head_hidden,
            reliability_min=reliability_min,
            logvar_clip=reliability_logvar_clip,
            softmax_temperature=reliability_softmax_temperature,
            uniform_mix=reliability_uniform_mix,
            ema_beta=reliability_ema_beta,
        )

        # 4. C2-c: Reliability-Weighted Alignment Loss
        self.alignment_loss_fn = ReliabilityWeightedAlignmentLoss(
            feature_dim=feature_dim,
            proj_dim=alignment_proj_dim,
            pair_conf_threshold=reliability_pair_conf_threshold,
        )

        # 5. C2-d: Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(
            feature_dim=feature_dim,
            num_fusion_tokens=num_fusion_tokens,
            num_heads=num_attention_heads,
            dropout=fusion_dropout,
            ffn_hidden=ffn_hidden_dim,
        )

        # 6. C3: Backbone
        backbone_name = llm_name.lower()
        self.use_vanilla_transformer = backbone_name in {
            "vanilla_transformer", "vanilla", "transformer"
        }
        if self.use_vanilla_transformer:
            self.llm_backbone = VanillaTransformerBackbone(
                feature_dim=feature_dim,
                hidden_dim=vanilla_hidden_dim,
                T=T,
                num_layers=vanilla_num_layers,
                num_heads=vanilla_num_heads,
                ffn_hidden_dim=vanilla_ffn_hidden_dim,
                dropout=vanilla_dropout,
            )
        else:
            self.llm_backbone = PEFTLLMBackbone(
                model_name=llm_name,
                num_unfrozen_layers=num_unfrozen_layers,
                num_adapter_layers=num_adapter_layers,
                feature_dim=feature_dim,
                llm_hidden_dim=llm_hidden_dim,
                T=T,
                num_fusion_tokens=num_fusion_tokens,
                num_experts=num_experts,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                num_modalities=4,
                num_regimes=num_delay_regimes,
                router_extra_dim=self.router_extra_dim,
            )

        # 7. Beam classification head (2-layer MLP for better expressiveness)
        self.pred_dropout = nn.Dropout(0.1)
        self.beam_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(feature_dim * 2, num_beams),
        )

        # Optional extension branch (disabled by default to keep proposal-faithful path).
        self.beam_history_conditioner = (
            BeamHistoryConditioner(
                num_beams=num_beams,
                feature_dim=feature_dim,
                H=H,
                T=T,
                dropout=beam_history_dropout,
            )
            if self.use_beam_history else None
        )
        self.ar_beam_decoder = (
            AutoregressiveBeamDecoder(
                num_beams=num_beams,
                feature_dim=feature_dim,
                hidden_dim=ar_decoder_hidden_dim,
                dropout=ar_decoder_dropout,
            )
            if self.use_autoregressive_decoder else None
        )
        self.topk_pairwise_reranker = (
            TopKPairwiseReranker(
                feature_dim=pairwise_reranker_feature_dim,
                hidden_dim=pairwise_reranker_hidden_dim,
                dropout=pairwise_reranker_dropout,
            )
            if self.use_pairwise_reranker else None
        )

        # C3: SSL objectives (for Stage 1)
        self.ssl_objectives = SSLObjectives(feature_dim=feature_dim)

        # Initialize alignment projections to identity when possible so
        # alignment starts as a meaningful distance (instead of a collapsed
        # zero projection that can encourage trivial feature shrinkage).
        for proj in self.alignment_loss_fn.projections.values():
            if proj.weight.shape[0] == proj.weight.shape[1]:
                nn.init.eye_(proj.weight)
            else:
                nn.init.xavier_uniform_(proj.weight)

    def _build_quality_cues(
        self,
        reliability_input_ref: Dict[str, torch.Tensor],
        staleness_ms: Optional[Dict[str, torch.Tensor]] = None,
        degradation_cues: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build 1-D quality cues q_ω(t); combine staleness and degradation cues."""
        mods = ["image", "radar", "lidar", "gps"]
        B, H, _ = reliability_input_ref[mods[0]].shape
        device = reliability_input_ref[mods[0]].device
        if staleness_ms is None and degradation_cues is None:
            return {m: torch.zeros(B, H, 1, device=device) for m in mods}

        denom = max(float(getattr(self.async_alignment, "max_staleness_ms", 300.0)), 1e-6)
        cues = {}
        for m in mods:
            stale_cue = torch.zeros(B, H, 1, device=device)
            if staleness_ms is not None:
                stale = staleness_ms.get(m)
                if stale is not None:
                    stale_cue = (stale / denom).clamp(0.0, 1.0).unsqueeze(-1)
            degr_cue = torch.zeros(B, H, 1, device=device)
            if degradation_cues is not None:
                dc = degradation_cues.get(m)
                if dc is not None:
                    degr_cue = dc.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            mix = self.reliability_degradation_cue_mix
            cues[m] = torch.maximum(stale_cue, degr_cue) if mix >= 0.999 else (
                (1.0 - mix) * stale_cue + mix * degr_cue
            )
        return cues

    def _apply_degradation_reliability_prior(
        self,
        reliability_weights: Dict[str, torch.Tensor],
        aligned_features: Dict[str, torch.Tensor],
        degradation_cues: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mods = ["image", "radar", "lidar", "gps"]
        if degradation_cues is None or self.reliability_degradation_prior_scale <= 0:
            gated = {m: aligned_features[m] * reliability_weights[m].unsqueeze(-1) for m in mods}
            return reliability_weights, gated

        stacked_w = torch.stack([reliability_weights[m] for m in mods], dim=-1)  # (B,H,4)
        stacked_c = torch.stack(
            [
                degradation_cues[m].to(stacked_w.device, dtype=stacked_w.dtype).squeeze(-1)
                if m in degradation_cues else torch.zeros_like(reliability_weights[m])
                for m in mods
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        prior = torch.exp(-self.reliability_degradation_prior_scale * stacked_c)
        adjusted = (stacked_w * prior).clamp(min=float(self.reliability_estimator.reliability_min))
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        out_weights = {}
        gated = {}
        for i, m in enumerate(mods):
            w = adjusted[:, :, i]
            out_weights[m] = w
            gated[m] = aligned_features[m] * w.unsqueeze(-1)
        return out_weights, gated

    def _compute_observation_keep(
        self,
        quality_cues: Dict[str, torch.Tensor],
        reliability_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute how much to trust the current observation stack as a whole.
        Relative reliability renormalization alone cannot help when all modalities
        are degraded together, so we derive a global keep factor from quality cues.
        """
        mods = ["image", "radar", "lidar", "gps"]
        stacked_q = torch.stack(
            [quality_cues[m].squeeze(-1) for m in mods], dim=-1
        ).clamp(0.0, 1.0)  # (B, H, 4)
        stacked_w = torch.stack([reliability_weights[m] for m in mods], dim=-1)
        mean_q = stacked_q.mean(dim=-1).mean(dim=1)             # (B,)
        max_q = stacked_q.max(dim=-1).values.mean(dim=1)        # (B,)
        max_w = stacked_w.max(dim=-1).values.mean(dim=1)        # (B,)
        cue_risk = 0.5 * (mean_q + max_q)
        risk = cue_risk + 0.10 * (1.0 - max_w) * (cue_risk > 1e-6).float()
        keep = torch.exp(-self.degradation_observation_keep_scale * risk)
        return keep.clamp(min=self.degradation_observation_keep_min, max=1.0)

    def _build_degradation_temporal_prior(
        self,
        beam_history: Optional[torch.Tensor],
        gps: Optional[torch.Tensor],
        observation_keep: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Build a simple continuity prior centered on the last historical beam.
        This is only used more strongly when observation quality is poor.
        """
        if beam_history is None or beam_history.numel() == 0:
            return None

        B = beam_history.size(0)
        device = beam_history.device
        dtype = observation_keep.dtype
        last_beam = beam_history[:, -1].long().clamp(min=0, max=self.num_beams - 1)
        beam_ids = torch.arange(self.num_beams, device=device).view(1, -1)
        raw_dist = (beam_ids - last_beam.unsqueeze(1)).abs()
        circ_dist = torch.minimum(raw_dist, self.num_beams - raw_dist)

        jump_scale = torch.full((B,), self.degradation_temporal_prior_scale, device=device, dtype=dtype)
        if gps is not None and gps.ndim == 3 and gps.size(1) >= 2:
            vel = (gps[:, -1] - gps[:, -2]).norm(dim=-1).to(dtype=dtype)
            vel = vel.clamp(0.0, 1.0)
            jump_scale = jump_scale * (1.15 - 0.55 * vel)

        prior = -jump_scale.unsqueeze(1) * circ_dist.to(dtype=dtype)
        if self.degradation_temporal_neighbor_radius >= 0:
            big_jump = (circ_dist > self.degradation_temporal_neighbor_radius).to(dtype=dtype)
            prior = prior - self.degradation_temporal_big_jump_penalty * big_jump

        strength = (1.0 - observation_keep).clamp(0.0, 1.0) * self.degradation_temporal_prior_max_boost
        prior = strength.unsqueeze(1) * prior
        return prior.unsqueeze(1).expand(B, self.T, self.num_beams)

    def _build_reliability_and_regime_vectors(
        self, reliability_weights: Dict[str, torch.Tensor],
        regimes: Dict[str, torch.Tensor],
        log_variances: Dict[str, torch.Tensor],
        quality_cues: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build compact vectors for MoE routing."""
        mods = ["image", "radar", "lidar", "gps"]
        B = reliability_weights[mods[0]].shape[0]
        device = reliability_weights[mods[0]].device

        # Reliability vector: mean over time, per modality
        rel_vec = torch.stack(
            [reliability_weights[m].mean(dim=1) for m in mods], dim=-1
        )  # (B, 4)

        # Regime vector: aggregate all modality delay regimes (soft histogram over Ω).
        regime_oh = []
        for m in mods:
            idx = regimes.get(m, torch.zeros(B, dtype=torch.long, device=device))
            regime_oh.append(F.one_hot(idx, self.num_delay_regimes).float())
        regime_vec = torch.stack(regime_oh, dim=0).mean(dim=0)  # (B, R)

        quality_vec = torch.stack(
            [quality_cues[m].mean(dim=1).squeeze(-1) for m in mods], dim=-1
        )  # (B, 4)
        logvar_vec = torch.stack(
            [log_variances[m].mean(dim=1) for m in mods], dim=-1
        )  # (B, 4)
        router_extra_vec = torch.cat([quality_vec, logvar_vec], dim=-1)  # (B, 8)

        return rel_vec, regime_vec, router_extra_vec

    def _build_history_anchor_logits(
        self,
        beam_history: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Optional continuity prior centered on the last historical beam."""
        if self.history_anchor_scale <= 0.0 or beam_history is None or beam_history.numel() == 0:
            return None

        B = beam_history.size(0)
        device = beam_history.device
        last_beam = beam_history[:, -1].long().clamp(min=0, max=self.num_beams - 1)
        beam_ids = torch.arange(self.num_beams, device=device).view(1, -1)
        raw_dist = (beam_ids - last_beam.unsqueeze(1)).abs()
        circ_dist = torch.minimum(raw_dist, self.num_beams - raw_dist).float()

        base = -self.history_anchor_scale * circ_dist
        if self.history_anchor_big_jump_penalty > 0.0:
            far = (circ_dist > self.history_anchor_neighbor_radius).float()
            base = base - self.history_anchor_big_jump_penalty * far

        if self.T > 1:
            decay = torch.pow(
                torch.full((self.T,), self.history_anchor_decay, device=device),
                torch.arange(self.T, device=device, dtype=torch.float32),
            ).view(1, self.T, 1)
        else:
            decay = torch.ones(1, 1, 1, device=device)
        return base.unsqueeze(1) * decay.expand(B, -1, -1)

    def forward(self, images, radars, lidars, gps,
                staleness_ms=None, missing_masks=None,
                degradation_cues: Optional[Dict[str, torch.Tensor]] = None,
                beam_history: Optional[torch.Tensor] = None,
                beam_future_targets: Optional[torch.Tensor] = None,
                teacher_forcing: bool = False,
                teacher_forcing_ratio: float = 1.0,
                ar_logit_scale_override: Optional[float] = None):
        """
        Full forward pass.

        Args:
            images: (B, H, 3, 224, 224)
            radars: (B, H, 1, M_F, S_R)
            lidars: (B, H, 1, 256, 256)
            gps:    (B, H, 2)
            staleness_ms: Optional Dict[mod -> (B, H)] staleness per frame
            missing_masks: Optional Dict[mod -> (B, H)] missing indicators
            degradation_cues: Optional Dict[mod -> (B, H, 1)] quality degradation cues
            beam_history: Optional (B, H) past beam indices (extension branch)
            beam_future_targets: Optional (B, T) future beams for teacher forcing
            teacher_forcing: whether to use beam_future_targets in AR decoder
            teacher_forcing_ratio: scheduled-sampling TF ratio for AR decoder
            ar_logit_scale_override: optional external ramp multiplier for AR branch
        Returns:
            predictions: (B, T, num_beams)
            aux: Dict with intermediate features for loss computation
        """
        # 1. Encode
        raw_features = self.encoder(images, radars, lidars, gps)

        # 2. C1: Async Alignment
        aligned_features, align_info = self.async_alignment(
            raw_features, staleness_ms, missing_masks
        )

        # 3. C2: Reliability Estimation + Gating
        quality_cues = self._build_quality_cues(
            aligned_features, staleness_ms=staleness_ms, degradation_cues=degradation_cues
        )
        reliability_weights, log_variances, gated_features = \
            self.reliability_estimator(
                aligned_features, quality_cues=quality_cues, missing_masks=missing_masks
            )
        reliability_weights, gated_features = self._apply_degradation_reliability_prior(
            reliability_weights, aligned_features, degradation_cues
        )
        observation_keep = self._compute_observation_keep(quality_cues, reliability_weights)

        # 4. C2-d: Cross-Attention Fusion
        Z = self.fusion(gated_features)  # (B, H*num_tokens, feature_dim)

        # 5. C3: MoE-LoRA LLM
        rel_vec, regime_vec, router_extra_vec = self._build_reliability_and_regime_vectors(
            reliability_weights, align_info.get("regimes", {}), log_variances, quality_cues
        )
        pred_features, moe_balance_loss = self.llm_backbone(
            Z, rel_vec, regime_vec, router_extra_vec
        )  # (B, T, feature_dim)

        history_aux = {
            "enabled": False,
            "ctx_norm": 0.0,
            "feature_scale": 0.0,
            "logit_scale": 0.0,
        }
        if self.use_beam_history and self.beam_history_conditioner is not None and beam_history is not None:
            hist = self.beam_history_conditioner(beam_history.to(pred_features.device))
            pred_features = pred_features + (
                hist["feature_scale"] * hist["feature_gate"] * hist["feature_residual"]
            )
            history_aux = {
                "enabled": True,
                "ctx_norm": float(hist["ctx"].norm(dim=-1).mean().detach().item()),
                "feature_scale": float(hist["feature_scale"].detach().item()),
                "logit_scale": float(hist["logit_scale"].detach().item()),
                "logit_residual": hist["logit_residual"],
            }

        # 6. Beam classification
        predictions = self.beam_classifier(self.pred_dropout(pred_features))
        if history_aux.get("enabled", False):
            predictions = predictions + history_aux["logit_scale"] * history_aux["logit_residual"]

        ar_aux = {
            "enabled": False,
            "logit_scale": 0.0,
            "effective_logit_scale": 0.0,
            "state_norm": 0.0,
            "teacher_forcing_ratio": 0.0,
        }
        if self.use_autoregressive_decoder and self.ar_beam_decoder is not None:
            tf_targets = beam_future_targets if teacher_forcing else None
            ar_out = self.ar_beam_decoder(
                pred_features,
                beam_history=beam_history,
                teacher_forcing_targets=tf_targets,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            base_scale = torch.tanh(ar_out["logit_scale"])
            if ar_logit_scale_override is None:
                eff_scale = base_scale
            else:
                eff_scale = base_scale * max(float(ar_logit_scale_override), 0.0)
            predictions = predictions + eff_scale * ar_out["ar_logits"]
            ar_aux = {
                "enabled": True,
                "logit_scale": float(base_scale.detach().item()),
                "effective_logit_scale": float(eff_scale.detach().item()),
                "state_norm": float(ar_out["state_norm"].detach().item()),
                "teacher_forcing": bool(tf_targets is not None),
                "teacher_forcing_ratio": float(ar_out.get("teacher_forcing_ratio", 0.0)),
            }

        temporal_prior_logits = self._build_degradation_temporal_prior(
            beam_history=beam_history,
            gps=gps,
            observation_keep=observation_keep,
        )
        if temporal_prior_logits is not None:
            predictions = predictions + temporal_prior_logits

        history_anchor_logits = self._build_history_anchor_logits(beam_history=beam_history)
        if history_anchor_logits is not None:
            predictions = predictions + history_anchor_logits

        aux = {
            "raw_features": raw_features,
            "aligned_features": aligned_features,
            "reliability_weights": reliability_weights,
            "log_variances": log_variances,
            "gated_features": gated_features,
            "quality_cues": quality_cues,
            "observation_keep": observation_keep,
            "fused": Z,
            "moe_balance_loss": moe_balance_loss,
            "align_info": align_info,
            "history": {
                k: v for k, v in history_aux.items() if k != "logit_residual"
            },
            "history_anchor_scale": self.history_anchor_scale,
            "ar_decoder": ar_aux,
            # Reuse model-owned alignment loss module to avoid accidentally
            # training against a detached/random criterion-side projection.
            "alignment_loss_module": self.alignment_loss_fn,
        }
        return predictions, aux

    def predict_beam_indices(self, images, radars, lidars, gps,
                             staleness_ms=None, missing_masks=None,
                             beam_history: Optional[torch.Tensor] = None):
        predictions, _ = self.forward(
            images, radars, lidars, gps,
            staleness_ms=staleness_ms,
            missing_masks=missing_masks,
            beam_history=beam_history,
        )
        return predictions.argmax(dim=-1)


class RobustM2BeamLLMLoss(nn.Module):
    """
    Training loss for Stage 2:
        L_total = L_beam + λ_align · L_align^{rw} + λ_moe · L_balance

    L_align^{rw} = reliability-weighted alignment (C2-c)
    L_balance = MoE load-balancing loss
    """

    def __init__(self, feature_dim: int = 64, lambda_align: float = 1.0,
                 lambda_beam: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None,
                 class_prior: Optional[torch.Tensor] = None,
                 valid_beam_mask: Optional[torch.Tensor] = None,
                 step_weights: Optional[torch.Tensor] = None,
                 focal_gamma: float = 0.0,
                 label_smoothing: float = 0.0,
                 beam_soft_target_lambda: float = 0.0,
                 beam_soft_target_tau: float = 1.5,
                 lambda_moe_balance: float = 1.0,
                 lambda_prior_match: float = 0.0,
                 lambda_cvar: float = 0.0,
                 cvar_tail_fraction: float = 0.10,
                 lambda_reliability_monopoly: float = 0.0,
                 reliability_monopoly_cap: float = 0.75):
        super().__init__()
        self.lambda_align = lambda_align
        self.lambda_beam = float(lambda_beam)
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(max(0.0, min(0.3, label_smoothing)))
        self.beam_soft_target_lambda = float(max(0.0, min(1.0, beam_soft_target_lambda)))
        self.beam_soft_target_tau = float(max(0.1, beam_soft_target_tau))
        self.lambda_moe_balance = float(lambda_moe_balance)
        self.lambda_prior_match = float(lambda_prior_match)
        self.lambda_cvar = float(max(0.0, lambda_cvar))
        self.cvar_tail_fraction = float(max(0.01, min(0.5, cvar_tail_fraction)))
        self.lambda_reliability_monopoly = float(lambda_reliability_monopoly)
        self.reliability_monopoly_cap = float(reliability_monopoly_cap)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        if class_prior is not None:
            prior = class_prior.float().clamp_min(1e-12)
            prior = prior / prior.sum()
            self.register_buffer("class_prior", prior)
        else:
            self.class_prior = None
        if valid_beam_mask is not None:
            self.register_buffer("valid_beam_mask", valid_beam_mask.bool())
        else:
            self.valid_beam_mask = None
        if step_weights is not None:
            self.register_buffer("step_weights", step_weights.float())
        else:
            self.step_weights = None

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                aux: Dict, use_reliability_align: bool = True,
                align_loss_module: Optional[ReliabilityWeightedAlignmentLoss] = None,
                align_scale: float = 1.0,
                focal_gamma_override: Optional[float] = None,
                detach_reliability_align_weights: bool = False,
                moe_scale: float = 1.0,
                prior_match_scale: float = 1.0,
                cvar_scale: float = 1.0,
                reliability_monopoly_scale: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            predictions: (B, T, num_beams)
            targets: (B, T)
            aux: auxiliary outputs from model forward
            use_reliability_align: if False, use unweighted alignment (ablation A3)
        Returns:
            total_loss: scalar
            loss_dict: component losses
        """
        B, T, C = predictions.shape

        # L_beam: class-balanced CE (optional) + focal modulation (optional)
        logits = predictions.reshape(-1, C)
        flat_targets = targets.reshape(-1)
        if self.valid_beam_mask is not None:
            valid_mask = self.valid_beam_mask.to(device=logits.device)
            if valid_mask.numel() != C:
                raise ValueError(
                    f"valid_beam_mask has {valid_mask.numel()} classes but logits have {C}"
                )
            invalid_mask = (~valid_mask).unsqueeze(0)  # (1,C)
            if bool((~valid_mask).all()):
                raise ValueError("All classes are masked as invalid in valid_beam_mask.")
            logits = logits.masked_fill(invalid_mask, -1e4)

        ce_per = F.cross_entropy(
            logits,
            flat_targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        log_probs = F.log_softmax(logits, dim=-1)
        focal_gamma = self.focal_gamma if focal_gamma_override is None \
            else max(float(focal_gamma_override), 0.0)
        if focal_gamma > 0:
            true_logp = log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
            pt = true_logp.exp().clamp(1e-6, 1.0)
            focal = (1.0 - pt).pow(focal_gamma)
            beam_term = focal * ce_per
        else:
            beam_term = ce_per

        # Beam indices have neighborhood structure. Blend a circular soft-target
        # objective so near-miss beams contribute useful learning signal.
        if self.beam_soft_target_lambda > 0:
            cls = torch.arange(C, device=logits.device, dtype=torch.long).view(1, C)
            tgt = flat_targets.view(-1, 1)
            dist = (cls - tgt).abs()
            circ_dist = torch.minimum(dist, C - dist).float()
            soft_kernel = torch.exp(-circ_dist / self.beam_soft_target_tau)
            if self.valid_beam_mask is not None:
                vmask = self.valid_beam_mask.to(device=logits.device, dtype=soft_kernel.dtype).view(1, C)
                soft_kernel = soft_kernel * vmask
            soft_kernel = soft_kernel / soft_kernel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            soft_ce = -(soft_kernel * log_probs).sum(dim=-1)
            if self.class_weights is not None:
                soft_ce = soft_ce * self.class_weights[flat_targets]
            beam_term = (
                (1.0 - self.beam_soft_target_lambda) * beam_term
                + self.beam_soft_target_lambda * soft_ce
            )

        beam_matrix = beam_term.reshape(B, T)
        if self.step_weights is not None:
            sw = self.step_weights.to(device=beam_matrix.device, dtype=beam_matrix.dtype)
            if sw.numel() < T:
                pad = torch.ones(T - sw.numel(), device=sw.device, dtype=sw.dtype)
                sw = torch.cat([sw, pad], dim=0)
            elif sw.numel() > T:
                sw = sw[:T]
            sw = sw / sw.mean().clamp_min(1e-6)
            L_beam = (beam_matrix * sw.view(1, T)).mean()
        else:
            L_beam = beam_matrix.mean()

        cvar_coeff = self.lambda_cvar * max(float(cvar_scale), 0.0)
        if cvar_coeff > 0:
            per_sample_beam = beam_matrix.mean(dim=1)  # (B,)
            tail_k = max(1, int(round(self.cvar_tail_fraction * B)))
            top_vals, _ = torch.topk(per_sample_beam, k=tail_k, largest=True, sorted=False)
            L_cvar = top_vals.mean()
        else:
            L_cvar = predictions.new_zeros(())

        # Anti-collapse regularizer: match batch-average prediction distribution
        # to the empirical train prior (or uniform if unavailable).
        prior_coeff = self.lambda_prior_match * max(float(prior_match_scale), 0.0)
        if prior_coeff > 0:
            mean_probs = log_probs.exp().mean(dim=0)  # (C,)
            if self.class_prior is not None:
                target_prior = self.class_prior.to(mean_probs.device)
            else:
                target_prior = torch.full_like(mean_probs, 1.0 / float(C))
            L_prior = F.kl_div(
                (mean_probs.clamp_min(1e-8)).log(),
                target_prior,
                reduction="sum",
            )
        else:
            L_prior = predictions.new_zeros(())

        # L_align: reliability-weighted
        align_coeff = self.lambda_align * max(float(align_scale), 0.0)
        if align_coeff > 0:
            align_module = align_loss_module or aux.get("alignment_loss_module", None)
            if align_module is None:
                raise ValueError(
                    "Alignment loss module is required when alignment is enabled."
                )
            L_align = align_module(
                aux["aligned_features"],
                aux["reliability_weights"],
                use_reliability=use_reliability_align,
                detach_reliability=detach_reliability_align_weights,
            )
        else:
            L_align = predictions.new_zeros(())

        # L_balance: MoE
        L_balance = aux.get("moe_balance_loss", torch.tensor(0.0, device=predictions.device))

        moe_coeff = self.lambda_moe_balance * max(float(moe_scale), 0.0)
        rel_mono_coeff = self.lambda_reliability_monopoly * max(
            float(reliability_monopoly_scale), 0.0
        )
        if rel_mono_coeff > 0 and "reliability_weights" in aux:
            mods = ["image", "radar", "lidar", "gps"]
            w_stack = torch.stack([aux["reliability_weights"][m] for m in mods], dim=-1)  # (B,H,4)
            max_w = w_stack.max(dim=-1).values
            L_rel_mono = F.relu(max_w - self.reliability_monopoly_cap).mean()
        else:
            L_rel_mono = predictions.new_zeros(())
        total = (
            self.lambda_beam * L_beam
            + align_coeff * L_align
            + moe_coeff * L_balance
            + prior_coeff * L_prior
            + cvar_coeff * L_cvar
            + rel_mono_coeff * L_rel_mono
        )

        return total, {
            "total": total.item(),
            "beam": L_beam.item(),
            "alignment": L_align.item(),
            "moe_balance": L_balance.item() if torch.is_tensor(L_balance) else L_balance,
            "prior_match": L_prior.item(),
            "cvar": L_cvar.item(),
            "reliability_monopoly": L_rel_mono.item(),
            "beam_coeff": self.lambda_beam,
            "align_coeff": align_coeff,
            "moe_balance_coeff": moe_coeff,
            "prior_match_coeff": prior_coeff,
            "cvar_coeff": cvar_coeff,
            "reliability_monopoly_coeff": rel_mono_coeff,
            "focal_gamma": focal_gamma,
            "label_smoothing": self.label_smoothing,
            "step_weighting": 1.0 if self.step_weights is not None else 0.0,
        }


class EncoderPretrainModel(nn.Module):
    """Encoder pre-training model (unchanged from base, for Phase 1 encoder warmup)."""

    def __init__(self, feature_dim=64, num_beams=64, pretrained=True):
        super().__init__()
        self.encoder = MultimodalEncoder(feature_dim, pretrained)
        self.fusion_fc = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, num_beams)

    def forward(self, images, radars, lidars, gps):
        if images.dim() == 4: images = images.unsqueeze(1)
        if radars.dim() == 4: radars = radars.unsqueeze(1)
        if lidars.dim() == 4: lidars = lidars.unsqueeze(1)
        if gps.dim() == 2:    gps = gps.unsqueeze(1)
        raw_features = self.encoder(images, radars, lidars, gps)
        concat = torch.cat([
            F.normalize(raw_features["image"][:, 0, :], dim=-1),
            F.normalize(raw_features["radar"][:, 0, :], dim=-1),
            F.normalize(raw_features["lidar"][:, 0, :], dim=-1),
            F.normalize(raw_features["gps"][:, 0, :], dim=-1),
        ], dim=-1)
        fused = self.fusion_fc(concat)
        return self.classifier(fused), raw_features
