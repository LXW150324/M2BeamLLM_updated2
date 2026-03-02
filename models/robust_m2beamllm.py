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
        # Kept for backward compat but unused
        use_beam_history: bool = False,
        beam_history_dropout: float = 0.1,
        use_autoregressive_decoder: bool = False,
        ar_decoder_hidden_dim: int = 128,
        ar_decoder_dropout: float = 0.1,
        gps_ema_beta: float = 0.0,
        use_pairwise_reranker: bool = False,
        pairwise_reranker_feature_dim: int = 24,
        pairwise_reranker_hidden_dim: int = 64,
        pairwise_reranker_dropout: float = 0.1,
        # Unused legacy params (kept for checkpoint compat)
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
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_beams = num_beams
        self.T = T
        self.H = H
        self.num_delay_regimes = num_delay_regimes
        self.router_extra_dim = int(max(0, router_extra_dim))

        # 1. Multimodal Encoders
        self.encoder = MultimodalEncoder(feature_dim, pretrained_encoders)

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

        # 7. Beam classification head
        self.pred_dropout = nn.Dropout(0.1)
        self.beam_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(feature_dim * 2, num_beams),
        )

        # C3: SSL objectives (for Stage 1)
        self.ssl_objectives = SSLObjectives(feature_dim=feature_dim)

        # Initialize alignment projections to identity when possible
        for proj in self.alignment_loss_fn.projections.values():
            if proj.weight.shape[0] == proj.weight.shape[1]:
                nn.init.eye_(proj.weight)
            else:
                nn.init.xavier_uniform_(proj.weight)

    def _build_reliability_and_regime_vectors(
        self, reliability_weights: Dict[str, torch.Tensor],
        regimes: Dict[str, torch.Tensor],
        log_variances: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build compact vectors for MoE routing."""
        mods = ["image", "radar", "lidar", "gps"]
        B = reliability_weights[mods[0]].shape[0]
        device = reliability_weights[mods[0]].device

        # Reliability vector: mean over time, per modality
        rel_vec = torch.stack(
            [reliability_weights[m].mean(dim=1) for m in mods], dim=-1
        )  # (B, 4)

        # Regime vector: aggregate all modality delay regimes
        regime_oh = []
        for m in mods:
            idx = regimes.get(m, torch.zeros(B, dtype=torch.long, device=device))
            regime_oh.append(F.one_hot(idx, self.num_delay_regimes).float())
        regime_vec = torch.stack(regime_oh, dim=0).mean(dim=0)  # (B, R)

        logvar_vec = torch.stack(
            [log_variances[m].mean(dim=1) for m in mods], dim=-1
        )  # (B, 4)
        # Router extra context: logvar (quality signal)
        router_extra_vec = torch.cat([
            torch.zeros(B, 4, device=device),  # placeholder for quality_cues
            logvar_vec,
        ], dim=-1)  # (B, 8)

        return rel_vec, regime_vec, router_extra_vec

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
        reliability_weights, log_variances, gated_features = \
            self.reliability_estimator(
                aligned_features, missing_masks=missing_masks
            )

        # 4. C2-d: Cross-Attention Fusion
        Z = self.fusion(gated_features)  # (B, H*num_tokens, feature_dim)

        # 5. C3: MoE-LoRA LLM
        rel_vec, regime_vec, router_extra_vec = self._build_reliability_and_regime_vectors(
            reliability_weights, align_info.get("regimes", {}), log_variances
        )
        pred_features, moe_balance_loss = self.llm_backbone(
            Z, rel_vec, regime_vec, router_extra_vec
        )  # (B, T, feature_dim)

        # 6. Beam classification
        predictions = self.beam_classifier(self.pred_dropout(pred_features))

        aux = {
            "raw_features": raw_features,
            "aligned_features": aligned_features,
            "reliability_weights": reliability_weights,
            "log_variances": log_variances,
            "gated_features": gated_features,
            "fused": Z,
            "moe_balance_loss": moe_balance_loss,
            "align_info": align_info,
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
        )
        return predictions.argmax(dim=-1)


class RobustM2BeamLLMLoss(nn.Module):
    """
    Training loss for Stage 2:
        L_total = L_beam + lambda_align * L_align^{rw} + lambda_moe * L_balance

    L_align^{rw} = reliability-weighted alignment (C2-c)
    L_balance = MoE load-balancing loss
    """

    def __init__(self, feature_dim: int = 64, lambda_align: float = 1.0,
                 lambda_beam: float = 1.0,
                 label_smoothing: float = 0.0,
                 lambda_moe_balance: float = 1.0,
                 # Kept for backward compat but unused
                 class_weights: Optional[torch.Tensor] = None,
                 class_prior: Optional[torch.Tensor] = None,
                 valid_beam_mask: Optional[torch.Tensor] = None,
                 step_weights: Optional[torch.Tensor] = None,
                 focal_gamma: float = 0.0,
                 beam_soft_target_lambda: float = 0.0,
                 beam_soft_target_tau: float = 1.5,
                 lambda_prior_match: float = 0.0,
                 lambda_cvar: float = 0.0,
                 cvar_tail_fraction: float = 0.10,
                 lambda_reliability_monopoly: float = 0.0,
                 reliability_monopoly_cap: float = 0.75):
        super().__init__()
        self.lambda_align = lambda_align
        self.lambda_beam = float(lambda_beam)
        self.label_smoothing = float(max(0.0, min(0.3, label_smoothing)))
        self.lambda_moe_balance = float(lambda_moe_balance)

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

        # L_beam: cross-entropy loss
        logits = predictions.reshape(-1, C)
        flat_targets = targets.reshape(-1)

        L_beam = F.cross_entropy(
            logits,
            flat_targets,
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )

        # L_align: reliability-weighted alignment
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

        # L_balance: MoE load-balancing
        L_balance = aux.get("moe_balance_loss", torch.tensor(0.0, device=predictions.device))
        moe_coeff = self.lambda_moe_balance * max(float(moe_scale), 0.0)

        total = (
            self.lambda_beam * L_beam
            + align_coeff * L_align
            + moe_coeff * L_balance
        )

        return total, {
            "total": total.item(),
            "beam": L_beam.item(),
            "alignment": L_align.item(),
            "moe_balance": L_balance.item() if torch.is_tensor(L_balance) else L_balance,
            "beam_coeff": self.lambda_beam,
            "align_coeff": align_coeff,
            "moe_balance_coeff": moe_coeff,
            "label_smoothing": self.label_smoothing,
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
