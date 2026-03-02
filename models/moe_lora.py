"""
C3: Label-Efficient Two-Stage Training with PEFT.
Reference: Paper Section II-C.

Components:
  - LoRA adapter: low-rank weight updates for frozen LLM layers
  - MoE-LoRA: degradation-aware expert routing
  - SSL objectives (Stage 1): masked reconstruction, cross-modal prediction, temporal forecasting
  - Beam prediction head (Stage 2)

Degradation-aware MoE routing:
  e(t) = argmax_k R_k([o(t); r(t); w(t)])
  Routes each sample to specialist expert based on delay regime and reliability pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math
from transformers import GPT2Model


# ===========================================================================
# LoRA Adapter
# ===========================================================================

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) for a single linear layer.
    W' = W + (α/r) · B·A where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}

    Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns LoRA delta: (α/r) · x·A^T·B^T"""
        dropped = self.lora_dropout(x)
        return (dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling


# ===========================================================================
# MoE-LoRA with Degradation-Aware Routing
# ===========================================================================

class MoELoRA(nn.Module):
    """
    Mixture-of-Experts LoRA with degradation-aware routing.

    Each expert specializes in a failure-mode pattern:
    - Expert 0: clean conditions (all modalities reliable)
    - Expert 1: camera degraded (fog/rain/occlusion)
    - Expert 2: high-delay regime (GPS drift, stale LiDAR)
    - Expert 3: multi-modality degradation (severe conditions)

    Router input: [o(t); r(t); w(t)] where:
      o(t) = pooled LLM features
      r(t) = delay regime vector
      w(t) = reliability weight vector
    """

    def __init__(self, in_features: int, out_features: int,
                 num_experts: int = 4, rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.05, num_modalities: int = 4,
                 num_regimes: int = 3, router_hidden: int = 128,
                 extra_context_dim: int = 0):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        # E experts, each a LoRA adapter
        self.experts = nn.ModuleList([
            LoRALayer(in_features, out_features, rank, alpha, dropout)
            for _ in range(num_experts)
        ])

        # Router: R_k([o; r; w]) -> expert scores
        # o: feature_dim, r: num_regimes (one-hot), w: num_modalities
        self.extra_context_dim = int(max(0, extra_context_dim))
        router_input_dim = in_features + num_modalities + num_regimes + self.extra_context_dim
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts),
        )

        # Load-balancing auxiliary loss coefficient
        self.balance_coeff = 0.01
        # Paper-faithful inference behavior: top-1 sparse routing at eval time.
        self.sparse_inference_top1 = True
        # A4 ablation support: force all samples through one shared expert.
        self.force_shared_expert = False
        self.shared_expert_idx = 0

    def set_shared_expert(self, enabled: bool = True, expert_idx: int = 0):
        """Enable/disable single-expert routing (used by A4 ablation)."""
        self.force_shared_expert = enabled
        self.shared_expert_idx = int(max(0, min(expert_idx, self.num_experts - 1)))

    def forward(self, x: torch.Tensor,
                reliability_vec: Optional[torch.Tensor] = None,
                regime_vec: Optional[torch.Tensor] = None,
                extra_context_vec: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:               (B, seq_len, in_features)
            reliability_vec: (B, num_modalities) per-modality reliability
            regime_vec:      (B, num_regimes) one-hot delay regime
        Returns:
            output: (B, seq_len, out_features) MoE-LoRA output
            balance_loss: scalar load-balancing loss
        """
        B, S, D = x.shape
        device = x.device

        if self.force_shared_expert or self.num_experts == 1:
            expert_idx = int(max(0, min(self.shared_expert_idx, self.num_experts - 1)))
            output = self.experts[expert_idx](x)
            return output, x.new_zeros(())

        if reliability_vec is None:
            reliability_vec = torch.ones(B, 4, device=device) * 0.25
        if regime_vec is None:
            regime_vec = torch.zeros(B, 3, device=device)
            regime_vec[:, 0] = 1.0  # default: low delay
        if extra_context_vec is None and self.extra_context_dim > 0:
            extra_context_vec = torch.zeros(B, self.extra_context_dim, device=device)

        # Router input: pool x over seq_len, concat with reliability + regime
        x_pooled = x.mean(dim=1)  # (B, D)
        router_parts = [x_pooled, reliability_vec, regime_vec]
        if self.extra_context_dim > 0:
            router_parts.append(extra_context_vec)
        router_input = torch.cat(router_parts, dim=-1)
        gate_logits = self.router(router_input)  # (B, E)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B, E)

        # Top-1 routing (sparse: only one expert active at inference)
        top_expert = gate_probs.argmax(dim=-1)  # (B,)

        if (not self.training) and self.sparse_inference_top1:
            # True top-1 sparse routing at inference: each sample uses one expert.
            output = torch.zeros(B, S, self.out_features, device=device, dtype=x.dtype)
            for expert_idx, expert in enumerate(self.experts):
                sample_mask = (top_expert == expert_idx)
                if sample_mask.any():
                    output[sample_mask] = expert(x[sample_mask])
        else:
            # Compute all expert outputs (during training, for gradient flow)
            expert_outputs = torch.stack([
                expert(x) for expert in self.experts
            ], dim=1)  # (B, E, S, out_features)

            # Weighted sum with gate probs (soft routing during training)
            gate_expanded = gate_probs.unsqueeze(-1).unsqueeze(-1)  # (B, E, 1, 1)
            output = (gate_expanded * expert_outputs).sum(dim=1)  # (B, S, out_features)

        # Load-balancing loss (Switch Transformer style)
        # Encourage uniform expert utilization
        avg_gate = gate_probs.mean(dim=0)  # (E,)
        balance_loss = self.balance_coeff * (self.num_experts * (avg_gate ** 2).sum())

        return output, balance_loss


# ===========================================================================
# LLM Backbone with MoE-LoRA PEFT
# ===========================================================================

class PEFTLLMBackbone(nn.Module):
    """
    Pretrained GPT-2 backbone with MoE-LoRA PEFT.

    LoRA adapters are injected *inside* GPT-2 transformer blocks via forward
    hooks on the attention Q/K/V projections, attention output projection,
    and MLP layers.  This is the standard LoRA approach (Hu et al., ICLR 2022)
    and ensures the low-rank corrections participate in intermediate feature
    propagation rather than being applied as a post-hoc residual.
    """

    def __init__(self, model_name: str = "gpt2",
                 num_unfrozen_layers: int = 0,
                 num_adapter_layers: int = 2,
                 feature_dim: int = 64,
                 llm_hidden_dim: int = 768,
                 T: int = 5,
                 num_fusion_tokens: int = 4,
                 num_experts: int = 4,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.05,
                 num_modalities: int = 4,
                 num_regimes: int = 3,
                 router_extra_dim: int = 0):
        super().__init__()
        self.T = T
        self.llm_hidden_dim = llm_hidden_dim
        self.num_fusion_tokens = num_fusion_tokens

        # Input projection: feature_dim -> llm_hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim),
            nn.Dropout(0.1),
        )

        # Learnable prompt tokens for T future predictions
        self.prompt_tokens = nn.Parameter(
            torch.randn(1, T, llm_hidden_dim) * 0.02
        )

        # Pretrained LLM (frozen)
        self.llm = GPT2Model.from_pretrained(model_name)
        total_layers = len(self.llm.h)
        num_unfrozen_layers = int(max(0, min(int(num_unfrozen_layers), total_layers)))
        num_adapter_layers = int(max(0, min(int(num_adapter_layers), total_layers)))
        adapter_start = total_layers - num_adapter_layers

        # Freeze all
        for param in self.llm.parameters():
            param.requires_grad = False

        # Unfreeze last N layers
        if num_unfrozen_layers > 0:
            for layer in self.llm.h[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            if hasattr(self.llm, "ln_f"):
                for param in self.llm.ln_f.parameters():
                    param.requires_grad = True

        # --- Intra-layer MoE-LoRA adapters ---
        # We create LoRA adapters for attention (q, k, v, o) and FFN in each
        # adapted layer. During forward, we use hooks to inject LoRA deltas
        # into the GPT-2 linear projections at the correct positions.
        self.adapter_keys = ("q", "k", "v", "o", "ffn")
        self.moe_lora_layers = nn.ModuleList()
        self._adapter_layer_indices = []  # which GPT-2 layer indices get adapters
        for layer_idx in range(adapter_start, total_layers):
            layer_adapters = nn.ModuleDict({
                key: MoELoRA(llm_hidden_dim, llm_hidden_dim,
                             num_experts, lora_rank, lora_alpha, lora_dropout,
                             num_modalities, num_regimes,
                             extra_context_dim=router_extra_dim)
                for key in self.adapter_keys
            })
            self.moe_lora_layers.append(layer_adapters)
            self._adapter_layer_indices.append(layer_idx)

        # Output projection: llm_hidden_dim -> feature_dim
        self.output_proj = nn.Sequential(
            nn.Linear(llm_hidden_dim, feature_dim),
        )

        # Runtime context for MoE routing (set before each forward pass)
        self._moe_context = {
            "reliability_vec": None,
            "regime_vec": None,
            "router_extra_vec": None,
            "balance_loss": None,
        }

        # Register forward hooks on the adapted GPT-2 layers
        self._hooks = []
        for adapter_idx, layer_idx in enumerate(self._adapter_layer_indices):
            hook = self.llm.h[layer_idx].register_forward_hook(
                self._make_layer_hook(adapter_idx)
            )
            self._hooks.append(hook)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"PEFT LLM Backbone: {model_name}")
        print(f"  Unfrozen layers: {num_unfrozen_layers}/{total_layers}")
        print(f"  Adapter layers: {num_adapter_layers}/{total_layers} (intra-layer hooks)")
        print(f"  MoE-LoRA: {num_experts} experts, rank={lora_rank}")
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _make_layer_hook(self, adapter_idx: int):
        """
        Create a forward hook for a GPT-2 transformer block that applies
        MoE-LoRA corrections to the block's hidden states.

        GPT-2 block forward returns (hidden_states, ...).  We apply LoRA
        deltas to the full hidden-state tensor so the corrections propagate
        through subsequent layers.
        """
        def hook_fn(module, input, output):
            # output is a tuple: (hidden_states, present_kv, ...)
            hidden = output[0]  # (B, seq_len, d_LLM)
            moe_adapters = self.moe_lora_layers[adapter_idx]
            ctx = self._moe_context
            rel = ctx["reliability_vec"]
            reg = ctx["regime_vec"]
            rex = ctx["router_extra_vec"]

            for key in self.adapter_keys:
                delta, bl = moe_adapters[key](hidden, rel, reg, rex)
                hidden = hidden + delta
                if ctx["balance_loss"] is not None:
                    ctx["balance_loss"] = ctx["balance_loss"] + bl
                else:
                    ctx["balance_loss"] = bl

            return (hidden,) + output[1:]
        return hook_fn

    def forward(self, Z: torch.Tensor,
                reliability_vec: Optional[torch.Tensor] = None,
                regime_vec: Optional[torch.Tensor] = None,
                router_extra_vec: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z: (B, seq_len, feature_dim) fused token sequence from CrossAttentionFusion
            reliability_vec: (B, num_modalities) for MoE routing
            regime_vec: (B, num_regimes) for MoE routing
        Returns:
            predictions: (B, T, feature_dim)
            moe_balance_loss: scalar
        """
        B = Z.size(0)

        # Project to LLM dim
        embeddings = self.input_proj(Z)  # (B, seq_len, d_LLM)

        # Append prompt tokens
        prompts = self.prompt_tokens.expand(B, -1, -1)
        full_input = torch.cat([embeddings, prompts], dim=1)  # (B, seq_len+T, d_LLM)

        # Set MoE routing context for hooks
        self._moe_context = {
            "reliability_vec": reliability_vec,
            "regime_vec": regime_vec,
            "router_extra_vec": router_extra_vec,
            "balance_loss": torch.tensor(0.0, device=Z.device),
        }

        # Forward through LLM (hooks apply LoRA inside each adapted layer)
        llm_output = self.llm(inputs_embeds=full_input).last_hidden_state

        total_balance_loss = self._moe_context["balance_loss"]
        if total_balance_loss is None:
            total_balance_loss = torch.tensor(0.0, device=Z.device)

        # Extract prompt positions for future prediction
        prompt_output = llm_output[:, -self.T:, :]  # (B, T, d_LLM)

        # Output projection
        predictions = self.output_proj(prompt_output)  # (B, T, feature_dim)

        return predictions, total_balance_loss


# ===========================================================================
# Vanilla Transformer Backbone (A5 / E3 baseline)
# ===========================================================================

class VanillaTransformerBackbone(nn.Module):
    """
    Vanilla multimodal Transformer baseline used for A5/E3 comparisons.
    Keeps the same token interface as PEFTLLMBackbone but without pretrained LLM
    and without MoE-LoRA routing.
    """

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 768, T: int = 5,
                 num_layers: int = 6, num_heads: int = 8,
                 ffn_hidden_dim: int = 2048, dropout: float = 0.1,
                 max_seq_len: int = 256):
        super().__init__()
        self.T = T
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.prompt_tokens = nn.Parameter(
            torch.randn(1, T, hidden_dim) * 0.02
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.01
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, Z: torch.Tensor,
                reliability_vec: Optional[torch.Tensor] = None,
                regime_vec: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z: (B, seq_len, feature_dim)
            reliability_vec/regime_vec: unused (kept for API compatibility)
        Returns:
            predictions: (B, T, feature_dim)
            balance_loss: zero scalar for interface compatibility
        """
        B, S, _ = Z.shape
        x = self.input_proj(Z)
        prompts = self.prompt_tokens.expand(B, -1, -1)
        x = torch.cat([x, prompts], dim=1)  # (B, S+T, hidden)

        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.encoder(x)
        x = self.final_norm(x)

        prompt_output = x[:, -self.T:, :]
        predictions = self.output_proj(prompt_output)
        return predictions, predictions.new_zeros(())


# ===========================================================================
# Self-Supervised Learning Objectives (Stage 1)
# ===========================================================================

class SSLObjectives(nn.Module):
    """
    Stage 1 self-supervised objectives (no beam labels needed):
      (T1) Masked reconstruction: reconstruct randomly masked feature tokens
      (T2) Cross-modal prediction: predict modality ω from other modalities
      (T3) Temporal forecasting: predict next-step fused representation
    """

    MODALITIES = ["image", "radar", "lidar", "gps"]

    def __init__(self, feature_dim: int = 64, mask_ratio: float = 0.3):
        super().__init__()
        self.mask_ratio = mask_ratio

        # (T1) Per-modality reconstruction decoders
        self.decoders = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.GELU(),
                nn.Linear(feature_dim * 2, feature_dim),
            ) for m in self.MODALITIES
        })

        # (T2) Cross-modal prediction: predict ω from concatenation of others
        self.cross_predictors = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(feature_dim * 3, feature_dim * 2),
                nn.GELU(),
                nn.Linear(feature_dim * 2, feature_dim),
            ) for m in self.MODALITIES
        })

        # (T3) Temporal forecasting: predict Z(t+1) from Z(t)
        self.temporal_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def masked_reconstruction_loss(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(T1) Randomly mask features and reconstruct."""
        total_loss = 0.0
        for mod in self.MODALITIES:
            feat = features[mod]  # (B, H, M)
            B, H, M = feat.shape

            # Random mask per (sample, timestep)
            mask = (torch.rand(B, H, device=feat.device) < self.mask_ratio).float()
            masked_feat = feat * (1 - mask.unsqueeze(-1))

            # Reconstruct
            reconstructed = self.decoders[mod](masked_feat)
            loss = ((reconstructed - feat) ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum() * M + 1e-8)
            total_loss += loss

        return total_loss / len(self.MODALITIES)

    def cross_modal_prediction_loss(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(T2) Predict each modality from the other three."""
        total_loss = 0.0
        for target_mod in self.MODALITIES:
            other_mods = [m for m in self.MODALITIES if m != target_mod]
            # Concat other modalities: (B, H, 3*M)
            other_feats = torch.cat([features[m] for m in other_mods], dim=-1)
            predicted = self.cross_predictors[target_mod](other_feats)
            target = features[target_mod]
            total_loss += F.mse_loss(predicted, target.detach())

        return total_loss / len(self.MODALITIES)

    def temporal_forecasting_loss(self, fused: torch.Tensor) -> torch.Tensor:
        """(T3) Predict next-step fused representation."""
        # fused: (B, H, M)
        if fused.shape[1] < 2:
            return torch.tensor(0.0, device=fused.device)

        current = fused[:, :-1, :]   # (B, H-1, M)
        target = fused[:, 1:, :]     # (B, H-1, M)
        predicted = self.temporal_predictor(current)
        return F.mse_loss(predicted, target.detach())

    def forward(self, features: Dict[str, torch.Tensor],
                fused: torch.Tensor,
                lambda_mr: float = 1.0, lambda_cm: float = 0.5,
                lambda_tf: float = 0.5) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total SSL loss.
        Returns:
            total_loss: scalar
            loss_dict: individual loss components
        """
        l_mr = self.masked_reconstruction_loss(features)
        l_cm = self.cross_modal_prediction_loss(features)
        l_tf = self.temporal_forecasting_loss(fused)

        total = lambda_mr * l_mr + lambda_cm * l_cm + lambda_tf * l_tf
        return total, {
            "masked_recon": l_mr.item(),
            "cross_modal": l_cm.item(),
            "temporal_forecast": l_tf.item(),
            "ssl_total": total.item(),
        }
