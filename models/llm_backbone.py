"""
LLM Backbone for M²BeamLLM with Selective Fine-Tuning (SFT).
Reference: Section III-E, Fig. 2.

Uses a pretrained GPT-2 backbone. Most layers are frozen;
only the last few layers are unfrozen for task adaptation.
Components:
    - Normalization: Z → zero mean, unit variance (stats computed on-the-fly)
    - Input Projection: normalized Z → embeddings (H×d_LLM)
    - Layer Norm + Dropout
    - Pretrained LLM: frozen + unfrozen layers
    - Output Projection: embeddings → predictions (T×M)
    - Inverse Normalization: restore original scale using saved stats

BUG FIX: Added T learnable prompt tokens for future prediction.
    GPT-2 uses causal attention — without prompt tokens, the model has no
    mechanism to predict future timesteps beyond the input sequence.
    Prompt tokens give the LLM explicit "query slots" for each future step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, BertModel, BertConfig


class InputProjection(nn.Module):
    """
    Projects fused feature sequence Z into LLM embedding space.

    Steps (per paper Section III-E):
        1. Compute mean and std of Z across time dimension (per feature)
        2. Normalize Z to zero mean, unit variance
        3. Linear projection: M → d_LLM

    The mean and std are returned for use in inverse normalization.
    """

    def __init__(self, input_dim: int = 64, llm_dim: int = 768):
        super().__init__()
        self.projection = nn.Linear(input_dim, llm_dim)

    def forward(self, Z: torch.Tensor):
        """
        Args:
            Z: (B, H, M) fused feature sequence
        Returns:
            embeddings: (B, H, d_LLM) projected embeddings
            mean: (B, 1, M) per-sample mean for inverse norm
            std: (B, 1, M) per-sample std for inverse norm
        """
        mean = Z.mean(dim=1, keepdim=True)         # (B, 1, M)
        std = Z.std(dim=1, keepdim=True) + 1e-5     # (B, 1, M)
        Z_norm = (Z - mean) / std                   # (B, H, M)
        embeddings = self.projection(Z_norm)         # (B, H, d_LLM)
        return embeddings, mean, std


class OutputProjection(nn.Module):
    """
    Projects LLM output back to feature dimension M, then applies
    inverse normalization to restore original scale.

    Reference: Fig. 2 "Output Projection → Inverse Norm"
    """

    def __init__(self, llm_dim: int = 768, output_dim: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Linear(llm_dim, output_dim)

    def forward(self, llm_output: torch.Tensor,
                mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Args:
            llm_output: (B, T, d_LLM) LLM outputs at prompt positions
            mean: (B, 1, M) from input projection
            std: (B, 1, M) from input projection
        Returns:
            (B, T, M) predictions in original feature space
        """
        projected = self.projection(llm_output)      # (B, T, M)
        output = projected * std + mean               # inverse normalization
        return output


class LLMBackbone(nn.Module):
    """
    Pretrained LLM backbone with selective fine-tuning and prompt tokens.

    Supports:
        - GPT-2: decoder-only, autoregressive (paper default)
        - BERT: encoder-only, bidirectional (for comparison)
    """

    def __init__(self, model_name: str = "gpt2",
                 num_unfrozen_layers: int = 2,
                 feature_dim: int = 64,
                 llm_hidden_dim: int = 768,
                 T: int = 5):
        super().__init__()
        self.model_name = model_name
        self.T = T
        self.llm_hidden_dim = llm_hidden_dim

        # Input projection (with normalization)
        self.input_proj = InputProjection(feature_dim, llm_hidden_dim)

        # Learnable prompt tokens for T future prediction positions
        self.prompt_tokens = nn.Parameter(
            torch.randn(1, T, llm_hidden_dim) * 0.02
        )

        # Load pretrained LLM
        self.is_gpt2 = "gpt2" in model_name.lower()
        if self.is_gpt2:
            self.llm = GPT2Model.from_pretrained(model_name)
            total_layers = len(self.llm.h)
        else:
            self.llm = BertModel.from_pretrained(model_name)
            total_layers = len(self.llm.encoder.layer)

        # Freeze all parameters first
        for param in self.llm.parameters():
            param.requires_grad = False

        # Unfreeze the last `num_unfrozen_layers` transformer layers
        if self.is_gpt2:
            layers_to_unfreeze = self.llm.h[-num_unfrozen_layers:]
        else:
            layers_to_unfreeze = self.llm.encoder.layer[-num_unfrozen_layers:]

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        # Also unfreeze final layer norm
        if self.is_gpt2 and hasattr(self.llm, "ln_f"):
            for param in self.llm.ln_f.parameters():
                param.requires_grad = True

        # Output projection (with inverse normalization)
        self.output_proj = OutputProjection(llm_hidden_dim, feature_dim)

        # Layer norm and dropout before LLM (Fig. 2)
        self.pre_ln = nn.LayerNorm(llm_hidden_dim)
        self.pre_dropout = nn.Dropout(0.1)

        print(f"LLM Backbone: {model_name}")
        print(f"  Total layers: {total_layers}")
        print(f"  Unfrozen layers: {num_unfrozen_layers}")
        print(f"  Prompt tokens: {T} (for {T}-step future prediction)")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: (B, H, M) fused feature sequence
        Returns:
            P_hat: (B, T, M) beam prediction features
        """
        B = Z.size(0)

        # Input projection + normalization
        embeddings, mean, std = self.input_proj(Z)    # (B, H, d_LLM)
        embeddings = self.pre_ln(embeddings)
        embeddings = self.pre_dropout(embeddings)

        # Append T learnable prompt tokens for future prediction
        prompts = self.prompt_tokens.expand(B, -1, -1)
        full_input = torch.cat([embeddings, prompts], dim=1)  # (B, H+T, d_LLM)

        # Pass through LLM backbone
        if self.is_gpt2:
            llm_output = self.llm(inputs_embeds=full_input).last_hidden_state
        else:
            llm_output = self.llm(inputs_embeds=full_input).last_hidden_state

        # Extract only the T prompt positions (future predictions)
        future_output = llm_output[:, -self.T:, :]   # (B, T, d_LLM)
        # Output projection + inverse normalization
        predictions = self.output_proj(future_output, mean, std)

        return predictions  # (B, T, M)