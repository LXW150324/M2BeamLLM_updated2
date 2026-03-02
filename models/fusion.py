"""
Multimodal Feature Fusion Module for M²BeamLLM.
Reference: Section III-C, Eqs. (14-16).

Integrates features from heterogeneous modalities using transformer-based
self-attention to capture cross-modal dependencies.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class MultimodalFusion(nn.Module):
    """
    Transformer-based fusion module.

    For each time step t:
    1. Stack normalized features F[t] = [x_Image, x_Radar, x_LiDAR, x_GPS]^T  (M×4)
    2. Apply multi-head self-attention: A[t] = MultiHead(F^T, F^T, F^T)        (Eq. 14)
    3. Aggregate: z[t] = FFN(sum_w A[t](w,:))                                  (Eq. 15)
    4. Concatenate over time: Z = [z[-H+1], ..., z[0]]^T ∈ R^{H×M}            (Eq. 16)
    """

    def __init__(self, feature_dim: int = 64, num_heads: int = 4,
                 ffn_hidden_dim: int = 256, dropout: float = 0.1):
        """
        Args:
            feature_dim: M, the dimension of each modality's feature
            num_heads: Number of attention heads
            ffn_hidden_dim: Hidden dimension of the feed-forward network
            dropout: Dropout rate
        """
        super().__init__()
        self.feature_dim = feature_dim

        # Multi-head self-attention
        # Input: (seq_len=|Ω|, batch, feature_dim=M)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization after attention
        self.norm1 = nn.LayerNorm(feature_dim)

        # Position-wise Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization after FFN
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, normalized_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multimodal features across all time steps.

        Args:
            normalized_features: Dict of modality -> (B, H, M)

        Returns:
            Z: (B, H, M) fused time-series feature sequence
        """
        modalities = list(normalized_features.keys())
        B, H, M = normalized_features[modalities[0]].shape

        fused_sequence = []

        for t in range(H):
            # Stack modality features at time t: (B, |Ω|, M)
            F_t = torch.stack([normalized_features[m][:, t, :] for m in modalities], dim=1)
            # F_t shape: (B, |Ω|=4, M)

            # Multi-head self-attention (Eq. 14)
            # Q = K = V = F_t
            attn_out, _ = self.self_attention(F_t, F_t, F_t)  # (B, |Ω|, M)

            # Residual connection + LayerNorm
            attn_out = self.norm1(F_t + attn_out)

            # Aggregate across modalities by summing (Eq. 15)
            aggregated = attn_out.sum(dim=1)  # (B, M)

            # Feed-forward network
            ffn_out = self.ffn(aggregated)  # (B, M)

            # Residual + norm
            z_t = self.norm2(aggregated + ffn_out)  # (B, M)

            fused_sequence.append(z_t)

        # Stack over time: Z = [z[-H+1], ..., z[0]] (Eq. 16)
        Z = torch.stack(fused_sequence, dim=1)  # (B, H, M)

        return Z