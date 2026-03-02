"""
Multimodal Alignment Module for M²BeamLLM.
Reference: Section III-B, Eqs. (12-13).

Bridges semantic discrepancies among heterogeneous modalities by projecting
features onto a shared unit hypersphere and enforcing cosine similarity constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MultimodalAlignment(nn.Module):
    """
    CLIP-inspired contrastive alignment.

    1. L2-normalize each modality's features (Eq. 12)
    2. Compute pairwise cosine similarities (Eq. 13)
    3. Alignment loss encourages high cross-modal similarity (Eq. 21)
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: α in Eq. (21), controls similarity concentration
        """
        super().__init__()
        self.temperature = temperature

    def normalize_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        L2-normalize feature vectors onto unit hypersphere S^{M-1}.

        Reference: Eq. (12)
            x_bar = x_tilde / ||x_tilde||_2

        Args:
            features: Dict of modality -> (B, H, M) feature tensors

        Returns:
            Dict of modality -> (B, H, M) normalized features
        """
        normalized = {}
        for key, feat in features.items():
            normalized[key] = F.normalize(feat, p=2, dim=-1)
        return normalized

    def compute_similarity_matrix(self, normalized_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix between all modalities.

        Reference: Eq. (13)
            S_{w1,w2}[t] = x_bar_{w1}[t] · x_bar_{w2}[t]^T

        Args:
            normalized_features: Dict of modality -> (B, H, M) L2-normalized features

        Returns:
            Similarity matrix of shape (B, H, |Ω|, |Ω|)
        """
        modalities = list(normalized_features.keys())
        num_mod = len(modalities)

        # Stack: (B, H, |Ω|, M)
        stacked = torch.stack([normalized_features[m] for m in modalities], dim=2)
        B, H, _, M = stacked.shape

        # Cosine similarity: (B, H, |Ω|, |Ω|)
        sim_matrix = torch.matmul(stacked, stacked.transpose(-1, -2))

        return sim_matrix

    def alignment_loss(self, features: Dict[str, torch.Tensor],
                       beam_targets: torch.Tensor = None) -> torch.Tensor:
        """
        Compute multimodal alignment loss.

        Reference: Eq. (21) — encourage cross-modal similarity.

        Correct approach: for each pair of different modalities (w1, w2),
        maximize their cosine similarity. This pushes all modality features
        from the same sample/timestep to align in the shared space.

        L2 = 1 - mean_{t, w1≠w2}( cos_sim(x_w1[t], x_w2[t]) )

        Args:
            features: Dict of modality -> (B, H, M) raw features
            beam_targets: unused, kept for API compatibility

        Returns:
            Scalar alignment loss
        """
        normalized = self.normalize_features(features)
        modalities = list(normalized.keys())
        num_mod = len(modalities)

        # Compute mean cosine similarity between all distinct modality pairs
        total_sim = 0.0
        count = 0
        for i in range(num_mod):
            for j in range(i + 1, num_mod):
                # Cosine similarity (already L2-normalized, so just dot product)
                sim = (normalized[modalities[i]] * normalized[modalities[j]]).sum(dim=-1)
                # sim: (B, H), values in [-1, 1]
                total_sim += sim.mean()
                count += 1

        # Loss = 1 - mean_similarity (want to minimize, i.e., maximize similarity)
        mean_sim = total_sim / count
        return 1.0 - mean_sim

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalize features and return.

        Args:
            features: Dict of modality -> (B, H, M)

        Returns:
            Dict of modality -> (B, H, M) normalized features
        """
        return self.normalize_features(features)