"""
M²BeamLLM: Complete model integrating all components.
Reference: Fig. 2, Sections III-A through III-F.

Pipeline:
    1. Multimodal Feature Encoding  (encoders.py)
    2. Multimodal Alignment          (alignment.py) — L2-normalization per paper Eq.12
    3. Multimodal Feature Fusion     (fusion.py)
    4. LLM Backbone + Prediction     (llm_backbone.py)

BUG FIX: lambda_align=0 default in Phase 2 to prevent feature collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from models.encoders import MultimodalEncoder
from models.alignment import MultimodalAlignment
from models.fusion import MultimodalFusion
from models.llm_backbone import LLMBackbone


class M2BeamLLM(nn.Module):
    """Full M²BeamLLM framework for multimodal beam prediction."""

    def __init__(
        self,
        feature_dim: int = 64,
        num_beams: int = 64,
        llm_name: str = "gpt2",
        llm_hidden_dim: int = 768,
        num_unfrozen_layers: int = 2,
        temperature: float = 0.07,
        num_attention_heads: int = 4,
        ffn_hidden_dim: int = 256,
        fusion_dropout: float = 0.1,
        T: int = 5,
        H: int = 8,
        pretrained_encoders: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_beams = num_beams
        self.T = T
        self.H = H

        # 1. Multimodal Encoders (Section III-A)
        self.encoder = MultimodalEncoder(feature_dim, pretrained_encoders)

        # 2. Multimodal Alignment (Section III-B) — L2-normalization (Eq. 12)
        self.alignment = MultimodalAlignment(temperature)

        # 3. Multimodal Fusion (Section III-C)
        self.fusion = MultimodalFusion(
            feature_dim=feature_dim,
            num_heads=num_attention_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=fusion_dropout,
        )

        # 4. LLM Backbone + Prediction (Section III-D,E)
        self.llm_backbone = LLMBackbone(
            model_name=llm_name,
            num_unfrozen_layers=num_unfrozen_layers,
            feature_dim=feature_dim,
            llm_hidden_dim=llm_hidden_dim,
            T=T,
        )

        # 5. Classification head: M features → num_beams classes (Eq. 18)
        self.pred_dropout = nn.Dropout(0.1)
        self.beam_classifier = nn.Linear(feature_dim, num_beams)

    def forward(self, images, radars, lidars, gps):
        # 1. Encode multimodal data
        raw_features = self.encoder(images, radars, lidars, gps)

        # 2. L2-normalize features (Eq. 12)
        normalized_features = self.alignment(raw_features)

        # 3. Fuse features
        Z = self.fusion(normalized_features)

        # 4. LLM backbone prediction (with prompt tokens for future prediction)
        pred_features = self.llm_backbone(Z)

        # 5. Beam classification (Eq. 18)
        predictions = self.beam_classifier(self.pred_dropout(pred_features))

        aux = {
            "raw_features": raw_features,
            "normalized_features": normalized_features,
            "fused": Z,
        }
        return predictions, aux

    def predict_beam_indices(self, images, radars, lidars, gps):
        predictions, _ = self.forward(images, radars, lidars, gps)
        return predictions.argmax(dim=-1)


class M2BeamLLMLoss(nn.Module):
    """
    L = L1 + lambda*L2
    L1: Cross-entropy (Eq. 20)
    L2: Alignment loss (Eq. 21) — default lambda=0 in Phase 2 to prevent collapse
    """

    def __init__(self, temperature=0.07, lambda_align=0.0, label_smoothing=0.0):
        super().__init__()
        self.lambda_align = lambda_align
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.alignment = MultimodalAlignment(temperature)

    def forward(self, predictions, targets, raw_features):
        B, T, num_beams = predictions.shape
        L1 = self.ce_loss(predictions.reshape(-1, num_beams), targets.reshape(-1))

        if self.lambda_align > 0:
            L2 = self.alignment.alignment_loss(raw_features)
            total = L1 + self.lambda_align * L2
        else:
            L2 = torch.tensor(0.0, device=predictions.device)
            total = L1

        return total, {
            "total": total.item(),
            "prediction": L1.item(),
            "alignment": L2.item(),
        }


class EncoderPretrainModel(nn.Module):
    """Encoder pre-training model. Reference: Eq. (19)"""

    def __init__(self, feature_dim=64, num_beams=64, pretrained=True):
        super().__init__()
        self.encoder = MultimodalEncoder(feature_dim, pretrained)
        self.alignment = MultimodalAlignment()
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
        normalized = self.alignment(raw_features)
        concat = torch.cat([
            normalized["image"][:, 0, :], normalized["radar"][:, 0, :],
            normalized["lidar"][:, 0, :], normalized["gps"][:, 0, :],
        ], dim=-1)
        fused = self.fusion_fc(concat)
        return self.classifier(fused), raw_features