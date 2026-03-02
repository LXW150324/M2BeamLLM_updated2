"""
Multimodal sensing data encoders for M²BeamLLM.
Reference: Section III-A, Fig. 3.

Each encoder maps raw/preprocessed sensor data to an M-dimensional feature vector.
    - Vision: ResNet-18 backbone + FC layers → R^M         (Eq. 5)
    - Radar:  CNN on RA map → R^M                          (Eq. 7)
    - LiDAR:  Modified ResNet-18 on histogram → R^M        (Eq. 9)
    - GPS:    MLP on normalized coordinates → R^M           (Eq. 11)
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Fix macOS SSL certificate issue for downloading pretrained weights
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# ============================================================================
# Vision Feature Encoder — Fig. 3(a), Eq. (5)
# ============================================================================

class VisionEncoder(nn.Module):
    """
    Vision feature encoder using pretrained ResNet-18.

    Pipeline: Image (3×224×224) → ResNet-18 (no FC) → Linear(512→256) → ReLU
              → Linear(256→128) → ReLU → Linear(128→M) → feature (M×1)
    """

    def __init__(self, feature_dim: int = 64, pretrained: bool = True):
        super().__init__()
        self.feature_dim = feature_dim

        # Load pretrained ResNet-18 and remove classification head
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Remove avgpool and fc layers: output is (batch, 512, 7, 7) for 224x224 input
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # FC layers for dimension compression
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224) preprocessed image tensor
        Returns:
            (batch, M) vision feature vector
        """
        features = self.backbone(x)  # (B, 512, 7, 7)
        features = self.avgpool(features)  # (B, 512, 1, 1)
        features = features.flatten(1)  # (B, 512)
        return self.fc(features)  # (B, M)


# ============================================================================
# Radar Feature Encoder — Fig. 3(c), Eq. (7)
# ============================================================================

class RadarEncoder(nn.Module):
    """
    Radar feature encoder using CNN on Range-Angle map.

    Pipeline: RA Map (1×M_F×S_R) → [Conv2D→ReLU]×5 → AvgPool → Flatten
              → Linear(512→256) → ReLU → Linear(256→128) → ReLU
              → Linear(128→M) → feature (M×1)
    """

    def __init__(self, feature_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.feature_dim = feature_dim

        # 5 convolutional blocks
        self.conv_layers = nn.Sequential(
            self._conv_block(in_channels, 16),
            self._conv_block(16, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, M_F, S_R) Range-Angle map
        Returns:
            (batch, M) radar feature vector
        """
        features = self.conv_layers(x)  # (B, 256, H', W')
        features = self.avgpool(features)  # (B, 256, 2, 2)
        features = features.flatten(1)  # (B, 1024)
        return self.fc(features)  # (B, M)


# ============================================================================
# LiDAR Feature Encoder — Fig. 3(b), Eq. (9)
# ============================================================================

class LiDAREncoder(nn.Module):
    """
    LiDAR feature encoder using modified ResNet-18 on histogram map.

    Pipeline: Histogram (1×256×256) → Conv2D(1→3) → BatchNorm → ReLU → MaxPool
              → ResNet-18 (no FC) → GlobalAvgPool → Linear(512→M) → feature (M×1)
    """

    def __init__(self, feature_dim: int = 64, pretrained: bool = True):
        super().__init__()
        self.feature_dim = feature_dim

        # Initial conv to expand single channel to 3 channels for ResNet compatibility
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256→128
        )

        # ResNet-18 backbone (without FC)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 256, 256) LiDAR histogram map
        Returns:
            (batch, M) LiDAR feature vector
        """
        x = self.initial_conv(x)  # (B, 3, 128, 128)
        x = self.backbone(x)  # (B, 512, H', W')
        x = self.global_avgpool(x)  # (B, 512, 1, 1)
        x = x.flatten(1)  # (B, 512)
        return self.fc(x)  # (B, M)


# ============================================================================
# GPS Feature Encoder — Fig. 3(d), Eq. (11)
# ============================================================================

class GPSEncoder(nn.Module):
    """
    GPS feature encoder using MLP.

    Pipeline: GPS (2×1) → Linear(2→32) → LayerNorm → GeLU
              → Linear(32→64) → LayerNorm → GeLU
              → Linear(64→M) → LayerNorm → GeLU → feature (M×1)
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2) normalized GPS coordinates [lat, lon]
        Returns:
            (batch, M) GPS feature vector
        """
        return self.mlp(x)  # (B, M)


# ============================================================================
# Combined Encoder Module
# ============================================================================

class MultimodalEncoder(nn.Module):
    """
    Wraps all four modality-specific encoders.
    Processes H historical frames for each modality.
    """

    def __init__(self, feature_dim: int = 64, pretrained: bool = True,
                 gps_ema_beta: float = 0.0):
        super().__init__()
        self.vision_encoder = VisionEncoder(feature_dim, pretrained)
        self.radar_encoder = RadarEncoder(feature_dim)
        self.lidar_encoder = LiDAREncoder(feature_dim, pretrained)
        self.gps_encoder = GPSEncoder(feature_dim)
        self.gps_ema_beta = float(max(0.0, min(0.99, gps_ema_beta)))
        self.post_norms = nn.ModuleDict({
            "image": nn.LayerNorm(feature_dim),
            "radar": nn.LayerNorm(feature_dim),
            "lidar": nn.LayerNorm(feature_dim),
            "gps": nn.LayerNorm(feature_dim),
        })
        self.modality_scales = nn.ParameterDict({
            "image": nn.Parameter(torch.tensor(1.0)),
            "radar": nn.Parameter(torch.tensor(1.0)),
            "lidar": nn.Parameter(torch.tensor(1.0)),
            # GPS starts slightly down-weighted to reduce shortcut reliance.
            "gps": nn.Parameter(torch.tensor(0.5)),
        })

    def forward(self, images: torch.Tensor, radars: torch.Tensor,
                lidars: torch.Tensor, gps: torch.Tensor):
        """
        Process H frames of multimodal data.

        Args:
            images: (B, H, 3, 224, 224)
            radars: (B, H, 1, M_F, S_R)
            lidars: (B, H, 1, 256, 256)
            gps:    (B, H, 2)

        Returns:
            Dictionary of features, each (B, H, M)
        """
        B, H = images.shape[:2]

        if self.gps_ema_beta > 0.0 and H > 1:
            gps_s = [gps[:, 0]]
            beta = self.gps_ema_beta
            for t in range(1, H):
                gps_s.append(beta * gps_s[-1] + (1.0 - beta) * gps[:, t])
            gps = torch.stack(gps_s, dim=1)

        # Reshape to process all frames at once: (B*H, ...)
        img_flat = images.reshape(B * H, *images.shape[2:])
        rad_flat = radars.reshape(B * H, *radars.shape[2:])
        lid_flat = lidars.reshape(B * H, *lidars.shape[2:])
        gps_flat = gps.reshape(B * H, *gps.shape[2:])

        # Encode
        img_feat = self.vision_encoder(img_flat)  # (B*H, M)
        rad_feat = self.radar_encoder(rad_flat)  # (B*H, M)
        lid_feat = self.lidar_encoder(lid_flat)  # (B*H, M)
        gps_feat = self.gps_encoder(gps_flat)  # (B*H, M)

        # Reshape back: (B, H, M)
        out = {
            "image": img_feat.reshape(B, H, -1),
            "radar": rad_feat.reshape(B, H, -1),
            "lidar": lid_feat.reshape(B, H, -1),
            "gps": gps_feat.reshape(B, H, -1),
        }
        for mod, feat in out.items():
            out[mod] = self.post_norms[mod](feat) * self.modality_scales[mod]
        return out
