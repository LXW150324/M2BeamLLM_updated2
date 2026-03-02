"""
Configuration for Robust M²BeamLLM Framework.
Extends base M²BeamLLM config with parameters for all innovations (C1-C5).
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    dataset_name: str = "deepsense"  # {deepsense, deepmimo, viwi}
    data_root: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "deepsense", "scenario32")
    deepmimo_root: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "deepmimo")
    viwi_root: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "viwi")
    camera_dir: str = "unit1/camera_data"
    radar_dir: str = "unit1/radar_data"
    lidar_dir: str = "unit1/lidar_data"
    gps_dir: str = "unit1/GPS_data"
    gps_ue_dir: str = "unit2/GPS_data"
    beam_dir: str = "unit1/mmWave_data"

    image_width: int = 960
    image_height: int = 540
    image_channels: int = 3
    resnet_input_size: int = 224
    radar_antennas: int = 4
    radar_samples: int = 256
    radar_chirps: int = 128
    radar_fft_size: int = 64
    lidar_grid_size: int = 256
    lidar_clip_count: int = 5
    num_beams: int = 64
    train_ratio: float = 0.7
    test_ratio: float = 0.3
    window_size: int = 13
    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class AsyncAlignConfig:
    """C1: Delay-Regime-Aware Asynchronous Alignment parameters."""
    buffer_size: int = 3              # K: windowed candidate buffer size
    staleness_embed_dim: int = 32     # ϕ(Δt) embedding dimension
    max_staleness_ms: float = 300.0   # staleness embedding clamp range
    repair_stale_threshold_ms: Optional[float] = 150.0  # trigger repair when Δt > threshold
    repair_max_consecutive_missing: int = 3  # only short missing bursts use feature repair
    num_delay_regimes: int = 3        # R: delay regime bins (low/medium/high)
    delay_regime_boundaries: List[float] = field(
        default_factory=lambda: [50.0, 150.0]  # ms boundaries
    )
    repair_eta_init: float = 0.0      # sigmoid^{-1}(0.5) ≈ 0 for initial η ≈ 0.5
    residual_mlp_hidden: int = 128    # hidden dim of residual alignment MLP
    regime_adapter_dim: int = 32      # lightweight adapter dim per regime


@dataclass
class ReliabilityConfig:
    """C2: Reliability-Guided Fusion parameters."""
    variance_head_hidden: int = 64    # hidden dim of variance head f_var
    reliability_min: float = 1e-4      # proposal-faithful: near-pure precision weighting
    logvar_clip: Optional[float] = None
    softmax_temperature: float = 1.0
    uniform_mix: float = 0.0
    ema_beta: float = 0.0
    pair_conf_threshold: float = 0.0
    alignment_projection_dim: int = 64  # P_{ω←ω'} projection dimension
    degradation_cue_mix: float = 0.0
    degradation_prior_scale: float = 0.0


@dataclass
class MoELoRAConfig:
    """C3: MoE-LoRA PEFT parameters."""
    num_experts: int = 4              # E: number of MoE experts
    lora_rank: int = 32              # Increased from 16 for stronger adaptation
    lora_alpha: float = 64.0         # Scaled with rank (alpha/rank = 2)
    lora_dropout: float = 0.05
    router_hidden_dim: int = 128      # router MLP hidden dim
    use_router_extra_context: bool = True  # Enable extra context for better routing


@dataclass
class SSLConfig:
    """C3: Self-supervised pretraining (Stage 1) parameters."""
    mask_ratio: float = 0.3           # fraction of features to mask
    lambda_mr: float = 1.0            # masked reconstruction weight
    lambda_cm: float = 0.5            # cross-modal prediction weight
    lambda_tf: float = 0.5            # temporal forecasting weight
    # Proposal-faithful default: Stage 1 focuses on encoders/fusion/SSL heads;
    # explicit alignment/reliability are activated in Stage 2 supervision.
    use_explicit_alignment: bool = False
    use_reliability_gating: bool = False
    ssl_epochs: int = 20             # Moderate SSL pretraining
    ssl_lr: float = 5e-4              # Slightly lower to avoid feature space collapse


@dataclass
class StressTestConfig:
    """C5: Robustness Evaluation Suite parameters."""
    # S1: Asynchrony stress
    delay_values_ms: List[float] = field(
        default_factory=lambda: [0.0, 50.0, 100.0, 200.0]
    )
    s1_include_modality_specific_delays: bool = False
    burst_lengths: List[int] = field(
        default_factory=lambda: [1, 3, 5, 10]
    )
    # S2: Degradation stress
    corruption_severities: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    camera_blur_max: float = 12.0     # milder blur on normalized/ImageNet-preprocessed inputs
    lidar_dropout_max: float = 0.8    # occupancy dropout cap on histogram space
    radar_noise_max: float = 5.0      # interpreted as multiplier on normalized radar std
    gps_noise_max: float = 30.0       # milder normalized-position noise cap (meters before scaling)
    gps_noise_in_physical_space: bool = True
    # S3: Domain shift - holdout scenario ratio
    domain_shift_holdout: float = 0.2
    # Tail metrics
    tail_percentile: float = 0.1      # worst 10%
    num_seeds: int = 5


@dataclass
class ModelConfig:
    feature_dim: int = 128            # Increased from 64: wider bottleneck for 64-beam classification
    llm_name: str = "gpt2"
    llm_hidden_dim: int = 768
    num_unfrozen_layers: int = 2      # Unfreeze last 2 GPT-2 layers for task adaptation
    num_adapter_layers: int = 4       # MoE-LoRA on last 4 layers for stronger adaptation
    num_attention_heads: int = 4
    fusion_dropout: float = 0.1
    ffn_hidden_dim: int = 256
    temperature: float = 0.07
    num_modalities: int = 4
    # Cross-attention fusion tokens
    num_fusion_tokens: int = 4        # M tokens for LLM input
    # Vanilla Transformer baseline (A5 / E3)
    # Strict paper-aligned A5 default (~350M order-of-magnitude):
    # 24-layer transformer, wider hidden than GPT-2 backbone wrapper.
    vanilla_hidden_dim: int = 1024
    vanilla_num_layers: int = 24
    vanilla_num_heads: int = 16
    vanilla_ffn_hidden_dim: int = 5120
    vanilla_dropout: float = 0.1
    # Optional extension: history-conditioned residual branch (default off)
    use_beam_history: bool = False
    beam_history_dropout: float = 0.1
    # Optional extension: autoregressive beam decoder (default off)
    use_autoregressive_decoder: bool = False
    ar_decoder_hidden_dim: int = 128
    ar_decoder_dropout: float = 0.1
    # GPS corruption is unusually destructive in S2; smooth short spikes
    # before encoding and add a degradation-aware temporal prior fallback.
    gps_ema_beta: float = 0.0
    degradation_observation_keep_scale: float = 0.0
    degradation_observation_keep_min: float = 1.0
    degradation_temporal_prior_scale: float = 0.0
    degradation_temporal_big_jump_penalty: float = 0.0
    degradation_temporal_neighbor_radius: int = 2
    degradation_temporal_prior_max_boost: float = 0.0
    # Optional extension: direct history-anchor logits around the last beam.
    history_anchor_scale: float = 0.0
    history_anchor_decay: float = 0.8
    history_anchor_big_jump_penalty: float = 0.0
    history_anchor_neighbor_radius: int = 2
    # Optional extension: trainable Top-K pairwise reranker (default off)
    use_pairwise_reranker: bool = False
    pairwise_reranker_feature_dim: int = 24
    pairwise_reranker_hidden_dim: int = 64
    pairwise_reranker_dropout: float = 0.1


@dataclass
class TrainConfig:
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    seed: int = 20260222
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 5e-4
    llm_learning_rate: float = 2e-5
    lora_learning_rate: float = 1e-4
    lr_decay_factor: float = 0.5
    lr_decay_step: int = 8
    weight_decay: float = 1e-4

    lambda_align: float = 0.3
    lambda_beam: float = 1.0
    lambda_moe_balance: float = 0.01   # Enable MoE load-balancing to prevent expert collapse
    lambda_prior_match: float = 0.0
    lambda_reliability_monopoly: float = 0.0
    lambda_cvar: float = 0.0
    reliability_monopoly_cap: float = 0.75
    align_warmup_epochs: int = 3       # Warm up alignment loss over 3 epochs
    use_class_balanced_loss: bool = True  # Enable class-balanced loss for long-tail beam dist
    use_weighted_sampler: bool = False
    weighted_sampler_power: float = 0.5
    # Mask logits of beam IDs never seen in train_future labels.
    # Useful when codebook size is fixed (e.g., 64) but some IDs never occur.
    mask_unseen_beams: bool = True     # Mask unseen beam IDs to focus classifier capacity
    # Optional per-future-step loss weights (length T). If None, uniform.
    beam_step_weights: Optional[List[float]] = None
    focal_gamma: float = 1.0           # Mild focal loss to focus on hard examples
    label_smoothing: float = 0.05       # Mild label smoothing for regularization
    beam_soft_target_lambda: float = 0.1  # Soft neighborhood targets for beam structure
    beam_soft_target_tau: float = 1.5
    focal_warmup_epochs: int = 3       # Ramp focal loss over first 3 epochs
    moe_balance_warmup_epochs: int = 3  # Ramp MoE balance loss
    llm_warmup_epochs: int = 2         # Head-only warmup for 2 epochs
    moe_only_warmup_epochs: int = 2    # MoE-only warmup for 2 epochs
    llm_top1_warmup_epochs: int = 2    # Top LLM layer warmup
    prior_match_warmup_epochs: int = 0
    reliability_monopoly_warmup_epochs: int = 0
    cvar_warmup_epochs: int = 0
    cvar_tail_fraction: float = 0.10
    # AR decoder extension schedules (used only when enabled)
    ar_decoder_warmup_epochs: int = 6
    ar_teacher_forcing_decay_epochs: int = 8
    ar_teacher_forcing_min_ratio: float = 0.2
    # Pairwise reranker (R1) training schedule/loss (used only when enabled)
    pairwise_reranker_k: int = 5
    pairwise_reranker_lambda: float = 0.25
    pairwise_reranker_margin: float = 0.2
    pairwise_reranker_hardcase_weight: float = 2.0
    pairwise_reranker_warmup_epochs: int = 6
    pairwise_reranker_delay_epochs: int = 0
    reliability_align_detach_epochs: int = 0
    # Stage2 training augmentation (helps avoid GPS-only shortcuts)
    stage2_modality_dropout_enable: bool = False
    stage2_modality_dropout_start_epoch: int = 1
    moddrop_image_prob: float = 0.03
    moddrop_radar_prob: float = 0.05
    moddrop_lidar_prob: float = 0.05
    moddrop_gps_prob: float = 0.20
    stage2_corruption_enable: bool = False
    stage2_corruption_start_epoch: int = 4
    stage2_corruption_apply_prob: float = 0.50
    stage2_corruption_combined_prob: float = 0.25

    # Inference-time postprocessing (optional, default off)
    eval_temporal_smoothing_enable: bool = False
    eval_temporal_smoothing_alpha: float = 0.20
    eval_transition_rerank_enable: bool = False
    eval_transition_rerank_lambda: float = 0.35
    eval_transition_rerank_prev_temp: float = 1.0
    eval_transition_prior_pseudocount: float = 1.0
    # Layer-2: constrained top-K reranker (inference-time, default off)
    eval_constrained_reranker_enable: bool = False
    eval_constrained_reranker_k: int = 5
    eval_constrained_reranker_lambda: float = 0.35
    eval_rerank_rank_penalty: float = 0.08
    eval_rerank_dist_prev_penalty: float = 0.35
    eval_rerank_dist_hist_penalty: float = 0.20
    eval_rerank_neighbor_bonus: float = 0.12
    eval_rerank_hist_repeat_bonus: float = 0.12
    eval_rerank_margin_penalty: float = 0.05
    eval_rerank_staleness_scale: float = 0.20
    eval_rerank_lowconf_scale: float = 0.20
    # Layer-3: Viterbi-style temporal smoother over top-K lattice (inference-time)
    eval_viterbi_smoother_enable: bool = False
    eval_viterbi_k: int = 5
    eval_viterbi_path_bonus: float = 0.80
    eval_viterbi_base_alpha: float = 0.35          # beam-distance transition penalty
    eval_viterbi_big_jump_penalty: float = 0.50    # extra penalty for large beam jumps
    eval_viterbi_big_jump_threshold: int = 4
    eval_viterbi_staleness_gain: float = 0.50      # more stale => stronger smoothing
    eval_viterbi_lowconf_gain: float = 0.40        # lower confidence => stronger smoothing
    eval_viterbi_speed_relax_gain: float = 0.60    # faster motion => weaker smoothing
    # Trainable pairwise reranker (R1) inference-time application
    eval_pairwise_reranker_enable: bool = False
    eval_pairwise_reranker_k: int = 5
    eval_pairwise_reranker_lambda: float = 0.35
    eval_pairwise_reranker_stale_scale: float = 0.20
    eval_pairwise_reranker_lowconf_scale: float = 0.20

    # C4 benchmark helpers (optional runtime measurement)
    complexity_latency_warmup_iters: int = 10
    complexity_latency_iters: int = 30

    num_workers: int = 2
    pin_memory: bool = False

    H_standard: int = 8
    T_standard: int = 5
    H_fewshot: int = 3
    T_fewshot: int = 10

    encoder_epochs: int = 20
    encoder_lr: float = 1e-3
    encoder_batch_size: int = 16

    checkpoint_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
    log_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    save_every: int = 5

    top_k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    dba_delta_values: List[int] = field(default_factory=lambda: [1, 2])


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    async_align: AsyncAlignConfig = field(default_factory=AsyncAlignConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    moe_lora: MoELoRAConfig = field(default_factory=MoELoRAConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    stress_test: StressTestConfig = field(default_factory=StressTestConfig)

    def __post_init__(self):
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train.log_dir, exist_ok=True)

    def apply_dataset_preset(self, dataset_name: str, data_root: Optional[str] = None):
        """Select dataset protocol for training/eval (preprocessed .npy window format)."""
        name = str(dataset_name).strip().lower()
        if name not in {"deepsense", "deepmimo", "viwi"}:
            raise ValueError(f"Unsupported dataset_name='{dataset_name}'")
        self.data.dataset_name = name
        if data_root:
            self.data.data_root = data_root
            return
        if name == "deepsense":
            self.data.data_root = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "deepsense", "scenario32"
            )
            # DeepSense defaults preserved
            self.data.num_beams = 64
            self.data.window_size = 13
        elif name == "deepmimo":
            self.data.data_root = self.data.deepmimo_root
            # Paper simulation datasets may differ by scenario/config; keep overridable defaults.
            self.data.num_beams = max(int(self.data.num_beams), 64)
        elif name == "viwi":
            self.data.data_root = self.data.viwi_root
            self.data.num_beams = max(int(self.data.num_beams), 64)


def get_config() -> Config:
    return Config()


def get_device(cfg: Config) -> torch.device:
    if cfg.train.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif cfg.train.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
