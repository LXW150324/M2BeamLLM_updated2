# M²BeamLLM: Multimodal Sensing-empowered mmWave Beam Prediction with LLMs

## Reproduction Code for MacBook Pro (Apple Silicon M-series Max)

### Project Structure

```
M2BeamLLM/
├── README.md                   # This file
├── configs/
│   └── config.py               # All hyperparameters & paths
├── data/
│   └── deepsense/              # ← PUT YOUR DeepSense 6G Scenario 32 DATASET HERE
│       └── Scenario32/
│           ├── unit1/           # BS sensing data
│           │   ├── camera/      # RGB images (.jpg)
│           │   ├── radar/       # Radar data (.npy)
│           │   └── lidar/       # LiDAR point clouds (.npy)
│           ├── unit2/           # UE data
│           │   └── gps/         # GPS coordinates (.npy)
│           └── beam_index/      # Ground truth beam indices (.npy)
├── models/
│   ├── __init__.py
│   ├── encoders.py             # Vision, Radar, LiDAR, GPS encoders
│   ├── alignment.py            # Multimodal alignment (CLIP-style)
│   ├── fusion.py               # Multi-head attention fusion
│   ├── llm_backbone.py         # GPT-2 / BERT backbone with SFT
│   └── m2beamllm.py            # Full M²BeamLLM model
├── utils/
│   ├── __init__.py
│   ├── dataset.py              # DeepSense 6G dataset loader
│   ├── preprocessing.py        # Radar FFT, LiDAR histogram, etc.
│   ├── metrics.py              # Top-K accuracy, DBA-Score
│   └── visualization.py        # Training curves, data visualization
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
├── train_encoders.py           # Encoder pre-training script
├── checkpoints/                # Saved model weights
├── logs/                       # Training logs
└── requirements.txt            # Python dependencies
```

### Dataset Placement

1. Download the DeepSense 6G Scenario 32 dataset from:
   https://www.deepsense6g.net/scenario-32/

2. Place the dataset under `data/deepsense/Scenario32/`:
   - The dataset typically contains folders like `unit1/` (BS) and `unit2/` (UE)
   - If your dataset structure differs, update `configs/config.py` accordingly

3. The expected data format per sample:
   - Camera: RGB image (960×540×3) → `.jpg` or `.png`
   - Radar: Complex tensor (4×256×128) → `.npy`
   - LiDAR: Point cloud (N×3) → `.npy`
   - GPS: Coordinates (2,) → `.npy`
   - Beam index: Integer (0-63) → `.npy`

### Mac-Specific Adaptations

| Parameter         | Paper (A100 40GB) | Mac M-series Max   |
|-------------------|--------------------|--------------------|
| Batch size        | 16                 | 8                  |
| Device            | CUDA               | MPS                |
| Precision         | FP32               | FP32               |
| Num workers       | 4                  | 2                  |
| LLM backbone      | distilled-gpt2     | distilled-gpt2     |
| Unfrozen layers   | 2                  | 2                  |

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pre-train encoders
python train_encoders.py

# 3. Train M²BeamLLM (standard prediction: H=8, T=5)
python train.py --mode standard

# 4. Train M²BeamLLM (few-shot prediction: H=3, T=10)
python train.py --mode fewshot

# 5. Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt
```
