"""
Deep diagnostic: check data preprocessing for hidden bugs.
Run from M2BeamLLM_updated: python diagnose_deep.py
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import get_config

cfg = get_config()
root = cfg.data.data_root

print("="*60)
print("  Deep Data Diagnostic")
print("="*60)

# ---------------------------------------------------------------
# 1. Image normalization check (CRITICAL for ResNet-18)
# ---------------------------------------------------------------
print("\n[1] Image data (ResNet-18 expects ImageNet-normalized input)")
images = np.load(os.path.join(root, "train_images.npy"), mmap_mode="r")
print(f"  Shape: {images.shape}, dtype: {images.dtype}")

# Sample a few images
sample = np.array(images[0, 0])  # first sample, first frame, (3, 224, 224)
print(f"  Single image stats (sample[0,0]):")
print(f"    Overall: min={sample.min():.4f}, max={sample.max():.4f}, mean={sample.mean():.4f}")
for c, name in enumerate(["R", "G", "B"]):
    ch = sample[c]
    print(f"    {name} channel: min={ch.min():.4f}, max={ch.max():.4f}, mean={ch.mean():.4f}, std={ch.std():.4f}")

# Check if ImageNet-normalized (mean≈0, std≈1) or raw [0,1] or [0,255]
mean_per_ch = [np.array(images[:, :, c]).mean() for c in range(3)]
print(f"\n  Global channel means (across all samples):")
print(f"    R={mean_per_ch[0]:.4f}, G={mean_per_ch[1]:.4f}, B={mean_per_ch[2]:.4f}")
print(f"  ImageNet mean: R=0.485, G=0.456, B=0.406")
print(f"  If already normalized: means should be ~0")
print(f"  If [0,1] range: means should be ~0.4-0.5")
print(f"  If [0,255] range: means should be ~100-130")

if sample.max() > 2.0:
    print(f"  ⚠ VALUES > 2.0 detected! Likely [0,255] range, NOT normalized!")
elif abs(mean_per_ch[0]) < 0.5 and sample.min() < -1.0:
    print(f"  ✓ Looks ImageNet-normalized (mean≈0, has negative values)")
elif 0.2 < mean_per_ch[0] < 0.7 and sample.min() >= 0:
    print(f"  ⚠ Looks like [0,1] range, NOT ImageNet-normalized!")
    print(f"    ResNet-18 will produce suboptimal features!")

# ---------------------------------------------------------------
# 2. Radar data check
# ---------------------------------------------------------------
print(f"\n[2] Radar data")
radar = np.load(os.path.join(root, "train_radar.npy"), mmap_mode="r")
print(f"  Shape: {radar.shape}, dtype: {radar.dtype}")
sample_r = np.array(radar[0, 0])  # (1, 64, 256)
print(f"  Single sample: min={sample_r.min():.4f}, max={sample_r.max():.4f}, "
      f"mean={sample_r.mean():.4f}, std={sample_r.std():.4f}")
print(f"  Non-zero fraction: {(sample_r != 0).mean():.4f}")

# ---------------------------------------------------------------
# 3. LiDAR data check
# ---------------------------------------------------------------
print(f"\n[3] LiDAR data")
lidar = np.load(os.path.join(root, "train_lidar.npy"), mmap_mode="r")
print(f"  Shape: {lidar.shape}, dtype: {lidar.dtype}")
sample_l = np.array(lidar[0, 0])  # (1, 256, 256)
print(f"  Single sample: min={sample_l.min():.4f}, max={sample_l.max():.4f}, "
      f"mean={sample_l.mean():.4f}, std={sample_l.std():.4f}")
print(f"  Non-zero fraction: {(sample_l != 0).mean():.4f}")

# ---------------------------------------------------------------
# 4. GPS data check
# ---------------------------------------------------------------
print(f"\n[4] GPS data")
gps = np.load(os.path.join(root, "train_gps.npy"), mmap_mode="r")
print(f"  Shape: {gps.shape}, dtype: {gps.dtype}")
sample_g = np.array(gps[:5, :, :])  # first 5 samples
print(f"  First 5 samples, first frame GPS coords:")
for i in range(5):
    print(f"    Sample {i}: lat={gps[i,0,0]:.6f}, lon={gps[i,0,1]:.6f}")
print(f"  Global: min={np.array(gps).min():.6f}, max={np.array(gps).max():.6f}")

# ---------------------------------------------------------------
# 5. Beam data check
# ---------------------------------------------------------------
print(f"\n[5] Beam data")
beams = np.load(os.path.join(root, "train_beams.npy"))
print(f"  Shape: {beams.shape}, dtype: {beams.dtype}")
print(f"  Range: [{beams.min()}, {beams.max()}]")
print(f"  Unique values: {len(np.unique(beams))}")

# Check for per-window beam diversity
diversity = []
for i in range(len(beams)):
    diversity.append(len(np.unique(beams[i])))
diversity = np.array(diversity)
print(f"  Unique beams per window: mean={diversity.mean():.1f}, "
      f"min={diversity.min()}, max={diversity.max()}")
print(f"  Windows with ALL same beam: {(diversity == 1).sum()}")

# ---------------------------------------------------------------
# 6. Sliding window overlap check
# ---------------------------------------------------------------
print(f"\n[6] Sliding window temporal overlap")
# Check if consecutive windows overlap heavily
test_beams = np.load(os.path.join(root, "test_beams.npy"))
# Check if test_beams[i] and test_beams[i+1] differ by just one timestep shift
shifts_detected = 0
for i in range(min(100, len(beams) - 1)):
    if np.array_equal(beams[i, 1:], beams[i+1, :-1]):
        shifts_detected += 1
print(f"  Consecutive windows with 1-step shift: {shifts_detected}/100")
if shifts_detected > 50:
    print(f"  ⚠ Heavy overlap! Adjacent windows share 12/13 frames")
    print(f"    Effective diversity much lower than {len(beams)} samples")

# ---------------------------------------------------------------
# 7. Train/test leakage check
# ---------------------------------------------------------------
print(f"\n[7] Train/test beam sequence overlap")
train_set = set()
for i in range(len(beams)):
    train_set.add(tuple(beams[i]))
overlap = 0
for i in range(len(test_beams)):
    if tuple(test_beams[i]) in train_set:
        overlap += 1
print(f"  Identical beam sequences in both: {overlap}/{len(test_beams)} "
      f"({100*overlap/len(test_beams):.1f}%)")

# ---------------------------------------------------------------
# 8. Per-step accuracy analysis (which future steps are hardest?)
# ---------------------------------------------------------------
print(f"\n[8] Beam transition analysis")
all_beams = np.concatenate([beams, test_beams], axis=0)
H, T = 8, 5
for t in range(T):
    # How much does beam change from step t-1 to step t in future
    if t == 0:
        prev = all_beams[:, H-1]  # last historical
    else:
        prev = all_beams[:, H+t-1]
    curr = all_beams[:, H+t]
    same_pct = (prev == curr).mean()
    # Also check distance
    dist = np.abs(prev.astype(float) - curr.astype(float))
    print(f"  Future step {t+1}: same_beam={100*same_pct:.1f}%, "
          f"mean_beam_distance={dist.mean():.1f}, median={np.median(dist):.0f}")

print(f"\n{'='*60}")
print(f"  Diagnostic Complete")
print(f"{'='*60}")