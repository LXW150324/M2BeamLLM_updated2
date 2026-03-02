"""
Quick diagnostic to check beam index range and data integrity.
Run: python diagnose_quick.py
"""
import numpy as np
import os

data_root = "data/deepsense/scenario32"

# 1. Check beam index range
for split in ["train", "test"]:
    bp = os.path.join(data_root, f"{split}_beams.npy")
    if os.path.isfile(bp):
        beams = np.load(bp)
        flat = beams.flatten()
        print(f"\n[{split}_beams.npy]")
        print(f"  Shape: {beams.shape}")
        print(f"  Min: {flat.min()}, Max: {flat.max()}")
        print(f"  Unique: {len(np.unique(flat))}")
        print(f"  Mean: {flat.mean():.2f}")

        # CRITICAL: check if 1-indexed (1-64) vs 0-indexed (0-63)
        if flat.min() >= 1 and flat.max() >= 64:
            print(f"  *** PROBLEM: Beam indices are 1-indexed (1-64)! ***")
            print(f"  *** Need to subtract 1 to get 0-indexed (0-63) ***")
        elif flat.min() >= 0 and flat.max() <= 63:
            print(f"  OK: Beam indices are 0-indexed (0-63)")
        else:
            print(f"  *** UNUSUAL range - check data! ***")

        # Show histogram
        hist, _ = np.histogram(flat, bins=64, range=(0, 63))
        top5 = np.argsort(hist)[-5:][::-1]
        print(f"  Top-5 beams: {[(int(b), int(hist[b])) for b in top5]}")

# 2. Check CSV beam column directly
import pandas as pd

csv_path = os.path.join(data_root, "scenario32_dev.csv")
if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path)
    if "unit1_beam" in df.columns:
        bv = df["unit1_beam"]
        print(f"\n[CSV unit1_beam column]")
        print(f"  Min: {bv.min()}, Max: {bv.max()}")
        print(f"  Unique: {bv.nunique()}")
        print(f"  First 10 values: {bv.head(10).tolist()}")
        if bv.min() >= 1:
            print(f"  *** 1-INDEXED! Must subtract 1 in preprocessing ***")

# 3. Check radar data for complex number issues
rp = os.path.join(data_root, "train_radar.npy")
if os.path.isfile(rp):
    radar = np.load(rp, mmap_mode="r")
    sample = np.array(radar[0, 0])  # First sample, first frame
    print(f"\n[Radar check]")
    print(f"  Shape: {radar.shape}, dtype: {radar.dtype}")
    print(f"  Sample range: [{sample.min():.4f}, {sample.max():.4f}]")
    print(f"  All zeros? {(sample == 0).all()}")
    print(f"  Non-zero fraction: {(sample != 0).mean():.3f}")