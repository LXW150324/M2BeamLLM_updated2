"""
Diagnostic script: check beam index distribution in preprocessed data.
Run this to verify data quality before training.

Usage: python diagnose_data.py --data_root data/deepsense/scenario32
"""

import os
import sys
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/deepsense/scenario32")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"  Data Diagnostic for: {args.data_root}")
    print(f"{'=' * 60}\n")

    # ============================================================
    # 1. Check preprocessed .npy beam files
    # ============================================================
    for split in ["train", "test"]:
        beam_path = os.path.join(args.data_root, f"{split}_beams.npy")
        if not os.path.isfile(beam_path):
            print(f"[{split}] {split}_beams.npy NOT FOUND")
            continue

        beams = np.load(beam_path)
        flat = beams.flatten()
        unique_vals = np.unique(flat)

        print(f"[{split}_beams.npy]")
        print(f"  Shape: {beams.shape}")
        print(f"  Unique beam indices: {len(unique_vals)}")
        print(f"  Min: {flat.min()}, Max: {flat.max()}, Mean: {flat.mean():.2f}")

        if len(unique_vals) <= 5:
            print(f"  *** WARNING: Only {len(unique_vals)} unique values! ***")
            print(f"  Values: {unique_vals}")
            counts = {v: (flat == v).sum() for v in unique_vals}
            print(f"  Counts: {counts}")
        else:
            # Show distribution
            hist, _ = np.histogram(flat, bins=64, range=(0, 63))
            nonzero = np.count_nonzero(hist)
            print(f"  Beam bins used: {nonzero}/64")
            top5 = np.argsort(hist)[-5:][::-1]
            print(f"  Top-5 most common beams: {[(int(b), int(hist[b])) for b in top5]}")
        print()

    # ============================================================
    # 2. Check raw CSV and mmWave data
    # ============================================================
    csv_files = [f for f in os.listdir(args.data_root) if f.endswith(".csv")]
    if csv_files:
        import pandas as pd
        csv_path = os.path.join(args.data_root, csv_files[0])
        df = pd.read_csv(csv_path)
        print(f"[CSV: {csv_files[0]}]")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        # Check if beam_index column exists
        beam_cols = [c for c in df.columns if "beam" in c.lower()]
        if beam_cols:
            print(f"\n  Beam-related columns: {beam_cols}")
            for col in beam_cols:
                vals = df[col].dropna()
                print(f"    {col}: unique={vals.nunique()}, min={vals.min()}, max={vals.max()}, mean={vals.mean():.2f}")

        # Check mmWave column
        pwr_cols = [c for c in df.columns if any(k in c.lower() for k in ["pwr", "mmwave", "power"])]
        print(f"\n  mmWave-related columns: {pwr_cols}")

        # Try loading a sample mmWave file
        for col in pwr_cols:
            sample_ref = str(df.iloc[0][col])
            fpath = os.path.join(args.data_root, sample_ref)
            if os.path.isfile(fpath):
                data = np.load(fpath)
                print(f"\n  Sample mmWave file: {sample_ref}")
                print(f"    Shape: {data.shape}")
                print(f"    Dtype: {data.dtype}")
                print(f"    Min: {data.min():.6f}, Max: {data.max():.6f}")
                flat_pwr = data.flatten()
                if len(flat_pwr) >= 64:
                    pwr64 = flat_pwr[:64]
                    print(f"    First 64 values range: [{pwr64.min():.6f}, {pwr64.max():.6f}]")
                    print(f"    argmax of first 64: {np.argmax(pwr64)}")
                    print(f"    argmax of full array: {np.argmax(flat_pwr)}")
                    print(f"    Full array length: {len(flat_pwr)}")
                    if len(flat_pwr) > 64:
                        print(f"    *** NOTE: File has {len(flat_pwr)} elements, not 64! ***")
                        print(f"    This might mean the power vector format is different.")
                        print(f"    Last 64 values range: [{flat_pwr[-64:].min():.6f}, {flat_pwr[-64:].max():.6f}]")
                        print(f"    argmax of last 64: {np.argmax(flat_pwr[-64:])}")
            else:
                print(f"\n  Could not find mmWave file: {fpath}")
                # Try alternative paths
                for prefix in ["unit1/mmWave_data/", "unit1/"]:
                    alt = os.path.join(args.data_root, prefix, os.path.basename(sample_ref))
                    if os.path.isfile(alt):
                        print(f"  Found at alternative path: {alt}")
                        break

        # Show first few rows of relevant columns
        print(f"\n  First 3 rows (selected columns):")
        show_cols = [c for c in df.columns if any(k in c.lower() for k in ["beam", "pwr", "mmwave", "index"])]
        if show_cols:
            print(df[show_cols].head(3).to_string(index=True))

    # ============================================================
    # 3. Directly inspect mmWave_data directory
    # ============================================================
    mmwave_dir = os.path.join(args.data_root, "unit1", "mmWave_data")
    if os.path.isdir(mmwave_dir):
        files = sorted(os.listdir(mmwave_dir))[:5]
        print(f"\n[unit1/mmWave_data/] ({len(os.listdir(mmwave_dir))} files)")
        for f in files:
            fpath = os.path.join(mmwave_dir, f)
            try:
                data = np.load(fpath)
                beam_idx = int(np.argmax(data.flatten()[:64]))
                print(f"  {f}: shape={data.shape}, dtype={data.dtype}, "
                      f"beam_idx(first64)={beam_idx}, "
                      f"beam_idx(all)={int(np.argmax(data.flatten()))}")
            except Exception as e:
                print(f"  {f}: ERROR - {e}")

    print(f"\n{'=' * 60}")
    print("  Diagnostic complete. Check for:")
    print("  1. Only 1-2 unique beam values → preprocessing bug")
    print("  2. mmWave file shape != (64,) → need to fix beam extraction")
    print("  3. Beam column in CSV with direct values → use that instead")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()