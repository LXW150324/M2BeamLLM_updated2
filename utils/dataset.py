"""
Windowed NPY dataset loader used by training/evaluation.

Supports preprocessed outputs for DeepSense / DeepMIMO / ViWi as long as they
follow the same split_npy naming convention:
  train_{images,radar,lidar,gps,beams}[,powers,domain_ids].npy
  test_{...}.npy
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, Tuple


class WindowNPYDataset(Dataset):
    """
    Dataset that loads preprocessed sliding-window .npy arrays.
    Each window has `window_size` frames. The first H frames are input
    (historical), the remaining T frames are prediction targets.
    """

    def __init__(self, data_root: str, split: str = "train",
                 H: int = 8, T: int = 5, cfg=None):
        super().__init__()
        self.H = H
        self.T = T
        self.cfg = cfg
        self.dataset_name = getattr(getattr(cfg, "data", None), "dataset_name", "unknown")

        # Load arrays
        self.images = self._load(data_root, split, "images")
        self.radar  = self._load(data_root, split, "radar")
        self.lidar  = self._load(data_root, split, "lidar")
        self.gps    = self._load(data_root, split, "gps")
        self.beams  = self._load(data_root, split, "beams")
        self.domain_ids = self._load(data_root, split, "domain_ids")
        self.powers = self._load(data_root, split, "powers")

        # Load GPS normalizer and apply min-max normalization (Eq. 10)
        gps_norm_path = os.path.join(data_root, "gps_normalizer.npz")
        if self.gps is not None and os.path.isfile(gps_norm_path):
            norm_data = np.load(gps_norm_path)
            gps_min = norm_data["gps_min"]   # (2,)
            gps_max = norm_data["gps_max"]   # (2,)
            denom = gps_max - gps_min
            denom = np.where(denom < 1e-8, 1.0, denom)

            # Normalize in-place: (N, W, 2)
            # Need to materialize if memory-mapped
            self.gps = np.array(self.gps)
            self.gps = (self.gps - gps_min) / denom
            self.gps = self.gps.astype(np.float32)
            print(f"  GPS normalized: range [{self.gps.min():.4f}, {self.gps.max():.4f}]")
        elif self.gps is not None:
            print(f"  WARNING: gps_normalizer.npz not found, GPS NOT normalized!")
            self.gps = np.array(self.gps).astype(np.float32)

        # Validate
        if self.beams is not None:
            self.n_samples = len(self.beams)
            self.window_size = self.beams.shape[1]
            assert self.window_size >= H + T, \
                f"Window size {self.window_size} < H+T={H+T}"
            print(f"[{split}/{self.dataset_name}] Loaded {self.n_samples} windows, "
                  f"window_size={self.window_size}, H={H}, T={T}")
        else:
            print(f"[{split}/{self.dataset_name}] WARNING: No beam labels found. Using dummy data.")
            self.n_samples = 100 if split == "train" else 30
            self.window_size = H + T
            self._create_dummy()

    def _load(self, root, split, name):
        """Load a .npy file, return None if not found."""
        path = os.path.join(root, f"{split}_{name}.npy")
        if os.path.isfile(path):
            arr = np.load(path, mmap_mode="r")
            print(f"  Loaded {split}_{name}.npy: {arr.shape}")
            return arr
        return None

    def _create_dummy(self):
        """Create dummy data for pipeline testing."""
        W = self.window_size
        N = self.n_samples
        self.images = np.random.randn(N, W, 3, 224, 224).astype(np.float32)
        self.radar  = np.random.randn(N, W, 1, 64, 256).astype(np.float32)
        self.lidar  = np.random.randn(N, W, 1, 256, 256).astype(np.float32)
        self.gps    = np.random.rand(N, W, 2).astype(np.float32)  # already [0,1]
        self.beams  = np.random.randint(0, 64, (N, W))
        self.domain_ids = np.zeros((N,), dtype=np.int64)
        self.powers = np.random.rand(N, W, 64).astype(np.float32)

    def _expected_shapes(self):
        """Expected per-frame tensor shapes for zero-filling absent modalities."""
        data_cfg = getattr(self.cfg, "data", None)
        img_size = getattr(data_cfg, "resnet_input_size", 224)
        radar_fft = getattr(data_cfg, "radar_fft_size", 64)
        lidar_grid = getattr(data_cfg, "lidar_grid_size", 256)
        num_beams = getattr(data_cfg, "num_beams", 64)
        return {
            "images": (self.H, 3, img_size, img_size),
            "radars": (self.H, 1, radar_fft, 256),
            "lidars": (self.H, 1, lidar_grid, lidar_grid),
            "gps": (self.H, 2),
            "power_future": (self.T, num_beams),
        }

    def _modality_slice_or_zeros(self, arr, key: str):
        if arr is not None:
            return np.array(arr)
        shape = self._expected_shapes()[key]
        dtype = np.float32
        return np.zeros(shape, dtype=dtype)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        H, T = self.H, self.T
        img_hist = self._modality_slice_or_zeros(
            None if self.images is None else self.images[idx, :H], "images"
        )
        rad_hist = self._modality_slice_or_zeros(
            None if self.radar is None else self.radar[idx, :H], "radars"
        )
        lid_hist = self._modality_slice_or_zeros(
            None if self.lidar is None else self.lidar[idx, :H], "lidars"
        )
        gps_hist = self._modality_slice_or_zeros(
            None if self.gps is None else self.gps[idx, :H], "gps"
        )
        item = {
            "images":       torch.from_numpy(img_hist).float(),
            "radars":       torch.from_numpy(rad_hist).float(),
            "lidars":       torch.from_numpy(lid_hist).float(),
            "gps":          torch.from_numpy(gps_hist).float(),
            "beam_history": torch.from_numpy(np.array(self.beams[idx, :H])).long(),
            "beam_future":  torch.from_numpy(np.array(self.beams[idx, H:H+T])).long(),
        }
        if self.powers is not None:
            p = np.array(self.powers[idx, H:H+T])
            # Expected shape: (T, num_beams). If malformed, silently omit.
            if p.ndim == 2:
                item["power_future"] = torch.from_numpy(p).float()
        if self.domain_ids is not None:
            did = np.array(self.domain_ids[idx]).reshape(-1)
            if did.size > 0:
                item["domain_id"] = torch.tensor(int(did[0]), dtype=torch.long)
        return item


def create_dataloaders(cfg, H: int, T: int) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders."""
    train_ds = WindowNPYDataset(cfg.data.data_root, "train", H=H, T=T, cfg=cfg)
    test_ds  = WindowNPYDataset(cfg.data.data_root, "test",  H=H, T=T, cfg=cfg)

    train_sampler = None
    if bool(getattr(cfg.train, "use_weighted_sampler", False)) and getattr(train_ds, "beams", None) is not None:
        future = np.array(train_ds.beams[:, H:H + T], dtype=np.int64)
        if future.size > 0:
            counts = np.bincount(future.reshape(-1), minlength=getattr(cfg.data, "num_beams", 64)).astype(np.float64)
            freq = counts / max(counts.sum(), 1.0)
            freq = np.clip(freq, 1e-8, None)
            power = float(max(0.0, getattr(cfg.train, "weighted_sampler_power", 0.5)))
            inv = 1.0 / np.power(freq, power)
            sample_w = inv[future].mean(axis=1)
            sample_w = sample_w / max(sample_w.mean(), 1e-8)
            train_sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_w, dtype=torch.double),
                num_samples=len(sample_w),
                replacement=True,
            )
            print(
                f"  Weighted sampler: on (power={power:.2f}, "
                f"min={sample_w.min():.2f}, max={sample_w.max():.2f})"
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )

    return train_loader, test_loader


# Backward-compatible alias used by diagnostics/scripts.
DeepSenseDataset = WindowNPYDataset
