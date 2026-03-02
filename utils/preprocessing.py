"""
Preprocessing utilities for multimodal sensing data.
Handles all DeepSense 6G file formats: .jpg, .npy, .txt, .ply

Reference: M²BeamLLM paper, Section III-A.
"""

import os
import struct
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# Smart file loaders for various formats
# ============================================================

def smart_load_array(filepath: str) -> np.ndarray:
    """Load numeric array from .npy, .txt, .csv, .dat files."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".npy":
        return np.load(filepath)
    elif ext in (".txt", ".csv", ".dat"):
        try:
            return np.loadtxt(filepath)
        except ValueError:
            try:
                return np.loadtxt(filepath, delimiter=",")
            except:
                values = []
                with open(filepath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                values.append(float(line))
                            except ValueError:
                                pass
                return np.array(values)
    else:
        return np.load(filepath)


def load_ply(filepath: str) -> np.ndarray:
    """
    Load a PLY point cloud file. Returns Nx3 array (x, y, z).
    Handles both ASCII and binary_little_endian PLY formats.
    """
    with open(filepath, "rb") as f:
        # Parse header
        header_lines = []
        properties = []
        vertex_count = 0
        is_binary = False

        while True:
            line = f.readline()
            try:
                line_str = line.decode("ascii").strip()
            except:
                line_str = line.decode("latin-1").strip()
            header_lines.append(line_str)

            if line_str.startswith("format"):
                if "binary" in line_str:
                    is_binary = True
            elif line_str.startswith("element vertex"):
                vertex_count = int(line_str.split()[-1])
            elif line_str.startswith("property"):
                parts = line_str.split()
                # e.g. "property float x" or "property float32 x"
                if len(parts) >= 3:
                    properties.append((parts[-1], parts[1]))
            elif line_str == "end_header":
                break

        if vertex_count == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Find x, y, z property indices
        prop_names = [p[0] for p in properties]

        if is_binary:
            # Binary format
            dtype_map = {
                "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
                "int": "i4", "int32": "i4", "uint8": "u1", "uchar": "u1",
                "short": "i2", "int16": "i2", "uint16": "u2", "ushort": "u2",
                "char": "i1",
            }
            dt = np.dtype([(p[0], dtype_map.get(p[1], "f4")) for p in properties])
            raw = np.frombuffer(f.read(vertex_count * dt.itemsize), dtype=dt, count=vertex_count)

            coords = []
            for name in ["x", "y", "z"]:
                if name in raw.dtype.names:
                    coords.append(raw[name].astype(np.float32))
            if len(coords) >= 3:
                return np.column_stack(coords)
            elif len(coords) == 2:
                return np.column_stack(coords)
            else:
                # Fallback: take first 3 columns
                arr = np.zeros((vertex_count, min(3, len(properties))), dtype=np.float32)
                for ci, p in enumerate(properties[:3]):
                    arr[:, ci] = raw[p[0]].astype(np.float32)
                return arr
        else:
            # ASCII format
            points = []
            for _ in range(vertex_count):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line:
                    vals = line.split()
                    try:
                        point = [float(v) for v in vals[:3]]
                        points.append(point)
                    except ValueError:
                        pass

            if not points:
                return np.zeros((0, 3), dtype=np.float32)
            return np.array(points, dtype=np.float32)


# ============================================================
# Image preprocessing (Eq. 4)
# ============================================================

def get_image_transform(input_size: int = 224,
                        mean: list = None,
                        std: list = None) -> transforms.Compose:
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def preprocess_image(image_path: str, transform=None) -> torch.Tensor:
    if transform is None:
        transform = get_image_transform()
    img = Image.open(image_path).convert("RGB")
    return transform(img)


# ============================================================
# Radar preprocessing (Eq. 6): 2D-FFT -> Range-Angle map
# ============================================================

def radar_2d_fft(radar_data: np.ndarray, fft_size: int = 64) -> np.ndarray:
    """Process raw radar to Range-Angle map via 2D-FFT."""
    if radar_data.ndim == 1:
        total = radar_data.size
        for shape in [(4, 256, 128), (4, 256, 250)]:
            if np.prod(shape) == total:
                radar_data = radar_data.reshape(shape)
                break
        else:
            chirps = total // (4 * 256)
            if chirps > 0:
                radar_data = radar_data.reshape(4, 256, chirps)
            else:
                return np.zeros((fft_size, 256), dtype=np.float32)
    elif radar_data.ndim == 2:
        radar_data = radar_data[:, :, np.newaxis]

    M_R, S_R, A_R = radar_data.shape
    ra_map = np.zeros((fft_size, S_R), dtype=np.float32)

    for a in range(A_R):
        chirp_data = radar_data[:, :, a]
        padded = np.zeros((fft_size, S_R), dtype=complex)
        padded[:min(M_R, fft_size), :] = chirp_data[:min(M_R, fft_size), :]
        fft_result = np.fft.fft2(padded)
        fft_result = np.fft.fftshift(fft_result, axes=0)
        ra_map += np.abs(fft_result).astype(np.float32)

    return ra_map


def preprocess_radar(radar_data: np.ndarray, fft_size: int = 64) -> torch.Tensor:
    ra_map = radar_2d_fft(radar_data, fft_size)
    ra_min, ra_max = ra_map.min(), ra_map.max()
    if ra_max - ra_min > 1e-8:
        ra_map = (ra_map - ra_min) / (ra_max - ra_min)
    return torch.from_numpy(ra_map).unsqueeze(0).float()


# ============================================================
# LiDAR preprocessing (Eq. 8): Point cloud -> 2D histogram
# ============================================================

def point_cloud_to_histogram(point_cloud: np.ndarray,
                             grid_size: int = 256,
                             clip_count: int = 5,
                             x_range: tuple = (-100, 100),
                             y_range: tuple = (-100, 100)) -> np.ndarray:
    """Convert point cloud (Nx2 or Nx3) to 2D histogram."""
    if point_cloud.ndim == 1:
        if len(point_cloud) % 3 == 0:
            point_cloud = point_cloud.reshape(-1, 3)
        elif len(point_cloud) % 2 == 0:
            point_cloud = point_cloud.reshape(-1, 2)
        else:
            return np.zeros((1, grid_size, grid_size), dtype=np.float32)

    if len(point_cloud) == 0:
        return np.zeros((1, grid_size, grid_size), dtype=np.float32)

    x = point_cloud[:, 0]
    y = point_cloud[:, 1] if point_cloud.shape[1] >= 2 else np.zeros_like(x)

    mask = (x >= x_range[0]) & (x < x_range[1]) & (y >= y_range[0]) & (y < y_range[1])
    x_filtered = x[mask]
    y_filtered = y[mask]

    x_bins = np.clip(
        ((x_filtered - x_range[0]) / (x_range[1] - x_range[0]) * grid_size).astype(int),
        0, grid_size - 1)
    y_bins = np.clip(
        ((y_filtered - y_range[0]) / (y_range[1] - y_range[0]) * grid_size).astype(int),
        0, grid_size - 1)

    hist = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(hist, (x_bins, y_bins), 1)
    hist = np.clip(hist, 0, clip_count) / clip_count

    return hist[np.newaxis, :, :]


def preprocess_lidar(point_cloud: np.ndarray, grid_size: int = 256, clip_count: int = 5) -> torch.Tensor:
    hist = point_cloud_to_histogram(point_cloud, grid_size, clip_count)
    return torch.from_numpy(hist).float()


# ============================================================
# GPS preprocessing (Eq. 10): Min-max normalization
# ============================================================

class GPSNormalizer:
    def __init__(self):
        self.gps_min = None
        self.gps_max = None

    def fit(self, all_gps_data: np.ndarray):
        self.gps_min = all_gps_data.min(axis=0)
        self.gps_max = all_gps_data.max(axis=0)

    def transform(self, gps_data: np.ndarray) -> torch.Tensor:
        assert self.gps_min is not None, "Call fit() first"
        denom = self.gps_max - self.gps_min
        denom = np.where(denom < 1e-8, 1.0, denom)
        normalized = (gps_data - self.gps_min) / denom
        return torch.from_numpy(normalized).float()

    def save(self, path: str):
        np.savez(path, gps_min=self.gps_min, gps_max=self.gps_max)

    def load(self, path: str):
        data = np.load(path)
        self.gps_min = data["gps_min"]
        self.gps_max = data["gps_max"]