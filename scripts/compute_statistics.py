#!/usr/bin/env python3
"""
データセット（H5）からチャンネルごとの mean / std を計算し、
data/statistics/mu.npy, sigma.npy として保存する。
dino_module の NormalizeMeanStd で利用する。

使い方:
  python scripts/compute_statistics.py [--data_path data/raw/scene] [--output_dir data/statistics]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# プロジェクトルートを path に追加
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.read_h5 import get_h5_image_items


def _welford_combine(n1, mean1, m2_1, n2, mean2, m2_2):
    """2 つの Welford 統計 (n, mean, M2) を結合する。"""
    if n2 == 0:
        return n1, mean1, m2_1
    if n1 == 0:
        return n2, mean2, m2_2
    n = n1 + n2
    mean = (n1 * mean1 + n2 * mean2) / n
    m2 = m2_1 + m2_2 + n1 * n2 * (mean1 - mean2) ** 2 / n
    return n, mean, m2


def _welford_update(n, mean, m2, x_flat):
    """1 枚分のピクセル x_flat (C, N) を Welford のオンライン更新に反映。"""
    # x_flat: (C, N)
    count = x_flat.shape[1]
    if count == 0:
        return n, mean, m2
    mean2 = x_flat.mean(axis=1)
    m2_2 = ((x_flat - mean2[:, None]) ** 2).sum(axis=1)
    return _welford_combine(n, mean, m2, count, mean2, m2_2)


def compute_statistics(data_path: str, output_dir: str, num_bands: int | None) -> tuple[np.ndarray, np.ndarray]:
    """
    data_path 以下の全 H5 画像についてチャンネルごとの mean / std を計算する。
    返り値: (mean, std), 各 (C,)。std は母標準偏差（n で割る）。
    """
    import h5py

    data_root = Path(data_path)
    if not data_root.is_absolute():
        data_root = _project_root / data_path
    h5_files = sorted(data_root.rglob("*.h5")) + sorted(data_root.rglob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 under: {data_root}")

    n_total = 0
    mean = None
    m2 = None
    C = None

    for h5_path in h5_files:
        items = get_h5_image_items(str(h5_path))
        with h5py.File(h5_path, "r") as f:
            for key, index in items:
                dset = f[key]
                sh = dset.shape
                if len(sh) == 3:
                    # (H, W, C) → (C, H*W)
                    arr = np.asarray(dset[...], dtype=np.float64)
                    arr = np.transpose(arr, (2, 0, 1)).reshape(arr.shape[2], -1)
                elif len(sh) == 4:
                    i = index if index is not None else 0
                    arr = np.asarray(dset[i], dtype=np.float64)
                    arr = np.transpose(arr, (2, 0, 1)).reshape(arr.shape[2], -1)
                else:
                    continue
                if C is None:
                    C = arr.shape[0]
                    if num_bands is not None and C != num_bands:
                        raise ValueError(f"Expected {num_bands} bands, got {C} in {h5_path}/{key}")
                    mean = np.zeros(C, dtype=np.float64)
                    m2 = np.zeros(C, dtype=np.float64)
                n_total, mean, m2 = _welford_update(n_total, mean, m2, arr)

    if mean is None:
        raise RuntimeError("No valid image found in data_path.")

    # 母分散 → 母標準偏差（std が 0 のチャンネルは 1 にして除算エラーを防ぐ）
    variance = m2 / max(n_total, 1)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compute per-channel mean/std from H5 data.")
    parser.add_argument("--data_path", type=str, default="data/raw/scene", help="Directory containing .h5 files")
    parser.add_argument("--output_dir", type=str, default="data/statistics", help="Output directory for mu.npy, sigma.npy")
    parser.add_argument("--num_bands", type=int, default=None, help="Expected number of channels (optional check)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = compute_statistics(args.data_path, args.output_dir, args.num_bands)
    np.save(out_dir / "mu.npy", mean)
    np.save(out_dir / "sigma.npy", std)
    print(f"Saved mu.npy and sigma.npy to {out_dir}")
    print(f"  shape: {mean.shape}, mean range [{mean.min():.4f}, {mean.max():.4f}], std range [{std.min():.4f}, {std.max():.4f}]")


if __name__ == "__main__":
    main()
