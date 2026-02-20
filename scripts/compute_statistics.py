#!/usr/bin/env python3
"""
H5 データからチャンネルごとの mean / std を計算し、mu.npy / sigma.npy として保存する。
DINOModule 等の NormalizeMeanStd で利用。実行: python scripts/compute_statistics.py [--data_path ...] [--output_dir ...]
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _collect_image_items(f):
    """開いた h5py.File f から (key, index) のリストを返す。3D は (key, None)、4D は (key, 0..N-1)。"""
    items = []

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            sh = obj.shape
            if len(sh) == 3:
                items.append((name, None))
            elif len(sh) == 4:
                for i in range(sh[0]):
                    items.append((name, i))
            else:
                items.append((name, None))

    f.visititems(walk)
    return items


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
    count = x_flat.shape[1]
    if count == 0:
        return n, mean, m2
    mean2 = x_flat.mean(axis=1)
    m2_2 = ((x_flat - mean2[:, None]) ** 2).sum(axis=1)
    return _welford_combine(n, mean, m2, count, mean2, m2_2)


# 3D をチャンク読みするときの行数（メモリと I/O のバランス）
_ROW_CHUNK = 512


def _process_dset_3d_chunked(dset, n_total, mean, m2, C, num_bands, h5_path, key):
    """3D (H, W, C) を行チャンクで読み、Welford 更新。"""
    H, W, C_d = dset.shape
    if C is not None and C != C_d:
        raise ValueError(f"Channel mismatch: expected {C}, got {C_d} in {h5_path}/{key}")
    if num_bands is not None and C_d != num_bands:
        raise ValueError(f"Expected {num_bands} bands, got {C_d} in {h5_path}/{key}")
    C_out = C if C is not None else C_d
    mean = mean if mean is not None else np.zeros(C_out, dtype=np.float64)
    m2 = m2 if m2 is not None else np.zeros(C_out, dtype=np.float64)

    for row_start in range(0, H, _ROW_CHUNK):
        row_end = min(row_start + _ROW_CHUNK, H)
        # (chunk_H, W, C) → (C, chunk_H*W)、float64 で 1 回だけコピー
        chunk = np.asarray(dset[row_start:row_end, :, :], dtype=np.float64)
        arr = chunk.transpose(2, 0, 1).reshape(C_d, -1)
        n_total, mean, m2 = _welford_update(n_total, mean, m2, arr)
    return n_total, mean, m2, C_out


def compute_statistics(data_path: str, output_dir: str, num_bands: int | None) -> tuple[np.ndarray, np.ndarray]:
    """
    data_path 以下の全 H5 画像でチャンネルごとの mean / std を計算する。
    返り値: (mean, std)、各 (C,)。std は母標準偏差（分散を n で割る）。
    """
    data_root = Path(data_path)
    if not data_root.is_absolute():
        data_root = _project_root / data_path
    h5_files = sorted(data_root.rglob("*.h5")) + sorted(data_root.rglob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 under: {data_root}")

    n_files = len(h5_files)
    print(f"Computing statistics from {n_files} H5 file(s) under {data_root}", flush=True)

    n_total = 0
    mean = None
    m2 = None
    C = None

    for file_i, h5_path in enumerate(h5_files, start=1):
        print(f"  [{file_i}/{n_files}] {h5_path.name} ...", flush=True)
        with h5py.File(h5_path, "r") as f:
            items = _collect_image_items(f)
            for key, index in items:
                dset = f[key]
                sh = dset.shape
                if len(sh) == 3:
                    n_total, mean, m2, C = _process_dset_3d_chunked(
                        dset, n_total, mean, m2, C, num_bands, h5_path, key
                    )
                elif len(sh) == 4:
                    i = index if index is not None else 0
                    arr = np.asarray(dset[i], dtype=np.float64)
                    arr = arr.transpose(2, 0, 1).reshape(arr.shape[2], -1)
                    if C is None:
                        C = arr.shape[0]
                        if num_bands is not None and C != num_bands:
                            raise ValueError(f"Expected {num_bands} bands, got {C} in {h5_path}/{key}")
                        mean = np.zeros(C, dtype=np.float64)
                        m2 = np.zeros(C, dtype=np.float64)
                    n_total, mean, m2 = _welford_update(n_total, mean, m2, arr)
                else:
                    continue

    if mean is None:
        raise RuntimeError("No valid image found in data_path.")

    print(f"  Done. Total pixels: {n_total:,}, channels: {C}", flush=True)

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

    print("Computing per-channel mean/std (Welford)...", flush=True)
    mean, std = compute_statistics(args.data_path, args.output_dir, args.num_bands)
    np.save(out_dir / "mu.npy", mean)
    np.save(out_dir / "sigma.npy", std)
    print(f"Saved mu.npy and sigma.npy to {out_dir}")
    print(f"  shape: {mean.shape}, mean range [{mean.min():.4f}, {mean.max():.4f}], std range [{std.min():.4f}, {std.max():.4f}]")


if __name__ == "__main__":
    main()
