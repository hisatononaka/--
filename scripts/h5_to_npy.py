#!/usr/bin/env python3
"""
HDF5 (.h5) を「日付 / グループサブフォルダ / .npy」に展開するスクリプト（Spectral Earth 風）。
前提: トップレベルは Group の列。各 Group 内に Dataset が 1 つ。
出力: {output_dir}/{sensor}/{日付}/{グループキー}/0.npy または 00000.npy, 00001.npy, ...
（グループ = 常にサブフォルダ。中に 1 枚または複数 .npy。）
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def find_h5_files(data_dir: str, recursive: bool = True) -> list[tuple[str, str]]:
    """data_dir 以下で .h5/.hdf5 を探し、(フルパス, 相対パス) のリストを返す。"""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return []
    if recursive:
        files = [(str(p.resolve()), str(p.relative_to(data_path)))
                 for p in data_path.rglob("*") if p.suffix.lower() in (".h5", ".hdf5")]
    else:
        files = [(str(f.resolve()), f.name) for f in sorted(data_path.iterdir())
                 if f.suffix.lower() in (".h5", ".hdf5")]
    return sorted(files, key=lambda x: x[1])


def _datasets_in_group(grp) -> list[tuple[str, h5py.Dataset]]:
    """Group 内の全 Dataset を (キー, Dataset) のリストで返す（1 グループに複数枚ある場合に対応）。"""
    out = []
    for k in sorted(grp.keys()):
        obj = grp[k]
        if isinstance(obj, h5py.Dataset):
            out.append((k, obj))
    return out


def collect_groups(f) -> list[tuple[str, list[tuple[str, h5py.Dataset]]]]:
    """File のトップレベルを走査し、(グループキー, [(サブキー, Dataset), ...]) のリストを返す。"""
    out = []
    for name in sorted(f.keys()):
        obj = f[name]
        if isinstance(obj, h5py.Dataset):
            out.append((name, [(name, obj)]))
        elif isinstance(obj, h5py.Group):
            dsets = _datasets_in_group(obj)
            if dsets:
                out.append((name, dsets))
    return out


def get_h5_info(path: str) -> tuple[int, tuple[int, ...], int]:
    """Group 数、最初の shape、総パッチ数を返す。1 グループ内の全 Dataset を数える。"""
    with h5py.File(path, "r") as f:
        items = collect_groups(f)
        if not items:
            return (0, (), 0)
        n = 0
        for _gname, dsets in items:
            for _sk, dset in dsets:
                n += dset.shape[0] if dset.ndim == 4 else 1
        first_sh = items[0][1][0][1].shape if items[0][1] else ()
        return (len(items), first_sh, n)


def expand_one_h5(h5_path: str, out_base: str, sensor: str, dry_run: bool = False) -> str | None:
    """
    1 つの .h5 を展開。日付 = ファイル名（拡張子なし）。
    グループは常にサブフォルダ（Spectral Earth の patch_id 相当）。
    出力: {out_base}/{sensor}/{日付}/{グループキー}/0.npy（3D）または 00000.npy, 00001.npy, ...（4D）。
    """
    date_id = Path(h5_path).stem.replace(".hdf5", "").replace(".h5", "")
    out_root = Path(out_base) / sensor / date_id

    with h5py.File(h5_path, "r") as f:
        items = collect_groups(f)
        if not items:
            return f"No group/dataset in {h5_path}. Keys: {list(f.keys())}"
        if dry_run:
            n = sum(
                dset.shape[0] if dset.ndim == 4 else 1
                for _, dsets in items for _, dset in dsets
            )
            return f"Would write {len(items)} group(s), {n} patch(es) -> {out_root}"

    out_root.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        for gname, dsets in collect_groups(f):
            sub = out_root / gname
            sub.mkdir(exist_ok=True)
            idx = 0
            for _sk, dset in dsets:
                sh = dset.shape
                if len(sh) == 3:
                    np.save(sub / f"{idx}.npy", dset[:].astype(np.float32))
                    idx += 1
                elif len(sh) == 4:
                    for i in range(sh[0]):
                        np.save(sub / f"{idx:05d}.npy", dset[i].astype(np.float32))
                        idx += 1
                else:
                    return f"Unsupported shape {sh} in group '{gname}'"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Expand HDF5 (group-based) to date-dir + group_key.npy")
    p.add_argument("--data-dir", "--h5-dir", default="data", dest="data_dir", help=".h5 を探すディレクトリ")
    p.add_argument("--output-dir", default="data", help="出力ルート → {output_dir}/{sensor}/{日付}/")
    p.add_argument("--sensor", default="scene", help="出力サブディレクトリ名")
    p.add_argument("--no-recursive", action="store_true", help="data-dir 直下のみスキャン")
    p.add_argument("--all", action="store_true", help="全 .h5 を選択プロンプトなしで展開")
    p.add_argument("--dry-run", action="store_true", help="書き出さず一覧のみ")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    files = find_h5_files(args.data_dir, recursive=not args.no_recursive)
    if not files:
        print(f"No .h5/.hdf5 under {data_dir}")
        sys.exit(0)

    print(f"Found {len(files)} HDF5 file(s) under {data_dir}\n")
    for idx, (full_path, rel_path) in enumerate(files):
        try:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
        except OSError:
            size_mb = 0
        try:
            n_grp, shape, n_patch = get_h5_info(full_path)
            shape_str = str(shape) if shape else "?"
        except Exception as e:
            n_grp, shape_str, n_patch = 0, str(e), 0
        print(f"  [{idx}]  {rel_path}  ({size_mb:.1f} MB)  groups={n_grp}  patches≈{n_patch}  shape={shape_str}")

    if args.all:
        indices = list(range(len(files)))
    else:
        line = input("\nIndices to expand (e.g. 0 2), or 'all', or 'q': ").strip()
        if line.lower() == "q":
            sys.exit(0)
        indices = list(range(len(files))) if line.lower() == "all" else [int(x) for x in line.split()]

    for idx in indices:
        full_path, rel_path = files[idx]
        print(f"\nExpanding [{idx}] {rel_path} ...")
        err = expand_one_h5(full_path, args.output_dir, args.sensor, dry_run=args.dry_run)
        if err:
            if args.dry_run:
                print(f"  (dry-run) {err}")
            else:
                print(f"  Error: {err}", file=sys.stderr)
        else:
            date_id = Path(rel_path).stem.replace(".h5", "").replace(".hdf5", "")
            out_dir = Path(args.output_dir) / args.sensor / date_id
            n = len(list(out_dir.rglob("*.npy"))) if out_dir.exists() else 0
            print(f"  -> {out_dir}  ({n} .npy files)")
    print("\nDone.")


if __name__ == "__main__":
    main()
