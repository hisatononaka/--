#!/usr/bin/env python3
"""
read_h5 の返り値を確認するスクリプト。
使い方: python scripts/check_h5_returns.py [path_to.h5]
        path 省略時は data/raw/scene 以下で最初の .h5 を使用。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.read_h5 import get_h5_keys, get_h5_image_items, read_h5_item


def find_one_h5(data_path: str) -> str | None:
    """data_path がディレクトリならその下の最初の .h5 を返す。ファイルならそのまま返す。"""
    path = ROOT / data_path
    if path.is_file():
        return str(path)
    if path.is_dir():
        for p in sorted(path.rglob("*.h5")) + sorted(path.rglob("*.hdf5")):
            return str(p)
    return None


def main() -> None:
    if len(sys.argv) >= 2:
        h5_path = sys.argv[1]
        if not os.path.isfile(h5_path):
            print(f"Error: not a file: {h5_path}", file=sys.stderr)
            sys.exit(1)
    else:
        h5_path = find_one_h5("data/raw/scene")
        if not h5_path:
            print("Error: no .h5 under data/raw/scene. Pass path: python scripts/check_h5_returns.py <path.h5>", file=sys.stderr)
            sys.exit(1)
        print(f"Using first .h5 under data/raw/scene: {h5_path}\n")

    print("=" * 60)
    print("1. get_h5_keys(path)")
    print("=" * 60)
    keys = get_h5_keys(h5_path)
    print(f"  type: {type(keys).__name__}, len: {len(keys)}")
    for i, k in enumerate(keys[:10]):
        print(f"  [{i}] {k!r}")
    if len(keys) > 10:
        print(f"  ... and {len(keys) - 10} more")
    print()

    print("=" * 60)
    print("2. get_h5_image_items(path)")
    print("=" * 60)
    items = get_h5_image_items(h5_path)
    print(f"  type: {type(items).__name__}, len: {len(items)}")
    for i, (key, idx) in enumerate(items[:8]):
        print(f"  [{i}] key={key!r}, index={idx}")
    if len(items) > 8:
        print(f"  ...")
        for i in range(max(0, len(items) - 2), len(items)):
            key, idx = items[i]
            print(f"  [{i}] key={key!r}, index={idx}")
    print()

    if len(items) == 0:
        print("No items -> skip read_h5_item check.")
        return

    print("=" * 60)
    print("3. read_h5_item(path, key, as_tensor=False, index=index)")
    print("=" * 60)
    path, key, index = h5_path, items[0][0], items[0][1]
    out = read_h5_item(path, key, as_tensor=False, index=index)
    print(f"  first item: key={key!r}, index={index}")
    print(f"  return type: {type(out).__name__}")
    if isinstance(out, list):
        print(f"  return value (list): len={len(out)}")
    else:
        print(f"  return shape: {getattr(out, 'shape', 'N/A')}")
        print(f"  return dtype: {getattr(out, 'dtype', 'N/A')}")
    print()

    print("=" * 60)
    print("4. read_h5_item(..., as_tensor=True)")
    print("=" * 60)
    out_t = read_h5_item(path, key, as_tensor=True, index=index)
    print(f"  return type: {type(out_t).__name__}")
    if hasattr(out_t, "shape"):
        print(f"  return shape: {out_t.shape}, dtype: {out_t.dtype}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
