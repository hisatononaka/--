import os
import h5py
import numpy as np
import torch


def get_h5_keys(path_file):
  """H5 内の Dataset キーを列挙（visititems）。"""
  keys = []
  def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
      keys.append(name)
  try:
    with h5py.File(path_file, "r") as f:
      f.visititems(walk)
  except (OSError, FileNotFoundError, Exception):
    pass
  return keys

def get_h5_image_items(path_file):
  """全画像を (key, index) で列挙。3D=1件、4D=N件。"""
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
  try:
    with h5py.File(path_file, "r") as f:
      f.visititems(walk)
  except (OSError, FileNotFoundError, Exception):
    pass
  return items


def read_h5_item(path_file, key, as_tensor=False, index=None):
  """
  path_file の H5 を開き、key が一致する Dataset を読み込んで返す
  key が見つからない場合、またはファイルが読み込めない場合は [] を返す
  戻り値: numpy (as_tensor=False) または Tensor (as_tensor=True)
  3D (H,W,C) → (C, H, W)。→ index 指定時は (C,H,W)、未指定時は (N,C,H,W)
  """
  try:
    with h5py.File(path_file, "r") as f:
      try:
        dset = f[key]
        arr = np.asarray(dset[:])
      except KeyError:
        return []
  except (OSError, FileNotFoundError, Exception):
    return []
  if arr.ndim == 3:
    arr = np.transpose(arr, (2, 0, 1))
  elif arr.ndim == 4:
    if index is not None:
      arr = np.transpose(arr[index], (2, 0, 1))
    else:
      arr = np.transpose(arr, (0, 3, 1, 2))
  if as_tensor:
    return torch.from_numpy(arr.astype(np.float32))
  return arr