import os
import h5py
import numpy as np
import torch


def get_h5_keys(path_file):
  """H5 内の全 Dataset キーを visititems で列挙する。"""
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
  """全画像を (key, index) のリストで列挙。3D は 1 件 (key, None)、4D は N 件 (key, 0..N-1)。"""
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
  path_file の H5 から key の Dataset を読み、numpy または Tensor で返す。key 不在・読み込み失敗時は []。
  3D (H,W,C) → (C,H,W)。4D: index 指定時 (C,H,W)、未指定時 (N,C,H,W)。as_tensor=True で Tensor。
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