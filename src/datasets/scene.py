"""
H5 から 1 枚ずつ画像を返す Dataset。Lightning 用に batch["image"] (B,C,H,W) で使う。
"""
import torch
from torch.utils.data import Dataset

from ..utils.read_h5 import get_h5_image_items, read_h5_item


class H5Dataset(Dataset):
  """
  H5 ファイル（または複数）内の全画像を 1 枚ずつ返す。3D=1枚、4D=N枚。
  __getitem__(idx) -> (C, H, W) Tensor。
  """
  def __init__(self, path_file, as_tensor=True):
    self.as_tensor = as_tensor
    if isinstance(path_file, (list, tuple)):
      self._items = [
        (p, k, idx) for p in path_file for k, idx in get_h5_image_items(p)
      ]
    else:
      self._items = [(path_file, k, idx) for k, idx in get_h5_image_items(path_file)]

  def __len__(self):
    return len(self._items)

  def __getitem__(self, idx):
    path, key, index = self._items[idx]
    out = read_h5_item(path, key, as_tensor=self.as_tensor, index=index)
    if isinstance(out, list) and len(out) == 0:
      raise RuntimeError(f"read_h5_item returned [] for key={key} in {path}")
    return out
