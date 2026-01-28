"""
H5 から 1 枚ずつ画像を返す Dataset。遅延読み込みでメモリを抑える。
"""
import torch
from torch.utils.data import Dataset

import read_h5


class H5Dataset(Dataset):
  """
  1 つの H5 ファイル（または複数）内の全画像を 1 枚ずつ返す。
  __getitem__ でその都度 H5 を開いて読み込むため、メモリに全画像を載せない。
  __getitem__(idx) -> tensor (C, H, W)。as_tensor=True で Tensor。
  """
  def __init__(self, path_file, as_tensor=True):
    self.as_tensor = as_tensor
    if isinstance(path_file, (list, tuple)):
      self._items = [
        (p, key) for p in path_file for key in read_h5.get_h5_keys(p)
      ]
    else:
      self._items = [
        (path_file, key) for key in read_h5.get_h5_keys(path_file)
      ]

  def __len__(self):
    return len(self._items)

  def __getitem__(self, idx):
    path, key = self._items[idx]
    return read_h5.read_h5_item(path, key, as_tensor=self.as_tensor)
