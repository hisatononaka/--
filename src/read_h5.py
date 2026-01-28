"""
H5 ファイルの読み込み。遅延読み込み用の get_h5_keys / read_h5_item と、
後方互換の read_h5 / iter_batches を提供する。
"""
import h5py
import numpy as np
import torch


def get_h5_keys(path_file):
  """H5 ファイル内の Dataset キーを順序付きで返す。"""
  keys = []

  def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
      keys.append(name)

  with h5py.File(path_file, "r") as f:
    f.visititems(walk)
  return keys


def read_h5_item(path_file, key, as_tensor=False):
  """1 枚の画像だけ読み込む。戻り値 (C, H, W)。遅延読み込み用。"""
  with h5py.File(path_file, "r") as f:
    arr = np.transpose(f[key][:], (2, 0, 1))
  if as_tensor:
    return torch.from_numpy(arr.astype(np.float32))
  return arr


def read_h5(path_file, as_tensor=False):
  """
  ファイル全体をメモリに読み込んで dict で返す（後方互換用）。
  大量データでは H5Dataset の遅延読み込みか iter_batches を使うこと。
  """
  data = {}
  with h5py.File(path_file, "r") as f:

    def walk(name, obj):
      if isinstance(obj, h5py.Dataset):
        arr = np.transpose(obj[:], (2, 0, 1))
        if as_tensor:
          data[name] = torch.from_numpy(arr.astype(np.float32))
        else:
          data[name] = arr

    f.visititems(walk)
  return data


def iter_batches(path_file, batch_size, as_tensor=True):
  """
  H5 を遅延読み込みし、batch_size 枚ずつ yield する。
  最後のバッチだけ len < batch_size になりうる。
  """
  keys = get_h5_keys(path_file)
  for i in range(0, len(keys), batch_size):
    batch_keys = keys[i : i + batch_size]
    yield [read_h5_item(path_file, k, as_tensor=as_tensor) for k in batch_keys]
