"""
Lightning DataModule: H5 を読み、batch["image"] (B, C, H, W) を渡す。
"""
import os
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..datasets.scene import H5Dataset


def _resolve_data_path(root, data_path):
    if isinstance(data_path, (list, tuple)):
        return [os.path.join(root, p) for p in data_path]
    path = os.path.join(root, data_path)
    if os.path.isdir(path):
        files = list(Path(path).rglob("*.h5")) + list(Path(path).rglob("*.hdf5"))
        return sorted([str(p) for p in files])
    return path if os.path.isfile(path) else []


class SceneDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        batch_size=64,
        num_workers=0,
        project_root=None,
        shuffle=False,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.project_root = project_root or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.shuffle = shuffle

    def setup(self, stage=None):
        path = _resolve_data_path(self.project_root, self.data_path)
        if isinstance(path, list) and len(path) == 0:
            raise FileNotFoundError(f"No .h5 under data_path: {self.data_path}")
        self._dataset = H5Dataset(path, as_tensor=True)

    def train_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=lambda batch: {"image": torch.stack(batch, dim=0)},
        )

    def val_dataloader(self):
        return []  # 検証なし（None だと Lightning が iter(None) でエラーになる）

    def test_dataloader(self):
        return []
