"""
Scene 用 Lightning DataModule。

- H5 から画像を読み、batch["image"] (B, C, H, W) を返す。
- metadata: metadata_path で JSON を指定（未指定時は statistics_dir/metadata.json）。batch に scene_tags / object_tags を付与。
- 単一ラベル: single_label_classes 等を指定すると複数ラベルを単一に絞り、batch["label"] (B,) を返す（SingleLabelClassificationModule 用）。
"""
import json
import os
from pathlib import Path
from typing import List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from ..datasets.scene import H5Dataset


def _collate_image_batch(batch):
    """画像 Tensor のリストを (B, C, H, W) に stack し {"image": ...} で返す。"""
    return {"image": torch.stack(batch, dim=0)}


def _collate_with_metadata(batch):
    """{"image", "scene_tags", "object_tags"} の dict リストを 1 バッチにまとめる。"""
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "scene_tags": [b["scene_tags"] for b in batch],
        "object_tags": [b["object_tags"] for b in batch],
    }


def _collate_single_label(batch):
    """単一ラベル用。image を (B,C,H,W)、label を (B,) に stack して返す。"""
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
    }


def _resolve_data_path(root, data_path):
    """data_path を root 基準で解決し、.h5/.hdf5 のリスト、または単一ファイルパスを返す。ディレクトリ指定時はその下を再帰検索。"""
    if isinstance(data_path, (list, tuple)):
        return [os.path.join(root, p) for p in data_path]
    path = os.path.join(root, data_path)
    if os.path.isdir(path):
        files = list(Path(path).rglob("*.h5")) + list(Path(path).rglob("*.hdf5"))
        return sorted([str(p) for p in files])
    return path if os.path.isfile(path) else []


def _load_metadata_by_id(metadata_path):
    """metadata.json を読み、id → {scene_tags, object_tags} の辞書を返す。"""
    with open(metadata_path, encoding="utf-8") as f:
        rows = json.load(f)
    return {
        str(r["id"]): {
            "scene_tags": r.get("scene_tags", []),
            "object_tags": r.get("object_tags", []),
        }
        for r in rows
        if "id" in r
    }


class SceneDataModule(LightningDataModule):
    """
    H5 画像と、任意で metadata（scene/object tags）を返す DataModule。

    - metadata_path: JSON のパス（project_root 基準）。未指定時は statistics_dir/metadata.json。
    - 単一ラベル時: single_label_classes / label_source / multi_to_single / max_samples / seed でデータを絞る。
      val_ratio, test_ratio で train/val/test を分割（0 ならその分割なし）。残りが train。val_ratio + test_ratio < 1 にすること。
    """

    def __init__(
        self,
        data_path,
        batch_size=64,
        num_workers=0,
        project_root=None,
        shuffle=False,
        statistics_dir=None,
        metadata_path: str | None = None,
        *,
        single_label_classes: List[str] | None = None,
        label_source: str = "scene",
        multi_to_single: str = "first",
        max_samples: int | None = None,
        seed: int | None = None,
        val_ratio: float = 0.0,
        test_ratio: float = 0.0,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.project_root = project_root or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.shuffle = shuffle
        self.statistics_dir = statistics_dir
        self.metadata_path = metadata_path
        self.single_label_classes = single_label_classes
        self.label_source = label_source
        self.multi_to_single = multi_to_single
        self.max_samples = max_samples
        self.seed = seed
        self.val_ratio = max(0.0, min(1.0, val_ratio)) if val_ratio else 0.0
        self.test_ratio = max(0.0, min(1.0, test_ratio)) if test_ratio else 0.0

    @property
    def num_classes(self) -> int:
        """単一ラベルモード時のみ。クラス数（モデルの num_classes に渡す）。"""
        if hasattr(self, "_dataset") and getattr(self._dataset, "classes", None):
            return len(self._dataset.classes)
        return 0

    @property
    def classes(self) -> List[str]:
        """単一ラベルモード時のクラス名のリスト（インデックス順）。"""
        if hasattr(self, "_dataset") and getattr(self._dataset, "classes", None):
            return self._dataset.classes
        return []

    def setup(self, stage=None):
        path = _resolve_data_path(self.project_root, self.data_path)
        if isinstance(path, list) and len(path) == 0:
            raise FileNotFoundError(f"No .h5 under data_path: {self.data_path}")
        metadata_lookup = {}
        resolved_metadata = None
        if self.metadata_path:
            resolved_metadata = (
                self.metadata_path
                if os.path.isabs(self.metadata_path)
                else os.path.join(self.project_root, self.metadata_path)
            )
        elif self.statistics_dir:
            resolved_metadata = os.path.join(
                self.project_root, self.statistics_dir, "metadata.json"
            )
        if resolved_metadata and os.path.isfile(resolved_metadata):
            metadata_lookup = _load_metadata_by_id(resolved_metadata)

        single_label_config = None
        if self.single_label_classes and metadata_lookup:
            single_label_config = {
                "classes": self.single_label_classes,
                "label_source": self.label_source,
                "multi_to_single": self.multi_to_single,
                "max_samples": self.max_samples,
                "seed": self.seed,
            }
        self._dataset = H5Dataset(
            path,
            as_tensor=True,
            metadata_lookup=metadata_lookup,
            single_label_config=single_label_config,
        )
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        if (
            getattr(self._dataset, "_single_label_mode", False)
            and len(self._dataset) > 0
            and (self.val_ratio > 0 or self.test_ratio > 0)
        ):
            n = len(self._dataset)
            n_val = max(0, int(n * self.val_ratio)) if self.val_ratio > 0 else 0
            n_test = max(0, int(n * self.test_ratio)) if self.test_ratio > 0 else 0
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                n_val = min(n_val, n - 1 - n_test)
                n_test = n - n_train - n_val
            gen = torch.Generator().manual_seed(int(self.seed or 42))
            if n_val > 0 and n_test > 0:
                self._train_dataset, self._val_dataset, self._test_dataset = random_split(
                    self._dataset, [n_train, n_val, n_test], generator=gen
                )
            elif n_val > 0:
                self._train_dataset, self._val_dataset = random_split(
                    self._dataset, [n_train, n_val], generator=gen
                )
            else:
                self._train_dataset, self._test_dataset = random_split(
                    self._dataset, [n_train, n_test], generator=gen
                )

    def train_dataloader(self):
        dataset = self._train_dataset if self._train_dataset is not None else self._dataset
        if getattr(self._dataset, "_single_label_mode", False):
            collate_fn = _collate_single_label
        elif self._dataset.metadata_lookup:
            collate_fn = _collate_with_metadata
        else:
            collate_fn = _collate_image_batch
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        if self._val_dataset is None:
            return []
        if getattr(self._dataset, "_single_label_mode", False):
            collate_fn = _collate_single_label
        else:
            collate_fn = _collate_with_metadata if self._dataset.metadata_lookup else _collate_image_batch
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        if self._test_dataset is None:
            return []
        if getattr(self._dataset, "_single_label_mode", False):
            collate_fn = _collate_single_label
        else:
            collate_fn = _collate_with_metadata if self._dataset.metadata_lookup else _collate_image_batch
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )
