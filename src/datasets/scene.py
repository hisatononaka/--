"""
H5 形式の画像データセット。

- 通常: __getitem__ は画像 Tensor (C, H, W)。metadata あり時は scene_tags / object_tags も返す。
- single_label_config 指定時: 単一ラベル分類用にサンプルを絞り、{"image", "label"} を返す（label はクラスインデックス）。
- 3D は 1 枚、4D は N 枚として扱う。
"""
from __future__ import annotations

import os
import random
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from ..utils.read_h5 import get_h5_image_items, read_h5_item


def id_from_h5_path(path):
    """H5 のファイルパスから metadata 用 id (YYYYMMDD_HHMMSS) を導出する。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return stem


def id_from_h5_key(path, key):
    """
    (path, key) から metadata 用 id を導出する。
    key が "group/YYYYMMDDHHMMSS" 形式なら末尾を YYYYMMDD_HHMMSS に変換（12/14 桁対応）。
    それ以外は path の basename から導出。
    """
    if key and "/" in key:
        suffix = key.split("/")[-1]
        if suffix.isdigit():
            if len(suffix) in (12, 14):
                return f"{suffix[:8]}_{suffix[8:]}"
            return suffix
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return stem


def _metadata_tags_key(label_source: str) -> str:
    """label_source 'scene' / 'object' を metadata のキー 'scene_tags' / 'object_tags' に変換する。"""
    if label_source == "object":
        return "object_tags"
    return "scene_tags"


def _allocate_by_ratio(
    max_samples: int,
    ratio: list[float],
    num_classes: int,
) -> list[int]:
    """比率 ratio（長さ num_classes）に従い、合計 max_samples になるよう各クラスのサンプル数を分配する。"""
    total_ratio = sum(ratio)
    if total_ratio <= 0:
        return [max_samples // num_classes] * num_classes
    counts = [int(max_samples * r / total_ratio) for r in ratio]
    remainder = max_samples - sum(counts)
    for i in range(remainder):
        counts[i % num_classes] += 1
    return counts


def _build_single_label_items(
    items: list[tuple[str, str, int]],
    metadata_lookup: dict[str, dict[str, list[str]]],
    id_from_key: Callable[[str, str], str],
    classes: list[str],
    label_source: str,
    multi_to_single: str,
    max_samples: int | None,
    seed: int | None = None,
) -> list[tuple[str, str, int, int]]:
    """
    複数ラベルを単一ラベルに絞り、(path, key, index, label_index) のリストを構築する。
    - ラベルは classes の並び順で 0..num_classes-1。
    - multi_to_single == "first": タグのうち classes に含まれる最初のものを採用。どれも含まれないサンプルは除外。
    - max_samples is None: 全件を元の順で返す。
    - max_samples 指定: 合計をクラス件数比で分配し、各クラスからランダムに選択（seed で再現可能）。
    """
    class_set = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    tags_key = _metadata_tags_key(label_source)

    by_class: dict[int, list[tuple[str, str, int]]] = {i: [] for i in range(num_classes)}
    for path, key, index in items:
        sample_id = id_from_key(path, key)
        meta = metadata_lookup.get(sample_id, {})
        tags = meta.get(tags_key, [])
        if not tags:
            continue
        for t in tags:
            if t in class_set:
                label_index = class_set[t]
                by_class[label_index].append((path, key, index))
                break

    if max_samples is None:
        result: list[tuple[str, str, int, int]] = []
        for path, key, index in items:
            sample_id = id_from_key(path, key)
            meta = metadata_lookup.get(sample_id, {})
            tags = meta.get(tags_key, [])
            if not tags:
                continue
            for t in tags:
                if t in class_set:
                    result.append((path, key, index, class_set[t]))
                    break
        return result

    rng = random.Random(seed)
    ratio = [float(len(by_class[i])) for i in range(num_classes)]
    if sum(ratio) <= 0:
        ratio = [1.0] * num_classes
    counts = _allocate_by_ratio(max_samples, ratio, num_classes)
    result = []
    for label_index in range(num_classes):
        pool = by_class[label_index]
        n = min(counts[label_index], len(pool))
        chosen = rng.sample(pool, n)
        for (path, key, index) in chosen:
            result.append((path, key, index, label_index))
    rng.shuffle(result)
    return result


class H5Dataset(Dataset):
    """
    H5（単体または複数）内の画像を 1 枚ずつ返す Dataset。

    - metadata_lookup なし: __getitem__ は Tensor (C, H, W) のみ。
    - metadata_lookup あり: __getitem__ は {"image", "scene_tags", "object_tags"}。
    - single_label_config あり: 複数ラベルを単一に絞り {"image", "label"} を返す。
      classes / label_source / multi_to_single / max_samples / seed で挙動を指定。
    """

    def __init__(
        self,
        path_file,
        as_tensor=True,
        metadata_lookup=None,
        *,
        single_label_config: dict[str, Any] | None = None,
    ):
        self.as_tensor = as_tensor
        self.metadata_lookup = metadata_lookup or {}
        if isinstance(path_file, (list, tuple)):
            raw_items = [
                (p, k, idx) for p in path_file for k, idx in get_h5_image_items(p)
            ]
        else:
            raw_items = [(path_file, k, idx) for k, idx in get_h5_image_items(path_file)]

        self._single_label_config = single_label_config
        if single_label_config and self.metadata_lookup:
            classes = single_label_config["classes"]
            label_source = single_label_config.get("label_source", "scene")
            multi_to_single = single_label_config.get("multi_to_single", "first")
            max_samples = single_label_config.get("max_samples")
            seed = single_label_config.get("seed")
            self._items = _build_single_label_items(
                raw_items,
                self.metadata_lookup,
                id_from_h5_key,
                classes=classes,
                label_source=label_source,
                multi_to_single=multi_to_single,
                max_samples=max_samples,
                seed=seed,
            )
            self._single_label_mode = True
            self.classes = classes
        else:
            self._items = raw_items
            self._single_label_mode = False
            self.classes = []

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        if self._single_label_mode:
            path, key, index, label_index = self._items[idx]
        else:
            path, key, index = self._items[idx]
            label_index = None
        out = read_h5_item(path, key, as_tensor=self.as_tensor, index=index)
        if isinstance(out, list) and len(out) == 0:
            raise RuntimeError(f"read_h5_item returned [] for key={key} in {path}")
        if self._single_label_mode:
            return {"image": out, "label": torch.tensor(label_index, dtype=torch.long)}
        if self.metadata_lookup:
            sample_id = id_from_h5_key(path, key)
            meta = self.metadata_lookup.get(sample_id, {})
            scene_tags = meta.get("scene_tags", [])
            object_tags = meta.get("object_tags", [])
            return {"image": out, "scene_tags": scene_tags, "object_tags": object_tags}
        return out
