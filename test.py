#!/usr/bin/env python3
"""
train_singlelabel.py で DataLoader が空になる原因を特定するスクリプト。
config を読み、data_path / metadata / ID 対応 / 単一ラベル絞り込み を順にチェックする。

原因の例:
- label_source が "scene" のとき、metadata のキーは "scene_tags" なので、
  scene -> scene_tags / object -> object_tags のマッピングを src/datasets/scene.py で行う必要がある。
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.datamodules.scene import (
    _resolve_data_path,
    _load_metadata_by_id,
    SceneDataModule,
)
from src.datasets.scene import id_from_h5_key, _metadata_tags_key
from src.utils.read_h5 import get_h5_image_items


def main():
    # 1. config 読み込み（train_singlelabel と同じ）
    config = {}
    for name in (
        "configs/data/scene.yaml",
        "configs/model/singlelabel_classification.yaml",
        "configs/train.yaml",
    ):
        path = os.path.join(ROOT, name)
        if os.path.isfile(path):
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            config.update(data)

    print("=== 1. Config ===\n")
    data_path = config.get("data_path")
    metadata_path = config.get("metadata_path")
    statistics_dir = config.get("statistics_dir")
    single_label_classes = config.get("single_label_classes") or []
    label_source = config.get("label_source", "scene")
    print(f"  data_path: {data_path}")
    print(f"  metadata_path: {metadata_path}")
    print(f"  statistics_dir: {statistics_dir}")
    print(f"  single_label_classes: {single_label_classes}")
    print(f"  label_source: {label_source}")

    # 2. data_path 解決
    print("\n=== 2. Data path 解決 ===\n")
    path = _resolve_data_path(ROOT, data_path)
    if isinstance(path, list):
        print(f"  .h5 ファイル数: {len(path)}")
        for i, p in enumerate(path[:5]):
            print(f"    [{i}] {p}")
        if len(path) > 5:
            print(f"    ... 他 {len(path) - 5} 件")
        if len(path) == 0:
            print("  → 原因: data_path に .h5 が存在しません。")
            return
    else:
        print(f"  単一ファイル: {path}")
        print(f"  存在: {os.path.isfile(path)}")
        if not os.path.isfile(path):
            print("  → 原因: 指定した .h5 が存在しません。")
            return

    # 3. metadata 読み込み
    print("\n=== 3. Metadata ===\n")
    resolved_metadata = None
    if metadata_path:
        resolved_metadata = (
            metadata_path if os.path.isabs(metadata_path)
            else os.path.join(ROOT, metadata_path)
        )
    elif statistics_dir:
        resolved_metadata = os.path.join(ROOT, statistics_dir, "metadata.json")
    if not resolved_metadata or not os.path.isfile(resolved_metadata):
        print(f"  metadata ファイル: {resolved_metadata}")
        print("  → 原因: metadata ファイルが存在しません。single_label では metadata 必須です。")
        return
    print(f"  パス: {resolved_metadata}")
    metadata_lookup = _load_metadata_by_id(resolved_metadata)
    print(f"  id 数: {len(metadata_lookup)}")
    if len(metadata_lookup) == 0:
        print("  → 原因: metadata に 'id' を持つレコードがありません。")
        with open(resolved_metadata) as f:
            raw = json.load(f)
        if isinstance(raw, list) and len(raw) > 0:
            print(f"  先頭レコードのキー: {list(raw[0].keys())}")
        return
    sample_ids = list(metadata_lookup.keys())[:5]
    print(f"  サンプル id (先頭5件): {sample_ids}")
    tags_key = _metadata_tags_key(label_source)
    print(f"  label_source={label_source!r} -> metadata のキー: {tags_key!r}")
    for sid in sample_ids[:2]:
        meta = metadata_lookup[sid]
        print(f"    id={sid} -> {tags_key}={meta.get(tags_key, [])[:5]}...")

    # 4. H5 の key と metadata id の対応
    print("\n=== 4. H5 キー → metadata id 対応 ===\n")
    files = path if isinstance(path, list) else [path]
    raw_items = []
    for p in files:
        items = get_h5_image_items(p)
        for k, idx in items:
            raw_items.append((p, k, idx))
    print(f"  raw_items 総数: {len(raw_items)}")
    if len(raw_items) == 0:
        print("  → 原因: H5 内に 3D/4D の画像 Dataset がありません。")
        return

    # 先頭いくつかで id を計算し、metadata に存在するか確認
    in_meta = 0
    sample_key_ids = []
    for path_i, key, index in raw_items[:100]:
        sid = id_from_h5_key(path_i, key)
        if sid in metadata_lookup:
            in_meta += 1
            if len(sample_key_ids) < 5:
                sample_key_ids.append((key, sid, metadata_lookup[sid].get(label_source, [])))
    print(f"  先頭100件のうち metadata に存在する id: {in_meta} 件")
    if in_meta == 0:
        print("  → 原因: H5 の key/path から導出した id が metadata の id と一致していません。")
        print("  H5 側の id 例 (先頭3件):")
        for path_i, key, index in raw_items[:3]:
            print(f"    path={os.path.basename(path_i)}, key={key} -> id={id_from_h5_key(path_i, key)}")
        print("  metadata 側の id 例:", sample_ids)
        return
    print("  対応例 (key -> id -> tags):")
    for key, sid, tags in sample_key_ids:
        print(f"    key={key!r} -> id={sid} -> {tags_key}={tags[:5]}")

    # 5. 単一ラベル絞り込み
    print("\n=== 5. 単一ラベル絞り込み ===\n")
    if not single_label_classes:
        print("  single_label_classes が未設定のためスキップ。")
        return
    class_set = {c: i for i, c in enumerate(single_label_classes)}
    matched = 0
    by_class = {i: 0 for i in range(len(single_label_classes))}
    tags_key = _metadata_tags_key(label_source)
    for path_i, key, index in raw_items:
        sid = id_from_h5_key(path_i, key)
        meta = metadata_lookup.get(sid, {})
        tags = meta.get(tags_key, [])
        for t in tags:
            if t in class_set:
                matched += 1
                by_class[class_set[t]] += 1
                break
    print(f"  single_label_classes に一致したサンプル数: {matched}")
    for i, c in enumerate(single_label_classes):
        print(f"    '{c}': {by_class[i]} 件")
    if matched == 0:
        print("  → 原因: metadata のタグに single_label_classes のいずれかが含まれるサンプルがありません。")
        print(f"  実際に使われている {tags_key} の例（raw_items から収集）:")
        seen = set()
        for path_i, key, index in raw_items[:200]:
            sid = id_from_h5_key(path_i, key)
            meta = metadata_lookup.get(sid, {})
            for t in meta.get(tags_key, []):
                seen.add(t)
        print("   ", list(seen)[:30] if seen else " (なし。label_source と metadata のキーが一致しているか確認してください。)")
        return

    # 6. DataModule で実際の dataset 長を確認
    print("\n=== 6. SceneDataModule 実測 ===\n")
    dm_kwargs = {
        k: config[k] for k in (
            "data_path", "batch_size", "num_workers", "shuffle", "statistics_dir", "metadata_path",
            "single_label_classes", "label_source", "multi_to_single", "max_samples",
            "seed", "val_ratio", "test_ratio",
        ) if k in config
    }
    dm = SceneDataModule(project_root=ROOT, **dm_kwargs)
    dm.setup("fit")
    train_ds = dm._train_dataset if dm._train_dataset is not None else dm._dataset
    train_loader = dm.train_dataloader()
    print(f"  num_classes: {dm.num_classes}")
    print(f"  train 用 dataset の長さ: {len(train_ds)}")
    print(f"  train_dataloader の長さ: {len(train_loader)}")
    if len(train_loader) == 0:
        print("  → DataLoader が 0 のため、train でバッチが回りません。上記のいずれかで 0 になっている箇所を確認してください。")
    else:
        batch = next(iter(train_loader))
        print(f"  最初のバッチ: image {batch['image'].shape}, label {batch['label'].shape}")
    print("\n--- 以上で原因特定を終了 ---")


if __name__ == "__main__":
    main()
