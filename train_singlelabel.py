#!/usr/bin/env python3
"""
単一ラベル分類のエントリポイント。プロジェクトルートで python train_singlelabel.py。
data / model / train の config を SceneDataModule / SingleLabelClassificationModule に渡す。
single_label_classes は data config で必須。val_ratio / test_ratio で train/val/test に分割し、学習後に test で最終精度を表示。
学習ごとに run_log.yaml で使用 ID・checkpoint・設定を記録する。
"""
import os
import sys
from datetime import datetime, timezone

import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.models.singlelabel_classification_module import SingleLabelClassificationModule
from src.datamodules.scene import SceneDataModule

DM_KEYS = (
    "data_path", "batch_size", "num_workers", "shuffle", "statistics_dir", "metadata_path",
    "single_label_classes", "label_source", "multi_to_single", "max_samples",
    "seed", "val_ratio", "test_ratio",
)
MODEL_KEYS = (
    "backbone_name", "token_patch_size", "use_adapter",
    "linear_eval", "lr", "weight_decay", "warmup_epochs",
    "checkpoint_path",
)


def load_config():
    config = {}
    for name in (
        "configs/data/scene.yaml",
        "configs/model/singlelabel_classification.yaml",
        "configs/train.yaml",
    ):
        path = os.path.join(ROOT, name)
        if os.path.isfile(path):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            config.update(data)
    return config


def main():
    cfg = load_config()

    if not cfg.get("single_label_classes"):
        raise ValueError(
            "単一ラベル分類には configs/data/scene.yaml で single_label_classes を指定してください。"
            "例: single_label_classes: [\"class_a\", \"class_b\", ...]"
        )

    dm_kwargs = {k: cfg[k] for k in DM_KEYS if k in cfg}
    dm = SceneDataModule(project_root=ROOT, **dm_kwargs)
    dm.setup("fit")

    # 学習ログ用: 使用した ID と設定を記録
    run_log = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "num_classes": dm.num_classes,
        "classes": getattr(dm, "classes", []),
        "split_sample_ids": dm.get_split_sample_ids(),
    }

    model_kwargs = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
    model_kwargs["num_classes"] = dm.num_classes
    if "num_bands" in cfg:
        model_kwargs["in_channels"] = cfg["num_bands"]
    model_kwargs["img_size"] = cfg.get("patch_size", 128)
    if cfg.get("statistics_dir"):
        model_kwargs["statistics_dir"] = os.path.join(ROOT, cfg["statistics_dir"])
    if "checkpoint_path" in model_kwargs and model_kwargs["checkpoint_path"]:
        model_kwargs["checkpoint_path"] = os.path.join(
            ROOT, model_kwargs["checkpoint_path"]
        )
    elif "checkpoint_path" in model_kwargs:
        del model_kwargs["checkpoint_path"]

    model = SingleLabelClassificationModule(**model_kwargs)

    max_epochs = cfg.get("max_epochs", 100) or 100
    log_dir = os.path.join(ROOT, cfg.get("log_dir", "logs"))
    logger_name = cfg.get("logger_name", "singlelabel")
    accelerator = cfg.get("accelerator", "auto")
    logger = CSVLogger(save_dir=log_dir, name=logger_name)
    val_ratio = cfg.get("val_ratio") or 0.0
    test_ratio = cfg.get("test_ratio") or 0.0
    monitor = "val_loss" if val_ratio > 0 else "train_loss"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{" + monitor + ":.4f}",
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=dm)

    # checkpoint パスをログに追加
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            run_log["checkpoint_best"] = getattr(cb, "best_model_path", None) or ""
            run_log["checkpoint_last"] = getattr(cb, "last_model_path", None) or ""
            break

    if val_ratio > 0:
        trainer.validate(model, datamodule=dm)
    test_results = None
    if test_ratio > 0:
        test_results = trainer.test(model, datamodule=dm)
        if test_results:
            r = test_results[0]
            loss = r.get("test_loss")
            acc = r.get("test_acc")
            loss = float(loss) if hasattr(loss, "item") else loss
            acc = float(acc) if hasattr(acc, "item") else acc
            run_log["test_loss"] = loss
            run_log["test_acc"] = acc
            print("--- Test (最終精度) ---")
            print(f"  test_loss: {loss:.4f}" if isinstance(loss, (int, float)) else f"  test_loss: {loss}")
            print(f"  test_acc:  {acc:.4f}" if isinstance(acc, (int, float)) else f"  test_acc:  {acc}")

    # run_log を CSV ログと同じディレクトリに保存
    run_log_dir = getattr(logger, "log_dir", None)
    if not run_log_dir:
        run_log_dir = os.path.join(
            ROOT, cfg.get("log_dir", "logs"), logger_name,
            f"version_{getattr(logger, 'version', 0)}",
        )
    os.makedirs(run_log_dir, exist_ok=True)
    run_log_path = os.path.join(run_log_dir, "run_log.yaml")
    with open(run_log_path, "w", encoding="utf-8") as f:
        yaml.dump(run_log, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"Run log saved: {run_log_path}")


if __name__ == "__main__":
    main()
