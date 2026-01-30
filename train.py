#!/usr/bin/env python3
"""
DINO 事前学習のエントリポイント。
実行: プロジェクトルートで python train.py
config のパラメータをそのまま DataModule / DINOModule に渡す。
"""
import os
import sys
import yaml

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.models.dino_module import DINOModule
from src.datamodules.scene import SceneDataModule

DM_KEYS = ("data_path", "batch_size", "num_workers", "shuffle")
MODEL_KEYS = (
    "backbone_name", "in_channels", "hidden_dim", "bottleneck_dim", "output_dim",
    "lr", "warmup_epochs", "weight_decay", "momentum", "warmup_teacher_temp_epochs",
    "size", "multicrop", "n_views", "token_patch_size", "use_adapter",
)


def load_config():
    config = {}
    for name in ("configs/data/scene.yaml", "configs/model/dino.yaml", "configs/train.yaml"):
        path = os.path.join(ROOT, name)
        if os.path.isfile(path):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            config.update(data)
    return config


def main():
    cfg = load_config()

    dm_kwargs = {k: cfg[k] for k in DM_KEYS if k in cfg}
    dm = SceneDataModule(project_root=ROOT, **dm_kwargs)

    model_kwargs = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
    model = DINOModule(**model_kwargs)

    max_epochs = cfg.get("max_epochs", 100) or 100
    log_dir = os.path.join(ROOT, cfg.get("log_dir", "logs"))
    logger_name = cfg.get("logger_name", "dino")
    accelerator = cfg.get("accelerator", "auto")
    logger = CSVLogger(save_dir=log_dir, name=logger_name)
    # 学習した重みを保存（train_loss 最小のベスト + 最終エポック）
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{train_loss:.4f}",
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
