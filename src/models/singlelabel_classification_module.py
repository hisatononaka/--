"""
単一ラベル分類用 Lightning モジュール。
事前学習チェックポイント（DINO 等）からバックボーンを読み込み、線形ヘッドで分類。
linear_eval=True 時はバックボーンを freeze して線形評価のみ。statistics_dir で正規化用 mu/sigma を指定可能。
"""
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import timm
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from ..transforms.normalize import NormalizeMeanStd
from ..backbones.spectral_adapter import SpectralAdapter
from ..backbones.registry import BACKBONE_REGISTRY

ADAPTER_OUT_CHANNELS = 128


def _load_backbone_state(
    module: torch.nn.Module,
    state_dict: dict,
    prefix_src: str,
) -> int:
    """state_dict のうち prefix_src で始まるキーを module にロードする。戻り値はロードしたキー数。"""
    to_load = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix_src):
            continue
        rest = k[len(prefix_src) :].lstrip(".")
        to_load[rest] = v
    if not to_load:
        return 0
    missing, unexpected = module.load_state_dict(to_load, strict=False)
    return len(to_load) - len(missing)


class SingleLabelClassificationModule(LightningModule):
    """単一ラベル分類。入力 (B, C, H, W)、ラベル (B,) のクラスインデックス。checkpoint_path で事前学習重みを読み込める。"""

    def __init__(
        self,
        backbone_name: str = "vit_small",
        in_channels: int = 151,
        num_classes: int = 10,
        token_patch_size: int = 4,
        use_adapter: bool = False,
        img_size: int = 128,
        *,
        checkpoint_path: str | None = None,
        linear_eval: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        statistics_dir: str | None = None,
        ) -> None:
        """
        backbone_name: レジストリまたは timm のバックボーン名。
        checkpoint_path: 事前学習チェックポイント。state_dict の student_backbone / spectral_adapter をコピー。
        linear_eval: True ならバックボーンと adapter を freeze し線形層のみ学習。
        statistics_dir: 正規化用 mu.npy / sigma.npy のディレクトリ（絶対パス）。None なら mean=0, std=1。
        """
        super().__init__()
        self.save_hyperparameters(ignore=["checkpoint_path"])
        self.num_classes = num_classes
        self.img_size = int(img_size)
        self.linear_eval = linear_eval
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.use_adapter = use_adapter
        backbone_in_chans = ADAPTER_OUT_CHANNELS if use_adapter else in_channels

        if use_adapter:
            mean = torch.zeros(backbone_in_chans)
            std = torch.ones(backbone_in_chans)
        else:
            if statistics_dir:
                mu_path = os.path.join(statistics_dir, "mu.npy")
                sigma_path = os.path.join(statistics_dir, "sigma.npy")
            else:
                mu_path = sigma_path = None
            try:
                if mu_path and sigma_path:
                    mean = torch.tensor(np.load(mu_path))
                    std = torch.tensor(np.load(sigma_path))
                else:
                    mean = torch.zeros(in_channels)
                    std = torch.ones(in_channels)
            except FileNotFoundError:
                mean = torch.zeros(in_channels)
                std = torch.ones(in_channels)
        self.normalize = NormalizeMeanStd(mean=mean, std=std)

        if use_adapter:
            self.spectral_adapter = SpectralAdapter()
        else:
            self.spectral_adapter = None

        if backbone_name in BACKBONE_REGISTRY:
            if "vit" in backbone_name:
                backbone = BACKBONE_REGISTRY[backbone_name](
                    num_classes=0,
                    in_chans=backbone_in_chans,
                    token_patch_size=token_patch_size,
                    patch_size=self.img_size,
                )
            else:
                backbone = BACKBONE_REGISTRY[backbone_name](
                    num_classes=0,
                    in_chans=backbone_in_chans,
                )
        else:
            backbone = timm.create_model(
                backbone_name,
                in_chans=backbone_in_chans,
                num_classes=0,
                pretrained=False,
                img_size=self.img_size,
            )
        self.backbone = backbone
        feat_dim = getattr(backbone, "num_features", backbone.num_classes if hasattr(backbone, "num_classes") else 384)
        self.head = torch.nn.Linear(feat_dim, num_classes)

        if checkpoint_path:
            self._load_pretrained(checkpoint_path)
        if linear_eval:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if self.spectral_adapter is not None:
                for p in self.spectral_adapter.parameters():
                    p.requires_grad = False

    def _load_pretrained(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        prefix = "model." if any(k.startswith("model.") for k in state.keys()) else ""
        n1 = _load_backbone_state(self.backbone, state, f"{prefix}student_backbone.")
        if n1 > 0:
            self.print(f"Loaded {n1} backbone weights from {checkpoint_path}")
        if self.spectral_adapter is not None:
            n2 = _load_backbone_state(self.spectral_adapter, state, f"{prefix}spectral_adapter.")
            if n2 > 0:
                self.print(f"Loaded {n2} spectral_adapter weights from {checkpoint_path}")

    def forward(self, x: Tensor) -> Tensor:
        """ロジット (B, num_classes) を返す。入力 (B,C,H,W) を img_size×img_size にリサイズしてから backbone に渡す。"""
        x = F.interpolate(
            x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        if self.spectral_adapter is not None:
            x = self.spectral_adapter(x)
        x = self.normalize(x)
        feat = self.backbone(x)
        if hasattr(feat, "flatten"):
            feat = feat.flatten(start_dim=1)
        return self.head(feat)

    def _shared_step(self, batch: dict, prefix: str) -> Tensor:
        x = batch["image"].float()
        y = batch["label"].long()
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log(f"{prefix}_loss", loss, on_step=(prefix == "train"), on_epoch=True)
        self.log(f"{prefix}_acc", acc, on_step=(prefix == "train"), on_epoch=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.trainer.max_epochs is None or self.trainer.max_epochs <= 0:
            return [optimizer], []
        warmup = self.warmup_epochs
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1.0 / max(1, warmup), total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=max(1, self.trainer.max_epochs - warmup)),
            ],
            milestones=[warmup],
        )
        return [optimizer], [scheduler]
