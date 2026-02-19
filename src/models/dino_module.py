import copy
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import timm
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
import kornia.augmentation as K

from ..transforms.normalize import NormalizeMeanStd
from ..backbones.spectral_adapter import SpectralAdapter
from ..backbones.registry import BACKBONE_REGISTRY

# SpectralAdapter 出力チャンネル数（バックボーン入力・正規化の次元に使用）
ADAPTER_OUT_CHANNELS = 128


class DINOModule(LightningModule):
    """
    DINO（self-DIstillation with NO labels）のハイパースペクトル向け実装。
    student-teacher で teacher は momentum で student を追従。標準／マルチクロップ両対応。
    """

    def __init__(
        self,
        backbone_name: str = "spec_resnet50",
        in_channels: int = 202,
        hidden_dim: float = 2048,
        bottleneck_dim: float = 256,
        output_dim: int = 32768,
        lr: float = 9.6,
        warmup_epochs: int = 20,
        weight_decay: float = 1e-6,
        momentum: float = 0.9,
        warmup_teacher_temp_epochs: int = 10,
        global_size: int = 128,
        local_size: int = 48,
        multicrop: bool = False,
        n_views: int = 0,
        token_patch_size: int = 4,
        use_adapter: bool = False,
        dynamic_img_size: bool = False,
        statistics_dir: str | None = None,
        ) -> None:
        """
        backbone_name: レジストリまたは timm のバックボーン名。
        in_channels: 入力スペクトルチャンネル数。
        use_adapter: True ならデータ拡張前に SpectralAdapter で C → 128ch に変換。
        statistics_dir: 正規化用 mu.npy / sigma.npy があるディレクトリ（絶対パス推奨）。None なら data/statistics または 0/1。
        """
        super().__init__()
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.global_size = global_size
        self.local_size = local_size
        self.multicrop = multicrop
        self.n_views = n_views
        self.in_channels = in_channels
        self.use_adapter = use_adapter
        backbone_in_chans = ADAPTER_OUT_CHANNELS if use_adapter else in_channels

        if use_adapter:
            self.spectral_adapter = SpectralAdapter()
            mean = torch.zeros(backbone_in_chans)
            std = torch.ones(backbone_in_chans)
        else:
            self.spectral_adapter = None
            if statistics_dir:
                mu_path = os.path.join(statistics_dir, "mu.npy")
                sigma_path = os.path.join(statistics_dir, "sigma.npy")
            else:
                mu_path = "data/statistics/mu.npy"
                sigma_path = "data/statistics/sigma.npy"
            try:
                mean = torch.tensor(np.load(mu_path))
                std = torch.tensor(np.load(sigma_path))
            except FileNotFoundError:
                mean = torch.zeros(in_channels)
                std = torch.ones(in_channels)

        global_ks = global_size // 10 // 2 * 2 + 1
        local_ks = local_size // 10 // 2 * 2 + 1
        if multicrop:
            max_scale_global = 1.0
            max_scale_local = 0.4
        else:
            max_scale_global = 1.0
            max_scale_local = 0.05

        global_pipeline = [
            K.RandomResizedCrop(size=(global_size, global_size), scale=(0.05, max_scale_global)),
            K.RandomGaussianBlur(kernel_size=(global_ks, global_ks), sigma=(0.1, 2), p=0.5),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            NormalizeMeanStd(mean=mean, std=std),
        ]
        local_pipeline = [
            K.RandomResizedCrop(size=(local_size, local_size), scale=(0.05, max_scale_local)),
            K.RandomGaussianBlur(kernel_size=(local_ks, local_ks), sigma=(0.1, 2), p=0.5),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            NormalizeMeanStd(mean=mean, std=std),
        ]
        self.augmentation1 = K.AugmentationSequential(*global_pipeline, data_keys=["input"])
        self.augmentation2 = K.AugmentationSequential(*local_pipeline, data_keys=["input"])

        if backbone_name in BACKBONE_REGISTRY:
            if "vit" in backbone_name:
                backbone = BACKBONE_REGISTRY[backbone_name](
                    num_classes=0,
                    in_chans=backbone_in_chans,
                    token_patch_size=token_patch_size,
                    patch_size=global_size,
                    dynamic_img_size=dynamic_img_size,
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
                img_size=global_size,
                dynamic_img_size=dynamic_img_size,
            )

        self.student_backbone = backbone
        self.teacher_backbone = copy.deepcopy(backbone)
        self.student_head = DINOProjectionHead(backbone.num_features, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1)
        self.teacher_head = DINOProjectionHead(backbone.num_features, hidden_dim, bottleneck_dim, output_dim)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=output_dim, warmup_teacher_temp_epochs=warmup_teacher_temp_epochs)
        self.avg_output_std = 0.0

    def forward(self, x: Tensor) -> Tensor:
        """Student の forward。"""
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x: Tensor) -> Tensor:
        """Teacher の forward。"""
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def _augment(self, x: Tensor, aug_module) -> Tensor:
        """Kornia でデータ拡張。MPS の場合は幾何変換を CPU で実行（linalg.solve 未実装のため）。"""
        device = x.device
        if device.type == "mps":
            x = x.cpu()
            out = aug_module(x)
            return out.to(device)
        return aug_module(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        max_epochs = max(1, self.trainer.max_epochs or 1)
        momentum = cosine_schedule(self.current_epoch, max_epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        if "image1" in batch and "image2" in batch:
            x1 = batch["image1"].float()
            x2 = batch["image2"].float()
            assert x1.size(1) == self.in_channels
        else:
            x = batch["image"].float()
            assert x.size(1) == self.in_channels
            x1 = x2 = x

        if self.use_adapter:
            x1 = self.spectral_adapter(x1)
            x2 = self.spectral_adapter(x2)

        with torch.no_grad():
            x1 = self._augment(x1, self.augmentation1)
            x2 = self._augment(x2, self.augmentation1)

        views = [x1, x2]
        global_views = views[:]
        if self.multicrop:
            local_views = []
            for i in range(self.n_views):
                if "image1" in batch and "image2" in batch:
                    x_raw = batch["image1"].float() if np.random.rand() > 0.5 else batch["image2"].float()
                else:
                    x_raw = batch["image"].float()
                if self.use_adapter:
                    x_raw = self.spectral_adapter(x_raw)
                local_views.append(self._augment(x_raw, self.augmentation2))
            views = global_views + local_views

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]

        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss", loss)

        with torch.no_grad():
            features = self.student_backbone(global_views[0]).flatten(start_dim=1)
            norm_features = F.normalize(features, dim=1)
            output_std = torch.std(norm_features, dim=0).mean().item()
            self.avg_output_std = 0.9 * self.avg_output_std + 0.1 * output_std
            self.log("train_ssl_std", self.avg_output_std)

        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        lr = float(self.lr)
        weight_decay = float(self.weight_decay)
        warmup_epochs = int(self.warmup_epochs)
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1 / warmup_epochs, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs),
            ],
            milestones=[warmup_epochs],
        )
        return [optimizer], [lr_scheduler]
