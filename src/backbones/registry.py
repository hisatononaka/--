"""
Backbone のレジストリ。DINOModule が backbone_name で取得する。
"""
import timm

from .vit import ViTSmall, ViTTiny

BACKBONE_REGISTRY = {}


def _register(name, fn):
    BACKBONE_REGISTRY[name] = fn


def _spec_resnet50(num_classes=0, in_chans=202, **kwargs):
    m = timm.create_model(
        "resnet50", num_classes=num_classes, in_chans=in_chans, pretrained=False, **kwargs
    )
    if not hasattr(m, "num_features"):
        m.num_features = 2048
    return m


_register("spec_resnet50", _spec_resnet50)


def _vit_small(num_classes=0, token_patch_size=8, **kwargs):
    return ViTSmall(num_classes=num_classes, token_patch_size=token_patch_size, **kwargs)


def _vit_tiny(num_classes=0, token_patch_size=8, **kwargs):
    return ViTTiny(num_classes=num_classes, token_patch_size=token_patch_size, **kwargs)


_register("vit_small", _vit_small)
_register("vit_tiny", _vit_tiny)
