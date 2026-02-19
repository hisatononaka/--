"""timm の VisionTransformer を用いた ViT-Small / ViT-Tiny ラッパー。

- patch_size: 入力画像の空間サイズ（H=W）。config の global_size や img_size に対応。
- token_patch_size: 1 patch のピクセル数（4, 8, 16 など）。timm の patch_size に渡す。
- dynamic_img_size: True にすると可変解像度に対応（position embedding を補間）。
"""

from timm.models.vision_transformer import VisionTransformer


def ViTSmall(
    patch_size=128,
    token_patch_size=8,
    in_chans=202,
    num_classes=0,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    global_pool="token",
    dynamic_img_size=False,
    **kwargs,
):
    return VisionTransformer(
        img_size=patch_size,
        patch_size=token_patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
        dynamic_img_size=dynamic_img_size,
        **kwargs,
    )


def ViTTiny(
    patch_size=128,
    token_patch_size=8,
    in_chans=202,
    num_classes=0,
    embed_dim=192,
    depth=12,
    num_heads=3,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    global_pool="token",
    dynamic_img_size=False,
    **kwargs,
):
    return VisionTransformer(
        img_size=patch_size,
        patch_size=token_patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
        dynamic_img_size=dynamic_img_size,
        **kwargs,
    )
