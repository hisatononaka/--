import torch
import utils

def batch(image, config=None, config_path=None):
  """1枚の画像から multi-crop。戻り値の各 tensor は (C,H,W)。"""
  if config is None:
    config = utils.load_config(config_path)
  global_crops = utils.crops(image, config["global"])
  local_crops = utils.crops(image, config["local"])
  return {"global": global_crops, "local": local_crops}


def make_crop_batch(images, config=None, config_path=None):
  """
  B枚の元画像から DINO 用の crop バッチを組む。
  戻り値の各 tensor は (B, C, H, W)。B=len(images)。

  Args:
    images: list of B tensors、各 (C, H, W)
    config: 省略時は config_path から読み込み。両方 None ならプロジェクトルートの config を使用。
  Returns:
    {"global": [tensor(B,C,224,224), tensor(B,C,224,224)],
     "local": [tensor(B,C,96,96), ...]}
  """
  if config is None:
    config = utils.load_config(config_path)
  B = len(images)
  n_global = config["global"]["num"]
  n_local = config["local"]["num"]
  global_stacks = [[] for _ in range(n_global)]
  local_stacks = [[] for _ in range(n_local)]
  for i in range(B):
    g_list = utils.crops(images[i], config["global"])
    l_list = utils.crops(images[i], config["local"])
    for j in range(n_global):
      global_stacks[j].append(g_list[j])
    for j in range(n_local):
      local_stacks[j].append(l_list[j])
  return {
    "global": [torch.stack(s, dim=0) for s in global_stacks],
    "local": [torch.stack(s, dim=0) for s in local_stacks],
  }


def apply_spectral_adapter(crops_dict, adapter):
  """
  学習前に 151ch → 128ch に変換する。
  spectral_earth の 1D SpectralAdapter を挟み、各 crop を [*, 128, H, W] に変換する。

  Args:
    crops_dict: batch() の戻り値 {"global": [tensor,...], "local": [tensor,...]}。各 tensor は (C,H,W) または (B,C,H,W)。
    adapter: SpectralAdapter のインスタンス（backbones.SpectralAdapter）

  Returns:
    同じキー構成で、各 tensor が (128,H,W) または (B,128,H,W) になった dict。
  """
  out = {}
  for key, tensors in crops_dict.items():
    out[key] = [_run_adapter(t, adapter) for t in tensors]
  return out


def _run_adapter(t, adapter):
  if t.dim() == 3:
    return adapter(t.unsqueeze(0)).squeeze(0)
  return adapter(t)