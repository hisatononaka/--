import os
from torchvision import transforms
import yaml


def get_project_root():
  """utils が src/ にある前提で、プロジェクトルートの絶対パスを返す。"""
  return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _prob_for_crop(value, crop_index, num_crops):
  """値がない→None。配列→numごとに異なる(インデックスで取得)。スカラ→全cropで同じ。"""
  if value is None:
    return None
  if isinstance(value, (list, tuple)):
    if crop_index < len(value):
      return value[crop_index]
    return value[-1] if value else None
  return value


def _build_transforms_for_crop(transforms_dict, crop_index, num_crops):
  """
  config["transforms"] から crop_index 用の Compose を組み立てる。
  - キーが無い/値が無い: その拡張はかけない
  - 配列: num ごとに異なる（値は確率）
  - スカラ: 全 num で同じ（値は確率）
  - gaussian_blur: 確率 p、強さは uniform(0.1, 2.0)
  - flip: 確率 p で水平反転
  """
  if not transforms_dict:
    return None
  out = []
  for key, value in transforms_dict.items():
    p = _prob_for_crop(value, crop_index, num_crops)
    if p is None:
      continue
    if key == "gaussian_blur":
      out.append(
        transforms.RandomApply(
          [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
          p=p,
        )
      )
    elif key == "flip":
      out.append(transforms.RandomHorizontalFlip(p=p))
  return transforms.Compose(out) if out else None


def crops(image, config):
  """config: crop_scale, size, num, transforms(optional)。transforms の値は確率。"""
  num = config["num"]
  crops_out = []
  base = transforms.RandomResizedCrop(
    config["size"],
    scale=tuple(config["crop_scale"]),
    interpolation=transforms.InterpolationMode.BICUBIC,
  )
  for i in range(num):
    img = base(image)
    extra = _build_transforms_for_crop(
      config.get("transforms"), i, num
    )
    crops_out.append(extra(img) if extra else img)
  return crops_out

def load_config(config_path=None):
  """
  YAML 設定を読み込む。
  config_path が None のときはプロジェクトルートの param/config.yaml を使う（CWD に依存しない）。
  """
  if config_path is None:
    config_path = os.path.join(get_project_root(), "param", "config.yaml")
  with open(config_path, "r") as yml:
    return yaml.safe_load(yml)