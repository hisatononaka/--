# Spectral adapter from https://github.com/AABNassim/spectral_earth
# スペクトル次元で 1D 畳み込みし、ハイパースペクトル入力を 128ch に変換する。

import torch.nn as nn


class SpectralAdapter(nn.Sequential):
  """
  スペクトル次元で 1D 畳み込みを行い、ハイパースペクトル入力を 128ch に変換する。
  空間サイズは保ったまま、[B, C, H, W] → [B, 128, H, W]。
  """
  def __init__(self):
    super(SpectralAdapter, self).__init__(
      nn.Conv3d(
        1, 32, kernel_size=(7, 1, 1), stride=(5, 1, 1), padding=(1, 0, 0)
      ),
      nn.BatchNorm3d(32),
      nn.ReLU(),
      nn.Conv3d(
        32, 64, kernel_size=(7, 1, 1), stride=(5, 1, 1), padding=(1, 0, 0)
      ),
      nn.BatchNorm3d(64),
      nn.ReLU(),
      nn.Conv3d(
        64, 128, kernel_size=(5, 1, 1), stride=(3, 1, 1), padding=(1, 0, 0)
      ),
      nn.BatchNorm3d(128),
      nn.ReLU(),
      nn.AdaptiveAvgPool3d((1, None, None)),
    )

  def forward(self, x):
    """
    Args:
      x: [B, C, H, W] 例: [B, 151, H, W]。C がスペクトル次元。
    Returns:
      [B, 128, H, W]
    """
    x = x.unsqueeze(1)  # [B, 1, C, H, W]
    x = super(SpectralAdapter, self).forward(x)
    x = x.squeeze(2)    # [B, 128, H, W]
    return x
