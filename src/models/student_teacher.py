"""
DINO 用の Student / Teacher。
入力は SpectralAdapter を通した (B, 128, H, W)。Adapter は別に組み合わせる。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
  """
  DINO の projection head。
  embed_dim -> hidden -> bottleneck -> (L2 norm) -> last_layer -> out_dim
  """
  def __init__(
    self,
    in_dim,
    out_dim,
    hidden_dim=2048,
    bottleneck_dim=256,
    nlayers=3,
    use_bn=False,
    norm_last_layer=True,
  ):
    super().__init__()
    nlayers = max(nlayers, 1)
    if nlayers == 1:
      self.mlp = nn.Linear(in_dim, bottleneck_dim)
    else:
      layers = [nn.Linear(in_dim, hidden_dim)]
      if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
      layers.append(nn.GELU())
      for _ in range(nlayers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
          layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
      layers.append(nn.Linear(hidden_dim, bottleneck_dim))
      self.mlp = nn.Sequential(*layers)
    self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
    self.last_layer.weight_g.data.fill_(1)
    if norm_last_layer:
      self.last_layer.weight_g.requires_grad = False
    self._init_weights(self.mlp)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      nn.init.trunc_normal_(m.weight, std=0.02)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.mlp(x)
    x = F.normalize(x, dim=-1, p=2)
    x = self.last_layer(x)
    return x


class Backbone2D(nn.Module):
  """128ch 入力の小型 CNN。出力は (B, embed_dim)。"""
  def __init__(self, in_channels=128, embed_dim=384):
    super().__init__()
    self.embed_dim = embed_dim
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, 256, 3, stride=2, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 512, 3, stride=2, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, embed_dim, 3, stride=2, padding=1),
      nn.BatchNorm2d(embed_dim),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d(1),
    )

  def forward(self, x):
    # x: (B, 128, H, W) -> (B, embed_dim)
    x = self.conv(x)
    return x.flatten(1)


class StudentNetwork(nn.Module):
  """
  Student: adapter + backbone + head。
  forward は tensor (B, C, H, W) または list of tensors（multi-crop）を受け取り、
  各 crop の (B, out_dim) を返す。list の場合は list で返す。
  """
  def __init__(self, adapter, backbone, head):
    super().__init__()
    self.adapter = adapter
    self.backbone = backbone
    self.head = head

  def forward(self, x):
    if isinstance(x, (list, tuple)):
      return [self.forward_one(t) for t in x]
    return self.forward_one(x)

  def forward_one(self, x):
    x = self.adapter(x)
    x = self.backbone(x)
    return self.head(x)


class TeacherNetwork(nn.Module):
  """Teacher: Student と同じ構造。学習時は EMA で更新し、勾配は切る。"""
  def __init__(self, adapter, backbone, head):
    super().__init__()
    self.adapter = adapter
    self.backbone = backbone
    self.head = head
    for p in self.parameters():
      p.requires_grad = False

  def forward(self, x):
    if isinstance(x, (list, tuple)):
      return [self.forward_one(t) for t in x]
    return self.forward_one(x)

  def forward_one(self, x):
    x = self.adapter(x)
    x = self.backbone(x)
    return self.head(x)

  @torch.no_grad()
  def update_from_student(self, student, m):
    """EMA: teacher = m * teacher + (1-m) * student"""
    for pt, ps in zip(self.parameters(), student.parameters()):
      pt.data.mul_(m).add_(ps.data, alpha=1 - m)


def build_student_teacher(
  adapter_out_channels=128,
  embed_dim=384,
  out_dim=65536,
  use_adapter=True,
):
  """
  Student と Teacher を組み立てる。
  use_adapter=True なら SpectralAdapter を先頭に使う（151ch→128ch）。
  実行はプロジェクトルートから python src/train.py などで src を path に含めること。
  """
  from backbones import SpectralAdapter

  adapter = SpectralAdapter() if use_adapter else nn.Identity()
  backbone = Backbone2D(in_channels=adapter_out_channels, embed_dim=embed_dim)
  head = DINOHead(
    in_dim=embed_dim,
    out_dim=out_dim,
    hidden_dim=2048,
    bottleneck_dim=256,
    nlayers=3,
  )
  student = StudentNetwork(adapter, backbone, head)
  teacher = TeacherNetwork(
    SpectralAdapter() if use_adapter else nn.Identity(),
    Backbone2D(in_channels=adapter_out_channels, embed_dim=embed_dim),
    DINOHead(embed_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3),
  )
  teacher.load_state_dict(student.state_dict())
  for p in teacher.parameters():
    p.requires_grad = False
  return student, teacher
