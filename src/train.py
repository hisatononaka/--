"""
DINO 事前学習: 100 epoch で student/teacher を学習する。
実行: cd src && python train.py  または プロジェクトルートで python src/train.py
"""
import glob
import os
import sys

# プロジェクトルートから python src/train.py で実行したとき src を path に追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

import utils
import dino
from dataset import H5Dataset
from models import build_student_teacher
from models.dino_loss import DINOLoss


def _project_root():
  return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_data_path(project_root, raw_data_path):
  """
  data_path を H5Dataset に渡す形に解決する。
  - 文字列でディレクトリ → その下の全 .h5（サブフォルダ含む）のリスト
  - 文字列でファイル → そのパス 1 つ
  - リスト → 各要素を project_root 基準で結合したリスト
  """
  if isinstance(raw_data_path, (list, tuple)):
    return [os.path.join(project_root, p) for p in raw_data_path]
  path = os.path.join(project_root, raw_data_path)
  if os.path.isdir(path):
    files = glob.glob(os.path.join(path, "**", "*.h5"), recursive=True)
    return sorted(files)
  return path


def _cosine_schedule(base_value, final_value, epochs, niter_per_epoch, warmup_epochs=0):
  """epoch ごとの値の cosine schedule。warmup あり。"""
  warmup_iters = warmup_epochs * niter_per_epoch
  total_iters = epochs * niter_per_epoch
  schedule = []
  for i in range(total_iters):
    if i < warmup_iters:
      schedule.append(base_value + (final_value - base_value) * i / warmup_iters)
    else:
      progress = (i - warmup_iters) / (total_iters - warmup_iters)
      schedule.append(final_value + 0.5 * (base_value - final_value) * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))
  return schedule


def main():
  project_root = _project_root()
  config_path = os.path.join(project_root, "param", "config.yaml")
  config = utils.load_config(config_path)

  batch_size = config["batch_size"]
  train_cfg = config.get("train", {})
  raw_data_path = train_cfg.get("data_path", "data/2024820.h5")
  data_path = _resolve_data_path(project_root, raw_data_path)
  if isinstance(data_path, list) and len(data_path) == 0:
    raise FileNotFoundError(
      f"data_path に .h5 が 1 件もありません: {raw_data_path}"
    )
  epochs = train_cfg.get("epochs", 100)
  lr = train_cfg.get("lr", 0.0005)
  min_lr = train_cfg.get("min_lr", 1e-6)
  warmup_epochs = train_cfg.get("warmup_epochs", 10)
  weight_decay = train_cfg.get("weight_decay", 0.04)
  momentum_teacher = train_cfg.get("momentum_teacher", 0.996)
  teacher_temp = train_cfg.get("teacher_temp", 0.04)
  warmup_teacher_temp = train_cfg.get("warmup_teacher_temp", 0.04)
  warmup_teacher_temp_epochs = train_cfg.get("warmup_teacher_temp_epochs", 0)
  num_workers = train_cfg.get("num_workers", 0)

  model_cfg = config.get("model", {})
  embed_dim = model_cfg.get("embed_dim", 384)
  out_dim = model_cfg.get("out_dim", 65536)
  use_adapter = model_cfg.get("use_adapter", True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("device:", device)

  dataset = H5Dataset(data_path, as_tensor=True)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=len(dataset) >= batch_size,
    collate_fn=lambda x: x,
  )
  niter_per_epoch = len(dataloader)

  student, teacher = build_student_teacher(
    embed_dim=embed_dim,
    out_dim=out_dim,
    use_adapter=use_adapter,
  )
  student = student.to(device)
  teacher = teacher.to(device)

  n_params = sum(p.numel() for p in student.parameters())
  n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)

  ncrops = config["global"]["num"] + config["local"]["num"]
  dino_loss = DINOLoss(
    out_dim=out_dim,
    ncrops=ncrops,
    warmup_teacher_temp=warmup_teacher_temp,
    teacher_temp=teacher_temp,
    warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
    nepochs=epochs,
  ).to(device)

  optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=lr,
    weight_decay=weight_decay,
  )

  lr_schedule = _cosine_schedule(lr, min_lr, epochs, niter_per_epoch, warmup_epochs)
  momentum_schedule = _cosine_schedule(momentum_teacher, 1.0, epochs, niter_per_epoch, warmup_epochs=0)

  global_step = 0
  for epoch in range(epochs):
    student.train()
    teacher.train()
    epoch_loss = 0.0
    n_batches = 0
    for batch_images in dataloader:
      for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedule[global_step]
      m = momentum_schedule[global_step]

      crops_batch = dino.make_crop_batch(batch_images, config=config)
      all_crops = [t.to(device) for t in crops_batch["global"]] + [t.to(device) for t in crops_batch["local"]]
      global_crops = [t.to(device) for t in crops_batch["global"]]

      student_out = student(all_crops)
      with torch.no_grad():
        teacher_out = teacher(global_crops)

      loss = dino_loss(student_out, teacher_out, epoch)
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
      optimizer.step()

      with torch.no_grad():
        teacher.update_from_student(student, m)

      epoch_loss += loss.item()
      n_batches += 1
      global_step += 1

    avg_loss = epoch_loss / n_batches if n_batches else 0.0
    current_lr = lr_schedule[global_step - 1] if global_step else lr
    print(f"epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  lr={current_lr:.2e}")

  print("done.")
  ckpt_path = os.path.join(project_root, "checkpoint_dino.pth")
  torch.save(
    {
      "student": student.state_dict(),
      "teacher": teacher.state_dict(),
      "epoch": epochs,
      "config": config,
    },
    ckpt_path,
  )
  print("saved:", ckpt_path)

  # teacher の重みのみを別ファイルで保存（特徴抽出などで利用しやすい）
  teacher_path = os.path.join(project_root, "teacher_weights.pth")
  torch.save(
    {
      "teacher": teacher.state_dict(),
      "epoch": epochs,
      "config": config,
    },
    teacher_path,
  )
  print("saved teacher:", teacher_path)


if __name__ == "__main__":
  main()
