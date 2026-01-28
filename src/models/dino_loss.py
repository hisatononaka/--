"""
DINO Loss: teacher の centering + temperature sharpening、student との cross-entropy。
同じ view 同士は比較しない。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
  def __init__(
    self,
    out_dim,
    ncrops,
    warmup_teacher_temp=0.04,
    teacher_temp=0.04,
    warmup_teacher_temp_epochs=0,
    nepochs=100,
    student_temp=0.1,
    center_momentum=0.9,
  ):
    super().__init__()
    self.student_temp = student_temp
    self.center_momentum = center_momentum
    self.ncrops = ncrops
    self.register_buffer("center", torch.zeros(1, out_dim))
    # teacher temperature schedule: warmup then constant
    teacher_temp_schedule = torch.cat(
      (
        torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
        torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
      )
    )
    self.register_buffer("teacher_temp_schedule", teacher_temp_schedule)

  def forward(self, student_output, teacher_output, epoch):
    """
    student_output: list of (B, out_dim), length = ncrops (2 global + 6 local)
    teacher_output: list of (B, out_dim), length = 2 (global only)
    """
    student_out = torch.cat(student_output, dim=0) / self.student_temp
    student_out = student_out.chunk(self.ncrops)

    temp = self.teacher_temp_schedule[epoch].item()
    teacher_out = torch.cat(teacher_output, dim=0)
    teacher_out = F.softmax((teacher_out - self.center) / temp, dim=-1)
    teacher_out = teacher_out.detach().chunk(2)

    total_loss = 0.0
    n_terms = 0
    for iq, q in enumerate(teacher_out):
      for v, s in enumerate(student_out):
        if v == iq:
          continue
        loss = torch.sum(-q * F.log_softmax(s, dim=-1), dim=-1)
        total_loss += loss.mean()
        n_terms += 1
    total_loss /= n_terms
    self.update_center(teacher_output)
    return total_loss

  @torch.no_grad()
  def update_center(self, teacher_output):
    """teacher 出力の running average で center を更新。"""
    batch_center = torch.cat(teacher_output, dim=0).mean(dim=0, keepdim=True)
    self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
