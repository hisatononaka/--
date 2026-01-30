import torch


class NormalizeMeanStd(torch.nn.Module):
    """平均・標準偏差で正規化。x = (x - mean) / std。mean, std は (C,) で channel 次元にブロードキャスト。"""
    def __init__(self, mean, std):
        super().__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor):
            std = torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std", std.view(1, -1, 1, 1).clamp(min=1e-6))

    def forward(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std
