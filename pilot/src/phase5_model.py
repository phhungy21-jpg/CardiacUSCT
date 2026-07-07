"""Phase 5 — deliberately small model. Per protocol 5.1: with low N (150
patients), a large architecture guarantees overfitting and makes the
generalization gap (Phase 6) uninterpretable. A shallow 2D CNN operating
per-slice (through-plane already excluded from the target, see Phase 4) is
appropriate."""

import torch
import torch.nn as nn


class SmallMotionCNN(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 2, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
