"""
models/cnn1d.py
---------------
Baseline 1D Convolutional Neural Network for ECG classification.

Architecture:
  5 × ConvBlock(Conv1d → BatchNorm → ReLU → MaxPool)
  kernel sizes: 3, 5, 7, 9, 11
  → GlobalAveragePooling
  → FC(256) → Dropout(0.5) → FC(64) → Dropout(0.3) → FC(num_classes)
"""

import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1D(nn.Module):
    """
    Parameters
    ----------
    in_channels : int
        Number of input leads (1 for single-lead ECG).
    num_classes : int
        Number of output classes (4 for PhysioNet 2017).
    base_filters : int
        Number of filters in the first convolutional block; doubled each block.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_filters: int = 32,
    ) -> None:
        super().__init__()

        kernels:  List[int] = [3, 5, 7, 9, 11]
        channels: List[int] = [
            in_channels,
            base_filters,
            base_filters * 2,
            base_filters * 4,
            base_filters * 8,
            base_filters * 16,
        ]

        self.conv_blocks = nn.Sequential(
            *[
                ConvBlock(channels[i], channels[i + 1], kernels[i])
                for i in range(len(kernels))
            ]
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, channels, time)
        Returns
        -------
        logits : (batch, num_classes)
        """
        x = self.conv_blocks(x)       # (B, C, T')
        x = self.gap(x).squeeze(-1)   # (B, C)
        return self.classifier(x)

    def get_cam_target_layer(self) -> nn.Module:
        """Return the last Conv layer for Grad-CAM."""
        return self.conv_blocks[-1].block[0]


if __name__ == "__main__":
    model = CNN1D(in_channels=1, num_classes=4)
    x = torch.randn(8, 1, 3000)
    out = model(x)
    print("CNN1D output shape:", out.shape)   # (8, 4)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
